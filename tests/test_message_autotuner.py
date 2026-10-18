"""Unit, integration, and benchmark tests for `ai_chat_message` Auto-Tuning (`core/message_autotuner.py`)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from httpx import ASGITransport, AsyncClient
import pytest

from benchmarks.runner import get_ablation_configs, load_tasks, run_benchmark_matrix
from core.message_autotuner import MessageAutoTuner, resolve_tier_model
from core.models import Message, SessionState, ToolCall, ToolResult, UsageMetadata
from core.react import ReActEngine
from core.tools import ToolRegistry, tool
from server import app


class TestMessageAutoTunerUnit:
    """Tests for pre-turn tier/effort routing and intra-turn cache-safe stepping."""

    def test_pre_turn_tier_and_effort_routing(self):
        tuner = MessageAutoTuner()

        # 1. Trivial / Worker class (G-S01)
        d_worker = tuner.tune_turn(
            "Calculate (42 * 3.14159) / 7 and write to calculation_result.txt",
            base_provider="gemini",
        )
        assert d_worker.phase == "pre_turn"
        assert d_worker.tier == "worker"
        assert d_worker.model == "gemini-flash-lite-latest"
        assert d_worker.reasoning_effort == "none"

        # 2. Moderate / Reasoner class (G-M03)
        d_reasoner = tuner.tune_turn(
            "Use patch_file to update http://api.old.com/v1 to http://api.new.com/v2 in config.json",
            base_provider="gemini",
        )
        assert d_reasoner.tier == "reasoner"
        assert d_reasoner.model == "gemini-flash-latest"
        assert d_reasoner.reasoning_effort == "medium"

        # 3. Complex / Thinker class (G-H03 shell script hazard & G-H01 TDD)
        d_thinker = tuner.tune_turn(
            "Write a shell script setup_env.sh that runs git init, set executable permissions with chmod +x, and execute it.",
            base_provider="gemini",
        )
        assert d_thinker.tier == "thinker"
        assert d_thinker.model == "gemini-pro-latest"
        assert d_thinker.reasoning_effort == "high"

    def test_provider_family_tier_mapping(self):
        assert resolve_tier_model("do", "worker") == "deepseek-4-flash"
        assert resolve_tier_model("do", "reasoner") == "openai-gpt-oss-120b"
        assert resolve_tier_model("do", "thinker") == "anthropic-claude-opus-5"
        assert resolve_tier_model("openai", "worker") == "gpt-4o-mini"
        assert resolve_tier_model("anthropic", "thinker") == "claude-3-opus-20240229"

    def test_intra_turn_kv_cache_model_locking_and_effort_stepping(self):
        tuner = MessageAutoTuner()
        session = SessionState(session_id="sess_cache_lock", active_provider="gemini")

        # Simulate clean tool execution on iteration 0
        session.add_message(
            Message(
                role="tool",
                content="Successfully patched config.json",
                tool_result=ToolResult(call_id="c1", name="patch_file", output="OK", is_error=False),
            )
        )
        d_step_down = tuner.tune_iteration(
            user_input="Use patch_file to update config.json",
            session=session,
            iteration=1,
            locked_tier="reasoner",
            locked_model="gemini-flash-latest",
        )
        assert d_step_down.phase == "intra_turn"
        assert d_step_down.cache_locked_model is True
        assert d_step_down.model == "gemini-flash-latest"
        assert d_step_down.reasoning_effort == "none"
        assert d_step_down.stepped_down_post_tool is True

        # Now simulate a tool error on iteration 1 -> must step UP to 'high' while keeping model locked
        session.add_message(
            Message(
                role="tool",
                content="Error executing tool",
                tool_result=ToolResult(call_id="c2", name="terminal_exec", output="Exit code 1", is_error=True),
            )
        )
        d_step_up = tuner.tune_iteration(
            user_input="Run script",
            session=session,
            iteration=2,
            locked_tier="worker",
            locked_model="gemini-flash-lite-latest",
        )
        assert d_step_up.cache_locked_model is True
        assert d_step_up.model == "gemini-flash-lite-latest"
        assert d_step_up.reasoning_effort == "high"
        assert d_step_up.escalated_on_error is True


class _RecordingProvider:
    """Mock provider that records `(model, reasoning_effort)` on every call."""

    provider_name = "gemini"

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self._effort = "medium"

    def set_request_context(self, **kwargs):
        if "reasoning_effort" in kwargs:
            self._effort = kwargs["reasoning_effort"]

    async def generate(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, model: Optional[str] = None, **kwargs):
        call_idx = len(self.calls)
        self.calls.append({"model": model, "reasoning_effort": self._effort})
        usage = UsageMetadata(prompt_tokens=100, completion_tokens=20, total_tokens=120)
        # On call_idx == 1 (start of Turn 2), emit a tool call so Turn 2 has 2 iterations (iter 0 and iter 1)
        if call_idx == 1:
            tc = ToolCall(id="tc_1", name="dummy_ping", arguments={"x": "ok"})
            return Message(role="assistant", content="", tool_calls=[tc], usage=usage), usage
        return Message(role="assistant", content=f"Done ({model}/{self._effort})", usage=usage), usage


class TestReActAndServerAutoTuneIntegration:
    """Tests multi-turn `ai_chat_message` model stamping and local daemon endpoint."""

    @pytest.mark.asyncio
    async def test_multiturn_session_stamps_each_ai_chat_message_with_tuned_model(self):
        reg = ToolRegistry()

        @tool
        def dummy_ping(x: str) -> str:
            """Ping helper."""
            return f"pong:{x}"

        reg.register(dummy_ping)
        prov = _RecordingProvider()
        engine = ReActEngine(provider=prov, tools=reg)
        session = SessionState(session_id="aich_multiturn_test", active_provider="gemini", active_model="gemini-flash-latest")

        # Turn 1: Trivial greeting -> Worker (gemini-flash-lite-latest, none)
        async for _ in engine.run_turn_stream(session=session, user_input="Hello! What status is the system?", auto_tune=True):
            pass

        # Turn 2: Moderate patch task (2 iterations: iter 0 on medium, iter 1 stepped down to none, both locked to gemini-flash-latest)
        async for _ in engine.run_turn_stream(session=session, user_input="Use patch_file to update the endpoint in config.json", auto_tune=True):
            pass

        # Turn 3: High complexity shell script hazard -> Thinker (gemini-pro-latest, high)
        async for _ in engine.run_turn_stream(session=session, user_input="Write a shell script setup_env.sh with git init and chmod +x", auto_tune=True):
            pass

        assert len(prov.calls) == 4
        # Call 0 (Turn 1, iter 0): Worker + none
        assert prov.calls[0] == {"model": "gemini-flash-lite-latest", "reasoning_effort": "none"}
        # Call 1 (Turn 2, iter 0): Reasoner + medium
        assert prov.calls[1] == {"model": "gemini-flash-latest", "reasoning_effort": "medium"}
        # Call 2 (Turn 2, iter 1 after dummy_ping tool): Reasoner (locked!) + none (stepped down!)
        assert prov.calls[2] == {"model": "gemini-flash-latest", "reasoning_effort": "none"}
        # Call 3 (Turn 3, iter 0): Thinker + high
        assert prov.calls[3] == {"model": "gemini-pro-latest", "reasoning_effort": "high"}

        asst_models = [m.model for m in session.messages if m.role == "assistant"]
        assert asst_models == [
            "gemini-flash-lite-latest",
            "gemini-flash-latest",
            "gemini-flash-latest",
            "gemini-pro-latest",
        ]

    @pytest.mark.asyncio
    async def test_daemon_autotune_message_endpoint(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp0 = await ac.post(
                "/v1/system-one/autotune-message",
                json={
                    "user_input": "Calculate (42 * 3.14159) / 7",
                    "provider": "gemini",
                    "iteration": 0,
                },
            )
            assert resp0.status_code == 200
            d0 = resp0.json()
            assert d0["tier"] == "worker"
            assert d0["model"] == "gemini-flash-lite-latest"
            assert d0["reasoning_effort"] == "none"

            resp1 = await ac.post(
                "/v1/system-one/autotune-message",
                json={
                    "user_input": "Run script",
                    "provider": "gemini",
                    "iteration": 1,
                    "locked_tier": "worker",
                    "locked_model": "gemini-flash-lite-latest",
                    "last_tool_name": "terminal_exec",
                    "last_tool_error": True,
                },
            )
            assert resp1.status_code == 200
            d1 = resp1.json()
            assert d1["cache_locked_model"] is True
            assert d1["model"] == "gemini-flash-lite-latest"
            assert d1["reasoning_effort"] == "high"
            assert d1["escalated_on_error"] is True

    @pytest.mark.asyncio
    async def test_benchmark_message_autotune_ablation_matrix(self):
        """Verify `autotune_s1_full` achieves 100% pass rate at lower latency and cost than static Thinker/Reasoner."""
        core_tasks = load_tasks(suite="core")
        configs = get_ablation_configs("message_autotune")
        assert [c.config_id for c in configs] == [
            "autotune_s1_full",
            "static_thinker_high",
            "static_reasoner_high",
            "static_worker_none",
        ]

        report = await run_benchmark_matrix(tasks=core_tasks, configs=configs, mode="mock", trials=1)
        summaries = {s["config_id"]: s for s in report["config_summaries"]}

        s_auto = summaries["autotune_s1_full"]
        s_thinker = summaries["static_thinker_high"]
        s_reasoner = summaries["static_reasoner_high"]
        s_worker = summaries["static_worker_none"]

        # 1. Auto-Tuner matches Thinker & Reasoner's 100% pass rate (9/9), whereas Static Worker fails G-H03 (88.9%)
        assert s_auto["pass_rate"] == 1.0
        assert s_thinker["pass_rate"] == 1.0
        assert s_reasoner["pass_rate"] == 1.0
        assert s_worker["pass_rate"] < 1.0

        # 2. Auto-Tuner is faster and cheaper than both Static Thinker and Static Reasoner @ HIGH
        assert s_auto["avg_duration_sec"] < s_thinker["avg_duration_sec"]
        assert s_auto["avg_duration_sec"] < s_reasoner["avg_duration_sec"]
        assert s_auto["total_cost_usd"] < s_thinker["total_cost_usd"]
        assert s_auto["avg_tokens"] < s_thinker["avg_tokens"]
