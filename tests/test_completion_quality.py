"""Regression tests for the completion-quality fixes (see 00_AGENT_COMPLETION_QUALITY_TESTING.md).

Covers: persistent working directory across tools, the continuation guard, finish-reason handling,
per-call vs. per-turn completion events, partial-turn preservation, parallel tool-result pairing,
compaction digest placement, and the Gemini provider's thinking config / valstorm pass-through fallback.
"""

import os
from typing import Any, Dict, List, Optional

import pytest

from core.models import Message, SessionState, StreamEventType, ToolCall, ToolResult, UsageMetadata
from core.react import ReActEngine


# --------------------------------------------------------------------------- helpers
class _Registry:
    def __init__(self, names=("terminal_exec", "read_file")):
        self.names = list(names)
        self.calls: List[str] = []

    def get_schemas(self) -> List[Dict[str, Any]]:
        return [{"name": n, "description": n, "parameters": {"type": "object", "properties": {}}} for n in self.names]

    def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        self.calls.append(name)
        return "ok"


class _Provider:
    def __init__(self, responses: List[Message]):
        self.responses = list(responses)
        self.seen: List[List[Message]] = []

    async def generate(self, messages, tools=None, model=None):
        self.seen.append(list(messages))
        if self.responses:
            return self.responses.pop(0)
        return Message(role="assistant", content="All done and verified.")


async def _run(engine: ReActEngine, prompt: str, session: Optional[SessionState] = None):
    session = session or SessionState(active_model="m", active_provider="p")
    events = []
    async for ev in engine.run_turn_stream(session=session, user_input=prompt, model="m"):
        events.append(ev)
    return session, events


# --------------------------------------------------------------------------- finish reasons
@pytest.mark.asyncio
async def test_truncated_output_asks_model_to_continue():
    provider = _Provider([
        Message(role="assistant", content="Here is the first half of", finish_reason="length"),
        Message(role="assistant", content="the answer, completed."),
    ])
    engine = ReActEngine(provider=provider, tools=_Registry())
    session, _ = await _run(engine, "Explain")
    assert len(provider.seen) == 2
    assert any("cut off" in (m.content or "") for m in session.messages if m.role == "user")


@pytest.mark.asyncio
async def test_malformed_tool_call_is_retried():
    provider = _Provider([
        Message(role="assistant", content=None, finish_reason="malformed_tool_call"),
        Message(role="assistant", content="Recovered."),
    ])
    engine = ReActEngine(provider=provider, tools=_Registry())
    session, _ = await _run(engine, "Do it")
    assert len(provider.seen) == 2
    assert any("malformed" in (m.content or "") for m in session.messages if m.role == "user")


# --------------------------------------------------------------------------- event labelling
@pytest.mark.asyncio
async def test_provider_turn_complete_is_relabelled_per_call():
    from core.models import StreamEvent

    class _StreamProvider:
        def __init__(self):
            self.n = 0

        async def generate_stream(self, messages, tools=None, model=None):
            self.n += 1
            if self.n == 1:
                msg = Message(role="assistant", tool_calls=[ToolCall(id="c1", name="read_file", arguments={})])
            else:
                msg = Message(role="assistant", content="Finished.")
            yield StreamEvent(event_type=StreamEventType.TURN_COMPLETE, message=msg)

    engine = ReActEngine(provider=_StreamProvider(), tools=_Registry())
    _, events = await _run(engine, "go")
    kinds = [e.event_type for e in events]
    assert kinds.count(StreamEventType.TURN_COMPLETE) == 1
    assert kinds.count(StreamEventType.LLM_CALL_COMPLETE) == 2
    assert kinds[-1] == StreamEventType.TURN_COMPLETE


# --------------------------------------------------------------------------- sandbox cwd
@pytest.mark.asyncio
async def test_cd_persists_and_file_tools_follow(tmp_path):
    from core.sandbox import HostSandbox, set_current_sandbox
    from tools.developer_tools import patch_file, read_file, terminal_exec

    sub = tmp_path / "bench" / "pkg"
    sub.mkdir(parents=True)
    (sub / "mod.py").write_text("x = 1\n")
    sb = HostSandbox(base_dir=str(tmp_path))
    token = set_current_sandbox(sb)
    try:
        out = await terminal_exec(command="cd bench/pkg && echo hi")
        assert "hi" in out
        assert "Working directory is now" in out
        assert "__VSAGENT_CWD__" not in out
        assert sb.cwd == sub.resolve()

        pwd_out = await terminal_exec(command="pwd")
        assert str(sub.resolve()) in pwd_out
        assert "Working directory is now" not in pwd_out

        res = await patch_file(path="mod.py", old_string="x = 1", new_string="x = 2")
        assert str((sub / "mod.py").resolve()) in res
        assert (sub / "mod.py").read_text() == "x = 2\n"

        content = await read_file(path="mod.py")
        assert str((sub / "mod.py").resolve()) in content

        fail = await terminal_exec(command="exit 3")
        assert fail.startswith("Command exited with code 3")
    finally:
        from core.sandbox import _current_sandbox
        _current_sandbox.reset(token)


# --------------------------------------------------------------------------- partial turns
def test_preserve_partial_turn_keeps_completed_work():
    from cli.helpers import preserve_partial_turn

    s = SessionState(active_model="m", active_provider="p")
    s.add_message(Message(role="system", content="sys"))
    initial = len(s.messages)
    s.add_message(Message(role="user", content="do it"))
    s.add_message(Message(role="assistant", tool_calls=[ToolCall(id="a", name="read_file")]))
    s.add_message(Message(role="tool", content="r", tool_result=ToolResult(call_id="a", name="read_file", output="r")))
    s.add_message(Message(role="assistant", tool_calls=[ToolCall(id="b", name="terminal_exec"), ToolCall(id="c", name="read_file")]))
    s.add_message(Message(role="tool", content="r", tool_result=ToolResult(call_id="b", name="terminal_exec", output="r")))

    preserve_partial_turn(s, initial, "API error: boom")
    roles = [m.role for m in s.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert "interrupted" in s.messages[-1].content


# --------------------------------------------------------------------------- openai pairing
def test_openai_parallel_tool_results_stay_tool_messages():
    from providers.openai import OpenAIProvider

    p = OpenAIProvider(api_key="k", base_url="http://x", provider_name="valstorm")
    msgs = [
        Message(role="user", content="go"),
        Message(role="assistant", tool_calls=[ToolCall(id=f"t{i}", name="read_file") for i in range(3)]),
    ] + [
        Message(role="tool", content=f"r{i}", tool_result=ToolResult(call_id=f"t{i}", name="read_file", output=f"r{i}"))
        for i in range(3)
    ]
    out = p._format_messages(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "tool", "tool", "tool"]


# --------------------------------------------------------------------------- compaction
def test_compaction_digest_is_not_a_second_system_message():
    from core.compaction import ContextCompactor

    s = SessionState(active_model="gemini-flash-latest", active_provider="valstorm")
    s.add_message(Message(role="system", content="REAL SYSTEM PROMPT"))
    for i in range(40):
        s.add_message(Message(role="user", content=f"question {i} " + "x" * 2000))
        s.add_message(Message(role="assistant", content=f"answer {i} " + "y" * 2000))
    res = ContextCompactor().compact(s, target_tokens=5000)
    assert res.compacted
    systems = [m for m in s.messages if m.role == "system"]
    assert len(systems) == 1 and systems[0].content == "REAL SYSTEM PROMPT"
    assert any("HISTORICAL CONTEXT DIGEST" in (m.content or "") for m in s.messages if m.role == "user")


# --------------------------------------------------------------------------- gemini provider
def test_gemini_thinking_config_default_and_off(monkeypatch):
    from providers.gemini import GeminiProvider

    monkeypatch.delenv("VALSTORM_THINKING_LEVEL", raising=False)
    p = GeminiProvider(api_key="k", backend="aistudio")
    cfg = p._build_config("sys", None, {})
    assert cfg.thinking_config is not None

    monkeypatch.setenv("VALSTORM_THINKING_LEVEL", "off")
    cfg2 = p._build_config("sys", None, {})
    assert cfg2.thinking_config is None


def test_gemini_backend_env_override(monkeypatch):
    from providers.gemini import GeminiProvider

    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    assert GeminiProvider.is_enterprise_mode() is True
    monkeypatch.setenv("VALSTORM_GEMINI_BACKEND", "aistudio")
    assert GeminiProvider.is_enterprise_mode() is False


@pytest.mark.asyncio
async def test_valstorm_passthrough_falls_back_to_openai_gateway_on_404():
    from providers.gemini import GeminiProvider

    class _NotFound(Exception):
        code = 404

    class _Models:
        async def generate_content_stream(self, **kw):
            raise _NotFound("404 Not Found")

    class _Aio:
        models = _Models()

    class _Client:
        aio = _Aio()

    class _Compat:
        called = 0

        def set_request_context(self, **kw):
            pass

        async def generate_stream(self, messages, tools=None, model=None, **kw):
            _Compat.called += 1
            from core.models import StreamEvent
            yield StreamEvent(event_type=StreamEventType.TURN_COMPLETE, message=Message(role="assistant", content="via compat"))

    p = GeminiProvider(api_key="tok", backend="valstorm", valstorm_base_url="http://x/v1/ai/gemini",
                       client=_Client(), compat_provider=_Compat())
    items = [i async for i in p.generate_stream(messages=[Message(role="user", content="hi")])]
    assert _Compat.called == 1
    assert p._passthrough_unavailable is True
    assert p.backend_label() == "valstorm-compat"
    assert any(getattr(i, "message", None) and i.message.content == "via compat" for i in items)
