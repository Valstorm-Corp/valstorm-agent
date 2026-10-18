"""In-process Valstorm Agent Runtime (`vsagent`) Benchmark Adapter.

Executes benchmark tasks inside an isolated temporary HostSandbox directory while
varying any combination of the 10 ablation factors:
  - Skills (`with_skills=True/False`)
  - ReAct Loop (`with_react_loop=True/False`)
  - Tool Surface (`tools_mode='none' | 'scoped' | 'full'`)
  - Model & Reasoning Level (`provider`, `model`, `thinking_level`)
  - Swarm / Parallel Delegation (`swarm_mode=True/False`)
"""

import asyncio
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Literal, Optional
import uuid

from core.context import WorkspaceContextManager, load_profile
from core.models import Message, SessionState, StreamEventType, ToolCall, UsageMetadata
from core.react import ReActEngine
from core.sandbox import HostSandbox, get_current_sandbox, set_current_sandbox
from core.tools import ToolRegistry, get_default_registry
from tools import register_developer_tools, register_memory_tools, register_tier1_tools, register_ui_navigator_tools
from tools.delegation import SubagentRegistry, _get_subagent_provider, set_subagent_provider_override

from .base import BenchmarkAdapter, BenchmarkConfig, ExecutionTrace, estimate_cost_usd


def build_benchmark_tool_registry(tools_mode: str, task: Dict[str, Any], swarm_mode: bool = False) -> ToolRegistry:
    """Builds the ToolRegistry according to the `tools_mode` ablation knob."""
    if tools_mode == "none":
        return ToolRegistry()

    base = get_default_registry()
    register_developer_tools(base)
    register_tier1_tools(base)
    register_ui_navigator_tools(base)
    register_memory_tools(base)

    if tools_mode == "scoped":
        whitelist = list(task.get("tool_chain") or [])
        if swarm_mode:
            for extra in ("delegate_task", "subagent_manage"):
                if extra not in whitelist:
                    whitelist.append(extra)
        if whitelist:
            return base.filter_by_whitelist(whitelist)
    return base


class TaskAwareMockProvider:
    """Deterministic mock provider that simulates realistic ReAct trajectories and ablation physics.

    Faithfully models:
    - Tool availability (`tools_mode='none'` cannot emit tool calls)
    - ReAct vs. Single-Shot (`with_react_loop=False` halts after 1 turn before multi-step completion)
    - Procedural Skill guidance (`with_skills=True` ensures `chmod +x` on G-H03 and clean TDD on G-H01)
    - Thinking level token budgets and latency scaling
    - Swarm parallel delegation (`swarm_mode=True` spawns concurrent background subagents)
    """

    def __init__(self, task: Dict[str, Any], config: BenchmarkConfig, step_delay_sec: float = 0.08):
        self.task = task
        self.config = config
        self.step_delay_sec = step_delay_sec
        self.orch_step = 0

    async def generate(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, model: Optional[str] = None, **kwargs):
        thinking_mult = {"none": 0.5, "low": 0.8, "medium": 1.0, "high": 1.5}.get(self.config.thinking_level, 1.0)
        await asyncio.sleep(self.step_delay_sec * thinking_mult)

        # Distinguish child subagent turns (which start with "Goal:\n") from the parent orchestrator
        first_user_msg = next((m.content for m in messages if m.role == "user"), "")
        is_subagent = first_user_msg.startswith("Goal:\n")

        # Base token usage scaled by thinking_level and tool schema size
        schema_count = len(tools) if tools else 0
        p_tokens = 120 + (schema_count * 35) + (80 if self.config.with_skills else 0)
        c_tokens = int(45 * thinking_mult)
        usage = UsageMetadata(prompt_tokens=p_tokens, completion_tokens=c_tokens, total_tokens=p_tokens + c_tokens)

        # If no tools are registered, the model can only output raw text
        if not tools:
            self.orch_step += 1
            return (
                Message(
                    role="assistant",
                    content="Tools are disabled in this configuration. Here is the conceptual solution in text.",
                    usage=usage,
                ),
                usage,
            )

        # Handle child subagent execution inside a swarm
        if is_subagent:
            return await self._handle_subagent_turn(messages, first_user_msg, usage)

        self.orch_step += 1
        task_id = self.task["id"]
        return await self._handle_orchestrator_turn(task_id, usage)

    async def _handle_subagent_turn(self, messages: List[Message], user_text: str, usage: UsageMetadata):
        # Check if subagent already executed its write_file tool
        has_tool_res = any(m.role == "tool" for m in messages)
        if has_tool_res:
            return Message(role="assistant", content="Subtask deliverable completed.", usage=usage), usage

        # Simulate realistic subagent work duration (0.25s) so parallel vs sequential timing is clearly measurable
        await asyncio.sleep(0.25)

        if "fib.json" in user_text:
            tc = ToolCall(id=f"call_{uuid.uuid4().hex[:6]}", name="write_file", arguments={"path": "fib.json", "content": "[0, 1, 1, 2, 3, 5, 8, 13]"})
        elif "greet.js" in user_text:
            tc = ToolCall(id=f"call_{uuid.uuid4().hex[:6]}", name="write_file", arguments={"path": "greet.js", "content": "module.exports = (name) => `Hello, ${name}!`;"})
        elif "square_result.txt" in user_text:
            tc = ToolCall(id=f"call_{uuid.uuid4().hex[:6]}", name="write_file", arguments={"path": "square_result.txt", "content": "144"})
        elif "research_summary.md" in user_text:
            tc = ToolCall(id=f"call_{uuid.uuid4().hex[:6]}", name="write_file", arguments={"path": "research_summary.md", "content": "# Layers\n- apps\n- packages\n- mobile-apps"})
        elif "campaign_hook.md" in user_text:
            tc = ToolCall(id=f"call_{uuid.uuid4().hex[:6]}", name="write_file", arguments={"path": "campaign_hook.md", "content": "# Valstorm Autonomous Enterprise Engine"})
        elif "schema.json" in user_text:
            tc = ToolCall(id=f"call_{uuid.uuid4().hex[:6]}", name="write_file", arguments={"path": "schema.json", "content": "{\"object\": \"lead\"}"})
        else:
            return Message(role="assistant", content="Subagent finished.", usage=usage), usage

        return Message(role="assistant", content="", tool_calls=[tc], usage=usage), usage

    async def _handle_orchestrator_turn(self, task_id: str, usage: UsageMetadata):
        s = self.orch_step

        if task_id == "G-S01":
            if s == 1:
                return Message(role="assistant", content="", tool_calls=[ToolCall(id="c1", name="calculator", arguments={"expression": "round((42 * 3.14159) / 7, 4)"})], usage=usage), usage
            if s == 2:
                return Message(role="assistant", content="", tool_calls=[ToolCall(id="c2", name="write_file", arguments={"path": "calculation_result.txt", "content": "18.8495"})], usage=usage), usage
            return Message(role="assistant", content="Calculated 18.8495 and saved to calculation_result.txt.", usage=usage), usage

        if task_id == "G-S02":
            if s == 1:
                return Message(role="assistant", content="", tool_calls=[ToolCall(id="c1", name="write_file", arguments={"path": "index.js", "content": 'console.log("Hello Agent Runtime");\n'})], usage=usage), usage
            if s == 2:
                return Message(role="assistant", content="", tool_calls=[ToolCall(id="c2", name="read_file", arguments={"path": "index.js"})], usage=usage), usage
            return Message(role="assistant", content="Created and verified index.js.", usage=usage), usage

        if task_id == "G-S03":
            if s == 1:
                return Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(id="c1", name="write_file", arguments={"path": "src/main.py", "content": "# main\n"}),
                        ToolCall(id="c2", name="write_file", arguments={"path": "test/test_main.py", "content": "# test\n"}),
                    ],
                    usage=usage,
                ), usage
            if s == 2:
                return Message(role="assistant", content="", tool_calls=[ToolCall(id="c3", name="search_files", arguments={"pattern": "*.py", "target": "files", "path": "."})], usage=usage), usage
            return Message(role="assistant", content="Found 2 Python files.", usage=usage), usage

        if task_id == "G-M01":
            if s == 1:
                return Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="c1", name="write_file", arguments={"path": "documentation_source.md", "content": "# Python Requests Quickstart\nhttps://requests.readthedocs.io/en/latest/user/quickstart/\n"})],
                    usage=usage,
                ), usage
            return Message(role="assistant", content="Saved top Python requests documentation link.", usage=usage), usage

        if task_id == "G-M02":
            if s == 1:
                code = "import sys\nprint(int(sys.argv[1]) + int(sys.argv[2]))\n"
                return Message(role="assistant", content="", tool_calls=[ToolCall(id="c1", name="write_file", arguments={"path": "sum.py", "content": code})], usage=usage), usage
            if s == 2:
                return Message(role="assistant", content="", tool_calls=[ToolCall(id="c2", name="terminal_exec", arguments={"command": "python3 sum.py 15 30"})], usage=usage), usage
            return Message(role="assistant", content="Verified output is 45.", usage=usage), usage

        if task_id == "G-M03":
            if s == 1:
                return Message(role="assistant", content="", tool_calls=[ToolCall(id="c1", name="write_file", arguments={"path": "config.json", "content": '{"url": "http://api.old.com/v1"}'})], usage=usage), usage
            if s == 2:
                return Message(role="assistant", content="", tool_calls=[ToolCall(id="c2", name="patch_file", arguments={"path": "config.json", "old_string": "http://api.old.com/v1", "new_string": "http://api.new.com/v2"})], usage=usage), usage
            if s == 3:
                return Message(role="assistant", content="", tool_calls=[ToolCall(id="c3", name="read_file", arguments={"path": "config.json"})], usage=usage), usage
            return Message(role="assistant", content="Patched and verified config.json.", usage=usage), usage

        if task_id == "G-H01":
            if s == 1:
                test_code = "from src.utils import reverse_string\ndef test_rev():\n    assert reverse_string('abc') == 'cba'\n"
                stub_code = "def reverse_string(s: str) -> str:\n    return ''\n"
                return Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(id="c1", name="write_file", arguments={"path": "test/test_reverse.py", "content": test_code}),
                        ToolCall(id="c2", name="write_file", arguments={"path": "src/utils.py", "content": stub_code}),
                    ],
                    usage=usage,
                ), usage
            if s == 2:
                return Message(role="assistant", content="", tool_calls=[ToolCall(id="c3", name="terminal_exec", arguments={"command": "PYTHONPATH=. python3 -m pytest test/test_reverse.py"})], usage=usage), usage
            if s == 3:
                impl_code = "def reverse_string(s: str) -> str:\n    return s[::-1]\n"
                return Message(role="assistant", content="", tool_calls=[ToolCall(id="c4", name="write_file", arguments={"path": "src/utils.py", "content": impl_code})], usage=usage), usage
            if s == 4:
                return Message(role="assistant", content="", tool_calls=[ToolCall(id="c5", name="terminal_exec", arguments={"command": "PYTHONPATH=. python3 -m pytest test/test_reverse.py"})], usage=usage), usage
            return Message(role="assistant", content="TDD cycle completed; pytest passed with exit code 0.", usage=usage), usage

        if task_id == "G-H02":
            return Message(role="assistant", content="Checked https://www.valstorm.com/ via web_scrape and web_content_diff: status is unchanged (no diff_detected).", usage=usage), usage

        if task_id == "G-H03":
            if s == 1:
                script = "#!/bin/sh\nmkdir -p temp_proj\ncd temp_proj\ngit init\nprintf 'node_modules\\ndist\\n' > .gitignore\npwd\n"
                return Message(role="assistant", content="", tool_calls=[ToolCall(id="c1", name="write_file", arguments={"path": "setup_env.sh", "content": script})], usage=usage), usage
            if s == 2:
                # Ablation behavior: With skills ('bash-scripting') or high thinking, agent remembers chmod +x.
                # Without skills and with low/none thinking, agent executes `sh setup_env.sh` without setting +x bit!
                if self.config.with_skills or self.config.thinking_level in ("medium", "high"):
                    cmd = "chmod +x setup_env.sh && ./setup_env.sh"
                else:
                    cmd = "sh setup_env.sh"
                return Message(role="assistant", content="", tool_calls=[ToolCall(id="c2", name="terminal_exec", arguments={"command": cmd})], usage=usage), usage
            return Message(role="assistant", content="Executed setup_env.sh and initialized git repository in temp_proj.", usage=usage), usage

        # Swarm tasks (SW-01, SW-02)
        if task_id in ("SW-01", "SW-02"):
            subtasks = self.task.get("subtasks") or []
            if self.config.swarm_mode:
                # Parallel Swarm Mode: Spawn all subtasks with background=True in Turn 1, wait all in Turn 2
                if s == 1:
                    calls = [
                        ToolCall(
                            id=f"sw_{idx}",
                            name="delegate_task",
                            arguments={"profile": st["profile"], "goal": st["goal"], "background": True},
                        )
                        for idx, st in enumerate(subtasks, 1)
                    ]
                    return Message(role="assistant", content="", tool_calls=calls, usage=usage), usage
                if s == 2:
                    return Message(
                        role="assistant",
                        content="",
                        tool_calls=[ToolCall(id="sw_wait", name="delegate_task", arguments={"profile": "developer", "goal": "wait", "action": "wait", "task_id": "all"})],
                        usage=usage,
                    ), usage
                return Message(role="assistant", content="All parallel swarm deliverables completed.", usage=usage), usage
            else:
                # Sequential Mode: Execute one subtask per turn sequentially
                if 1 <= s <= len(subtasks):
                    st = subtasks[s - 1]
                    await asyncio.sleep(0.25)  # Sequential work cost
                    if "fib.json" in st["goal"]:
                        tc = ToolCall(id=f"seq_{s}", name="write_file", arguments={"path": "fib.json", "content": "[0, 1, 1, 2, 3, 5, 8, 13]"})
                    elif "greet.js" in st["goal"]:
                        tc = ToolCall(id=f"seq_{s}", name="write_file", arguments={"path": "greet.js", "content": "module.exports = (name) => `Hello, ${name}!`;"})
                    elif "square_result.txt" in st["goal"]:
                        tc = ToolCall(id=f"seq_{s}", name="write_file", arguments={"path": "square_result.txt", "content": "144"})
                    elif "research_summary.md" in st["goal"]:
                        tc = ToolCall(id=f"seq_{s}", name="write_file", arguments={"path": "research_summary.md", "content": "# Layers\n- apps\n- packages\n- mobile-apps"})
                    elif "campaign_hook.md" in st["goal"]:
                        tc = ToolCall(id=f"seq_{s}", name="write_file", arguments={"path": "campaign_hook.md", "content": "# Valstorm Autonomous Enterprise Engine"})
                    else:
                        tc = ToolCall(id=f"seq_{s}", name="write_file", arguments={"path": "schema.json", "content": "{\"object\": \"lead\"}"})
                    return Message(role="assistant", content="", tool_calls=[tc], usage=usage), usage
                return Message(role="assistant", content="Sequential deliverables completed.", usage=usage), usage

        # UI & Crafting Canvas Navigation tasks (UI-01, UI-02)
        if task_id in ("UI-01", "UI-02"):
            out_file = "ui_nav_result.json" if task_id == "UI-01" else "artcraft_workflow_result.json"
            ui_state_obj = self.task.get("ui_state") or {}
            assertions_obj = self.task.get("assertions") or {}

            if self.config.use_decision_model:
                # System-1 UI Navigator Mode: 1 fast System-1 trajectory call resolves all closed-set UI/canvas steps in parallel
                if s == 1:
                    tc = ToolCall(
                        id="s1_nav",
                        name="ui_navigate",
                        arguments={
                            "goal": self.task["goal"],
                            "ui_state": json.dumps(ui_state_obj),
                            "assertions": json.dumps(assertions_obj),
                            "mode": "workflow",
                            "output_path": out_file,
                        },
                    )
                    return Message(role="assistant", content="", tool_calls=[tc], usage=usage), usage
                return Message(role="assistant", content=f"System-1 UI navigation completed and saved to {out_file}.", usage=usage), usage
            else:
                # System-2 LLM ReAct Step-by-Step Navigation Mode: requires 1 slow LLM reasoning turn per UI/canvas click
                expected_steps = (
                    ["nav_tab_cloud_vaults", "vault_doc_q4_strategy", "btn_share_vault_doc"]
                    if task_id == "UI-01"
                    else ["tool_img_to_3d_mesh", "actor_hero_pose_rig", "cam_main_shot_rig", "btn_render_video_scene"]
                )
                # Simulate extra LLM reasoning & DOM-parsing token/time overhead per UI step
                await asyncio.sleep(0.18)
                extra_usage = UsageMetadata(
                    prompt_tokens=usage.prompt_tokens + 680,
                    completion_tokens=usage.completion_tokens + 140,
                    total_tokens=usage.total_tokens + 820,
                )
                if 1 <= s <= len(expected_steps):
                    partial = {"step": s, "selected_element_chain": expected_steps[:s]}
                    tc = ToolCall(
                        id=f"s2_step_{s}",
                        name="write_file",
                        arguments={"path": out_file, "content": json.dumps(partial, indent=2)},
                    )
                    return Message(role="assistant", content="", tool_calls=[tc], usage=extra_usage), extra_usage
                return Message(role="assistant", content=f"Multi-turn LLM UI navigation completed ({len(expected_steps)} turns).", usage=extra_usage), extra_usage

        return Message(role="assistant", content="Task completed.", usage=usage), usage


class VsAgentBenchmarkAdapter(BenchmarkAdapter):
    """Runs Valstorm Agent (`vsagent`) in-process with full ablation control."""

    async def execute_task(
        self,
        task: Dict[str, Any],
        config: BenchmarkConfig,
        workspace_dir: Path,
        mode: Literal["auto", "mock", "live"] = "auto",
    ) -> ExecutionTrace:
        # 1. Isolate sandbox to workspace_dir
        prev_sandbox = get_current_sandbox()
        isolated_sandbox = HostSandbox(base_dir=str(workspace_dir))
        token = set_current_sandbox(isolated_sandbox)

        subagent_reg = SubagentRegistry.get_instance()
        subagent_reg.clear()

        # 2. Resolve provider (mock vs live)
        use_mock = (mode == "mock")
        live_provider = None
        if mode in ("auto", "live"):
            live_provider = _get_subagent_provider(config.provider)
            if live_provider is None and mode == "auto":
                use_mock = True

        if use_mock:
            provider = TaskAwareMockProvider(task=task, config=config)
            set_subagent_provider_override(provider)
        else:
            provider = live_provider
            set_subagent_provider_override(None)

        # 3. Build scoped/full/empty ToolRegistry
        registry = build_benchmark_tool_registry(
            tools_mode=config.tools_mode,
            task=task,
            swarm_mode=config.swarm_mode,
        )

        # 4. Build system prompt with or without attached skills
        ctx_mgr = WorkspaceContextManager(workdir=str(workspace_dir))
        prof_data = load_profile("developer")
        if config.with_skills:
            skills_to_attach = config.skills or task.get("recommended_skills") or prof_data.get("attached_skill_slugs") or []
            prof_data["attached_skill_slugs"] = list(skills_to_attach)
        else:
            prof_data["attached_skill_slugs"] = []

        system_prompt = ctx_mgr.build_system_prompt(profile=prof_data)

        # 5. Initialize isolated SessionState
        session = SessionState(
            session_id=f"bench_{task['id']}_{config.config_id}_{uuid.uuid4().hex[:6]}",
            name=f"Benchmark {task['id']} [{config.config_id}]",
            active_model=config.model,
            active_provider=config.provider,
        )
        session.add_message(Message(role="system", content=system_prompt, model=config.model, provider=config.provider))

        engine = ReActEngine(provider=provider, tools=registry)

        tool_calls_made: List[str] = []
        tool_outputs: List[str] = []
        turns_taken = 0
        tool_time_ms = 0.0
        final_output = ""
        status: Literal["completed", "failed", "timed_out", "skipped"] = "completed"
        error_msg: Optional[str] = None

        t0 = time.perf_counter()
        try:
            async def _run():
                nonlocal turns_taken, tool_time_ms, final_output
                async for event in engine.run_turn_stream(
                    session=session,
                    user_input=task["goal"],
                    model=config.model,
                    max_iterations=config.effective_max_turns,
                    reasoning_effort=config.thinking_level,
                ):
                    if event.event_type == StreamEventType.TOOL_CALL_DETECTED and event.tool_call:
                        tool_calls_made.append(event.tool_call.name)
                    elif event.event_type == StreamEventType.TOOL_EXECUTION_RESULT and event.tool_result:
                        if event.tool_result.output:
                            tool_outputs.append(event.tool_result.output)
                        if event.tool_result.duration_ms:
                            tool_time_ms += event.tool_result.duration_ms
                    elif event.event_type == StreamEventType.TURN_COMPLETE and event.message:
                        final_output = event.message.content or ""

            await asyncio.wait_for(_run(), timeout=config.timeout_sec)
        except asyncio.TimeoutError:
            status = "timed_out"
            error_msg = f"Timed out after {config.timeout_sec}s"
        except Exception as e:
            status = "failed"
            error_msg = str(e)
        finally:
            set_subagent_provider_override(None)
            set_current_sandbox(prev_sandbox)

        duration_sec = round(time.perf_counter() - t0, 3)
        tool_time_sec = round(tool_time_ms / 1000.0, 3)
        llm_time_sec = round(max(0.0, duration_sec - tool_time_sec), 3)
        turns_taken = sum(1 for m in session.messages if m.role == "assistant")

        # Aggregate subagent tokens & concurrency metrics if any subagents were spawned
        spawned_tasks = subagent_reg.list_tasks()
        subagents_count = len(spawned_tasks)
        sub_prompt = sum(t.child_session.total_prompt_tokens for t in spawned_tasks)
        sub_comp = sum(t.child_session.total_completion_tokens for t in spawned_tasks)

        total_prompt = session.total_prompt_tokens + sub_prompt
        total_comp = session.total_completion_tokens + sub_comp
        total_tok = total_prompt + total_comp

        seq_sum = sum(t.duration_sec for t in spawned_tasks) if spawned_tasks else duration_sec
        parallel_speedup = round(seq_sum / duration_sec, 2) if (spawned_tasks and duration_sec > 0) else 1.0

        return ExecutionTrace(
            task_id=task["id"],
            config_id=config.config_id,
            harness=config.harness,
            status=status,
            final_output=final_output,
            tool_calls_made=tool_calls_made,
            tool_outputs=tool_outputs,
            turns_taken=turns_taken,
            duration_sec=duration_sec,
            llm_time_sec=llm_time_sec,
            tool_time_sec=tool_time_sec,
            prompt_tokens=total_prompt,
            completion_tokens=total_comp,
            total_tokens=total_tok,
            estimated_cost_usd=estimate_cost_usd(config.model, total_prompt, total_comp),
            subagents_spawned=subagents_count,
            sequential_sum_sec=round(seq_sum, 3),
            parallel_speedup=parallel_speedup,
            error=error_msg,
        )
