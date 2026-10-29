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
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Literal, Optional
import uuid

from core.context import WorkspaceContextManager, load_profile
from core.keystore import KeyStore
from core.models import Message, SessionState, StreamEventType, ToolCall, UsageMetadata
from core.react import ReActEngine
from core.sandbox import HostSandbox, get_current_sandbox, set_current_sandbox
from core.tools import ToolRegistry, get_default_registry
from providers import resolve_provider_instance
from tools import register_developer_tools, register_memory_tools, register_tier1_tools, register_ui_navigator_tools, register_vision_tools
from tools.delegation import SubagentRegistry, _get_subagent_provider, set_subagent_provider_override
from tools.valstorm_tools import create_valstorm_tools
from tools.vision_tool import set_vision_provider_override

from ..sandbox_valstorm import BenchmarkValstormClient, provision_craft_suite_sandbox
from .base import BenchmarkAdapter, BenchmarkConfig, ExecutionTrace, estimate_cost_usd

# External side-effect tools strictly excluded from benchmark sandboxes
_EXCLUDED_EXTERNAL_TOOLS = frozenset({
    "valstorm_send_sms",
    "slack_post_message",
    "slack_list_channels",
    "slack_get_channel_history",
    "slack_list_users",
    "slack_get_user_profile",
    "slack_add_reaction",
    "slack_update_message",
    "slack_delete_message",
    "slack_get_auth_status",
})


def build_benchmark_tool_registry(
    tools_mode: str,
    task: Dict[str, Any],
    swarm_mode: bool = False,
    use_decision_model: bool = True,
    workspace_dir: Optional[Path] = None,
) -> ToolRegistry:
    """Builds the ToolRegistry according to the `tools_mode` and `use_decision_model` ablation knobs."""
    if tools_mode == "none":
        return ToolRegistry()

    base = get_default_registry()
    register_developer_tools(base)
    register_tier1_tools(base)
    register_vision_tools(base)
    if use_decision_model:
        register_ui_navigator_tools(base)
    register_memory_tools(base)

    # Register safe, 100% sandboxed Valstorm Office/CRM/VFS tools (excluding SMS & Slack)
    if workspace_dir is not None:
        sandbox_client = BenchmarkValstormClient(workspace_dir=workspace_dir)
        for vt in create_valstorm_tools(client=sandbox_client):  # type: ignore[arg-type]
            t_name = getattr(vt, "name", "") or getattr(vt, "__name__", "")
            if t_name not in _EXCLUDED_EXTERNAL_TOOLS:
                base.register(vt)

    if tools_mode == "scoped":
        whitelist = list(task.get("tool_chain") or [])
        if not use_decision_model:
            whitelist = [t for t in whitelist if t not in ("ui_navigate", "system_one_decide")]
            for fallback_t in ("read_file", "write_file"):
                if fallback_t not in whitelist:
                    whitelist.append(fallback_t)
        elif task.get("ui_state") and "ui_navigate" in whitelist:
            whitelist = ["ui_navigate"]
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
        self._active_effort: str = config.thinking_level

    def set_request_context(self, **kwargs):
        if "reasoning_effort" in kwargs and kwargs["reasoning_effort"]:
            self._active_effort = str(kwargs["reasoning_effort"]).strip().lower()

    async def generate(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, model: Optional[str] = None, **kwargs):
        eff = self._active_effort if (self.config.thinking_level == "adaptive" or self.config.auto_tune) else self.config.thinking_level
        thinking_mult = {"none": 0.5, "low": 0.8, "medium": 1.0, "high": 1.5}.get(eff, 1.0)
        eff_model = (model or self.config.model or "").lower()
        model_speed_mult = 0.72 if "lite" in eff_model else (1.55 if "pro" in eff_model else 1.0)
        await asyncio.sleep(self.step_delay_sec * thinking_mult * model_speed_mult)

        # Distinguish child subagent turns (which start with "Goal:\n") from the parent orchestrator
        first_user_msg = next((m.content for m in messages if m.role == "user"), "")
        is_subagent = first_user_msg.startswith("Goal:\n")

        # Base token usage scaled by thinking_level and tool schema size
        schema_count = len(tools) if tools else 0
        p_tokens = 120 + (schema_count * 35) + (80 if self.config.with_skills else 0)
        c_tokens = int(45 * thinking_mult)
        usage = UsageMetadata(prompt_tokens=p_tokens, completion_tokens=c_tokens, total_tokens=p_tokens + c_tokens)

        # If no tools are registered, the model can only output raw text (or respond to internal analyze_image calls)
        if not tools:
            if any(getattr(m, "images", None) for m in messages):
                return (
                    Message(
                        role="assistant",
                        content="Visual QA Verified: 9:16 safe zones, WCAG AA contrast, and bounded text boxes passed inspection.",
                        usage=usage,
                    ),
                    usage,
                )
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
            return await self._handle_subagent_turn(messages, first_user_msg, usage, eff_model)

        self.orch_step += 1
        task_id = self.task["id"]
        return await self._handle_orchestrator_turn(task_id, usage)

    async def _handle_subagent_turn(self, messages: List[Message], user_text: str, usage: UsageMetadata, eff_model: str = ""):
        # Check if subagent already executed its write_file tool
        has_tool_res = any(m.role == "tool" for m in messages)
        if has_tool_res:
            return Message(role="assistant", content="Subtask deliverable completed.", usage=usage), usage

        # Simulate realistic subagent work duration (0.25s base, scaled by model tier speed)
        sub_mult = 0.72 if "lite" in eff_model else 1.0
        await asyncio.sleep(0.25 * sub_mult)

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
                # Ablation behavior: With skills ('bash-scripting') or medium/high thinking (including System-1 adaptive/autotune escalation), agent remembers chmod +x.
                # Without skills and with low/none thinking, agent executes `sh setup_env.sh` without setting +x bit!
                eff = self._active_effort if (self.config.thinking_level == "adaptive" or self.config.auto_tune) else self.config.thinking_level
                if self.config.with_skills or eff in ("medium", "high"):
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

        # UI & Crafting Canvas Navigation tasks (UI-01 .. UI-04)
        if task_id in ("UI-01", "UI-02", "UI-03", "UI-04"):
            out_file = (self.task.get("success_criteria") or {}).get("file") or "ui_nav_result.json"
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
                step_chains = {
                    "UI-01": ["nav_tab_cloud_vaults", "vault_doc_q4_strategy", "btn_share_vault_doc"],
                    "UI-02": ["tool_img_to_3d_mesh", "actor_hero_pose_rig", "cam_main_shot_rig", "btn_render_video_scene"],
                    "UI-03": ["btn_dismiss_sync_modal", "select_high_value_leads_filter", "btn_pin_filter_view"],
                    "UI-04": ["tool_bg_remove_alpha", "layer_3d_studio_backdrop", "timeline_scrub_kf_24", "btn_render_composited_shot"],
                }
                expected_steps = step_chains.get(task_id, ["nav_tab_cloud_vaults"])
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

        # Office, CRM & VFS Sandbox Tasks (OF-01 .. OF-05)
        if task_id == "OF-01":
            sql = "SELECT id, name, status, loan_amount FROM lead WHERE owner = ME AND status = 'Active' AND created_date = last_n_days:14 ORDER BY loan_amount DESC"
            if s == 1:
                return Message(role="assistant", content="", tool_calls=[ToolCall(id="of1_1", name="valstorm_sql_query", arguments={"query": sql})], usage=usage), usage
            if s == 2:
                return Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="of1_2",
                            name="valstorm_create_list_filter",
                            arguments={"name": "High-Value Active Leads (14d)", "object_api_name": "lead", "sql_query": sql, "is_pinned": True},
                        )
                    ],
                    usage=usage,
                ), usage
            return Message(role="assistant", content="Queried active leads and saved pinned list filter 'High-Value Active Leads (14d)'.", usage=usage), usage

        if task_id == "OF-02":
            if s == 1:
                return Message(role="assistant", content="", tool_calls=[ToolCall(id="of2_1", name="valstorm_schema_inspect", arguments={"api_name": "campaign"})], usage=usage), usage
            if s == 2:
                rec = {
                    "name": "Q4 Enterprise Open Model Launch",
                    "status": "Active",
                    "content_type": "Carousel",
                    "utm_source": "linkedin",
                    "owner": "user_qBulInDu18OcylWP",
                }
                return Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="of2_2", name="valstorm_record_cud", arguments={"api_name": "campaign", "action": "create", "records": [rec]})],
                    usage=usage,
                ), usage
            return Message(role="assistant", content="Created campaign record with string lookup ID and tag labels.", usage=usage), usage

        if task_id == "OF-03":
            if s == 1:
                return Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="of3_1",
                            name="valstorm_record_cud",
                            arguments={"api_name": "vault", "action": "create", "records": [{"name": "09_Standard_Operating_Procedures", "parent_vault": "vaul_root_hub"}]},
                        )
                    ],
                    usage=usage,
                ), usage
            if s == 2:
                sop_md = (
                    "# Open Model Failover SOP\n\n"
                    "| Model | Tier | Max Output |\n"
                    "| :--- | :--- | :--- |\n"
                    "| `gemini-4-argon` | GEAP Thinker Baseline | 1,000,000 |\n"
                    "| `qwen3.8-max` | DO Open Thinker | 262,144 |\n"
                )
                return Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="of3_2",
                            name="valstorm_vfs_write_file",
                            arguments={"name": "Open_Model_Failover_SOP.md", "content": sop_md, "vault_id": "vaul_1_0001"},
                        )
                    ],
                    usage=usage,
                ), usage
            return Message(role="assistant", content="Created 09_Standard_Operating_Procedures vault and Open_Model_Failover_SOP.md.", usage=usage), usage

        if task_id == "OF-04":
            if s == 1:
                return Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="of4_1", name="valstorm_sql_query", arguments={"query": "SELECT id, name, email, company FROM contact WHERE company = 'Acme Corp'"})],
                    usage=usage,
                ), usage
            if s == 2:
                return Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="of4_2",
                            name="valstorm_record_merge",
                            arguments={"master_record_id": "con_master_1", "duplicate_record_ids": ["con_dup_2"], "schema_api_name": "contact"},
                        )
                    ],
                    usage=usage,
                ), usage
            return Message(role="assistant", content="Merged duplicate contact con_dup_2 into con_master_1 and re-linked child tasks.", usage=usage), usage

        if task_id == "OF-05":
            if s == 1:
                fn_code = "def route_drip_step(step: int, lead_status: str) -> dict:\n    return {'action': 'send_email', 'step': step}\n"
                return Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="of5_1", name="valstorm_validate_function", arguments={"function_name": "route_drip_step", "code": fn_code})],
                    usage=usage,
                ), usage
            if s == 2:
                steps = [{"step": 1, "delay_minutes": 0}, {"step": 2, "delay_minutes": 2880}, {"step": 3, "delay_minutes": 7200}]
                return Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="of5_2",
                            name="valstorm_create_drip_cadence",
                            arguments={
                                "name": "14-Day Enterprise Nurture",
                                "target_schema": "lead",
                                "steps": steps,
                                "target_type": "function",
                                "exit_conditions": {"status": ["Closed Won"]},
                            },
                        )
                    ],
                    usage=usage,
                ), usage
            return Message(role="assistant", content="Validated AST-safe step router function and created 14-Day Enterprise Nurture drip cadence.", usage=usage), usage

        # Creative Studio & Craft Suite Tasks (CR-01 .. CR-04)
        if task_id == "CR-01":
            if s == 1:
                psd_spec = {
                    "document": {"name": "AdPlate", "width": 1080, "height": 1920, "mode": "rgb", "background": "#090d16"},
                    "commands": [
                        {"id": "shape.create", "params": {"name": "Card", "kind": "roundedRect", "rect": [80, 520, 920, 600], "radii": 36, "fill": "#111827"}},
                        {"id": "layer.layerStyle.dropShadow", "params": {"color": "#000000", "opacity": 80, "distance": 16, "size": 32}},
                        {"id": "layer.newAdjustmentLayer.curves", "params": {"points": [[0, 0], [50, 40], [200, 220], [255, 255]]}},
                        {"id": "type.create", "params": {"name": "Headline", "box": [120, 300, 840, 200], "text": "Valstorm Open AI Engine", "size": 48, "color": "#ffffff"}},
                    ],
                }
                return Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="cr1_1", name="write_file", arguments={"path": "ad_plate.psd.json", "content": json.dumps(psd_spec, indent=2)})],
                    usage=usage,
                ), usage
            if s == 2:
                return Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="cr1_2", name="analyze_image", arguments={"path": "ad_preview.png", "prompt": "Audit 9:16 safe zones, contrast, and text box wrapping."})],
                    usage=usage,
                ), usage
            return Message(role="assistant", content="Created PhotoCraft 9:16 layered PSD spec and completed visual QA via analyze_image.", usage=usage), usage

        if task_id == "CR-02":
            if s == 1:
                fc_spec = {
                    "project": "VerticalReel",
                    "commands": [
                        {"cmd": "file.newSequence", "params": {"name": "Reel_9x16", "width": 1080, "height": 1920, "fps": 30.0, "video": 2, "audio": 2}},
                        {"cmd": "timeline.place", "params": {"items": ["select_1"], "track": "V1", "seconds": 0.0, "video": True, "audio": False}},
                        {"cmd": "timeline.place", "params": {"item": "vo_1", "track": "A1", "seconds": 0.0, "video": False, "audio": True}},
                        {"cmd": "clip.audioGain", "params": {"clips": ["vo_1"], "mode": "normalizeMax", "db": -1.0}},
                        {"cmd": "captions.newTrack", "params": {"format": "Subtitle", "name": "Reels Captions", "language": "en"}},
                        {"cmd": "captions.setStyle", "params": {"track": "C1", "size": 28.0, "color": "#ffffff", "margin": 0.22, "outline": 2.0}},
                        {"cmd": "file.exportFcpxml", "params": {"path": "edit.fcpxml"}},
                    ],
                }
                return Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="cr2_1", name="write_file", arguments={"path": "reel_timeline.fcproj", "content": json.dumps(fc_spec, indent=2)})],
                    usage=usage,
                ), usage
            if s == 2:
                return Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="cr2_2", name="terminal_exec", arguments={"command": "./bin/filmcraft-cli export edit.fcpxml"})],
                    usage=usage,
                ), usage
            return Message(role="assistant", content="Assembled FilmCraft 9:16 vertical reel timeline and exported edit.fcpxml.", usage=usage), usage

        if task_id == "CR-03":
            if s == 1:
                ec_spec = {
                    "composition": "Main",
                    "keyframes": {
                        "transform/position": [{"time": 0.0, "value": [540, 960]}],
                        "transform/scale": [{"time": 1.5, "value": [105, 105]}],
                    },
                    "expressions": {"transform/opacity": "wiggle(2, 15)"},
                    "commands": [{"cmd": "file.exportLottie", "params": {"comp": "Main", "path": "overlay.json", "includeExpressions": True}}],
                }
                lottie_json = {"v": "5.7.4", "fr": 60, "ip": 0, "op": 180, "w": 1080, "h": 1920, "nm": "Main", "layers": []}
                return Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(id="cr3_1", name="write_file", arguments={"path": "lower_third.ecproj", "content": json.dumps(ec_spec, indent=2)}),
                        ToolCall(id="cr3_2", name="write_file", arguments={"path": "overlay.json", "content": json.dumps(lottie_json, indent=2)}),
                    ],
                    usage=usage,
                ), usage
            if s == 2:
                return Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="cr3_3", name="analyze_image", arguments={"path": "check.png", "prompt": "Verify lower third motion graphic layout."})],
                    usage=usage,
                ), usage
            return Message(role="assistant", content="Created EffectCraft lower_third.ecproj, exported Lottie overlay.json, and verified check.png.", usage=usage), usage

        if task_id == "CR-04":
            if s == 1:
                sc_spec = {
                    "cmd": "session.new",
                    "params": {"name": "PodcastMix", "sample_rate": 48000, "bit_depth": 24, "template": "blank"},
                    "imports": {"cmd": "file.import_audio", "paths": ["vo.wav", "music.wav"], "new_tracks": True},
                    "dialogue_chain": [
                        {"plugin": "High-Pass Filter", "freq_hz": 80},
                        {"plugin": "De-Esser", "band_hz": [5000, 8000]},
                        {"plugin": "Parametric EQ", "scoop_hz": 300, "air_shelf_hz": 11000},
                        {"plugin": "Compressor", "ratio": "3:1", "attack_ms": 20, "release_ms": 100},
                        {"plugin": "True-Peak Limiter", "ceiling_dbfs": -1.0, "target_lufs": -14.0},
                    ],
                }
                return Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="cr4_1", name="write_file", arguments={"path": "podcast_master.scraft", "content": json.dumps(sc_spec, indent=2)})],
                    usage=usage,
                ), usage
            return Message(role="assistant", content="Saved 48kHz/24-bit SoundCraft dialogue sweetening session to podcast_master.scraft.", usage=usage), usage

        return Message(role="assistant", content="Task completed.", usage=usage), usage


def _resolve_isolated_live_provider(provider_name: str, model_name: str) -> Optional[Any]:
    """Resolves a single provider/model in strict isolation (no fallback cascade) for head-to-head model benchmarks."""
    ks = KeyStore()
    p_norm = (provider_name or "gemini").strip().lower()
    api_key = ks.get_api_key(p_norm)
    if not api_key and p_norm in ("do", "digitalocean"):
        for alias in ("digitalocean", "do"):
            api_key = ks.get_api_key(alias)
            if api_key:
                break
    if not api_key and p_norm in ("gemini", "geap", "vertex", "aistudio", "valstorm"):
        for alias in ("gemini", "valstorm"):
            api_key = ks.get_api_key(alias)
            if api_key:
                break
        if not api_key:
            try:
                from tools.valstorm_client import resolve_valstorm_credentials
                api_key, _ = resolve_valstorm_credentials()
            except Exception:
                pass
    if not api_key:
        return None
    inst, _, _ = resolve_provider_instance(provider_name=p_norm, api_key=api_key, model=model_name)
    return inst


class _BenchmarkVisionInspector:
    """Deterministic vision inspector for `analyze_image` tool calls during benchmarks.

    Prevents nested secondary-LLM calls inside `analyze_image` from skewing the
    target model's wall-clock latency and token measurements.
    """

    default_model = "benchmark-vision-qa"

    async def generate(self, messages: List[Message], model: Optional[str] = None, **kwargs: Any):
        usage = UsageMetadata(prompt_tokens=64, completion_tokens=32, total_tokens=96)
        return (
            Message(
                role="assistant",
                content="Visual QA Verified: 1080x1920 9:16 safe zones, WCAG AA contrast, and bounded text boxes passed inspection.",
                usage=usage,
            ),
            usage,
        )


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

        # Provision Craft Suite CLI shims & valid PNG assets for Creative tasks
        if str(task.get("id", "")).startswith("CR-") or str(task.get("category", "")).startswith("creative_"):
            provision_craft_suite_sandbox(workspace_dir)

        subagent_reg = SubagentRegistry.get_instance()
        subagent_reg.clear()

        # 2. Resolve provider (mock vs live)
        use_mock = (mode == "mock")
        live_provider = None
        if mode in ("auto", "live"):
            if config.isolate_provider:
                live_provider = _resolve_isolated_live_provider(config.provider, config.model)
            else:
                live_provider = _get_subagent_provider(config.provider)
            if live_provider is None and mode == "auto":
                use_mock = True

        if use_mock:
            provider = TaskAwareMockProvider(task=task, config=config)
            set_subagent_provider_override(provider)
            set_vision_provider_override(_BenchmarkVisionInspector())
        else:
            provider = live_provider
            set_subagent_provider_override(live_provider if config.isolate_provider else None)
            set_vision_provider_override(_BenchmarkVisionInspector())

        # 3. Build scoped/full/empty ToolRegistry (with safe sandboxed Valstorm client)
        registry = build_benchmark_tool_registry(
            tools_mode=config.tools_mode,
            task=task,
            swarm_mode=config.swarm_mode,
            use_decision_model=config.use_decision_model,
            workspace_dir=workspace_dir,
        )

        # Materialize ui_state.json for UI/Canvas benchmark tasks
        effective_user_input = task["goal"]
        if task.get("ui_state"):
            ui_file = workspace_dir / "ui_state.json"
            ui_file.write_text(json.dumps(task["ui_state"], indent=2), encoding="utf-8")
            out_target = (task.get("success_criteria") or {}).get("file") or "ui_nav_result.json"
            if not use_mock:
                if config.use_decision_model:
                    effective_user_input = (
                        f"{task['goal']}\n\n"
                        f"The UI state is stored in 'ui_state.json'. Call `ui_navigate` once with "
                        f"`goal`, `ui_state='ui_state.json'`, `mode='workflow'`, and `output_path='{out_target}'`."
                    )
                else:
                    effective_user_input = (
                        f"{task['goal']}\n\n"
                        f"The UI state is stored in 'ui_state.json'. First call `read_file` on 'ui_state.json', "
                        f"then write the JSON result containing `selected_element_chain` (list of chosen element IDs in order) "
                        f"to '{out_target}' using `write_file`."
                    )

        # 4. Build system prompt with task-specific profile (developer, chief-of-staff, creative-director) and attached skills
        ctx_mgr = WorkspaceContextManager(workdir=str(workspace_dir))
        target_profile_slug = task.get("profile") or "developer"
        prof_data = load_profile(target_profile_slug)
        if config.with_skills:
            if config.skills:
                skills_to_attach = config.skills
            elif "recommended_skills" in task:
                skills_to_attach = task["recommended_skills"] or []
            else:
                skills_to_attach = prof_data.get("attached_skill_slugs") or []
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
        prev_sub_autotune = os.environ.get("VALSTORM_SUBAGENT_AUTO_TUNE")
        os.environ["VALSTORM_SUBAGENT_AUTO_TUNE"] = "1" if config.auto_tune else "0"
        try:
            async def _run():
                nonlocal turns_taken, tool_time_ms, final_output
                async for event in engine.run_turn_stream(
                    session=session,
                    user_input=effective_user_input,
                    model=config.model,
                    max_iterations=config.effective_max_turns,
                    reasoning_effort=config.thinking_level,
                    auto_tune=config.auto_tune,
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
            if prev_sub_autotune is None:
                os.environ.pop("VALSTORM_SUBAGENT_AUTO_TUNE", None)
            else:
                os.environ["VALSTORM_SUBAGENT_AUTO_TUNE"] = prev_sub_autotune
            set_subagent_provider_override(None)
            set_vision_provider_override(None)
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

        if config.auto_tune:
            per_msg_cost = 0.0
            for m in session.messages:
                if m.role == "assistant" and m.usage:
                    m_mod = m.model or session.active_model or config.model
                    per_msg_cost += estimate_cost_usd(m_mod, m.usage.prompt_tokens, m.usage.completion_tokens)
            for t in spawned_tasks:
                per_msg_cost += estimate_cost_usd(
                    t.model_name or config.model,
                    t.child_session.total_prompt_tokens,
                    t.child_session.total_completion_tokens,
                )
            est_cost = round(per_msg_cost, 6)
        else:
            est_cost = estimate_cost_usd(config.model, total_prompt, total_comp)

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
            estimated_cost_usd=est_cost,
            subagents_spawned=subagents_count,
            sequential_sum_sec=round(seq_sum, 3),
            parallel_speedup=parallel_speedup,
            error=error_msg,
        )
