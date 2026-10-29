"""Multi-Agent Subagent Delegation Tool for Valstorm Agent Runtime.

Provides:
- delegate_task: Spawns isolated child subagent turns using named profiles
  from ~/.valstorm/profiles/ (or built-in defaults) with tool scoping and context isolation.
  Supports both synchronous awaiting and non-blocking asynchronous background execution (background=True)
  with task tracking, polling, and batch waiting.
- subagent_manage: Manages background subagents (list, poll, wait, kill, spawn).
"""

import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Literal, Optional
import uuid

from core.context import (
    BUILTIN_PROFILES,
    WorkspaceContextManager,
    get_current_profile,
    invalidate_tenant_profiles_cache,
    list_available_profiles,
    load_profile,
)
from core.keystore import KeyStore
from core.models import Message, SessionState, ToolCall, ToolResult
from core.tools import ToolRegistry, get_default_registry, tool


def _build_base_subagent_tool_registry() -> ToolRegistry:
    """Builds a full tool registry that can be whitelisted for subagents."""
    from tools.clarify import register_clarify_tools
    from tools.confirmation import register_confirmation_tools
    from tools.developer_tools import register_developer_tools
    from tools.execute_code import register_execute_code_tools
    from tools.memory_tool import register_memory_tools
    from tools.process_manager import register_process_manager_tools
    from tools.skill_tool import register_skill_tools
    from tools.ui_navigator_tools import register_ui_navigator_tools
    from tools.vision_tool import register_vision_tools
    from tools.valstorm_client import resolve_valstorm_credentials
    from tools.valstorm_tools import register_valstorm_tools
    from tools.web_tools import register_web_tools

    registry = get_default_registry()
    register_developer_tools(registry)
    register_web_tools(registry)
    register_ui_navigator_tools(registry)
    register_clarify_tools(registry)
    register_confirmation_tools(registry)
    register_vision_tools(registry)
    register_execute_code_tools(registry)
    register_process_manager_tools(registry)
    register_skill_tools(registry)
    register_memory_tools(registry)

    token, base_url = resolve_valstorm_credentials()
    if token:
        try:
            from tools.valstorm_client import ValstormApiClient
            client = ValstormApiClient(token=token, base_url=base_url)
            register_valstorm_tools(registry=registry, client=client)
        except Exception:
            pass

    return registry


_SUBAGENT_PROVIDER_OVERRIDE: Optional[Any] = None


def set_subagent_provider_override(provider: Optional[Any]) -> None:
    """Sets a global provider override for testing subagent execution."""
    global _SUBAGENT_PROVIDER_OVERRIDE
    _SUBAGENT_PROVIDER_OVERRIDE = provider


def _get_subagent_provider(provider_name: str, api_key: Optional[str] = None) -> Optional[Any]:
    """Resolves provider instance for subagent execution, supporting both direct keys and Valstorm gateway auth."""
    if _SUBAGENT_PROVIDER_OVERRIDE is not None:
        return _SUBAGENT_PROVIDER_OVERRIDE

    keystore = KeyStore()
    p_name = (provider_name or "gemini").lower()
    resolved_key = keystore.get_api_key(p_name, override_key=api_key)
    clean_valstorm_base = None

    try:
        from tools.valstorm_client import get_current_valstorm_auth
        scoped_auth = get_current_valstorm_auth()
    except Exception:
        scoped_auth = None

    if p_name in ("valstorm", "hosted", "managed") and scoped_auth and scoped_auth[0]:
        p_name = "valstorm"
        resolved_key = scoped_auth[0]
        base = (scoped_auth[1] or "").rstrip("/")
        if base:
            if base.endswith("/ai"):
                base = base[:-3].rstrip("/")
            if not base.endswith("/v1") and "/v1" not in base:
                base = f"{base}/v1"
            clean_valstorm_base = f"{base}/ai"

    if not resolved_key:
        for fallback in ["gemini", "digitalocean", "openai", "anthropic"]:
            k = keystore.get_api_key(fallback)
            if k:
                resolved_key = k
                p_name = fallback
                break

    if not resolved_key and scoped_auth and scoped_auth[0]:
        p_name = "valstorm"
        resolved_key = scoped_auth[0]
        base = (scoped_auth[1] or "").rstrip("/")
        if base:
            if base.endswith("/ai"):
                base = base[:-3].rstrip("/")
            if not base.endswith("/v1") and "/v1" not in base:
                base = f"{base}/v1"
            clean_valstorm_base = f"{base}/ai"

    if not resolved_key:
        return None

    from providers import build_fallback_chain
    return build_fallback_chain(
        primary_provider=p_name,
        keystore=keystore,
        api_key=resolved_key if p_name == "valstorm" else None,
        base_url=clean_valstorm_base if p_name == "valstorm" else None,
    )


class SubagentTaskSession:
    """Represents a tracked asynchronous subagent task session."""

    def __init__(
        self,
        task_id: str,
        child_session_id: str,
        profile: str,
        profile_name: str,
        goal: str,
        context: Optional[str],
        model_name: str,
        provider_name: str,
        scoped_tools: List[str],
        child_session: SessionState,
        auto_tune: bool = True,
    ):
        self.task_id = task_id
        self.child_session_id = child_session_id
        self.profile = profile
        self.profile_name = profile_name
        self.goal = goal
        self.context = context
        self.model_name = model_name
        self.provider_name = provider_name
        self.scoped_tools = scoped_tools
        self.child_session = child_session
        self.auto_tune = auto_tune
        self.auto_tune_decision: Optional[Dict[str, Any]] = None
        self.status = "running"  # running, completed, failed, timed_out, killed
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.outcome: Optional[str] = None
        self.error: Optional[str] = None
        self.tools_executed: List[str] = []
        self.asyncio_task: Optional[asyncio.Task] = None

    @property
    def duration_sec(self) -> float:
        end = self.end_time if self.end_time else time.time()
        return round(end - self.start_time, 2)

    def format_summary(self) -> str:
        status_label = self.status.upper()
        header_action = "COMPLETE" if self.status == "completed" else status_label
        summary_lines = [
            f"🤖 [SUBAGENT DELEGATION {header_action} - PROFILE: {self.profile.upper()}]",
            f"Task ID: {self.task_id}",
            f"Child Session ID: {self.child_session_id}",
            f"Profile: {self.profile_name} ({self.profile})",
            f"Model: {self.model_name} via {self.provider_name}",
            f"Scoped Tools: [{', '.join(self.scoped_tools)}]",
            f"Status: {status_label}",
            f"Duration: {self.duration_sec}s",
            f"Tokens: Prompt={self.child_session.total_prompt_tokens} | Completion={self.child_session.total_completion_tokens} | Total={self.child_session.total_tokens}",
        ]

        if self.tools_executed:
            summary_lines.append(f"Tools Executed: [{', '.join(self.tools_executed)}]")
        if self.auto_tune_decision:
            summary_lines.append(
                f"Auto-Tune: Tier={self.auto_tune_decision.get('tier')} | "
                f"Model={self.model_name} | Effort={self.auto_tune_decision.get('reasoning_effort')}"
            )

        summary_lines.extend([
            "",
            "--- GOAL ---",
            self.goal,
        ])

        if self.context:
            summary_lines.extend(["", "--- CONTEXT PROVIDED ---", self.context.strip()])

        summary_lines.extend([
            "",
            "--- SUBAGENT OUTCOME & EXECUTION SUMMARY ---",
            self.outcome or (f"Subagent execution {self.status}." if self.status != "running" else "Subagent is currently running..."),
        ])

        return "\n".join(summary_lines)


class SubagentRegistry:
    """In-memory registry managing tracked subagent task sessions."""

    _instance: Optional["SubagentRegistry"] = None

    def __init__(self):
        self._tasks: Dict[str, SubagentTaskSession] = {}

    @classmethod
    def get_instance(cls) -> "SubagentRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, session: SubagentTaskSession) -> None:
        self._tasks[session.task_id] = session

    def get(self, task_id: str) -> Optional[SubagentTaskSession]:
        return self._tasks.get(task_id)

    def list_tasks(self) -> List[SubagentTaskSession]:
        return list(self._tasks.values())

    def clear(self) -> None:
        self._tasks.clear()


async def _run_subagent_coroutine(
    task_session: SubagentTaskSession,
    user_prompt: str,
    effective_max_turns: int,
    timeout_sec: float,
    scoped_registry: ToolRegistry,
) -> None:
    """Executes the subagent turn loop in the background and populates the task session."""
    provider_inst = _get_subagent_provider(task_session.provider_name)
    if provider_inst:
        from core.react import ReActEngine, StreamEventType
        child_engine = ReActEngine(provider=provider_inst, tools=scoped_registry)
        final_msg_content = ""
        try:
            from server import get_current_event_queue
            eq = get_current_event_queue()
        except Exception:
            eq = None

        try:
            if eq:
                await eq.put({
                    "event": "subagent.status",
                    "task_id": task_session.task_id,
                    "profile": task_session.profile,
                    "status": "running",
                    "goal": task_session.goal,
                    "timestamp": int(time.time() * 1000),
                })

            base_timeout = timeout_sec
            timeout_multiplier = [1.0, 1.5, 2.0]
            max_attempts = 3

            for attempt in range(max_attempts):
                current_timeout = base_timeout * timeout_multiplier[attempt]
                if attempt > 0:
                    if eq:
                        try:
                            await eq.put({
                                "event": "subagent.status",
                                "task_id": task_session.task_id,
                                "profile": task_session.profile,
                                "status": "retrying",
                                "goal": f"(Attempt {attempt + 1}/{max_attempts}) {task_session.goal}",
                                "timestamp": int(time.time() * 1000),
                            })
                        except Exception:
                            pass

                try:
                    final_msg_content = ""

                    async def _consume_stream():
                        nonlocal final_msg_content
                        effective_input = (
                            user_prompt
                            if attempt == 0
                            else f"{user_prompt}\n\n[SYSTEM DIRECTIVE: Previous execution attempt timed out. Please execute the required tool actions directly now without exploratory file browsing.]"
                        )
                        async for event in child_engine.run_turn_stream(
                            session=task_session.child_session,
                            user_input=effective_input,
                            model=task_session.model_name,
                            max_iterations=effective_max_turns,
                            auto_tune=task_session.auto_tune,
                        ):
                            if event.event_type == StreamEventType.TOOL_CALL_DETECTED and event.tool_call:
                                if event.tool_call.name not in task_session.tools_executed:
                                    task_session.tools_executed.append(event.tool_call.name)
                                if eq:
                                    await eq.put({
                                        "event": "subagent.status",
                                        "task_id": task_session.task_id,
                                        "profile": task_session.profile,
                                        "status": "running",
                                        "tool": event.tool_call.name,
                                        "goal": task_session.goal,
                                        "timestamp": int(time.time() * 1000),
                                    })
                            elif event.event_type == StreamEventType.TEXT_CHUNK and event.delta:
                                final_msg_content += event.delta
                            elif event.event_type == StreamEventType.TURN_COMPLETE and event.message:
                                if event.message.content:
                                    final_msg_content = event.message.content

                    await asyncio.wait_for(_consume_stream(), timeout=current_timeout)
                    task_session.outcome = final_msg_content or "(Subagent completed turn with no text response)"
                    task_session.status = "completed"
                    break
                except asyncio.TimeoutError:
                    if attempt == max_attempts - 1:
                        task_session.status = "timed_out"
                        task_session.outcome = f"Subagent execution timed out after {max_attempts} attempts (final timeout: {current_timeout:.1f}s)."
                    else:
                        continue
                except asyncio.CancelledError:
                    task_session.status = "killed"
                    task_session.outcome = "Subagent execution was cancelled."
                    break
                except Exception as err:
                    task_session.status = "failed"
                    task_session.error = str(err)
                    task_session.outcome = f"Subagent error during execution: {err}"
                    break
        finally:
            task_session.end_time = time.time()
            if task_session.child_session.active_model:
                task_session.model_name = task_session.child_session.active_model
            if (task_session.child_session.metadata or {}).get("last_autotune_decision"):
                task_session.auto_tune_decision = task_session.child_session.metadata["last_autotune_decision"]
            for m in task_session.child_session.messages:
                if m.tool_calls:
                    for tc in m.tool_calls:
                        if tc.name not in task_session.tools_executed:
                            task_session.tools_executed.append(tc.name)
            if eq:
                try:
                    await eq.put({
                        "event": "subagent.status",
                        "task_id": task_session.task_id,
                        "profile": task_session.profile,
                        "status": task_session.status,
                        "summary": task_session.outcome,
                        "goal": task_session.goal,
                        "tokens": {
                            "input": task_session.child_session.total_prompt_tokens,
                            "output": task_session.child_session.total_completion_tokens,
                            "total": task_session.child_session.total_tokens,
                        },
                        "timestamp": int(time.time() * 1000),
                    })
                except Exception:
                    pass
    else:
        # Fallback for offline/mock test environments without configured API key
        task_session.outcome = (
            f"Subagent '{task_session.profile}' initialized with scoped tools: [{', '.join(task_session.scoped_tools)}].\n"
            f"Ready to process goal: {task_session.goal}"
        )
        task_session.status = "completed"
        task_session.end_time = time.time()


@tool
async def delegate_task(
    profile: str,
    goal: str,
    context: Optional[str] = None,
    max_turns: Optional[int] = None,
    timeout_sec: Optional[float] = None,
    background: bool = False,
    action: str = "spawn",
    task_id: Optional[str] = None,
    auto_tune: Optional[bool] = None,
) -> str:
    """Spawns an isolated child subagent running a named profile to complete a delegated goal.

    Use this tool to delegate specialized, multi-step subtasks (e.g. coding, deep codebase search,
    test verification, or documentation authoring) to an isolated subagent without polluting
    the orchestrator's context window.

    Supports asynchronous background execution (`background=True`), allowing the orchestrator
    to spawn multiple specialized subagents concurrently in parallel without blocking!

    Actions:
      - 'spawn': Starts a subagent. Set `background=True` to run non-blocking in background, or `False` to wait.
      - 'poll': Checks status and intermediate progress of a background subagent (`task_id`).
      - 'wait': Awaits completion of a background subagent (`task_id`) or all running subagents (`task_id='all'`).
      - 'list': Lists all active and recent subagent tasks.
      - 'kill': Cancels a running background subagent task.

    Named profiles available:
      - 'developer': Specialized in monorepo coding, refactoring, test execution, and debugging.
      - 'researcher': Specialized in codebase search, architecture exploration, and data synthesis.
      - 'backend-tester': Specialized in test execution, regression diagnosis, and test suite verification.
      - 'writer': Specialized in technical documentation, API specifications, and architecture specs.
      - 'valstorm-assistant': General assistant for platform queries.
      - 'slack-agent': Specialist in Slack messaging, channel discovery, and notifications.
      - Any custom profile defined in ~/.valstorm/profiles/<name>.json.

    Args:
        profile: Name of the profile to execute the subagent under (e.g. 'developer', 'researcher').
        goal: Clear, actionable description of what the subagent must accomplish.
        context: Optional background context, file paths, constraints, or prior findings to guide the subagent.
        max_turns: Optional override for the subagent's maximum iterations (defaults to profile config, typically 25-30).
        timeout_sec: Optional execution timeout in seconds (default: 180s, or VALSTORM_SUBAGENT_TIMEOUT env var).
        background: If True, executes asynchronously in the background and returns task_id immediately.
        action: Management action ('spawn', 'poll', 'wait', 'list', 'kill'). Defaults to 'spawn'.
        task_id: The unique task identifier used with poll, wait, or kill.

    Returns:
        A structured summary report of the child subagent execution or background task notice.
    """
    clean_action = (action or "spawn").strip().lower()
    registry = SubagentRegistry.get_instance()

    # --- ACTION: LIST ---
    if clean_action == "list":
        tasks = registry.list_tasks()
        if not tasks:
            return "No subagent tasks have been spawned in this runtime session."
        lines = ["🤖 [ACTIVE & RECENT SUBAGENT SESSIONS]", ""]
        for t in tasks:
            lines.append(
                f"- [{t.status.upper()}] Task ID: {t.task_id} | Profile: {t.profile} | "
                f"Duration: {t.duration_sec}s | Tokens: {t.child_session.total_tokens} | Goal: {t.goal[:60]}"
            )
        return "\n".join(lines)

    # --- ACTION: POLL ---
    if clean_action == "poll":
        clean_task_id = (task_id or "").strip()
        if not clean_task_id:
            return "Error: A task_id must be provided to poll a subagent task."
        task_obj = registry.get(clean_task_id)
        if not task_obj:
            return f"Error: No subagent task found with ID '{clean_task_id}'."
        if task_obj.status == "running":
            tools_running = f" | Tools so far: [{', '.join(task_obj.tools_executed)}]" if task_obj.tools_executed else ""
            return (
                f"⏳ [SUBAGENT RUNNING] Task ID: {task_obj.task_id} | Profile: {task_obj.profile}\n"
                f"Elapsed: {task_obj.duration_sec}s | Tokens: {task_obj.child_session.total_tokens}{tools_running}\n"
                f"Goal: {task_obj.goal[:100]}..."
            )
        return task_obj.format_summary()

    # --- ACTION: WAIT ---
    if clean_action == "wait":
        clean_task_id = (task_id or "all").strip()
        wait_timeout = timeout_sec if timeout_sec is not None else 180.0

        if clean_task_id and clean_task_id != "all":
            task_obj = registry.get(clean_task_id)
            if not task_obj:
                return f"Error: No subagent task found with ID '{clean_task_id}'."
            if task_obj.status == "running" and task_obj.asyncio_task and not task_obj.asyncio_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(task_obj.asyncio_task), timeout=wait_timeout)
                except asyncio.TimeoutError:
                    return f"⏳ [SUBAGENT STILL RUNNING] Task '{clean_task_id}' did not complete within {wait_timeout}s. Elapsed: {task_obj.duration_sec}s."
            return task_obj.format_summary()
        else:
            # Wait for all running tasks
            running_tasks = [t for t in registry.list_tasks() if t.status == "running" and t.asyncio_task and not t.asyncio_task.done()]
            if not running_tasks:
                completed = [t.format_summary() for t in registry.list_tasks()]
                return "No active background subagents currently running." if not completed else "\n\n" + ("=" * 60) + "\n\n".join(completed)
            try:
                await asyncio.wait_for(
                    asyncio.gather(*[asyncio.shield(t.asyncio_task) for t in running_tasks], return_exceptions=True),
                    timeout=wait_timeout,
                )
            except asyncio.TimeoutError:
                pass
            summaries = [t.format_summary() for t in running_tasks]
            return "\n\n" + ("=" * 60) + "\n\n".join(summaries)

    # --- ACTION: KILL ---
    if clean_action == "kill":
        clean_task_id = (task_id or "").strip()
        if not clean_task_id:
            return "Error: A task_id must be provided to kill a subagent task."
        task_obj = registry.get(clean_task_id)
        if not task_obj:
            return f"Error: No subagent task found with ID '{clean_task_id}'."
        if task_obj.status == "running" and task_obj.asyncio_task and not task_obj.asyncio_task.done():
            task_obj.asyncio_task.cancel()
            task_obj.status = "killed"
            task_obj.end_time = time.time()
            return f"✔ Subagent task '{clean_task_id}' ({task_obj.profile}) was successfully cancelled."
        return f"Subagent task '{clean_task_id}' is not running (current status: {task_obj.status})."

    # --- ACTION: SPAWN (Default) ---
    clean_profile = (profile or "").strip().lower()
    if not clean_profile:
        return "Error: A target profile name must be provided for delegate_task."

    clean_goal = (goal or "").strip()
    if not clean_goal:
        return "Error: A goal must be provided for delegate_task."

    profile_cfg = load_profile(clean_profile)
    task_id_generated = f"subtask_{uuid.uuid4().hex[:8]}"
    child_session_id = f"aich_sub_{uuid.uuid4().hex[:12]}"
    model_name = profile_cfg.get("model", "gemini-flash-latest")
    provider_name = profile_cfg.get("provider", "gemini")
    allowed_tools = profile_cfg.get("allowed_tools") or [
        "terminal_exec",
        "patch_file",
        "write_file",
        "read_file",
        "search_files",
        "calculator",
        "execute_code",
        "skill_view",
        "skill_list",
    ]
    effective_max_turns = max_turns if max_turns is not None else profile_cfg.get("max_turns", 30)

    resolved_timeout = timeout_sec
    if resolved_timeout is None:
        try:
            resolved_timeout = float(os.environ.get("VALSTORM_SUBAGENT_TIMEOUT", str(profile_cfg.get("timeout_sec", 180.0))))
        except (ValueError, TypeError):
            resolved_timeout = 180.0

    # Filter registry to allowed tools (disallowing recursive delegation)
    safe_allowed_tools = [t for t in allowed_tools if t not in ("delegate_task", "subagent_manage")]
    base_registry = _build_base_subagent_tool_registry()
    scoped_registry = base_registry.filter_by_whitelist(safe_allowed_tools)

    ctx_mgr = WorkspaceContextManager()
    system_prompt = ctx_mgr.build_system_prompt(profile=profile_cfg)

    child_session = SessionState(
        session_id=child_session_id,
        name=f"Subagent Turn [{clean_profile}]: {clean_goal[:40]}",
        active_model=model_name,
        active_provider=provider_name,
        metadata={
            "parent_session_id": "orchestrator",
            "task_id": task_id_generated,
            "profile": clean_profile,
            "delegated_goal": clean_goal,
        },
    )

    system_msg = Message(
        role="system",
        content=system_prompt,
        model=model_name,
        provider=provider_name,
    )
    child_session.add_message(system_msg)

    user_prompt = f"Goal:\n{clean_goal}"
    if context:
        user_prompt += f"\n\nContext:\n{context.strip()}"

    env_auto_tune = os.environ.get("VALSTORM_SUBAGENT_AUTO_TUNE")
    if auto_tune is not None:
        effective_auto_tune = bool(auto_tune)
    elif env_auto_tune is not None and env_auto_tune.strip() != "":
        effective_auto_tune = env_auto_tune.strip().lower() not in ("0", "false", "no", "off")
    else:
        effective_auto_tune = bool(profile_cfg.get("auto_tune", True))

    task_session = SubagentTaskSession(
        task_id=task_id_generated,
        child_session_id=child_session_id,
        profile=clean_profile,
        profile_name=profile_cfg.get("name", clean_profile),
        goal=clean_goal,
        context=context,
        model_name=model_name,
        provider_name=provider_name,
        scoped_tools=safe_allowed_tools,
        child_session=child_session,
        auto_tune=effective_auto_tune,
    )

    # Launch execution as background task
    task_session.asyncio_task = asyncio.create_task(
        _run_subagent_coroutine(
            task_session=task_session,
            user_prompt=user_prompt,
            effective_max_turns=effective_max_turns,
            timeout_sec=resolved_timeout,
            scoped_registry=scoped_registry,
        )
    )
    registry.register(task_session)

    if background:
        # Non-blocking async spawn
        return (
            f"🤖 [SUBAGENT DELEGATION STARTED IN BACKGROUND]\n"
            f"Task ID: {task_id_generated}\n"
            f"Child Session ID: {child_session_id}\n"
            f"Profile: {profile_cfg.get('name', clean_profile)} ({clean_profile})\n"
            f"Model: {model_name} via {provider_name}\n"
            f"Scoped Tools: [{', '.join(safe_allowed_tools)}]\n"
            f"Status: RUNNING\n"
            f"Goal: {clean_goal}\n\n"
            f"The subagent is running concurrently in the background.\n"
            f"- Poll progress: delegate_task(action='poll', task_id='{task_id_generated}')\n"
            f"- Wait for completion: delegate_task(action='wait', task_id='{task_id_generated}')\n"
            f"- Wait for all subagents: delegate_task(action='wait', task_id='all')"
        )

    # Synchronous blocking wait
    await task_session.asyncio_task
    return task_session.format_summary()


@tool
async def subagent_manage(
    action: str,
    task_id: Optional[str] = None,
    profile: Optional[str] = None,
    goal: Optional[str] = None,
    context: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    max_turns: Optional[int] = None,
    background: bool = True,
    auto_tune: Optional[bool] = None,
) -> str:
    """Manages background subagents (list, poll, wait, kill, spawn).

    Actions:
      - 'list': Lists all tracked subagent sessions and their statuses.
      - 'poll': Checks progress and incremental output of a running subagent (`task_id`).
      - 'wait': Blocks until the specified subagent (`task_id`) or all running subagents (`task_id='all'`) complete.
      - 'kill': Cancels a running subagent task (`task_id`).
      - 'spawn': Starts a new background subagent session (`profile`, `goal`, `context`).

    Args:
        action: Management action ('list', 'poll', 'wait', 'kill', 'spawn').
        task_id: The unique task ID of the subagent session (required for poll, wait, kill).
        profile: Profile name for spawning a new subagent (e.g. 'developer', 'researcher').
        goal: Clear, actionable description of what the subagent must accomplish.
        context: Optional background context or file paths.
        timeout_sec: Maximum seconds to wait or execution timeout.
        max_turns: Max ReAct iterations for subagent.
        background: When spawning, whether to run in background (default True).

    Returns:
        Formatted subagent status, summary report, or list of sessions.
    """
    return await delegate_task(
        profile=profile,
        goal=goal,
        context=context,
        max_turns=max_turns,
        timeout_sec=timeout_sec,
        background=background,
        action=action,
        task_id=task_id,
        auto_tune=auto_tune,
    )


def _get_all_known_tool_names() -> set[str]:
    """Returns the complete set of valid runtime tool names for profile validation."""
    known: set[str] = {
        "delegate_task",
        "subagent_manage",
        "agent_manage",
        "skill_manage",
        "skill_view",
        "skill_list",
    }
    for prof in BUILTIN_PROFILES.values():
        for t_name in prof.get("allowed_tools", []):
            known.add(str(t_name))
    try:
        reg = _build_base_subagent_tool_registry()
        for t_name in reg.list_tools():
            known.add(str(t_name))
    except Exception:
        pass
    return known


def _resolve_writable_profiles_dir() -> Path:
    """Resolves the writable directory for saving custom profile JSON files on disk."""
    custom = os.environ.get("VALSTORM_PROFILES_DIR")
    if custom and custom.strip():
        p = Path(custom).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    prof_dir = (Path.home() / ".valstorm" / "profiles").resolve()
    prof_dir.mkdir(parents=True, exist_ok=True)
    return prof_dir


async def _persist_agent_to_cloud(
    record_payload: Dict[str, Any],
    existing_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Creates or updates an `ai_agent` record in the authenticated tenant's MongoDB."""
    try:
        from tools.valstorm_client import ValstormApiClient, get_current_valstorm_auth
        scoped_auth = get_current_valstorm_auth()
    except Exception:
        scoped_auth = None

    if not scoped_auth or not scoped_auth[0] or not scoped_auth[1]:
        return None

    token, base_url = scoped_auth
    try:
        async with ValstormApiClient(token=token, base_url=base_url) as client:
            if existing_id:
                upd = dict(record_payload)
                upd["id"] = existing_id
                res = await client.records_update("ai_agent", [upd])
            else:
                res = await client.records_create("ai_agent", [record_payload])
            if isinstance(res, list) and res:
                return res[0] if isinstance(res[0], dict) else {"id": existing_id}
            if isinstance(res, dict):
                recs = res.get("records") or res.get("data")
                if isinstance(recs, list) and recs and isinstance(recs[0], dict):
                    return recs[0]
                return res
    except Exception:
        pass
    return None


async def _delete_agent_from_cloud(record_id: str) -> bool:
    """Deletes an `ai_agent` record from the authenticated tenant's MongoDB."""
    if not record_id:
        return False
    try:
        from tools.valstorm_client import ValstormApiClient, get_current_valstorm_auth
        scoped_auth = get_current_valstorm_auth()
    except Exception:
        scoped_auth = None

    if not scoped_auth or not scoped_auth[0] or not scoped_auth[1]:
        return False

    token, base_url = scoped_auth
    try:
        async with ValstormApiClient(token=token, base_url=base_url) as client:
            await client.records_delete("ai_agent", [record_id])
            return True
    except Exception:
        return False


async def execute_agent_mutation(
    action: str,
    slug: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    model_tier: Optional[str] = None,
    provider: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    allowed_tools: Optional[List[str]] = None,
    skill_slugs: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    vaults: Optional[List[str]] = None,
    scoped_paths: Optional[List[str]] = None,
    max_turns: Optional[int] = None,
) -> Dict[str, Any]:
    """Executes a validated agent profile mutation and returns a structured receipt."""
    act = (action or "").strip().lower()
    if act not in ("create", "update", "attach_skills", "detach_skills", "delete", "list"):
        return {
            "status": "error",
            "message": (
                f"Error: Invalid action '{action}'. Must be one of: "
                "'create', 'update', 'attach_skills', 'detach_skills', 'delete', 'list'."
            ),
        }

    if act == "list":
        profs = list_available_profiles()
        lines = [f"🤖 Available Agent Profiles ({len(profs)} total):"]
        for p in profs:
            p_slug = p.get("api_name") or "unknown"
            p_name = p.get("name") or p_slug
            p_tools = len(p.get("allowed_tools") or [])
            p_skills = p.get("attached_skill_slugs") or []
            skills_str = f" | skills: [{', '.join(p_skills[:4])}]" if p_skills else ""
            lines.append(f"• {p_slug} ({p_name}) — {p_tools} tools{skills_str}")
        return {"status": "success", "action": "list", "profiles": profs, "message": "\n".join(lines)}

    clean_slug = re.sub(r"[^a-z0-9\-]", "-", (slug or name or "").strip().lower().replace("_", "-").replace(" ", "-").strip("-"))
    if not clean_slug or len(clean_slug) < 2:
        return {"status": "error", "message": "Error: A valid kebab-case profile `slug` is required."}

    try:
        from tools.valstorm_client import get_current_tenant_context
        ctx = get_current_tenant_context()
    except Exception:
        ctx = None

    is_cloud_mode = bool(ctx and ctx.is_cloud_mode)
    org_id = ctx.org_id if ctx else None

    all_profs = list_available_profiles()
    existing = next((p for p in all_profs if str(p.get("api_name", "")).lower() == clean_slug), None)

    # 1. Handle DELETE
    if act == "delete":
        if clean_slug in BUILTIN_PROFILES:
            return {
                "status": "error",
                "message": f"Error: Profile '{clean_slug}' is a built-in core profile and cannot be deleted.",
            }
        if not existing:
            return {"status": "error", "message": f"Error: Profile '{clean_slug}' not found."}

        prof_dir = _resolve_writable_profiles_dir()
        disk_file = prof_dir / f"{clean_slug}.json"
        deleted_disk = False
        if disk_file.is_file():
            disk_file.unlink()
            deleted_disk = True

        deleted_cloud = False
        if existing.get("id"):
            deleted_cloud = await _delete_agent_from_cloud(str(existing["id"]))

        invalidate_tenant_profiles_cache(org_id)
        return {
            "status": "success",
            "action": "delete",
            "slug": clean_slug,
            "deleted_disk": deleted_disk,
            "deleted_cloud": deleted_cloud,
            "message": f"Deleted agent profile '{clean_slug}'.",
        }

    # 2. Validate allowed_tools & enforce privilege escalation guard
    if allowed_tools is not None:
        known_tools = _get_all_known_tool_names()
        unknown = [t for t in allowed_tools if t not in known_tools]
        if unknown:
            return {
                "status": "error",
                "message": (
                    f"Error: Unknown tool name(s) in `allowed_tools`: {', '.join(unknown)}. "
                    f"Only registered runtime tools may be assigned to an agent profile."
                ),
            }

        caller_prof = get_current_profile()
        if caller_prof and isinstance(caller_prof.get("allowed_tools"), list):
            caller_slug = str(caller_prof.get("api_name") or "").lower()
            if caller_slug not in ("chief-of-staff", "orchestrator"):
                caller_allowed = set(caller_prof["allowed_tools"])
                escalated = [t for t in allowed_tools if t not in caller_allowed]
                if escalated:
                    return {
                        "status": "error",
                        "message": (
                            f"Error: Privilege escalation blocked. Active profile '{caller_slug}' "
                            f"cannot grant tool(s) outside its own whitelist: {', '.join(escalated)}."
                        ),
                    }

    # 3. Build skill slug -> cloud ID map
    try:
        from tools.skill_tool import _index_all_skills
        indexed_skills = await asyncio.to_thread(_index_all_skills)
        slug_to_skill_id = {s["slug"]: s.get("id") for s in indexed_skills if s.get("slug")}
    except Exception:
        slug_to_skill_id = {}

    if act in ("attach_skills", "detach_skills"):
        if not existing:
            return {"status": "error", "message": f"Error: Profile '{clean_slug}' not found."}
        current_skills = list(existing.get("attached_skill_slugs") or [])
        target_slugs = [str(s).strip().lower() for s in (skill_slugs or []) if str(s).strip()]
        if act == "attach_skills":
            for s in target_slugs:
                if s not in current_skills:
                    current_skills.append(s)
        else:
            current_skills = [s for s in current_skills if s not in target_slugs]
        updated_prof = dict(existing)
        updated_prof["attached_skill_slugs"] = current_skills
    else:
        # create or update
        base_prof = dict(existing) if existing else {}
        eff_name = (name or base_prof.get("name") or clean_slug.replace("-", " ").title()).strip()
        eff_desc = (description if description is not None else (base_prof.get("description") or f"Specialist agent for {eff_name}.")).strip()
        eff_prompt = (system_prompt if system_prompt is not None else (base_prof.get("system_prompt") or f"You are {eff_name}.")).strip()
        eff_provider = (provider or base_prof.get("provider") or "valstorm").strip().lower()
        eff_model = (model or base_prof.get("model") or "gemini-flash-latest").strip()
        eff_tier = (model_tier or base_prof.get("model_tier") or "tier_2").strip()
        eff_effort = reasoning_effort if reasoning_effort is not None else base_prof.get("reasoning_effort", "medium")
        eff_tools = (
            list(allowed_tools)
            if allowed_tools is not None
            else list(base_prof.get("allowed_tools") or ["valstorm_sql_query", "skill_view", "skill_list", "calculator"])
        )
        eff_skills = (
            [str(s).strip().lower() for s in skill_slugs if str(s).strip()]
            if skill_slugs is not None
            else list(base_prof.get("attached_skill_slugs") or [])
        )
        eff_tags = list(tags) if tags is not None else list(base_prof.get("tag") or [])
        eff_vaults = list(vaults) if vaults is not None else list(base_prof.get("knowledge_vaults") or base_prof.get("vaults") or [])
        eff_paths = list(scoped_paths) if scoped_paths is not None else list(base_prof.get("scoped_paths") or [])
        eff_turns = int(max_turns or base_prof.get("max_turns") or 25)

        updated_prof = {
            "id": base_prof.get("id"),
            "name": eff_name,
            "api_name": clean_slug,
            "description": eff_desc,
            "provider": eff_provider,
            "model": eff_model,
            "model_tier": eff_tier,
            "reasoning_effort": eff_effort,
            "system_prompt": eff_prompt,
            "allowed_tools": eff_tools,
            "attached_skill_slugs": eff_skills,
            "tag": eff_tags,
            "knowledge_vaults": eff_vaults,
            "vaults": eff_vaults,
            "scoped_paths": eff_paths,
            "max_turns": eff_turns,
        }

    # Resolve cloud ai_skills IDs where available
    cloud_ai_skill_ids = [
        str(slug_to_skill_id[s])
        for s in updated_prof.get("attached_skill_slugs", [])
        if slug_to_skill_id.get(s)
    ]
    if cloud_ai_skill_ids:
        updated_prof["ai_skills"] = cloud_ai_skill_ids

    target_path_str = ""
    if not is_cloud_mode or os.environ.get("VALSTORM_PROFILES_DIR"):
        prof_dir = _resolve_writable_profiles_dir()
        target_file = prof_dir / f"{clean_slug}.json"
        target_file.write_text(json.dumps(updated_prof, indent=2), encoding="utf-8")
        target_path_str = str(target_file)

    cloud_payload = {
        "name": updated_prof["name"],
        "api_name": clean_slug,
        "description": updated_prof.get("description", ""),
        "provider": "Valstorm" if updated_prof.get("provider") == "valstorm" else updated_prof.get("provider", "Gemini").title(),
        "model": updated_prof.get("model", "gemini-flash-latest"),
        "model_tier": updated_prof.get("model_tier", "tier_2"),
        "reasoning_effort": updated_prof.get("reasoning_effort") or "medium",
        "system_prompt": updated_prof.get("system_prompt", ""),
        "allowed_tools": updated_prof.get("allowed_tools", []),
        "ai_skills": cloud_ai_skill_ids,
        "tag": updated_prof.get("tag", []),
        "vaults": updated_prof.get("knowledge_vaults", []),
        "scoped_paths": updated_prof.get("scoped_paths", []),
        "is_active": True,
    }
    cloud_rec = await _persist_agent_to_cloud(
        cloud_payload,
        existing_id=str(updated_prof["id"]) if updated_prof.get("id") else None,
    )
    if isinstance(cloud_rec, dict) and cloud_rec.get("id") and not updated_prof.get("id"):
        updated_prof["id"] = cloud_rec["id"]
        if target_path_str:
            Path(target_path_str).write_text(json.dumps(updated_prof, indent=2), encoding="utf-8")

    invalidate_tenant_profiles_cache(org_id)
    verb = {
        "create": "Created",
        "update": "Updated",
        "attach_skills": "Attached skills to",
        "detach_skills": "Detached skills from",
    }.get(act, "Updated")
    receipt = {
        "status": "success",
        "action": act,
        "slug": clean_slug,
        "profile": updated_prof,
        "path": target_path_str,
        "record_id": updated_prof.get("id"),
        "message": (
            f"{verb} agent profile '{clean_slug}' ({updated_prof['name']}) "
            f"with {len(updated_prof.get('allowed_tools', []))} tools and "
            f"skills={updated_prof.get('attached_skill_slugs', [])}."
        ),
    }
    try:
        from server import get_current_event_queue
        eq = get_current_event_queue()
        if eq is not None:
            eq.put_nowait({
                "event": "agent.updated",
                "action": act,
                "slug": clean_slug,
                "name": updated_prof.get("name") or clean_slug,
            })
    except Exception:
        pass
    return receipt


@tool
async def agent_manage(
    action: str,
    slug: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    model_tier: Optional[str] = None,
    provider: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    allowed_tools: Optional[List[str]] = None,
    skill_slugs: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    vaults: Optional[List[str]] = None,
    scoped_paths: Optional[List[str]] = None,
    max_turns: Optional[int] = None,
) -> str:
    """Creates, updates, equips, lists, or deletes specialist agent profiles (`ai_agent`).

    Newly created or updated profiles are immediately available to spawn via `delegate_task(profile='<slug>', goal='...')`.

    Actions:
      - 'create': Creates a new specialist agent persona with scoped tools, attached skills, and knowledge vaults.
      - 'update': Updates an existing agent persona's prompt, tools, skills, or vaults.
      - 'attach_skills': Attaches one or more procedural `skill_slugs` to an existing agent profile.
      - 'detach_skills': Removes one or more `skill_slugs` from an existing agent profile.
      - 'delete': Deletes a custom agent profile (built-in core profiles cannot be deleted).
      - 'list': Lists all available agent profiles and their attached skills.

    Args:
        action: Mutation action ('create', 'update', 'attach_skills', 'detach_skills', 'delete', 'list').
        slug: Kebab-case identifier for the agent profile (e.g. 'finance-controller', 'migration-specialist').
        name: Human-readable display name for the agent.
        description: Short description of the specialist's domain and responsibilities.
        system_prompt: Core persona instructions and operating directives for the agent.
        model: Target LLM model identifier (default: 'gemini-flash-latest').
        model_tier: Complexity tier ('tier_1', 'tier_2', 'tier_3', 'tier_4').
        provider: LLM provider ('valstorm', 'gemini', 'openai', 'anthropic').
        reasoning_effort: Thinking budget ('off', 'low', 'medium', 'high').
        allowed_tools: List of valid runtime tool names this specialist is permitted to execute.
        skill_slugs: List of procedural skill slugs to attach to this agent's system prompt.
        tags: Optional list of Knowledge Graph domain tags.
        vaults: Optional list of Knowledge Vault IDs (`vaul_...`) scoped to this agent.
        scoped_paths: Optional list of repository path globs prioritized for this agent.
        max_turns: Maximum ReAct iterations per delegated run (default: 25).

    Returns:
        Confirmation summary of the created or updated agent profile.
    """
    receipt = await execute_agent_mutation(
        action=action,
        slug=slug,
        name=name,
        description=description,
        system_prompt=system_prompt,
        model=model,
        model_tier=model_tier,
        provider=provider,
        reasoning_effort=reasoning_effort,
        allowed_tools=allowed_tools,
        skill_slugs=skill_slugs,
        tags=tags,
        vaults=vaults,
        scoped_paths=scoped_paths,
        max_turns=max_turns,
    )
    return str(receipt.get("message") or json.dumps(receipt))


def create_delegation_tools() -> List[Any]:
    """Returns list of delegation and agent management tool functions."""
    return [delegate_task, subagent_manage, agent_manage]


def register_delegation_tools(registry: ToolRegistry) -> ToolRegistry:
    """Registers delegation and agent management tools into the provided ToolRegistry."""
    for tool_fn in create_delegation_tools():
        registry.register(tool_fn)
    return registry
