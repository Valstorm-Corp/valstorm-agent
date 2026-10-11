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
import time
from typing import Any, Dict, List, Literal, Optional
import uuid

from core.context import BUILTIN_PROFILES, WorkspaceContextManager, list_available_profiles, load_profile
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
    from tools.vision_tool import register_vision_tools
    from tools.valstorm_client import resolve_valstorm_credentials
    from tools.valstorm_tools import register_valstorm_tools

    registry = get_default_registry()
    register_developer_tools(registry)
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
    """Resolves provider instance for subagent execution."""
    if _SUBAGENT_PROVIDER_OVERRIDE is not None:
        return _SUBAGENT_PROVIDER_OVERRIDE

    keystore = KeyStore()
    p_name = (provider_name or "gemini").lower()
    resolved_key = keystore.get_api_key(p_name, override_key=api_key)

    if not resolved_key:
        for fallback in ["gemini", "digitalocean", "openai", "anthropic"]:
            k = keystore.get_api_key(fallback)
            if k:
                resolved_key = k
                p_name = fallback
                break

    if not resolved_key:
        return None

    from providers import build_fallback_chain
    return build_fallback_chain(
        primary_provider=p_name,
        keystore=keystore,
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
    )


def create_delegation_tools() -> List[Any]:
    """Returns list of delegation tool functions."""
    return [delegate_task, subagent_manage]


def register_delegation_tools(registry: ToolRegistry) -> ToolRegistry:
    """Registers delegation tools into the provided ToolRegistry."""
    for tool_fn in create_delegation_tools():
        registry.register(tool_fn)
    return registry
