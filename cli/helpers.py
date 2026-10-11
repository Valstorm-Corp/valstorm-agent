"""Helpers and execution pipelines for the Valstorm Agent Runtime CLI."""

import asyncio
import inspect
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from prompt_toolkit import PromptSession
from rich.console import Console

from core.context import WorkspaceContextManager, list_available_profiles, load_profile
from core.keystore import KeyStore
from core.memory import MemoryStore
from core.models import Message, SessionState, StreamEvent, StreamEventType, ToolResult, get_telemetry_badge
from core.react import ReActEngine
from core.storage import SessionStore
from core.tools import get_default_registry, ToolRegistry
from tools.developer_tools import register_developer_tools
from tools.memory_tool import register_memory_tools
from tools.valstorm_client import ValstormApiClient, resolve_valstorm_credentials
from tools.valstorm_tools import register_valstorm_tools
from tools import register_tier1_tools
from providers import (
    AnthropicProvider,
    BaseProvider,
    GeminiProvider,
    OpenAIProvider,
    resolve_provider_instance,
)

logger = logging.getLogger(__name__)

console = Console()


async def compress_session_context(
    session: SessionState,
    provider: Any,
    keep_last: int = 4,
) -> tuple[int, int]:
    """Summarizes older conversation turns to compress context window usage."""
    if len(session.messages) <= (keep_last + 2):
        return session.total_tokens, session.total_tokens

    to_compress: list[Message] = []
    to_keep: list[Message] = []

    initial_system: Optional[Message] = None
    for msg in session.messages:
        if msg.role == "system" and not initial_system:
            initial_system = msg
            break

    cutoff = max(1, len(session.messages) - keep_last)
    for idx, msg in enumerate(session.messages):
        if msg is initial_system:
            continue
        if idx < cutoff:
            to_compress.append(msg)
        else:
            to_keep.append(msg)

    if not to_compress:
        return session.total_tokens, session.total_tokens

    digest_parts = []
    for m in to_compress:
        role = m.role.upper()
        text = m.content or m.body or ""
        if m.tool_calls:
            text += f" [Tool Calls: {', '.join(tc.name for tc in m.tool_calls)}]"
        digest_parts.append(f"{role}: {text}")

    digest_text = "\n".join(digest_parts)

    prompt_msg = Message(
        role="user",
        content=(
            "Summarize the following historical conversation history into a concise, high-signal "
            "context summary. Capture key facts, decisions, file modifications, tool outputs, and goals:\n\n"
            f"{digest_text[:12000]}"
        ),
    )

    summary_msg, _ = await provider.generate(
        messages=[prompt_msg],
        model=session.active_model,
    )

    summary_content = summary_msg.content or "Historical context summary unavailable."

    compressed_sys_msg = Message(
        role="system",
        content=f"[COMPRESSED HISTORICAL CONTEXT SUMMARY]\n{summary_content}",
        model=session.active_model,
        provider=session.active_provider,
    )

    new_messages: list[Message] = []
    if initial_system:
        new_messages.append(initial_system)
    new_messages.append(compressed_sys_msg)
    new_messages.extend(to_keep)

    tokens_before = session.total_tokens
    session.messages = new_messages

    session.total_input_tokens = sum(m.usage.prompt_tokens for m in session.messages if m.usage)
    session.total_output_tokens = sum(m.usage.completion_tokens for m in session.messages if m.usage)

    tokens_after = session.total_tokens
    return tokens_before, tokens_after


def build_tool_registry(
    env: str = "local",
    token: Optional[str] = None,
    memory_store: Optional[MemoryStore] = None,
    session_store: Optional[SessionStore] = None,
) -> ToolRegistry:
    """Builds a ToolRegistry including built-in, developer, memory, and Valstorm platform tools."""
    registry = get_default_registry()

    # 1. Register Developer Toolbelt & Tier 1 Autonomy Suite
    register_developer_tools(registry)
    register_tier1_tools(registry)

    # 2. Register Memory & Session Search Tools
    register_memory_tools(registry, memory_store=memory_store, session_store=session_store)

    # 3. Register Valstorm Platform Tools if credentials available
    valstorm_token, base_url = resolve_valstorm_credentials(override_token=token, env=env)
    if valstorm_token:
        try:
            client = ValstormApiClient(token=valstorm_token, base_url=base_url)
            register_valstorm_tools(registry=registry, client=client)
        except Exception as e:
            console.print(f"[yellow][Warning][/yellow] Could not attach Valstorm tools: {e}")
    return registry


def ensure_provider_key(provider_name: str, api_key: Optional[str] = None, interactive: bool = True) -> str:
    """Resolves API key or interactively prompts user to enter & save it."""
    # 1. Explicit override passed in
    if api_key and api_key.strip():
        return api_key.strip()

    p_low = provider_name.strip().lower()
    # Vertex uses Google credentials (service account / ADC), not an API key.
    if p_low in ("vertex", "geap"):
        return ""
    # "aistudio" is the Gemini Developer API: same key as the "gemini" provider.
    if p_low in ("aistudio", "ai-studio"):
        provider_name = "gemini"

    # 2. For Valstorm provider, active CLI session credentials take precedence over stale static files
    if provider_name.lower() == "valstorm":
        token, _ = resolve_valstorm_credentials()
        if token:
            return token
        console.print("\n[bold red][Authentication Required][/bold red] You are running with the default Valstorm Managed Gateway (`valstorm`).")
        console.print("Please log in to your Valstorm organization via:")
        console.print("  [bold green]valstorm login pat <your_access_token>[/bold green]")
        console.print("\nOr to bring your own API key instead, specify a provider:")
        console.print("  [cyan]vsagent run \"your prompt\" --provider gemini[/cyan] (or openai/anthropic)\n")
        sys.exit(1)

    keystore = KeyStore()
    resolved = keystore.get_api_key(provider_name, override_key=api_key)
    if resolved:
        return resolved

    if provider_name.lower() == "valstorm":
        token, _ = resolve_valstorm_credentials()
        if token:
            return token

    if not interactive or not sys.stdin.isatty():
        env_var = KeyStore.PROVIDER_ENV_MAP.get(provider_name.lower(), [f"{provider_name.upper()}_API_KEY"])[0]
        console.print(f"\n[bold red][Error][/bold red] No API key found for provider '{provider_name}'.")
        console.print("You can set it via:")
        console.print(f"  1. Export environment variable: export {env_var}=your_key_here")
        console.print("  2. CLI option: --api-key your_key_here")
        console.print(f"  3. Save persistently: vsagent keys set {provider_name} your_key_here\n")
        sys.exit(1)

    console.print(f"\n[bold yellow][API Key Required][/bold yellow] No API key found for '{provider_name}'.")
    user_key = input(f"Enter your {provider_name.capitalize()} API key: ").strip()
    if not user_key:
        console.print("No key provided. Exiting.")
        sys.exit(1)

    save_choice = input("Save key to ~/.config/valstorm/keys.json for future runs? [Y/n]: ").strip().lower()
    if save_choice in ("", "y", "yes"):
        saved_path = keystore.save_api_key(provider_name, user_key, persist_to="config")
        console.print(f"[bold green][Saved][/bold green] Key saved to {saved_path}\n")

    return user_key


def get_provider(
    provider_name: str,
    api_key: Optional[str] = None,
    interactive: bool = True,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    enable_fallback: bool = True,
) -> BaseProvider:
    p_norm = provider_name.strip().lower()
    if p_norm in ("fallback", "cascade", "chained"):
        from providers import build_fallback_chain
        return build_fallback_chain()

    resolved_key = ensure_provider_key(provider_name, api_key=api_key, interactive=interactive)
    ks = KeyStore()

    from providers import build_fallback_chain, infer_cascade_tier
    cascade_tier = infer_cascade_tier(model_name=model)
    return build_fallback_chain(
        tier=cascade_tier,
        primary_provider=p_norm,
        primary_model=model,
        keystore=ks,
        enable_fallback=enable_fallback,
        base_url=base_url,
        api_key=resolved_key,
    )


async def sync_cli_turn_to_valstorm(
    session: SessionState,
    user_prompt: str,
    final_message: Message,
    valstorm_env: str = "prod",
    valstorm_token: Optional[str] = None,
    tool_calls_executed: Optional[List[Dict[str, Any]]] = None,
    turn_input_tokens: Optional[int] = None,
    turn_output_tokens: Optional[int] = None,
    turn_cached_tokens: Optional[int] = None,
) -> bool:
    """Asynchronously syncs a completed CLI turn and token telemetry to Valstorm backend."""
    try:
        from tools.valstorm_client import resolve_valstorm_credentials
        import httpx
        import uuid

        token, base_url = resolve_valstorm_credentials(
            override_token=valstorm_token,
            env=valstorm_env,
        )
        if not token or not base_url:
            return False

        # Find user message ID from session messages
        user_msg_id = None
        for m in reversed(session.messages):
            if m.role == "user" and (m.content == user_prompt or m.body == user_prompt):
                user_msg_id = m.id
                break

        # Format tool calls: prioritize full turn execution roster, fallback to final_message
        tool_calls_payload = None
        if tool_calls_executed:
            tool_calls_payload = tool_calls_executed
        elif final_message.tool_calls:
            tool_calls_payload = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else str(tc.arguments),
                    },
                    "name": tc.name,
                    "status": "completed",
                }
                for tc in final_message.tool_calls
            ]

        base = base_url.rstrip("/")
        if not base.endswith("/v1") and "/v1" not in base:
            base = f"{base}/v1"
        endpoint = f"{base}/ai/chat/{session.session_id}/desktop-sync"

        u = final_message.usage
        input_toks = turn_input_tokens if turn_input_tokens is not None else (u.prompt_tokens if u else 0)
        output_toks = turn_output_tokens if turn_output_tokens is not None else (u.completion_tokens if u else 0)

        payload = {
            "text": final_message.content or "",
            "status": "completed",
            "role": "assistant",
            "session_id": session.session_id,
            "run_id": f"cli_{uuid.uuid4().hex[:12]}",
            "message_id": getattr(final_message, "id", None),
            "user_text": user_prompt,
            "user_message_id": user_msg_id,
            "input_tokens": input_toks,
            "output_tokens": output_toks,
            "model": getattr(final_message, "model", None) or getattr(session, "active_model", None),
            "provider": getattr(final_message, "provider", None) or getattr(session, "active_provider", None),
            "device_pid": os.getpid(),
        }
        if turn_cached_tokens:
            payload["cached_input_tokens"] = int(min(turn_cached_tokens, input_toks or turn_cached_tokens))
        if tool_calls_payload:
            payload["tool_calls"] = tool_calls_payload

        auth_header = token if token.startswith("Bearer ") else f"Bearer {token}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": auth_header,
        }

        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(endpoint, json=payload, headers=headers)
            if resp.status_code < 400:
                print(f"\033[90m[Cloud Sync] Synced to Valstorm Cloud ({session.session_id})\033[0m")
                return True
            return False
    except Exception:
        return False


def describe_backend(provider: Any) -> str:
    """Returns the concrete backend the active provider talks to (e.g. 'vertex', 'aistudio', 'valstorm-passthrough')."""
    try:
        inner = getattr(getattr(provider, "active_tier", None), "provider", None) or provider
        label_fn = getattr(inner, "backend_label", None)
        if callable(label_fn):
            return label_fn()
        name = getattr(inner, "provider_name", None)
        return f"{name} (openai-compatible)" if name == "valstorm" else str(name or "unknown")
    except Exception:
        return "unknown"


def preserve_partial_turn(session: SessionState, initial_msg_count: int, reason: str) -> None:
    """Keeps completed work from an interrupted/failed turn instead of discarding the whole turn.

    Previously a single API error erased every tool call and result from the turn, so the agent
    "forgot" work it had already done (files were still changed on disk). Now only an incomplete
    trailing assistant tool-call message (and any partial results after it) is removed, and a short
    assistant note records that the turn was interrupted.
    """
    msgs = session.messages
    if len(msgs) <= initial_msg_count:
        return
    for idx in range(len(msgs) - 1, initial_msg_count - 1, -1):
        m = msgs[idx]
        if m.role == "assistant" and m.tool_calls:
            call_ids = {tc.id for tc in m.tool_calls}
            answered = {
                t.tool_result.call_id
                for t in msgs[idx + 1:]
                if t.role == "tool" and t.tool_result is not None
            }
            if not call_ids.issubset(answered):
                del msgs[idx:]
            break
        if m.role == "assistant":
            break
    if len(msgs) > initial_msg_count:
        msgs.append(
            Message(
                role="assistant",
                content=f"[Turn interrupted before completion: {reason}. Work completed above is preserved; resume from where it stopped.]",
                model=getattr(session, "active_model", None),
                provider=getattr(session, "active_provider", None),
            )
        )


async def stream_and_render_turn(
    engine: ReActEngine,
    session: SessionState,
    prompt: str,
    model: str,
    max_iterations: Optional[int] = None,
    valstorm_env: str = "prod",
    valstorm_token: Optional[str] = None,
    cloud_sync: bool = True,
) -> tuple[Optional[Message], bool]:
    """Streams a turn live, rendering real-time text chunks and tool lifecycle events. Returns (final_message, is_interrupted)."""
    is_first_text_chunk = True
    in_text_stream = False
    final_message: Optional[Message] = None
    was_interrupted = False
    tool_calls_executed: List[Dict[str, Any]] = []
    active_tool_calls_map: Dict[str, Dict[str, Any]] = {}
    start_prompt_tokens = session.total_prompt_tokens
    start_cached_tokens = session.total_cached_input_tokens
    start_completion_tokens = session.total_completion_tokens

    async def _runner():
        nonlocal is_first_text_chunk, in_text_stream, final_message, was_interrupted, tool_calls_executed, active_tool_calls_map
        try:
            kwargs = {}
            try:
                sig = inspect.signature(engine.run_turn_stream)
                if "max_iterations" in sig.parameters and max_iterations is not None:
                    kwargs["max_iterations"] = max_iterations
            except Exception:
                if max_iterations is not None:
                    kwargs["max_iterations"] = max_iterations

            async for event in engine.run_turn_stream(
                session=session,
                user_input=prompt,
                model=model,
                **kwargs,
            ):
                if event.event_type == StreamEventType.TEXT_CHUNK and event.delta:
                    if not in_text_stream:
                        if not is_first_text_chunk:
                            print()
                        in_text_stream = True
                    sys.stdout.write(event.delta)
                    sys.stdout.flush()
                    is_first_text_chunk = False

                elif event.event_type == StreamEventType.TOOL_CALL_DETECTED and event.tool_call:
                    if in_text_stream:
                        print()
                        in_text_stream = False
                    tc = event.tool_call
                    call_id = tc.id or f"call_{len(tool_calls_executed)}_{int(time.time() * 1000)}"
                    tc_record = {
                        "id": call_id,
                        "type": "function",
                        "name": tc.name,
                        "tool_name": tc.name,
                        "args": tc.arguments,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else str(tc.arguments),
                        },
                        "status": "executing",
                    }
                    active_tool_calls_map[tc.name] = tc_record
                    tool_calls_executed.append(tc_record)
                    print(f"\n\033[93m⚡ [Tool Requested]\033[0m {tc.name}(args={tc.arguments})")

                elif event.event_type == StreamEventType.TOOL_EXECUTION_START and event.tool_call:
                    if in_text_stream:
                        print()
                        in_text_stream = False
                    tc = event.tool_call
                    print(f"  \033[94m⚙ [Tool Executing]\033[0m Running {tc.name}...")

                elif event.event_type == StreamEventType.TOOL_EXECUTION_RESULT and event.tool_result:
                    if in_text_stream:
                        print()
                        in_text_stream = False
                    res = event.tool_result

                    # Update tracked tool call record with execution result
                    matched_record = active_tool_calls_map.get(res.name)
                    if matched_record:
                        matched_record["status"] = "error" if res.is_error else "completed"
                        matched_record["stdout"] = str(res.output)
                        matched_record["result"] = res.output
                        matched_record["duration_sec"] = res.duration_sec

                    # Precision timing formatting
                    if res.duration_ms < 1.0:
                        timing_str = f"{res.duration_ms:.2f}ms"
                    elif res.duration_ms < 1000:
                        timing_str = f"{res.duration_sec:.1f}s"
                    else:
                        timing_str = f"{res.duration_sec:.2f}s"

                    # Payload size formatting
                    if res.payload_bytes < 1024:
                        size_str = f"{res.payload_bytes} B"
                    else:
                        size_str = f"{res.payload_bytes / 1024:.1f} KB"

                    items_str = f" · {res.item_count} records" if res.item_count is not None else ""
                    trunc_str = " · \033[93mtruncated\033[0m" if res.truncated else ""

                    meta_badge = f"\033[90m[{timing_str} · {size_str}{items_str}{trunc_str}]\033[0m"
                    if res.is_error:
                        print(f"  \033[91m✖ [Tool Error {meta_badge}]\033[0m {res.output}")
                    else:
                        print(f"  \033[92m✔ [Tool Result {meta_badge}]\033[0m {res.output}")

                elif event.event_type == StreamEventType.CONTEXT_COMPACTED:
                    if in_text_stream:
                        print()
                        in_text_stream = False
                    if event.delta:
                        print(event.delta)

                elif event.event_type == StreamEventType.ITERATION_LIMIT_REACHED:
                    if in_text_stream:
                        print()
                        in_text_stream = False
                    if event.delta:
                        print(event.delta)

                elif event.event_type == StreamEventType.ERROR:
                    if in_text_stream:
                        print()
                        in_text_stream = False
                    err_text = event.delta or (event.metadata.get("error") if event.metadata else None) or "Unknown API error"
                    print(f"\n\033[91m✖ [API Error]\033[0m {err_text}")

                elif event.event_type == StreamEventType.TURN_COMPLETE and event.message:
                    if in_text_stream:
                        print()
                        in_text_stream = False
                    final_message = event.message
                    if is_first_text_chunk and final_message.content and final_message.content.strip():
                        print(final_message.content)
        except (KeyboardInterrupt, asyncio.CancelledError):
            was_interrupted = True
        except Exception as e:
            if in_text_stream:
                print()
                in_text_stream = False
            print(f"\n\033[91m✖ [API Error]\033[0m {e}")
            preserve_partial_turn(session, initial_msg_count, f"API error: {str(e)[:300]}")

    initial_msg_count = len(session.messages)
    task = asyncio.create_task(_runner())
    try:
        await task
    except (KeyboardInterrupt, asyncio.CancelledError):
        was_interrupted = True
        if not task.done():
            task.cancel()
            try:
                await task
            except BaseException:
                pass
    except Exception as e:
        if in_text_stream:
            print()
            in_text_stream = False
        print(f"\n\033[91m✖ [API Error]\033[0m {e}")
        preserve_partial_turn(session, initial_msg_count, f"API error: {str(e)[:300]}")

    if was_interrupted:
        preserve_partial_turn(session, initial_msg_count, "interrupted by the user")
        if in_text_stream:
            print()
        print("\n\033[93m⚡ [In-progress prompt stopped. Type 'exit' or press Ctrl+C again to leave session]\033[0m")
        return None, True

    if in_text_stream:
        print()

    # Print token telemetry and session resumption metadata
    if final_message and final_message.usage:
        u = final_message.usage
        active_model = getattr(final_message, "model", None) or getattr(session, "active_model", None) or "gemini-flash-latest"
        active_provider = getattr(final_message, "provider", None) or getattr(session, "active_provider", None) or "valstorm"
        print(f"\n\033[90m[Usage Stats] Turn Tokens: Prompt={u.prompt_tokens} | Completion={u.completion_tokens} | Total={u.total_tokens}\033[0m")
        print(f"\033[90m[Session Cumulative] Prompt={session.total_prompt_tokens} | Completion={session.total_completion_tokens} | Total={session.total_tokens}\033[0m")
        print(f"\033[90m[Session Info] ID: {session.session_id} | Model: {active_model} ({active_provider})\033[0m")
        print(f"\033[90m[Resume Command] vsagent chat --session {session.session_id}\033[0m")

    # Cloud sync to Valstorm if credentials are present and not disabled
    if final_message and cloud_sync:
        turn_prompt_tokens = max(0, session.total_prompt_tokens - start_prompt_tokens)
        turn_comp_tokens = max(0, session.total_completion_tokens - start_completion_tokens)
        turn_cached_tokens = max(0, session.total_cached_input_tokens - start_cached_tokens)
        try:
            await sync_cli_turn_to_valstorm(
                session=session,
                user_prompt=prompt,
                final_message=final_message,
                valstorm_env=valstorm_env,
                valstorm_token=valstorm_token,
                tool_calls_executed=tool_calls_executed if tool_calls_executed else None,
                turn_input_tokens=turn_prompt_tokens if turn_prompt_tokens > 0 else None,
                turn_output_tokens=turn_comp_tokens if turn_comp_tokens > 0 else None,
                turn_cached_tokens=turn_cached_tokens if turn_cached_tokens > 0 else None,
            )
        except Exception as sync_err:
            logger.debug(f"Cloud sync failed: {sync_err}")

    return final_message, False


async def run_single_prompt(
    prompt: str,
    model: str = "gemini-flash-latest",
    provider_name: str = "valstorm",
    api_key: Optional[str] = None,
    valstorm_env: str = "prod",
    valstorm_token: Optional[str] = None,
    resume_session_id: Optional[str] = None,
    max_iterations: Optional[int] = None,
    profile: Optional[str] = None,
    cloud_sync: bool = True,
):
    session_store = SessionStore()
    memory_store = MemoryStore()

    active_model = model
    active_provider = provider_name
    profile_cfg = None

    if profile:
        profile_cfg = load_profile(profile)
        if model in ("gemini-3.6-flash", "gemini-flash-latest") and profile_cfg.get("model"):
            active_model = profile_cfg["model"]
        if provider_name in ("gemini", "valstorm") and profile_cfg.get("provider"):
            active_provider = profile_cfg["provider"].lower()

    provider = get_provider(active_provider, api_key=api_key, interactive=True)
    full_tools = build_tool_registry(
        env=valstorm_env,
        token=valstorm_token,
        memory_store=memory_store,
        session_store=session_store,
    )

    if profile_cfg and profile_cfg.get("allowed_tools"):
        tools = full_tools.filter_by_whitelist(profile_cfg["allowed_tools"])
    else:
        tools = full_tools

    engine = ReActEngine(provider=provider, tools=tools)

    if resume_session_id:
        session = session_store.load_session(resume_session_id)
        if not session:
            console.print(f"[bold red][Error][/bold red] Session '{resume_session_id}' not found.")
            sys.exit(1)
        console.print(f"[bold green][Resumed Session][/bold green] {session.session_id} ({len(session.messages)} prior messages)")
        if engine.compactor and engine.compactor.should_auto_compact(session):
            console.print("[dim cyan]🧹 [Auto-Compacting Context] Resumed session exceeds context window; compacting...[/dim cyan]")
            c_res = engine.compactor.compact(session)
            if c_res.compacted:
                console.print(f"[green]✔ [Auto-Compacted on Resume][/green] {c_res.summary}")
                session_store.save_session(session)
    else:
        session = SessionState(active_model=active_model, active_provider=active_provider)
        # Inject system prompt with profile and attached skills if starting fresh session
        sys_prompt = WorkspaceContextManager(memory_store=memory_store).build_system_prompt(profile=profile_cfg)
        session.add_message(Message(role="system", content=sys_prompt, model=active_model, provider=active_provider))

    prof_label = f" | Profile: {profile_cfg.get('name')} ({profile})" if profile_cfg else ""
    console.print(f"\n[dim]--- Running Task with Model: {active_model} ({active_provider} → {describe_backend(provider)}){prof_label} ---[/dim]")
    console.print(f"[bold cyan]Prompt:[/bold cyan] {prompt}\n")

    _, _ = await stream_and_render_turn(
        engine=engine,
        session=session,
        prompt=prompt,
        model=active_model,
        max_iterations=max_iterations,
        valstorm_env=valstorm_env,
        valstorm_token=valstorm_token,
        cloud_sync=cloud_sync,
    )
    session_store.save_session(session)


async def run_interactive_repl(
    engine: ReActEngine,
    session: SessionState,
    profile_slug: str = "valstorm-assistant",
    model: str = "gemini-flash-latest",
    provider_name: str = "valstorm",
    max_turns: int = 50,
    session_store: Optional[SessionStore] = None,
):
    """Runs a rich interactive REPL terminal session with prompt_toolkit, supporting /undo and commands."""
    prompt_session = PromptSession()
    console.print(f"\n[bold cyan]Valstorm Agent REPL[/bold cyan] (Profile: [bold]{profile_slug}[/bold], Model: [dim]{model}[/dim])")
    console.print("[dim]Type your message, '/undo' to revert last turn, '/compact' to compress context, or '/exit' to quit.\n[/dim]")

    from core.snapshot import get_snapshot_manager
    snapshot_mgr = get_snapshot_manager()
    turn_counter = 0

    while True:
        try:
            user_text = await prompt_session.prompt_async(f"[{profile_slug}] > ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Exiting REPL.[/dim]")
            break

        cleaned = user_text.strip()
        if not cleaned:
            continue

        if cleaned in ("/exit", "/quit", "exit", "quit"):
            console.print("[dim]Goodbye![/dim]")
            break

        if cleaned == "/help":
            console.print("\n[bold]Available REPL Commands:[/bold]")
            console.print("  /undo      - Revert filesystem modifications and state from the last turn")
            console.print("  /compact   - Summarize and compress earlier conversation history")
            console.print("  /tokens    - Show cumulative token usage for current session")
            console.print("  /exit      - Exit the interactive REPL\n")
            continue

        if cleaned == "/undo":
            success, msg = await snapshot_mgr.revert_turn(session.session_id)
            if success:
                console.print(f"[bold green]✔ {msg}[/bold green]")
                if len(session.messages) >= 2:
                    session.messages = session.messages[:-2]
                    if session_store:
                        session_store.save_session(session)
            else:
                console.print(f"[yellow]⚠ {msg}[/yellow]")
            continue

        if cleaned == "/tokens":
            badge = get_telemetry_badge(session)
            console.print(f"\n[bold cyan]Telemetry:[/bold cyan] {badge}")
            console.print(f"Total Prompt Tokens: {session.total_prompt_tokens:,}")
            console.print(f"Total Completion Tokens: {session.total_output_tokens:,}")
            console.print(f"Total Tokens: {session.total_tokens:,}\n")
            continue

        if cleaned == "/compact":
            provider = get_provider(provider_name)
            before, after = await compress_session_context(session, provider)
            console.print(f"[bold green]✔ Context compressed:[/bold green] {before:,} -> {after:,} tokens.")
            continue

        turn_counter += 1
        # Capture pre-turn working tree snapshot
        await snapshot_mgr.create_snapshot(session.session_id, turn_index=turn_counter)

        final_msg, interrupted = await stream_and_render_turn(
            engine=engine,
            session=session,
            prompt=cleaned,
            model=model,
            max_iterations=max_turns,
        )

        if final_msg and session_store:
            session_store.save_session(session)
