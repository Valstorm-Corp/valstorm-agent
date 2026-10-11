"""Interactive REPL prompt for the Valstorm Agent Runtime."""

import asyncio
import json
import os
import sys
from typing import Optional

from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.table import Table

import cli
from core.context import WorkspaceContextManager, list_available_profiles, load_profile
from core.keystore import KeyStore
from core.memory import MemoryStore
from core.models import Message, SessionState, get_telemetry_badge
from core.react import ReActEngine

console = Console()


async def run_interactive_repl(
    initial_model: str = "gemini-flash-latest",
    initial_provider: str = "valstorm",
    api_key: Optional[str] = None,
    valstorm_env: str = "prod",
    valstorm_token: Optional[str] = None,
    resume_session_id: Optional[str] = None,
    max_iterations: Optional[int] = None,
    profile: Optional[str] = None,
    cloud_sync: bool = True,
):
    current_model = initial_model
    current_provider_name = initial_provider
    current_profile_slug = profile or "developer"
    profile_cfg = load_profile(current_profile_slug) if profile else None

    if profile_cfg:
        if initial_model in ("gemini-3.6-flash", "gemini-flash-latest") and profile_cfg.get("model"):
            current_model = profile_cfg["model"]
        if initial_provider in ("gemini", "valstorm") and profile_cfg.get("provider"):
            current_provider_name = profile_cfg["provider"].lower()

    keystore = KeyStore()
    session_store = cli.SessionStore()
    memory_store = MemoryStore()

    provider = cli.get_provider(current_provider_name, api_key=api_key, interactive=True)
    full_tools = cli.build_tool_registry(
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
        if session:
            current_model = session.active_model
            current_provider_name = session.active_provider
            provider = cli.get_provider(current_provider_name, interactive=True)
            engine = ReActEngine(provider=provider, tools=tools)
            print(f"\033[92m[Resumed Session]\033[0m {session.session_id} ({len(session.messages)} prior messages)")
            if engine.compactor and engine.compactor.should_auto_compact(session):
                print("\033[94m🧹 [Auto-Compacting Context...]\033[0m Prior session exceeds working context window; compacting...")
                c_res = engine.compactor.compact(session)
                if c_res.compacted:
                    print(f"\033[92m✔ [Auto-Compacted on Resume]\033[0m {c_res.summary}")
                    session_store.save_session(session)
        else:
            print(f"\033[93m[Warning]\033[0m Session '{resume_session_id}' not found. Starting fresh session.")
            session = SessionState(active_model=current_model, active_provider=current_provider_name)
            sys_prompt = WorkspaceContextManager(memory_store=memory_store).build_system_prompt(profile=profile_cfg)
            session.add_message(Message(role="system", content=sys_prompt, model=current_model, provider=current_provider_name))
    else:
        session = SessionState(active_model=current_model, active_provider=current_provider_name)
        sys_prompt = WorkspaceContextManager(memory_store=memory_store).build_system_prompt(profile=profile_cfg)
        session.add_message(Message(role="system", content=sys_prompt, model=current_model, provider=current_provider_name))

    profile_label = f" | Profile: {profile_cfg.get('name', current_profile_slug)}" if profile_cfg else ""
    print("\n=== Valstorm AI Agent Interactive REPL ===")
    print(f"Session ID: {session.session_id[:8]}{profile_label} | Active Model: {current_model} | Provider: {current_provider_name} → {cli.describe_backend(provider)}")
    print(f"Loaded Tools: {', '.join(tools.list_tools())}")
    print("Commands:")
    print("  /profile [slug]                 - Switch agent profile or list available profiles")
    print("  /model <model_name> [provider]  - Switch active model mid-session")
    print("  /key <provider> <api_key>       - Save or update an API key")
    print("  /sessions                       - List recent sessions")
    print("  /resume <session_id>            - Switch to a past session")
    print("  /memory                         - View persistent declarative memory facts")
    print("  /search <query>                 - Search past sessions via FTS5")
    print("  /stats                          - View cumulative token & session statistics")
    print("  /status                         - View telemetry, context usage %, & estimated cost")
    print("  /steer <message>                - Inject mid-turn instruction for next iteration")
    print("  /yolo                           - Toggle human-in-the-loop confirmation mode")
    print("  /compress [keep_last]           - Summarize historical turns to compress context")
    print("  /history                        - View message history & per-message models")
    print("  /exit, /quit                    - Exit REPL\n")

    kb = KeyBindings()

    @kb.add("c-c")
    def _(event):
        """Clear prompt buffer if text is present; exit session if buffer is empty."""
        buffer = event.app.current_buffer
        if buffer.text:
            buffer.text = ""
            buffer.cursor_position = 0
        else:
            event.app.exit(exception=KeyboardInterrupt)

    @kb.add("c-j")
    @kb.add("escape", "c-m")
    @kb.add("escape", "enter")
    def _(event):
        """Shift+Enter / Alt+Enter inserts a newline without submitting."""
        event.app.current_buffer.insert_text("\n")

    @kb.add("c-m")
    def _(event):
        """Plain Enter key submits the prompt."""
        event.app.current_buffer.validate_and_handle()

    prompt_session = cli.PromptSession(history=InMemoryHistory(), key_bindings=kb, multiline=True)

    while True:
        try:
            user_input = await prompt_session.prompt_async(f"\n[{current_model}] > ")
            user_input = user_input.strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\033[90m[Session Saved] ID: {session.session_id} | Resume: vsagent chat --session {session.session_id}\033[0m")
            print("Exiting...")
            session_store.save_session(session)
            break

        if not user_input:
            continue

        if user_input in ("/exit", "/quit", "exit", "quit"):
            session_store.save_session(session)
            print(f"\n\033[90m[Session Saved] ID: {session.session_id}\033[0m")
            print(f"\033[90m[Resume Command] vsagent chat --session {session.session_id}\033[0m")
            print("Goodbye!")
            break

        if user_input == "/sessions":
            sessions = session_store.list_sessions(limit=10)
            if not sessions:
                print("No past sessions stored.")
            else:
                print("\n--- Recent Stored Sessions ---")
                for s in sessions:
                    active_marker = " (Active)" if s["session_id"] == session.session_id else ""
                    print(f"  • {s['session_id'][:8]} | {s['title'][:35]:35} | {s['message_count']} msgs | {s['updated_at'][:16]}{active_marker}")
            continue

        if user_input.startswith("/resume"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /resume <session_id>")
                continue
            target_id = parts[1].strip()
            all_sess = session_store.list_sessions(limit=50)
            matched = [s["session_id"] for s in all_sess if s["session_id"].startswith(target_id)]
            if matched:
                target_id = matched[0]
            loaded = session_store.load_session(target_id)
            if loaded:
                session_store.save_session(session)
                session = loaded
                current_model = session.active_model
                current_provider_name = session.active_provider
                provider = cli.get_provider(current_provider_name, interactive=True)
                engine = ReActEngine(provider=provider, tools=tools)
                print(f"\033[92m[Switched Session]\033[0m Loaded {session.session_id} ({len(session.messages)} messages, {session.total_tokens} tokens)")
                if engine.compactor and engine.compactor.should_auto_compact(session):
                    print("\033[94m🧹 [Auto-Compacting Context...]\033[0m Session exceeds working context window; compacting...")
                    c_res = engine.compactor.compact(session)
                    if c_res.compacted:
                        print(f"\033[92m✔ [Auto-Compacted on Resume]\033[0m {c_res.summary}")
                        session_store.save_session(session)
            else:
                print(f"\033[91m[Error]\033[0m Session '{target_id}' not found.")
            continue

        if user_input.startswith("/memory"):
            facts = memory_store.get_facts()
            print("\n--- Persistent Declarative Memory ---")
            print(json.dumps(facts, indent=2))
            continue

        if user_input.startswith("/search"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /search <query>")
                continue
            q = parts[1].strip()
            results = session_store.search_sessions(query=q, limit=5)
            if not results:
                print(f"No results found for '{q}'.")
            else:
                print(f"\n--- FTS5 Search Results for '{q}' ---")
                for r in results:
                    print(f"  • [{r['title']}] ({r['session_id'][:8]} · {r['updated_at'][:10]}):")
                    print(f"    Snippet: {r['snippet']}")
            continue

        if user_input == "/status":
            print(f"\n{get_telemetry_badge(session)}")
            yolo_status = "ENABLED (bypassed)" if session.metadata.get("yolo") else "DISABLED (protected)"
            print(f"  • YOLO Approval Gate: {yolo_status}\n")
            continue

        if user_input.startswith("/steer"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /steer <message>")
                continue
            steer_text = parts[1].strip()
            engine.steer(steer_text)
            print(f"\033[96m🎯 [Steering Queued]\033[0m Message queued for next turn iteration: '{steer_text}'")
            continue

        if user_input == "/yolo":
            yolo_mode = not session.metadata.get("yolo", False)
            session.metadata["yolo"] = yolo_mode
            os.environ["YOLO_MODE"] = "true" if yolo_mode else "false"
            if yolo_mode:
                print("\033[93m⚡ [YOLO Mode: ENABLED]\033[0m High-risk confirmation prompts will be automatically bypassed.")
            else:
                print("\033[92m🛡 [YOLO Mode: DISABLED]\033[0m High-risk actions require human confirmation.")
            continue

        if user_input.startswith("/compress"):
            parts = user_input.split(maxsplit=1)
            keep_last = 4
            if len(parts) >= 2 and parts[1].strip().isdigit():
                keep_last = int(parts[1].strip())

            print("\033[94m⚙ [Compressing Context...]\033[0m Summarizing historical turns...")
            tb, ta = await cli.compress_session_context(session=session, provider=provider, keep_last=keep_last)
            print(f"\033[92m✔ [Compressed Context]\033[0m Reduced context token usage. Tokens before: {tb:,} -> Tokens after: {ta:,}")
            session_store.save_session(session)
            continue

        if user_input in ("/compact", "/prune") or user_input.startswith("/compact ") or user_input.startswith("/prune "):
            parts = user_input.split(maxsplit=1)
            keep_last = 6
            if len(parts) >= 2 and parts[1].strip().isdigit():
                keep_last = int(parts[1].strip())

            print("\033[94m🧹 [Compacting Context...]\033[0m Pruning verbose historical tool outputs...")
            from core.compaction import ContextCompactor
            compactor = ContextCompactor(keep_recent_messages=keep_last)
            res = compactor.compact(session)
            if res.compacted:
                print(f"\033[92m✔ [Compacted Context]\033[0m {res.summary}")
            else:
                print(f"\033[90m• [Compaction Skipped]\033[0m {res.summary}")
            session_store.save_session(session)
            continue

        if user_input in ("/tasks", "/todo"):
            if not getattr(session, "tasks", None):
                print("\nNo task checklist recorded in active session.")
            else:
                print("\n--- Session Task Checklist ---")
                for t in session.tasks:
                    status_icon = "✔" if t.get("status") == "completed" else ("⚙" if t.get("status") == "in_progress" else "○")
                    print(f"  {status_icon} [{t.get('status', 'pending')}] {t.get('title', '')}")
            continue

        if user_input == "/stats":
            print(f"\n--- Session Token Telemetry ---")
            print(f"Total Messages: {len(session.messages)}")
            print(f"Prompt Tokens:     {session.total_prompt_tokens}")
            print(f"Completion Tokens: {session.total_completion_tokens}")
            print(f"Total Tokens:      {session.total_tokens}")
            continue

        if user_input == "/history":
            print(f"\n--- Conversation History ({len(session.messages)} messages) ---")
            for i, msg in enumerate(session.messages, 1):
                model_tag = f"[{msg.model}]" if msg.model else ""
                tokens_tag = f"({msg.usage.total_tokens} tok)" if msg.usage else ""
                content_preview = (msg.content or "")[:80]
                if msg.tool_calls:
                    content_preview += f" [Tool Call: {', '.join(tc.name for tc in msg.tool_calls)}]"
                print(f"{i:2d}. {msg.role.upper():9} {model_tag:22} {tokens_tag:12} : {content_preview}")
            continue

        if user_input.startswith("/key"):
            parts = user_input.split(maxsplit=2)
            if len(parts) < 3:
                print("Usage: /key <provider> <api_key>")
                continue
            prov, k_val = parts[1], parts[2]
            saved = keystore.save_api_key(prov, k_val, persist_to="config")
            print(f"\033[92m[Saved]\033[0m API key for '{prov}' saved to {saved}")
            if prov.lower() == current_provider_name.lower():
                provider = cli.get_provider(current_provider_name, interactive=True)
                engine = ReActEngine(provider=provider, tools=tools)
                print(f"\033[92m[Reloaded]\033[0m Active provider reloaded with new key.")
            continue

        if user_input.startswith("/profile") or user_input.startswith("/profiles"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                all_profs = list_available_profiles()
                print("\n--- Available Agent Profiles ---")
                for p in all_profs:
                    slug = p.get("api_name", p.get("name", "unknown"))
                    display = p.get("name") or p.get("display_name") or slug.title()
                    p_model = p.get("model", "default")
                    active_marker = " (Active)" if slug == current_profile_slug else ""
                    print(f"  • \033[1m{slug:<18}\033[0m : {display:<25} [model: {p_model}]{active_marker}")
                    if p.get("description"):
                        print(f"    {p['description']}")
                print("\nUsage: /profile <slug> to switch active profile.")
                continue

            target_slug = parts[1].strip().lower()
            try:
                new_prof_cfg = load_profile(target_slug)
                current_profile_slug = target_slug
                profile_cfg = new_prof_cfg

                if new_prof_cfg.get("model"):
                    current_model = new_prof_cfg["model"]
                    session.active_model = current_model
                if new_prof_cfg.get("provider"):
                    current_provider_name = new_prof_cfg["provider"].lower()
                    session.active_provider = current_provider_name

                provider = cli.get_provider(current_provider_name, interactive=True)
                if new_prof_cfg.get("allowed_tools"):
                    tools = full_tools.filter_by_whitelist(new_prof_cfg["allowed_tools"])
                else:
                    tools = full_tools

                engine = ReActEngine(provider=provider, tools=tools)

                # Update system prompt
                new_sys_prompt = WorkspaceContextManager(memory_store=memory_store).build_system_prompt(profile=new_prof_cfg)
                has_sys = False
                for m in session.messages:
                    if m.role == "system":
                        m.content = new_sys_prompt
                        m.model = current_model
                        m.provider = current_provider_name
                        has_sys = True
                        break
                if not has_sys:
                    session.messages.insert(0, Message(role="system", content=new_sys_prompt, model=current_model, provider=current_provider_name))

                skills_badge = f" ({len(new_prof_cfg.get('attached_skill_slugs', []))} skills attached)" if new_prof_cfg.get("attached_skill_slugs") else ""
                print(f"\033[92m[Switched Profile]\033[0m Loaded \033[1m{new_prof_cfg.get('name', target_slug)}\033[0m ({target_slug}){skills_badge}")
                print(f"Active Model: {current_model} | Provider: {current_provider_name} | Scoped Tools: {len(tools.list_tools())}")
            except Exception as e:
                print(f"\033[91m[Error switching profile]\033[0m {e}")
            continue

        if user_input.startswith("/model"):
            parts = user_input.split(maxsplit=2)
            if len(parts) < 2:
                print("Usage: /model <model_name> [provider]")
                continue
            new_model = parts[1].strip()
            new_provider = parts[2].strip() if len(parts) > 2 else current_provider_name

            try:
                provider = cli.get_provider(new_provider, interactive=True)
                engine = ReActEngine(provider=provider, tools=tools)
                current_model = new_model
                current_provider_name = new_provider
                session.active_model = current_model
                session.active_provider = current_provider_name
                print(f"\033[92m[Switched]\033[0m Active model is now \033[1m{current_model}\033[0m ({current_provider_name})")
            except Exception as e:
                print(f"\033[91m[Error switching model]\033[0m {e}")
            continue

        # Execute conversation turn
        try:
            _, was_interrupted = await cli.stream_and_render_turn(
                engine=engine,
                session=session,
                prompt=user_input,
                model=current_model,
                max_iterations=max_iterations,
                valstorm_env=valstorm_env,
                valstorm_token=valstorm_token,
                cloud_sync=cloud_sync,
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        except Exception as e:
            print(f"\n\033[91m✖ [Turn Error]\033[0m {e}")

        # Auto-save session to SQLite after turn
        try:
            session_store.save_session(session)
        except Exception as e:
            print(f"\033[91m[Error saving session]\033[0m {e}")
