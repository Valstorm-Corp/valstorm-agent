"""ReAct (Reasoning and Acting) execution engine for multi-turn agent execution with streaming support."""

import asyncio
import inspect
import json
import os
import re
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

from .models import Message, SessionState, StreamEvent, StreamEventType, ToolCall, ToolResult, UsageMetadata, collapse_repeating_text
from .sanitizer import sanitize_text
from .security_guard import SecurityGuard
from .keystore import KeyStore
from .compaction import ContextCompactor

DEFAULT_MAX_ITERATIONS = int(os.environ.get("VALSTORM_MAX_ITERATIONS", "250"))

# Pure read-only / idempotent tools safe to launch speculatively the moment TOOL_CALL_DETECTED arrives in the stream
SPECULATIVE_SAFE_TOOLS = frozenset({
    "read_file",
    "read_local_file",
    "search_files",
    "code_outline",
    "find_symbols",
    "package_blast_radius",
    "calculator",
    "mock_db_lookup",
    "skill_view",
    "skill_list",
    "session_search",
    "valstorm_sql_query",
    "valstorm_mongo_query",
    "valstorm_vfs_search",
    "valstorm_vfs_browse",
    "valstorm_vfs_get_file",
    "valstorm_records_hydrate_batch",
    "valstorm_schema_inspect",
    "valstorm_function_list",
    "valstorm_get_execution_logs",
    "valstorm_get_log_detail",
    "valstorm_analytics_compute",
    "web_search",
    "web_scrape",
    "web_content_diff",
    "web_feed_poll",
    "system_one_decide",
    "analyze_image",
    "vision_analyze",
    "slack_list_channels",
    "slack_get_channel_history",
    "slack_list_users",
    "slack_get_user_profile",
    "slack_get_auth_status",
})


def is_speculative_safe_tool(tool_name: Optional[str]) -> bool:
    """Returns True if the tool is side-effect-free and safe to execute speculatively during LLM streaming."""
    if not tool_name:
        return False
    return tool_name.strip() in SPECULATIVE_SAFE_TOOLS

TRUNCATED_NUDGE = (
    "[System Notice: Your previous response was cut off by the output token limit. Continue exactly where you left "
    "off. If you were about to call a tool, call it now; if you were writing a large file, split it into smaller "
    "write_file/patch_file calls.]"
)
MALFORMED_CALL_NUDGE = (
    "[System Notice: Your previous tool call was malformed and could not be parsed. Retry the tool call with valid, "
    "complete JSON arguments. For very large content, split it across multiple smaller calls.]"
)
ACTION_INTENT_NUDGE = (
    "[System Directive: You stated your intent to perform an action, but did not execute a tool call. "
    "Please invoke the tool directly now to complete the action.]"
)
_ACTION_INTENT_RE = re.compile(
    r"\b(i will|i'll|let me|now i will|i am going to|i am about to|i'll go ahead and)\s+"
    r"(?:now\s+|next\s+|first\s+|also\s+|go ahead and\s+)?"
    r"(write|create|modify|patch|edit|run|execute|read|search|inspect|query|fetch|check|test|build|update|install|invoke|call|apply|compute|calculate|navigate)\b",
    re.IGNORECASE,
)

_ACTION_DEFERRAL_RE = re.compile(
    r"\b(once you confirm|after you confirm|once you approve|after you approve|"
    r"when you confirm|when you approve|when you are ready|when you're ready|"
    r"if you confirm|if you approve|if you'd like|if you would like|if you want|"
    r"let me know if|awaiting your|wait for your|before i proceed|upon your approval|"
    r"would you like me to|shall i proceed|in a future step)\b",
    re.IGNORECASE,
)


def evaluate_action_intent_s1(content: Optional[str]) -> Dict[str, Any]:
    """System-1 Noul + Choice evaluation for whether an assistant message is an unfulfilled tool preamble.

    Distinguishes genuine immediate tool-invocation preambles ("I will now write x.py.")
    from conditional/deferred statements ("I will modify those files once you confirm.")
    and completed markdown summaries.
    """
    if not content:
        return {"unfulfilled_intent": False, "noul": 0.01, "confidence": 0.98, "reason": "empty"}
    stripped = content.strip()
    if not stripped or stripped.endswith("?"):
        return {"unfulfilled_intent": False, "noul": 0.02, "confidence": 0.97, "reason": "question_or_empty"}
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", stripped) if p.strip()]
    if not paragraphs:
        return {"unfulfilled_intent": False, "noul": 0.02, "confidence": 0.97, "reason": "no_paragraphs"}
    last_para = paragraphs[-1]
    if len(last_para) > 600 or last_para.startswith(("|", "```", "#", ">", "-", "*")):
        return {"unfulfilled_intent": False, "noul": 0.03, "confidence": 0.95, "reason": "structural_block"}

    has_deferral = bool(_ACTION_DEFERRAL_RE.search(last_para))
    has_action_verb = bool(_ACTION_INTENT_RE.search(last_para))

    from .decision_model import ChoiceQuestion, NoulQuestion, SystemOneClient

    signal_tokens = (
        "immediate_action execute_tool_now unfulfilled_tool_intent"
        if (has_action_verb and not has_deferral)
        else "deferred_or_completed await_user_or_completed"
    )
    state = {
        "goal": signal_tokens,
        "last_paragraph": last_para,
        "signals": signal_tokens,
        "error": False,
    }
    s1 = SystemOneClient(backend="local")
    resp = s1.evaluate(
        state=state,
        questions={
            "intent_class": ChoiceQuestion(
                id="intent_class",
                instructions="Classify whether the assistant paragraph is an immediate unfulfilled tool call preamble or a completed/deferred response.",
                criteria={
                    "execute_tool_now": "immediate_action execute_tool_now unfulfilled_tool_intent",
                    "await_user_or_completed": "deferred_or_completed await_user_or_completed",
                },
            ),
            "unfulfilled_tool_intent": NoulQuestion(
                id="unfulfilled_tool_intent",
                instructions="immediate_action execute_tool_now unfulfilled_tool_intent",
            ),
        },
    )
    choice_ans = resp.answers["intent_class"]
    noul_ans = resp.answers["unfulfilled_tool_intent"]
    is_unfulfilled = (
        has_action_verb
        and not has_deferral
        and getattr(choice_ans, "choice", "") == "execute_tool_now"
        and getattr(noul_ans, "noul", 0.0) >= 0.55
    )
    return {
        "unfulfilled_intent": bool(is_unfulfilled),
        "noul": float(getattr(noul_ans, "noul", 0.0)),
        "confidence": float(getattr(choice_ans, "confidence", 0.0)),
        "choice": getattr(choice_ans, "choice", "await_user_or_completed"),
        "has_deferral": has_deferral,
    }


def _has_unfulfilled_action_intent(content: Optional[str]) -> bool:
    """True when an assistant response is an unfulfilled action preamble rather than a completed answer."""
    return bool(evaluate_action_intent_s1(content).get("unfulfilled_intent", False))


class ReActEngine:
    """Orchestrates the ReAct execution loop (Thought -> Action -> Observation -> Final Answer)."""

    def __init__(
        self,
        provider: Any,
        tools: Any,
        keystore: Optional[KeyStore] = None,
        compactor: Optional[ContextCompactor] = None,
    ):
        self.provider = provider
        self.tools = tools
        self.keystore = keystore
        self.compactor = compactor or ContextCompactor()
        self.steer_queue: List[str] = []

    def steer(self, message: str) -> None:
        """Enqueues an out-of-band steering instruction to inject on the next loop iteration."""
        if message and message.strip():
            self.steer_queue.append(message.strip())

    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Executes a single ToolCall against the tool registry safely with precision timing."""
        name = tool_call.name
        args = tool_call.arguments or {}
        start_t = time.perf_counter()

        try:
            if hasattr(self.tools, "execute_async"):
                res = await self.tools.execute_async(name, args, call_id=tool_call.id)
            elif hasattr(self.tools, "execute"):
                try:
                    if inspect.iscoroutinefunction(self.tools.execute):
                        res = await self.tools.execute(name, args, call_id=tool_call.id)
                    else:
                        res = await asyncio.to_thread(self.tools.execute, name, args, call_id=tool_call.id)
                    if inspect.isawaitable(res):
                        res = await res
                except TypeError:
                    try:
                        if inspect.iscoroutinefunction(self.tools.execute):
                            res = await self.tools.execute(name, args)
                        else:
                            res = await asyncio.to_thread(self.tools.execute, name, args)
                        if inspect.isawaitable(res):
                            res = await res
                    except TypeError:
                        if inspect.iscoroutinefunction(self.tools.execute):
                            res = await self.tools.execute(name, **args)
                        else:
                            res = await asyncio.to_thread(self.tools.execute, name, **args)
                        if inspect.isawaitable(res):
                            res = await res
            elif hasattr(self.tools, "get"):
                tool_fn = self.tools.get(name)
                if tool_fn is None:
                    raise ValueError(f"Tool '{name}' not found in registry.")
                if inspect.iscoroutinefunction(tool_fn):
                    res = await tool_fn(**args)
                else:
                    res = await asyncio.to_thread(tool_fn, **args)
                if inspect.isawaitable(res):
                    res = await res
            else:
                raise AttributeError("Tool registry does not have execute or get method.")

            dur_ms = round((time.perf_counter() - start_t) * 1000, 2)
            if isinstance(res, ToolResult):
                if not res.call_id:
                    res.call_id = tool_call.id
                if not res.duration_ms:
                    res.duration_ms = dur_ms
                if res.output:
                    res.output = sanitize_text(res.output)
                return res

            out_str = json.dumps(res, indent=2) if isinstance(res, (dict, list)) else str(res)
            out_str = sanitize_text(out_str)
            return ToolResult(
                call_id=tool_call.id,
                name=name,
                output=out_str,
                is_error=False,
                duration_ms=dur_ms,
                payload_bytes=len(out_str.encode("utf-8")),
                item_count=len(res) if isinstance(res, (list, dict)) else None,
            )
        except asyncio.CancelledError:
            curr = asyncio.current_task()
            if curr is None or getattr(curr, "cancelling", lambda: 1)() > 0:
                raise
            dur_ms = round((time.perf_counter() - start_t) * 1000, 2)
            err_msg = sanitize_text(f"Tool '{name}' internal subtask was cancelled.")
            return ToolResult(
                call_id=tool_call.id,
                name=name,
                output=err_msg,
                is_error=True,
                duration_ms=dur_ms,
                payload_bytes=len(err_msg.encode("utf-8")),
            )
        except Exception as e:
            dur_ms = round((time.perf_counter() - start_t) * 1000, 2)
            err_msg = sanitize_text(f"Error executing tool '{name}': {str(e)}")
            return ToolResult(
                call_id=tool_call.id,
                name=name,
                output=err_msg,
                is_error=True,
                duration_ms=dur_ms,
                payload_bytes=len(err_msg.encode("utf-8")),
            )

    async def run_turn_stream(
        self,
        session: SessionState,
        user_input: str,
        images: Optional[List[str]] = None,
        model: Optional[str] = None,
        max_iterations: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        auto_tune: Optional[bool] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Executes a full ReAct loop turn with real-time streaming of tokens and tool execution events."""
        iter_limit = max_iterations if max_iterations is not None else DEFAULT_MAX_ITERATIONS
        raw_req_model = model or session.active_model
        if reasoning_effort is not None:
            session.metadata["reasoning_effort"] = str(reasoning_effort).strip().lower()
        if auto_tune is not None:
            session.metadata["auto_tune"] = bool(auto_tune)

        is_auto_tune = bool(
            session.metadata.get("auto_tune")
            or str(raw_req_model or "").strip().lower() in ("auto", "autotune")
            or str((session.metadata or {}).get("reasoning_effort") or "").strip().lower() in ("auto", "autotune")
            or os.environ.get("VALSTORM_AUTO_TUNE", "").strip().lower() in ("1", "true", "yes")
        )
        target_model = (
            session.active_model
            if str(raw_req_model or "").strip().lower() in ("auto", "autotune")
            else (raw_req_model or "gemini-flash-latest")
        )
        locked_turn_tier = "reasoner"
        locked_turn_model = target_model or "gemini-flash-latest"
        # Avoid duplicating user prompt if already hydrated in session
        if not session.messages or session.messages[-1].content != user_input or session.messages[-1].role != "user":
            user_msg = Message(role="user", content=user_input, images=images)
            session.add_message(user_msg)

        # If a prior turn paused on confirmation_required for Lethal Trifecta, a new user turn satisfies the human-in-the-loop gate
        if session.metadata.get("pending_lethal_trifecta_confirmation"):
            session.metadata["lethal_trifecta_approved"] = True
            session.metadata["pending_lethal_trifecta_confirmation"] = False

        # 1. Pre-turn compaction check: clear context headroom before entering tool loop
        if self.compactor and self.compactor.should_auto_compact(session):
            c_res = self.compactor.compact(session)
            if c_res.compacted:
                yield StreamEvent(
                    event_type=StreamEventType.CONTEXT_COMPACTED,
                    delta=f"\n\033[93m🧹 [Auto-Compaction (Pre-Turn)]: Saved ~{c_res.tokens_saved:,} tokens ({c_res.compression_ratio:.1%} retained)\033[0m\n",
                    metadata={
                        "tokens_saved": c_res.tokens_saved,
                        "compression_ratio": c_res.compression_ratio,
                        "summary": c_res.summary,
                        "phase": "pre-turn",
                    },
                )

        assistant_msg: Optional[Message] = None
        usage_meta: Optional[UsageMetadata] = None
        empty_retries = 0
        max_empty_retries = 2
        truncation_retries = 0
        malformed_retries = 0
        action_nudge_retries = 0
        turn_tool_calls = 0

        for iteration in range(iter_limit):
            # 2. In-loop compaction check: emergency safeguard for runaway tool results
            if self.compactor and self.compactor.should_auto_compact(session):
                c_res = self.compactor.compact(session)
                if c_res.compacted:
                    yield StreamEvent(
                        event_type=StreamEventType.CONTEXT_COMPACTED,
                        delta=f"\n\033[93m🧹 [Auto-Compaction]: Saved ~{c_res.tokens_saved:,} tokens ({c_res.compression_ratio:.1%} retained)\033[0m\n",
                        metadata={
                            "tokens_saved": c_res.tokens_saved,
                            "compression_ratio": c_res.compression_ratio,
                            "summary": c_res.summary,
                            "phase": "in-loop",
                        },
                    )

            # Inject any queued out-of-band steering instructions
            if self.steer_queue:
                steer_msgs = list(self.steer_queue)
                self.steer_queue.clear()
                combined_steer = "\n".join(steer_msgs)
                steer_msg_obj = Message(
                    role="user",
                    content=f"[OUT-OF-BAND USER STEERING INSTRUCTION]\n{combined_steer}\n[/OUT-OF-BAND USER STEERING INSTRUCTION]",
                    model=target_model,
                    provider=getattr(self.provider, "provider_name", "unknown"),
                )
                session.add_message(steer_msg_obj)
                yield StreamEvent(
                    event_type=StreamEventType.TEXT_CHUNK,
                    delta=f"\n\033[96m🎯 [Steering Injected]: {combined_steer}\033[0m\n",
                )

            schemas = []
            if hasattr(self.tools, "get_schemas"):
                schemas = self.tools.get_schemas()
            elif hasattr(self.tools, "schemas"):
                schemas = self.tools.schemas

            assistant_msg = None
            usage_meta = None

            req_effort = (session.metadata or {}).get("reasoning_effort") or os.environ.get("VALSTORM_THINKING_LEVEL")
            effective_effort = req_effort
            if is_auto_tune:
                from .message_autotuner import MessageAutoTuner
                autotuner = getattr(self, "_message_autotuner", None)
                if autotuner is None:
                    autotuner = MessageAutoTuner()
                    self._message_autotuner = autotuner
                base_prov = session.active_provider or getattr(self.provider, "provider_name", "gemini")
                if iteration == 0:
                    tune_dec = autotuner.tune_turn(
                        user_input=user_input,
                        session=session,
                        base_provider=base_prov,
                        base_model=locked_turn_model,
                    )
                    locked_turn_tier = tune_dec.tier
                    locked_turn_model = tune_dec.model
                else:
                    tune_dec = autotuner.tune_iteration(
                        user_input=user_input,
                        session=session,
                        iteration=iteration,
                        locked_tier=locked_turn_tier,  # type: ignore[arg-type]
                        locked_model=locked_turn_model,
                        base_provider=base_prov,
                    )
                target_model = locked_turn_model
                session.active_model = locked_turn_model
                effective_effort = tune_dec.reasoning_effort
                session.metadata["last_autotune_decision"] = tune_dec.model_dump()
                session.metadata["last_autotune_model"] = locked_turn_model
                session.metadata["last_routed_thinking_level"] = effective_effort
                hist_tune = list(session.metadata.get("autotune_history") or [])
                hist_tune.append(tune_dec.model_dump())
                session.metadata["autotune_history"] = hist_tune
            elif req_effort and str(req_effort).strip().lower() == "adaptive":
                from .thinking_router import SystemOneThinkingRouter
                router = getattr(self, "_thinking_router", None)
                if router is None:
                    router = SystemOneThinkingRouter()
                    self._thinking_router = router
                route_dec = router.route_turn(user_input=user_input, session=session, iteration=iteration)
                effective_effort = route_dec.reasoning_effort
                session.metadata["last_routed_thinking_level"] = effective_effort
                hist_levels = list(session.metadata.get("routed_thinking_history") or [])
                hist_levels.append(effective_effort)
                session.metadata["routed_thinking_history"] = hist_levels

            set_ctx = getattr(self.provider, "set_request_context", None)
            if callable(set_ctx):
                ctx_kwargs: Dict[str, Any] = {
                    "chat_id": (session.metadata or {}).get("valstorm_chat_id") or session.session_id,
                }
                if effective_effort:
                    ctx_kwargs["reasoning_effort"] = effective_effort
                set_ctx(**ctx_kwargs)

            # Check if provider supports streaming
            speculative_tasks: Dict[str, asyncio.Task] = {}
            saw_mutating_tool_in_batch = False
            speculative_enabled = (
                os.environ.get("VALSTORM_SPECULATIVE_TOOLS", "1").strip().lower()
                not in ("0", "false", "no", "off")
                and bool((session.metadata or {}).get("speculative_tools", True))
            )

            if hasattr(self.provider, "generate_stream"):
                stream = self.provider.generate_stream(
                    messages=session.messages,
                    tools=schemas,
                    model=target_model,
                )
                async for item in stream:
                    if isinstance(item, StreamEvent):
                        if (
                            item.event_type == StreamEventType.TOOL_CALL_DETECTED
                            and item.tool_call is not None
                        ):
                            tc_stream = item.tool_call
                            if not is_speculative_safe_tool(tc_stream.name):
                                saw_mutating_tool_in_batch = True
                            elif (
                                speculative_enabled
                                and not saw_mutating_tool_in_batch
                                and tc_stream.id
                                and tc_stream.id not in speculative_tasks
                            ):
                                speculative_tasks[tc_stream.id] = asyncio.create_task(
                                    self._execute_tool(tc_stream)
                                )
                                session.metadata["speculative_tools_launched"] = (
                                    int(session.metadata.get("speculative_tools_launched", 0)) + 1
                                )
                        if item.event_type == StreamEventType.TURN_COMPLETE:
                            # A provider's TURN_COMPLETE only ends this single LLM call. Re-label it so
                            # consumers don't treat a mid-turn call as the end of the whole turn.
                            if item.message:
                                assistant_msg = item.message
                                if item.usage and not usage_meta:
                                    usage_meta = item.usage
                            yield StreamEvent(
                                event_type=StreamEventType.LLM_CALL_COMPLETE,
                                message=item.message,
                                usage=item.usage,
                                metadata=item.metadata,
                            )
                            continue
                        yield item
                    elif isinstance(item, tuple) and len(item) >= 2:
                        assistant_msg, usage_meta = item[0], item[1]
                    elif isinstance(item, Message):
                        assistant_msg = item
            else:
                if inspect.iscoroutinefunction(self.provider.generate):
                    res = await self.provider.generate(
                        messages=session.messages,
                        tools=schemas,
                        model=target_model,
                    )
                else:
                    res = self.provider.generate(
                        messages=session.messages,
                        tools=schemas,
                        model=target_model,
                    )
                    if inspect.isawaitable(res):
                        res = await res

                if isinstance(res, tuple):
                    assistant_msg, usage_meta = res[0], res[1]
                else:
                    assistant_msg = res

                if assistant_msg and assistant_msg.content:
                    yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta=assistant_msg.content)
                if assistant_msg and assistant_msg.tool_calls:
                    for tc in assistant_msg.tool_calls:
                        yield StreamEvent(event_type=StreamEventType.TOOL_CALL_DETECTED, tool_call=tc)

            if not isinstance(assistant_msg, Message):
                raise TypeError(f"Provider must produce a Message instance, got {type(assistant_msg)}")

            if is_auto_tune and target_model:
                assistant_msg.model = target_model
            elif not assistant_msg.model and target_model:
                assistant_msg.model = target_model
            if not assistant_msg.provider and hasattr(self.provider, "provider_name"):
                assistant_msg.provider = getattr(self.provider, "provider_name")
            if usage_meta and not assistant_msg.usage:
                assistant_msg.usage = usage_meta

            # Add to session (aggregates tokens)
            session.add_message(assistant_msg)

            # Process tool calls (executed concurrently in parallel)
            if assistant_msg.tool_calls and len(assistant_msg.tool_calls) > 0:
                empty_retries = 0
                truncation_retries = 0
                malformed_retries = 0
                action_nudge_retries = 0
                turn_tool_calls += len(assistant_msg.tool_calls)
                for tool_call in assistant_msg.tool_calls:
                    yield StreamEvent(
                        event_type=StreamEventType.TOOL_EXECUTION_START,
                        tool_call=tool_call,
                    )

                # Pre-taint session if any tool in the batch ingests untrusted external content
                for tc_item in assistant_msg.tool_calls:
                    if SecurityGuard.is_untrusted_ingestion_tool(tc_item.name):
                        session.metadata["untrusted_content_tainted"] = True

                # Parallel tool execution via asyncio.gather (reusing any in-flight speculative tasks)
                async def _resolve_or_execute_tool(tc: ToolCall) -> ToolResult:
                    allowed, block_reason = SecurityGuard.check_lethal_trifecta_guard(
                        tool_name=tc.name,
                        tool_args=tc.arguments,
                        untrusted_tainted=bool(session.metadata.get("untrusted_content_tainted", False)),
                        confirmation_approved=bool(session.metadata.get("lethal_trifecta_approved", False)),
                    )
                    if not allowed and block_reason:
                        spec_task = speculative_tasks.pop(tc.id, None) if tc.id else None
                        if spec_task is not None and not spec_task.done():
                            spec_task.cancel()
                        return ToolResult(
                            call_id=tc.id,
                            name=tc.name,
                            output=block_reason,
                            is_error=True,
                            duration_ms=0.0,
                            payload_bytes=len(block_reason.encode("utf-8")),
                        )

                    spec_task = speculative_tasks.pop(tc.id, None) if tc.id else None
                    if spec_task is not None:
                        res_obj = await spec_task
                    else:
                        res_obj = await self._execute_tool(tc)

                    if SecurityGuard.is_untrusted_ingestion_tool(tc.name) and not res_obj.is_error:
                        session.metadata["untrusted_content_tainted"] = True
                    if tc.name == "confirmation_required" and not res_obj.is_error:
                        session.metadata["pending_lethal_trifecta_confirmation"] = True
                    return res_obj

                try:
                    tool_results = await asyncio.gather(
                        *[_resolve_or_execute_tool(tc) for tc in assistant_msg.tool_calls]
                    )
                finally:
                    for orphan_task in speculative_tasks.values():
                        if not orphan_task.done():
                            orphan_task.cancel()
                    speculative_tasks.clear()

                for tool_call, tool_result in zip(assistant_msg.tool_calls, tool_results):
                    yield StreamEvent(
                        event_type=StreamEventType.TOOL_EXECUTION_RESULT,
                        tool_call=tool_call,
                        tool_result=tool_result,
                    )

                    if isinstance(tool_result.output, (str, int, float, bool)):
                        content_str = str(tool_result.output)
                    else:
                        try:
                            content_str = json.dumps(tool_result.output)
                        except Exception:
                            content_str = str(tool_result.output)

                    tool_msg = Message(
                        role="tool",
                        content=content_str,
                        tool_result=tool_result,
                        model=target_model,
                        provider=assistant_msg.provider,
                    )
                    session.add_message(tool_msg)
            else:
                has_content = bool(assistant_msg.content and assistant_msg.content.strip())
                finish = (assistant_msg.finish_reason or "").lower()
                can_continue = iteration < iter_limit - 1

                # Output was cut off by the token limit: ask the model to continue instead of ending.
                if finish in ("length", "max_tokens") and truncation_retries < 2 and can_continue:
                    truncation_retries += 1
                    session.add_message(Message(role="user", content=TRUNCATED_NUDGE, model=target_model, provider=assistant_msg.provider))
                    yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="\n\033[93m↻ [Output truncated — asking the model to continue]\033[0m\n")
                    continue

                # Malformed function call: retry the call instead of ending the turn silently.
                if "malformed" in finish and malformed_retries < 2 and can_continue:
                    malformed_retries += 1
                    session.add_message(Message(role="user", content=MALFORMED_CALL_NUDGE, model=target_model, provider=assistant_msg.provider))
                    yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="\n\033[93m↻ [Malformed tool call — retrying]\033[0m\n")
                    continue

                if not has_content:
                    if empty_retries < max_empty_retries and (iteration < iter_limit - 1):
                        empty_retries += 1
                        nudge_msg = Message(
                            role="user",
                            content="[System Notice: The previous response was empty. Please provide your response or next tool call.]",
                            model=target_model,
                            provider=assistant_msg.provider,
                        )
                        session.add_message(nudge_msg)
                        continue
                    else:
                        fallback_text = "(Model concluded turn without text response)"
                        assistant_msg.content = fallback_text
                        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta=f"\n\033[93m⚠️ {fallback_text}\033[0m\n")

                # Action intent safeguard: if the model explicitly stated it will execute an action
                # (or its stream was cut off right after the preamble before emitting the tool call),
                # nudge it to invoke the tool directly instead of ending prematurely.
                if (
                    can_continue
                    and action_nudge_retries < 2
                    and _has_unfulfilled_action_intent(assistant_msg.content)
                ):
                    action_nudge_retries += 1
                    nudge_msg = Message(
                        role="user",
                        content=ACTION_INTENT_NUDGE,
                        model=target_model,
                        provider=assistant_msg.provider,
                    )
                    session.add_message(nudge_msg)
                    yield StreamEvent(
                        event_type=StreamEventType.TEXT_CHUNK,
                        delta="\n\033[93m↻ [Prompting model to execute stated action tool call]\033[0m\n",
                    )
                    continue

                # No tool calls: final answer reached
                if assistant_msg and assistant_msg.content:
                    assistant_msg.content = collapse_repeating_text(assistant_msg.content)
                    assistant_msg.body = assistant_msg.content

                yield StreamEvent(
                    event_type=StreamEventType.TURN_COMPLETE,
                    message=assistant_msg,
                    usage=assistant_msg.usage,
                )
                return

        # Max iterations reached: yield limit notice event
        yield StreamEvent(
            event_type=StreamEventType.ITERATION_LIMIT_REACHED,
            delta=f"\n\033[93m⚠️ [Loop Bound]: Reached iteration limit ({iter_limit}).\033[0m\n",
            metadata={
                "iteration_count": iter_limit,
                "has_active_tool_calls": bool(assistant_msg and assistant_msg.tool_calls),
            },
        )

        if assistant_msg is not None:
            if assistant_msg.content:
                assistant_msg.content = collapse_repeating_text(assistant_msg.content)
                assistant_msg.body = assistant_msg.content
            yield StreamEvent(
                event_type=StreamEventType.TURN_COMPLETE,
                message=assistant_msg,
                usage=assistant_msg.usage,
            )
            return

        fallback_msg = Message(
            role="assistant",
            content=f"Terminated after reaching maximum iterations ({iter_limit}). Use --max-iterations or /continue to extend.",
            model=target_model,
            provider=getattr(self.provider, "provider_name", "unknown"),
        )
        session.add_message(fallback_msg)
        yield StreamEvent(
            event_type=StreamEventType.TURN_COMPLETE,
            message=fallback_msg,
            usage=fallback_msg.usage,
        )

    async def run_turn(
        self,
        session: SessionState,
        user_input: str,
        model: Optional[str] = None,
        max_iterations: Optional[int] = None,
        step_callback: Optional[Callable[[str, Any], None]] = None,
        auto_tune: Optional[bool] = None,
    ) -> Message:
        """Executes a full ReAct loop turn (non-streaming wrapper around run_turn_stream)."""
        final_message = None

        async for event in self.run_turn_stream(
            session=session,
            user_input=user_input,
            model=model,
            max_iterations=max_iterations,
            auto_tune=auto_tune,
        ):
            if step_callback:
                if event.event_type == StreamEventType.TEXT_CHUNK and event.delta:
                    step_callback("text_chunk", event.delta)
                elif event.event_type == StreamEventType.TOOL_CALL_DETECTED and event.tool_call:
                    step_callback("tool_call", event.tool_call)
                elif event.event_type == StreamEventType.TOOL_EXECUTION_RESULT and event.tool_result:
                    step_callback("tool_result", event.tool_result)

            if event.event_type == StreamEventType.TURN_COMPLETE and event.message:
                final_message = event.message

        if final_message is not None:
            return final_message
        if session.messages and session.messages[-1].role == "assistant":
            return session.messages[-1]
        return Message(role="assistant", content="No response generated.")
