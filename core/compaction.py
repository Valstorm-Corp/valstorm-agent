"""Context Compaction and Token Pruning Engine for Valstorm Agent Runtime.

Provides Hermes-style automated context compaction, hysteresis-based low-water mark pruning,
and manual `/compact` pruning to sustain long-running agent workflows without exhausting
context windows or degrading LLM reasoning.
"""

from dataclasses import dataclass
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from core.models import (
    MODEL_CONTEXT_LIMITS,
    Message,
    SessionState,
    ToolCall,
    ToolResult,
    UsageMetadata,
)


@dataclass
class CompactionResult:
    """Summary metrics of a context compaction run."""
    messages_before: int
    messages_after: int
    estimated_tokens_before: int
    estimated_tokens_after: int
    tokens_saved: int
    compression_ratio: float
    compacted: bool
    summary: str


DEFAULT_MAX_WORKING_CEILING: int = 640_000


def estimate_tokens_for_text(text: Optional[str], chars_per_token: float = 4.0) -> int:
    """Fast character-based heuristic token estimator (~4 chars per token for prose, ~2.4 for code/JSON)."""
    if not text:
        return 0
    ratio = max(1.0, chars_per_token)
    return max(1, int(len(text) / ratio))


def estimate_message_tokens(msg: Message) -> int:
    """Estimates tokens consumed by a single Message object with calibrated token densities."""
    tokens = 4  # overhead per message
    if msg.content:
        c = msg.content
        ratio = 2.4 if ("```" in c or c.strip().startswith(("{", "["))) else 4.0
        tokens += estimate_tokens_for_text(c, chars_per_token=ratio)
    elif msg.body:
        b = msg.body
        ratio = 2.4 if ("```" in b or b.strip().startswith(("{", "["))) else 4.0
        tokens += estimate_tokens_for_text(b, chars_per_token=ratio)

    if getattr(msg, "images", None):
        tokens += len(msg.images) * 258

    if msg.tool_calls:
        for tc in msg.tool_calls:
            args_str = json.dumps(tc.arguments or {}) if isinstance(tc.arguments, (dict, list)) else str(tc.arguments or "")
            tokens += 10 + estimate_tokens_for_text(tc.name, chars_per_token=3.0) + estimate_tokens_for_text(args_str, chars_per_token=2.4)

    if msg.tool_result:
        tr_out = str(msg.tool_result.output or "")
        tokens += estimate_tokens_for_text(tr_out, chars_per_token=2.4)

    return tokens


def estimate_session_history_tokens(session: SessionState) -> int:
    """Calculates total estimated tokens across all messages currently in session history."""
    return sum(estimate_message_tokens(m) for m in session.messages)


def get_model_context_ceiling(model_name: Optional[str], enforce_working_ceiling: bool = True) -> int:
    """Resolves context limit ceiling in tokens for a given model string.

    If enforce_working_ceiling is True, caps massive context windows (like Gemini's 1M)
    to a performant, cost-effective working ceiling (default 320k, triggers at 240k).
    Configurable via VALSTORM_MAX_WORKING_CEILING.
    """
    clean = (model_name or "gemini-flash-latest").lower()
    raw_limit = 1_048_576
    for k, v in MODEL_CONTEXT_LIMITS.items():
        if k in clean:
            raw_limit = v
            break

    if enforce_working_ceiling:
        max_working_str = os.environ.get("VALSTORM_MAX_WORKING_CEILING")
        max_working = int(max_working_str.strip()) if max_working_str and max_working_str.strip().isdigit() else DEFAULT_MAX_WORKING_CEILING
        return min(raw_limit, max_working)

    return raw_limit


class ContextCompactor:
    """Manages conversational history compression and tool result pruning with hysteresis."""

    def __init__(
        self,
        auto_threshold_ratio: float = 0.75,
        target_ratio: float = 0.40,
        min_messages_to_compact: int = 8,
        keep_recent_messages: int = 6,
        max_tool_chars: int = 16384,
    ):
        self.auto_threshold_ratio = float(
            os.environ.get("VALSTORM_COMPACT_RATIO")
            or os.environ.get("VALSTORM_COMPACT_HIGH_WATER_RATIO")
            or str(auto_threshold_ratio)
        )
        self.target_ratio = float(
            os.environ.get("VALSTORM_COMPACT_TARGET_RATIO")
            or os.environ.get("VALSTORM_COMPACT_LOW_WATER_RATIO")
            or str(target_ratio)
        )
        self.min_messages_to_compact = min_messages_to_compact
        self.keep_recent_messages = keep_recent_messages
        self.max_tool_chars = max_tool_chars

    @staticmethod
    def get_latest_prompt_tokens(session: SessionState) -> int:
        """Extracts the most recent ground-truth prompt token count reported by provider usage telemetry."""
        for msg in reversed(session.messages):
            if msg.role in ("assistant", "model") and msg.usage and msg.usage.prompt_tokens:
                if msg.usage.prompt_tokens > 0:
                    return msg.usage.prompt_tokens
        return 0

    def should_auto_compact(
        self,
        session: SessionState,
        force_threshold_tokens: Optional[int] = None,
    ) -> bool:
        """Determines if the session context has reached the compaction trigger point."""
        if len(session.messages) < self.min_messages_to_compact:
            return False

        # Effective tokens combines ground-truth LLM usage telemetry with calibrated message heuristics
        actual_tokens = self.get_latest_prompt_tokens(session)
        est_tokens = estimate_session_history_tokens(session)
        effective_tokens = max(actual_tokens, est_tokens)

        if force_threshold_tokens is not None:
            return effective_tokens >= force_threshold_tokens

        env_thresh = os.environ.get("VALSTORM_AUTO_COMPACT_THRESHOLD_TOKENS")
        if env_thresh and env_thresh.strip():
            try:
                thresh = int(env_thresh.strip())
                return effective_tokens >= thresh
            except ValueError:
                pass

        ceiling = get_model_context_ceiling(session.active_model)
        # Trigger when active message tokens exceed high-water mark ratio
        return effective_tokens >= (ceiling * self.auto_threshold_ratio)

    def compact(
        self,
        session: SessionState,
        keep_recent_messages: Optional[int] = None,
        max_tool_chars: Optional[int] = None,
        target_tokens: Optional[int] = None,
    ) -> CompactionResult:
        """Executes compaction on session.messages with low-water mark hysteresis.

        Retains:
          1. All root system prompt messages (at the top)
          2. The latest N messages (recent context window)
        Compaction Passes:
          - Pass 1: Prunes verbose tool output bodies and assistant reasoning in older turns.
          - Pass 2 (Hysteresis): If context is still above the target low-water mark,
            progressively distills/consolidates the oldest turns into a structured
            historical context digest, creating deep runway for subsequent tool iterations.
        """
        keep_n = keep_recent_messages if keep_recent_messages is not None else self.keep_recent_messages
        tool_char_limit = max_tool_chars if max_tool_chars is not None else self.max_tool_chars

        msgs = list(session.messages)
        total_msgs = len(msgs)

        if total_msgs <= keep_n + 1:
            tokens_current = estimate_session_history_tokens(session)
            return CompactionResult(
                messages_before=total_msgs,
                messages_after=total_msgs,
                estimated_tokens_before=tokens_current,
                estimated_tokens_after=tokens_current,
                tokens_saved=0,
                compression_ratio=1.0,
                compacted=False,
                summary="Session history is too short to require compaction.",
            )

        tokens_before = estimate_session_history_tokens(session)

        # Separate system messages, older messages, and recent messages
        system_msgs: List[Message] = []
        conversation_msgs: List[Message] = []

        for m in msgs:
            if m.role == "system":
                system_msgs.append(m)
            else:
                conversation_msgs.append(m)

        if len(conversation_msgs) <= keep_n:
            # Nothing to compact in conversation slice
            tokens_current = estimate_session_history_tokens(session)
            return CompactionResult(
                messages_before=total_msgs,
                messages_after=total_msgs,
                estimated_tokens_before=tokens_current,
                estimated_tokens_after=tokens_current,
                tokens_saved=0,
                compression_ratio=1.0,
                compacted=False,
                summary="Active conversation messages within retention threshold.",
            )

        older_slice = conversation_msgs[:-keep_n]
        recent_slice = conversation_msgs[-keep_n:]

        # --- PASS 1: Lightweight Tool & Assistant Pruning ---
        compacted_older: List[Message] = []
        tools_summarized = 0

        for msg in older_slice:
            if msg.role == "tool":
                raw_content = msg.content or msg.body or ""
                if len(raw_content) > tool_char_limit:
                    head = raw_content[: tool_char_limit // 2].rstrip()
                    tail = raw_content[- (tool_char_limit // 2) :].lstrip()
                    compact_text = (
                        f"[Compacted Output: {head}\n"
                        f"... ({len(raw_content)} chars omitted for context efficiency) ...\n"
                        f"{tail}]"
                    )
                    compacted_msg = msg.model_copy(deep=True)
                    compacted_msg.content = compact_text
                    compacted_msg.body = compact_text
                    compacted_older.append(compacted_msg)
                    tools_summarized += 1
                else:
                    compacted_older.append(msg)
            elif msg.role in ("assistant", "model"):
                raw_text = msg.content or msg.body or ""
                if len(raw_text) > 500 and msg.tool_calls:
                    trimmed_text = raw_text[:300].rstrip() + "\n[... historical reasoning trimmed ...]"
                    compacted_msg = msg.model_copy(deep=True)
                    compacted_msg.content = trimmed_text
                    compacted_msg.body = trimmed_text
                    compacted_older.append(compacted_msg)
                else:
                    compacted_older.append(msg)
            else:
                compacted_older.append(msg)

        # Calculate target token ceiling (Low-Water Mark)
        ceiling = get_model_context_ceiling(session.active_model)
        target_token_budget = target_tokens if target_tokens is not None else int(ceiling * self.target_ratio)

        current_trial_messages = system_msgs + compacted_older + recent_slice
        current_tokens = sum(estimate_message_tokens(m) for m in current_trial_messages)

        # --- PASS 2: Aggressive Low-Water Mark Compaction (Hysteresis Clearance) ---
        # If still exceeding target low-water mark, distill the oldest turns into a structured digest
        consolidated_summary_msg: Optional[Message] = None
        evicted_count = 0

        if current_tokens > target_token_budget and compacted_older:
            digest_lines: List[str] = []
            remaining_older = list(compacted_older)

            while remaining_older and current_tokens > target_token_budget:
                oldest_msg = remaining_older.pop(0)
                evicted_count += 1
                role = oldest_msg.role.upper()
                text = (oldest_msg.content or oldest_msg.body or "").strip()
                if oldest_msg.tool_calls:
                    tool_names = ", ".join(tc.name for tc in oldest_msg.tool_calls)
                    text = f"{text[:100]} [Called tools: {tool_names}]" if text else f"[Called tools: {tool_names}]"
                elif len(text) > 100:
                    text = text[:100] + "..."

                if text:
                    digest_lines.append(f"- {role}: {text}")

                temp_summary_text = (
                    f"[HISTORICAL CONTEXT DIGEST - {evicted_count} turns consolidated for context runway]\n"
                    + "\n".join(digest_lines[-12:])
                )
                temp_summary_msg = Message(
                    role="system",
                    content=temp_summary_text,
                    model=session.active_model,
                    provider=session.active_provider,
                )
                current_trial_messages = system_msgs + [temp_summary_msg] + remaining_older + recent_slice
                current_tokens = sum(estimate_message_tokens(m) for m in current_trial_messages)

            compacted_older = remaining_older
            if digest_lines:
                consolidated_summary_text = (
                    f"[HISTORICAL CONTEXT DIGEST - {evicted_count} turns consolidated for context runway]\n"
                    + "\n".join(digest_lines[-12:])
                )
                consolidated_summary_msg = Message(
                    role="system",
                    content=consolidated_summary_text,
                    model=session.active_model,
                    provider=session.active_provider,
                )

        # Assemble new message list
        active_conversation = list(compacted_older) + list(recent_slice)

        # Ensure active_conversation does not start with orphan tool results whose assistant call was evicted
        while active_conversation and active_conversation[0].role == "tool":
            active_conversation.pop(0)

        # The digest is delivered as conversation content (a user turn), NOT as an extra system
        # message: some backends keep only one system instruction, so a second system message
        # silently replaced the agent's real system prompt after compaction.
        if consolidated_summary_msg:
            digest_text = consolidated_summary_msg.content or ""
            if active_conversation and active_conversation[0].role == "user":
                first = active_conversation[0].model_copy(deep=True)
                merged_text = f"{digest_text}\n\n{first.content or first.body or ''}".strip()
                first.content = merged_text
                first.body = merged_text
                active_conversation[0] = first
            else:
                active_conversation.insert(
                    0,
                    Message(
                        role="user",
                        content=digest_text,
                        model=session.active_model,
                        provider=session.active_provider,
                    ),
                )
            consolidated_summary_msg = None

        # If active_conversation starts on an assistant message, prepend a user context message
        # so that conversation turns always start cleanly on a user turn.
        if active_conversation and active_conversation[0].role in ("assistant", "model"):
            bridge_msg = Message(
                role="user",
                content="[Context continues from previous turns. Please proceed with the current task.]",
                model=session.active_model,
                provider=session.active_provider,
            )
            active_conversation.insert(0, bridge_msg)

        new_messages: List[Message] = []
        new_messages.extend(system_msgs)
        if consolidated_summary_msg:
            new_messages.append(consolidated_summary_msg)
        new_messages.extend(active_conversation)

        session.messages = new_messages

        tokens_after = estimate_session_history_tokens(session)
        tokens_saved = max(0, tokens_before - tokens_after)
        ratio = round(tokens_after / tokens_before, 3) if tokens_before > 0 else 1.0

        summary = (
            f"Context compaction complete: saved ~{tokens_saved:,} tokens ({ratio:.1%} of original). "
            f"Retained {len(system_msgs)} system, {len(compacted_older)} compacted older, "
            f"{evicted_count} consolidated, and {len(recent_slice)} recent messages."
        )

        return CompactionResult(
            messages_before=total_msgs,
            messages_after=len(new_messages),
            estimated_tokens_before=tokens_before,
            estimated_tokens_after=tokens_after,
            tokens_saved=tokens_saved,
            compression_ratio=ratio,
            compacted=True,
            summary=summary,
        )
