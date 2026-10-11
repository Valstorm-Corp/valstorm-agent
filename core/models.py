"""Data contracts and state models for agent-runtime aligned with Valstorm schemas."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
import uuid

from pydantic import BaseModel, Field


def generate_prefix_id(prefix: str) -> str:
    """Generates a prefixed UUID identifier (e.g. aich_..., aicm_...)."""
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def collapse_repeating_text(text: Optional[str]) -> Optional[str]:
    """Detects and collapses exact N-fold repeating text blocks caused by LLM repetition loops.

    Checks contiguous exact repetitions (e.g. AAAA -> A) and delimiter-separated repetitions
    (e.g. A\\n\\nA\\n\\nA -> A) for multiples of 8, 6, 5, 4, 3, 2.
    """
    if not text or len(text) < 40:
        return text
    n = len(text)
    # 1. Exact contiguous multiple
    for k in (8, 6, 5, 4, 3, 2):
        if n % k == 0:
            chunk_len = n // k
            chunk = text[:chunk_len]
            if chunk * k == text:
                return collapse_repeating_text(chunk)
    # 2. Delimiter separated (e.g. \n\n or \n)
    for sep in ("\n\n", "\n"):
        parts = text.split(sep)
        if len(parts) >= 2:
            first = parts[0].strip()
            if len(first) >= 20 and all(p.strip() == first for p in parts):
                return collapse_repeating_text(first)
    return text



class UsageMetadata(BaseModel):
    """Token usage metadata for an LLM generation turn."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: Optional[int] = None
    # Reasoning ("thinking") tokens reported by the provider, when available.
    thoughts_tokens: Optional[int] = None

    @property
    def input_tokens(self) -> int:
        return self.prompt_tokens

    @property
    def output_tokens(self) -> int:
        return self.completion_tokens


class ToolCall(BaseModel):
    """Represents a tool call requested by an LLM."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    thought_signature: Optional[Any] = None


class ToolResult(BaseModel):
    """Result of executing a tool call with high-precision telemetry."""

    call_id: str
    name: str
    output: Any
    is_error: bool = False
    duration_ms: float = 0.0
    payload_bytes: int = 0
    item_count: Optional[int] = None
    truncated: bool = False
    raw_size_bytes: Optional[int] = None

    @property
    def duration_sec(self) -> float:
        return self.duration_ms / 1000.0


class Message(BaseModel):
    """A single message in conversation history mirroring Valstorm ai_chat_message schema."""

    id: str = Field(default_factory=lambda: generate_prefix_id("aicm"))
    ai_chat: Optional[str] = None
    role: Literal["user", "assistant", "model", "tool", "system"]
    body: Optional[str] = None
    content: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    system_prompt: Optional[str] = None
    images: Optional[List[str]] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_result: Optional[ToolResult] = None
    usage: Optional[UsageMetadata] = None
    # Why generation stopped, normalized: "stop", "tool_calls", "length", "malformed_tool_call",
    # "safety", or the provider's raw value. None when the provider didn't report one.
    finish_reason: Optional[str] = None
    # Concrete model version that actually served the request (e.g. resolved from a "-latest" alias).
    model_version: Optional[str] = None
    created_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    modified_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def model_post_init(self, __context: Any) -> None:
        """Synchronizes content and body fields."""
        if self.body is None and self.content is not None:
            self.body = self.content
        elif self.content is None and self.body is not None:
            self.content = self.body


class StreamEventType(str, Enum):
    """Types of real-time streaming events emitted during a ReAct turn."""
    TEXT_CHUNK = "text_chunk"
    THOUGHT_CHUNK = "thought_chunk"
    TOOL_CALL_DETECTED = "tool_call_detected"
    TOOL_EXECUTION_START = "tool_execution_start"
    TOOL_EXECUTION_RESULT = "tool_execution_result"
    TURN_COMPLETE = "turn_complete"
    # Emitted after each individual LLM call inside a turn. TURN_COMPLETE is reserved for
    # the end of the whole ReAct turn so clients don't mistake a mid-turn call for the end.
    LLM_CALL_COMPLETE = "llm_call_complete"
    ERROR = "error"
    CONTEXT_COMPACTED = "context_compacted"
    ITERATION_LIMIT_REACHED = "iteration_limit_reached"


class StreamEvent(BaseModel):
    """A single streaming event chunk emitted to consumer/UI."""
    event_type: StreamEventType
    delta: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None
    usage: Optional[UsageMetadata] = None
    message: Optional[Message] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionState(BaseModel):
    """State of an agent session mirroring Valstorm ai_chat schema."""

    session_id: str = Field(default_factory=lambda: generate_prefix_id("aich"))
    name: Optional[str] = None
    status: Literal["Active", "Waiting for Input", "Archived"] = "Active"
    messages: List[Message] = Field(default_factory=list)
    active_model: str = "gemini-flash-latest"
    active_provider: str = "valstorm"
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    # Cache-hit subset of prompt tokens (monotonic; not reset by context compaction)
    total_cached_input_tokens: int = 0
    tasks: List[Dict[str, Any]] = Field(default_factory=list)
    _total_tokens_accum: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    modified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def total_prompt_tokens(self) -> int:
        return self.total_input_tokens

    @total_prompt_tokens.setter
    def total_prompt_tokens(self, val: int) -> None:
        self.total_input_tokens = val

    @property
    def total_completion_tokens(self) -> int:
        return self.total_output_tokens

    @total_completion_tokens.setter
    def total_completion_tokens(self, val: int) -> None:
        self.total_output_tokens = val

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def add_message(self, msg: Message) -> None:
        """Appends a message and aggregates token usage if present."""
        if not msg.ai_chat:
            msg.ai_chat = self.session_id
        self.messages.append(msg)
        if msg.usage is not None:
            self.total_input_tokens += msg.usage.prompt_tokens
            self.total_output_tokens += msg.usage.completion_tokens
            self.total_cached_input_tokens += int(msg.usage.cached_tokens or 0)
        self.modified_at = datetime.now(timezone.utc)


MODEL_CONTEXT_LIMITS: Dict[str, int] = {
    "gemini-flash-latest": 1_048_576,
    "gemini-3.6-flash": 1_048_576,
    "gemini-flash-lite-latest": 1_048_576,
    "gemini-3.5-flash": 1_048_576,
    "gemini-2.5-flash": 2_097_152,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "claude-haiku-4.5": 200_000,
    "claude-3-5-sonnet": 200_000,
}

MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gemini-flash-latest": {"input": 0.10, "output": 0.40},
    "gemini-3.6-flash": {"input": 0.10, "output": 0.40},
    "gemini-flash-lite-latest": {"input": 0.075, "output": 0.30},
    "gemini-3.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-haiku-4.5": {"input": 0.80, "output": 4.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
}


def get_telemetry_badge(session: SessionState) -> str:
    """Returns formatted context window usage % and estimated cost telemetry badge."""
    model = (session.active_model or "gemini-3.6-flash").lower()

    limit = 1_048_576
    for k, v in MODEL_CONTEXT_LIMITS.items():
        if k in model:
            limit = v
            break

    pricing = {"input": 0.10, "output": 0.40}
    for k, v in MODEL_PRICING.items():
        if k in model:
            pricing = v
            break

    pct = (session.total_tokens / limit) * 100 if limit > 0 else 0.0
    cost = (
        (session.total_input_tokens / 1_000_000) * pricing["input"] +
        (session.total_output_tokens / 1_000_000) * pricing["output"]
    )

    return (
        f"📊 [Telemetry Status]\n"
        f"  • Model/Provider:   {session.active_model} ({session.active_provider})\n"
        f"  • Total Messages:   {len(session.messages)}\n"
        f"  • Prompt Tokens:    {session.total_input_tokens:,}\n"
        f"  • Completion Tok:   {session.total_output_tokens:,}\n"
        f"  • Total Tokens:     {session.total_tokens:,}\n"
        f"  • Context Usage:    {session.total_tokens:,} / {limit:,} tokens ({pct:.2f}% used)\n"
        f"  • Est. Session Cost: ${cost:.5f} USD"
    )
