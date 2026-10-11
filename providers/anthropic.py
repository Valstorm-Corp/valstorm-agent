"""Anthropic Claude Provider implementation."""

import base64
from typing import Any, Dict, List, Optional, Tuple
import json

from core.models import Message, ToolCall, UsageMetadata, collapse_repeating_text
from core.retry import execute_with_retry
from providers.base import BaseProvider, extract_and_resolve_images


# Friendly aliases -> GEAP (Vertex) Claude model ids. Current GEAP ids use dashes and no
# date pin (e.g. "claude-opus-5-5"); older ones carry "@YYYYMMDD" (e.g. "claude-sonnet-4-5@20250929").
# See https://platform.claude.com/docs/en/build-with-claude/claude-on-vertex-ai
CLAUDE_GEAP_NAME_MAP: Dict[str, str] = {
    "claude-opus-4-5": "claude-opus-4-5@20251101",
    "claude-sonnet-4-5": "claude-sonnet-4-5@20250929",
    "claude-haiku-4-5": "claude-haiku-4-5@20251001",
}


def normalize_geap_claude_model(model: str) -> str:
    """'claude-opus-5.5' / 'anthropic/claude-opus-5.5' -> 'claude-opus-5-5' (keeps '@date' pins)."""
    m = model.strip()
    if m.lower().startswith("anthropic/"):
        m = m.split("/", 1)[1]
    base, sep, pin = m.partition("@")
    base = base.lower().replace(".", "-").replace("_", "-")
    if sep:
        return f"{base}@{pin}"
    return CLAUDE_GEAP_NAME_MAP.get(base, base)


class AnthropicProvider(BaseProvider):
    """Adapter for Anthropic Claude API models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "claude-haiku-4.5",
        client: Optional[Any] = None,
        use_vertex_ai: bool = False, # ADDED
        region: Optional[str] = None, # ADDED
        project_id: Optional[str] = None, # ADDED
        **kwargs: Any,
    ):
        super().__init__(api_key=api_key, default_model=default_model, **kwargs)
        self._client = client
        self.use_vertex_ai = use_vertex_ai # ADDED
        self.region = region # ADDED
        self.project_id = project_id # ADDED


    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _get_client(self) -> Any:
        """Get or initialize the Anthropic client (standard or Vertex AI)."""
        if self._client is not None:
            return self._client
        
        if self.use_vertex_ai:
            from anthropic import AsyncAnthropicVertex  # lazy: needs the anthropic[vertex] extra (google-auth)
            if not self.region or not self.project_id:
                raise ValueError("AnthropicVertex client requires region and project_id for GEAP.")
            self._client = AsyncAnthropicVertex(region=self.region, project_id=self.project_id)
        else:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
            
        return self._client

    def _format_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        anthropic_tools = []
        for t in tools:
            anthropic_tools.append({
                "name": t.get("name"),
                "description": t.get("description", ""),
                "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
            })
        return anthropic_tools

    def _format_messages(self, messages: List[Message]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        system_parts = [m.content.strip() for m in messages if m.role == "system" and m.content]
        system_prompt = "\n\n".join(system_parts) if system_parts else None
        raw_msgs: List[Dict[str, Any]] = []

        for m in messages:
            if m.role == "system":
                continue

            if m.role == "user":
                images = extract_and_resolve_images(m)
                if not images:
                    raw_msgs.append({"role": "user", "content": m.content or ""})
                else:
                    content_blocks: List[Dict[str, Any]] = []
                    for img_bytes, mime_type in images:
                        if not mime_type or not (mime_type.startswith("image/") or mime_type == "application/pdf"):
                            continue
                        b64_str = base64.b64encode(img_bytes).decode("ascii")
                        if mime_type == "application/pdf":
                            content_blocks.append({
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": b64_str,
                                },
                            })
                        else:
                            content_blocks.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": b64_str,
                                },
                            })
                    if m.content:
                        content_blocks.append({"type": "text", "text": m.content})
                    raw_msgs.append({"role": "user", "content": content_blocks})
            elif m.role in ("assistant", "model"):
                content_blocks: List[Dict[str, Any]] = []
                if m.content:
                    content_blocks.append({"type": "text", "text": m.content})
                if m.tool_calls:
                    for tc in m.tool_calls:
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments or {},
                        })
                raw_msgs.append({"role": "assistant", "content": content_blocks or (m.content or "")})
            elif m.role == "tool":
                call_id = m.tool_result.call_id if m.tool_result else m.id
                raw_out = m.tool_result.output if m.tool_result else (m.content or "")
                out_str = raw_out if isinstance(raw_out, str) else json.dumps(raw_out)
                raw_msgs.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": out_str,
                        "is_error": m.tool_result.is_error if m.tool_result else False,
                    }]
                })

        if not raw_msgs:
            return system_prompt, [{"role": "user", "content": "Hello"}]

        # Merge consecutive turns of the same role for Anthropic
        merged: List[Dict[str, Any]] = []
        for m in raw_msgs:
            if merged and merged[-1]["role"] == m["role"]:
                c1 = merged[-1]["content"]
                c2 = m["content"]
                if isinstance(c1, str) and isinstance(c2, str):
                    if c1.strip() == c2.strip():
                        continue
                    merged[-1]["content"] = (c1 + "\n" + c2).strip()
                elif isinstance(c1, list) and isinstance(c2, list):
                    merged[-1]["content"].extend(c2)
                elif isinstance(c1, list) and isinstance(c2, str):
                    if c2.strip():
                        merged[-1]["content"].append({"type": "text", "text": c2})
                elif isinstance(c1, str) and isinstance(c2, list):
                    blocks = [{"type": "text", "text": c1}] if c1.strip() else []
                    blocks.extend(c2)
                    merged[-1]["content"] = blocks
            else:
                merged.append(m)

        # Ensure first turn is user
        if merged and merged[0]["role"] != "user":
            merged.insert(0, {"role": "user", "content": "[Session resumed]"})

        return system_prompt, merged

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Message, UsageMetadata]:
        """Generate response from Anthropic with exponential backoff retry."""
        import anthropic

        target_model = model or self.default_model or "claude-haiku-4.5"

        if self.use_vertex_ai:
            # AnthropicVertex builds .../publishers/anthropic/models/{model}:rawPredict itself,
            # so it needs the bare GEAP model id, never a fully-qualified resource path.
            if target_model.startswith("projects/"):
                target_model = target_model.rsplit("/models/", 1)[-1]
            target_model = normalize_geap_claude_model(target_model)
        client = self._get_client()

        system_prompt, formatted_msgs = self._format_messages(messages)
        anthropic_tools = self._format_tools(tools)

        req_kwargs: Dict[str, Any] = {
            "model": target_model,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "messages": formatted_msgs,
        }
        if system_prompt:
            req_kwargs["system"] = system_prompt
        if anthropic_tools:
            req_kwargs["tools"] = anthropic_tools

        max_retries = kwargs.get("max_retries", self.max_retries)
        initial_delay = kwargs.get("initial_delay", self.initial_delay)
        backoff_factor = kwargs.get("backoff_factor", self.backoff_factor)
        max_delay = kwargs.get("max_delay", self.max_delay)
        jitter = kwargs.get("jitter", True)

        async def _call():
            response = await client.messages.create(**req_kwargs)

            # Extract text and tool calls
            text_parts = []
            tool_calls: List[ToolCall] = []

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append(ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    ))

            # Extract token usage
            prompt_tokens = getattr(response.usage, "input_tokens", 0)
            completion_tokens = getattr(response.usage, "output_tokens", 0)
            cached_tokens = getattr(response.usage, "cache_read_input_tokens", None)

            usage = UsageMetadata(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cached_tokens=cached_tokens,
            )

            raw_content = "\n".join(text_parts) if text_parts else None
            content = collapse_repeating_text(raw_content) if raw_content else None
            msg = Message(
                role="assistant",
                content=content,
                model=target_model,
                provider="anthropic",
                tool_calls=tool_calls if tool_calls else None,
                usage=usage,
            )

            return msg, usage

        return await execute_with_retry(
            _call,
            provider_name=self.provider_name,
            max_retries=max_retries,
            initial_delay=initial_delay,
            backoff_factor=backoff_factor,
            max_delay=max_delay,
            jitter=jitter,
        )
