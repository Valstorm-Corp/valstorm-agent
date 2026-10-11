"""OpenAI provider adapter implementing BaseProvider."""

import base64
import json
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
import uuid

from core.models import Message, StreamEvent, StreamEventType, ToolCall, UsageMetadata
from core.retry import execute_stream_with_retry, execute_with_retry
from providers.base import BaseProvider, extract_and_resolve_images


def _normalize_finish_reason(raw: Any) -> Optional[str]:
    """Maps OpenAI-style finish reasons to the runtime's normalized vocabulary."""
    if not isinstance(raw, str) or not raw:
        return None
    r = raw.lower()
    if r in ("length", "max_tokens"):
        return "length"
    if r == "content_filter":
        return "safety"
    if "malformed" in r:
        return "malformed_tool_call"
    return r


class OpenAIProvider(BaseProvider):
    """Provider adapter for OpenAI models via official openai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gpt-4o",
        base_url: Optional[str] = None,
        client: Optional[Any] = None,
        provider_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key=api_key, default_model=default_model, **kwargs)
        self.base_url = base_url
        self._client = client
        self._provider_name = provider_name or "openai"

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def _get_client(self) -> Any:
        """Get or initialize the AsyncOpenAI client."""
        if self._client is not None:
            return self._client
        from openai import AsyncOpenAI
        client_kwargs: Dict[str, Any] = {}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self._client = AsyncOpenAI(**client_kwargs)
        return self._client

    def _is_auth_expired_error(self, exc: BaseException) -> bool:
        """Checks if exception is a 401 AuthenticationError or expired signature."""
        exc_type = type(exc).__name__
        if "AuthenticationError" in exc_type:
            return True
        err_msg = (str(exc) + " " + repr(exc)).lower()
        if "signature has expired" in err_msg or "expired" in err_msg:
            return True
        status = getattr(exc, "status_code", None)
        if status == 401:
            return True
        return False

    async def _refresh_valstorm_auth(self) -> bool:
        """Attempts to refresh token for Valstorm provider and updates the client."""
        try:
            from tools.valstorm_client import (
                resolve_valstorm_auth_context,
                refresh_valstorm_tokens_async,
                log_auth_debug,
            )
            _, base_url, refresh_token, auth_file = resolve_valstorm_auth_context()
            if not refresh_token:
                log_auth_debug("OpenAIProvider cannot refresh: No refresh token found.")
                return False

            tokens = await refresh_valstorm_tokens_async(base_url, refresh_token, auth_file_path=auth_file)
            if tokens:
                new_access, _ = tokens
                self.api_key = new_access
                self._client = None
                log_auth_debug("OpenAIProvider refreshed Valstorm token successfully.")
                return True
        except Exception as e:
            try:
                from tools.valstorm_client import log_auth_debug
                log_auth_debug(f"OpenAIProvider _refresh_valstorm_auth failed: {e}")
            except Exception:
                pass
        return False

    def _format_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Convert standard schemas or OpenAI schemas into OpenAI function calling format."""
        if not tools:
            return None

        formatted: List[Dict[str, Any]] = []
        for t in tools:
            if "type" in t and t["type"] == "function" and "function" in t:
                formatted.append(t)
            else:
                name = t.get("name", "")
                description = t.get("description", "")
                parameters = t.get("parameters", {"type": "object", "properties": {}})
                formatted.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": parameters,
                    },
                })
        return formatted

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert unified Message list into OpenAI chat messages payload."""
        formatted: List[Dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                formatted.append({
                    "role": "system",
                    "content": msg.content or "",
                })
            elif msg.role == "user":
                images = extract_and_resolve_images(msg)
                if not images:
                    formatted.append({
                        "role": "user",
                        "content": msg.content or "",
                    })
                else:
                    content_parts = []
                    if msg.content:
                        content_parts.append({"type": "text", "text": msg.content})
                    for img_bytes, mime_type in images:
                        if not mime_type or not mime_type.startswith("image/"):
                            continue
                        b64_str = base64.b64encode(img_bytes).decode("ascii")
                        data_uri = f"data:{mime_type};base64,{b64_str}"
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": data_uri},
                        })
                    formatted.append({
                        "role": "user",
                        "content": content_parts,
                    })
            elif msg.role == "assistant":
                asst_payload: Dict[str, Any] = {"role": "assistant"}
                if msg.content is not None:
                    asst_payload["content"] = msg.content
                else:
                    asst_payload["content"] = ""

                if msg.tool_calls:
                    asst_payload["tool_calls"] = []
                    for tc in msg.tool_calls:
                        tc_payload = {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": (
                                    json.dumps(tc.arguments)
                                    if isinstance(tc.arguments, dict)
                                    else str(tc.arguments)
                                ),
                            },
                        }
                        sig_val = getattr(tc, "thought_signature", None)
                        # thought_signature is a Gemini concept. Only the Valstorm gateway understands it; strict
                        # OpenAI-compatible endpoints (DigitalOcean, OpenAI, DeepSeek, Kimi) can reject the unknown
                        # field with a 400, which would break a mid-conversation failover from Gemini.
                        if sig_val is not None and self.provider_name == "valstorm":
                            if isinstance(sig_val, bytes):
                                sig_val = base64.b64encode(sig_val).decode("ascii")
                            tc_payload["thought_signature"] = str(sig_val)
                        asst_payload["tool_calls"].append(tc_payload)
                formatted.append(asst_payload)
            elif msg.role == "tool":
                tool_call_id = (
                    msg.tool_result.call_id
                    if msg.tool_result
                    else getattr(msg, "tool_call_id", str(uuid.uuid4()))
                )
                output_content = (
                    msg.tool_result.output
                    if msg.tool_result
                    else (msg.content or "")
                )
                if not isinstance(output_content, str):
                    output_content = json.dumps(output_content)

                # Check the most recent assistant message (skipping sibling tool results from the same
                # parallel batch) for a matching tool_call_id. Previously only formatted[-1] was checked,
                # so results 2..N of a parallel batch were downgraded to plain user text.
                valid_preceding_call = False
                for prev_msg in reversed(formatted):
                    prev_role = prev_msg.get("role")
                    if prev_role == "tool":
                        continue
                    if prev_role == "assistant":
                        for tc in prev_msg.get("tool_calls", []) or []:
                            if tc.get("id") == tool_call_id:
                                valid_preceding_call = True
                                break
                    break

                if valid_preceding_call:
                    formatted.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": output_content,
                    })
                else:
                    formatted.append({
                        "role": "user",
                        "content": f"[Historical tool result {getattr(msg.tool_result, 'name', '')}]: {output_content}",
                    })

        return formatted

    def _apply_valstorm_headers(self, req_kwargs: Dict[str, Any]) -> None:
        """Tags Valstorm gateway calls with the chat id so metered charges map back to the ai_chat."""
        if self.provider_name != "valstorm":
            return
        chat_id = (getattr(self, "_request_context", None) or {}).get("chat_id")
        if chat_id:
            headers = dict(req_kwargs.get("extra_headers") or {})
            headers["X-Valstorm-Chat-Id"] = str(chat_id)
            req_kwargs["extra_headers"] = headers

    def _parse_response(
        self, response: Any, model_name: str
    ) -> Tuple[Message, UsageMetadata]:
        """Extract content, tool calls, and token usage from OpenAI ChatCompletion."""
        # 1. Extract usage metadata
        usage = UsageMetadata()
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            usage.prompt_tokens = getattr(raw_usage, "prompt_tokens", 0) or 0
            usage.completion_tokens = getattr(raw_usage, "completion_tokens", 0) or 0
            usage.total_tokens = getattr(raw_usage, "total_tokens", 0) or (
                usage.prompt_tokens + usage.completion_tokens
            )
            # Optional cached tokens in prompt_tokens_details
            tokens_details = getattr(raw_usage, "prompt_tokens_details", None)
            if tokens_details is not None:
                usage.cached_tokens = getattr(tokens_details, "cached_tokens", None)

        # 2. Extract choice message
        tool_calls: List[ToolCall] = []
        content: Optional[str] = None

        finish_reason: Optional[str] = None
        if hasattr(response, "choices") and len(response.choices) > 0:
            choice_msg = response.choices[0].message
            content = getattr(choice_msg, "content", None)
            finish_reason = _normalize_finish_reason(getattr(response.choices[0], "finish_reason", None))

            raw_tool_calls = getattr(choice_msg, "tool_calls", None)
            if not content and not raw_tool_calls:
                extra = getattr(choice_msg, "model_extra", None) or {}
                reasoning = (
                    getattr(choice_msg, "reasoning_content", None)
                    or extra.get("reasoning_content")
                    or extra.get("thought")
                )
                if reasoning and isinstance(reasoning, str) and reasoning.strip():
                    content = reasoning
            if raw_tool_calls:
                for tc in raw_tool_calls:
                    tc_id = getattr(tc, "id", None) or str(uuid.uuid4())
                    func_obj = getattr(tc, "function", None)
                    func_name = getattr(func_obj, "name", "") if func_obj else ""
                    raw_args = getattr(func_obj, "arguments", "{}") if func_obj else "{}"

                    if isinstance(raw_args, str):
                        try:
                            parsed_args = json.loads(raw_args)
                        except Exception:
                            parsed_args = {"raw_arguments": raw_args}
                    elif isinstance(raw_args, dict):
                        parsed_args = raw_args
                    else:
                        parsed_args = {}

                    sig_val = getattr(tc, "thought_signature", None)
                    if not sig_val and isinstance(tc, dict):
                        sig_val = tc.get("thought_signature")
                    if not sig_val and func_obj:
                        sig_val = getattr(func_obj, "thought_signature", None) or (func_obj.get("thought_signature") if isinstance(func_obj, dict) else None)

                    tool_calls.append(
                        ToolCall(
                            id=tc_id,
                            name=func_name,
                            arguments=parsed_args,
                            thought_signature=sig_val,
                        )
                    )

        msg = Message(
            role="assistant",
            content=content,
            model=model_name,
            provider=self.provider_name,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            finish_reason=finish_reason,
        )
        return msg, usage

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Message, UsageMetadata]:
        """Generate response from OpenAI with exponential backoff retry."""
        model_name = model or self.default_model or "gpt-4o"
        client = self._get_client()

        formatted_messages = self._format_messages(messages)
        formatted_tools = self._format_tools(tools)

        req_kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": formatted_messages,
        }

        if formatted_tools:
            req_kwargs["tools"] = formatted_tools
        self._apply_valstorm_headers(req_kwargs)

        for k, v in kwargs.items():
            if k not in ("max_retries", "initial_delay", "backoff_factor", "max_delay", "jitter"):
                req_kwargs[k] = v

        max_retries = kwargs.get("max_retries", self.max_retries)
        initial_delay = kwargs.get("initial_delay", self.initial_delay)
        backoff_factor = kwargs.get("backoff_factor", self.backoff_factor)
        max_delay = kwargs.get("max_delay", self.max_delay)
        jitter = kwargs.get("jitter", True)

        async def _call():
            nonlocal client
            try:
                response = await client.chat.completions.create(**req_kwargs)
                return self._parse_response(response, model_name)
            except Exception as exc:
                if self.provider_name == "valstorm" and self._is_auth_expired_error(exc):
                    refreshed = await self._refresh_valstorm_auth()
                    if refreshed:
                        client = self._get_client()
                        response = await client.chat.completions.create(**req_kwargs)
                        return self._parse_response(response, model_name)
                raise

        return await execute_with_retry(
            _call,
            provider_name=self.provider_name,
            max_retries=max_retries,
            initial_delay=initial_delay,
            backoff_factor=backoff_factor,
            max_delay=max_delay,
            jitter=jitter,
        )

    async def generate_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream response chunks from OpenAI with exponential backoff retry."""
        model_name = model or self.default_model or "gpt-4o"
        client = self._get_client()

        formatted_messages = self._format_messages(messages)
        formatted_tools = self._format_tools(tools)

        req_kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": formatted_messages,
            "stream": True,
        }
        if formatted_tools:
            req_kwargs["tools"] = formatted_tools
        self._apply_valstorm_headers(req_kwargs)

        for k, v in kwargs.items():
            if k not in ("max_retries", "initial_delay", "backoff_factor", "max_delay", "jitter"):
                req_kwargs[k] = v

        async def _stream_call() -> AsyncIterator[StreamEvent]:
            nonlocal client
            try:
                stream = await client.chat.completions.create(**req_kwargs)
            except Exception as stream_err:
                if self.provider_name == "valstorm" and self._is_auth_expired_error(stream_err):
                    refreshed = await self._refresh_valstorm_auth()
                    if refreshed:
                        client = self._get_client()
                        try:
                            stream = await client.chat.completions.create(**req_kwargs)
                        except Exception:
                            stream = None
                    else:
                        stream = None
                else:
                    stream = None

                if stream is None:
                    # If stream fails immediately (e.g. streaming not supported on custom proxy), fallback to generate
                    res_msg, res_usage = await self.generate(messages=messages, tools=tools, model=model, **kwargs)
                    if res_msg.content:
                        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta=res_msg.content)
                    if res_msg.tool_calls:
                        for tc in res_msg.tool_calls:
                            yield StreamEvent(event_type=StreamEventType.TOOL_CALL_DETECTED, tool_call=tc)
                    yield StreamEvent(event_type=StreamEventType.TURN_COMPLETE, message=res_msg, usage=res_usage)
                    return

            accumulated_content: List[str] = []
            accumulated_reasoning: List[str] = []
            tool_calls_builder: Dict[int, Dict[str, Any]] = {}
            usage = UsageMetadata()
            stream_finish_reason: Optional[str] = None

            async for chunk in stream:
                raw_usage = getattr(chunk, "usage", None)
                if raw_usage is not None:
                    usage.prompt_tokens = getattr(raw_usage, "prompt_tokens", 0) or usage.prompt_tokens
                    usage.completion_tokens = getattr(raw_usage, "completion_tokens", 0) or usage.completion_tokens
                    usage.total_tokens = getattr(raw_usage, "total_tokens", 0) or usage.total_tokens
                    tokens_details = getattr(raw_usage, "prompt_tokens_details", None)
                    if tokens_details is not None:
                        cached = (
                            tokens_details.get("cached_tokens")
                            if isinstance(tokens_details, dict)
                            else getattr(tokens_details, "cached_tokens", None)
                        )
                        if cached is not None:
                            usage.cached_tokens = cached

                if not hasattr(chunk, "choices") or not chunk.choices:
                    continue

                chunk_finish = getattr(chunk.choices[0], "finish_reason", None)
                if chunk_finish:
                    stream_finish_reason = _normalize_finish_reason(chunk_finish)

                delta = chunk.choices[0].delta
                if not delta:
                    continue

                if getattr(delta, "content", None):
                    text_chunk = delta.content
                    accumulated_content.append(text_chunk)
                    yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta=text_chunk)
                else:
                    extra = getattr(delta, "model_extra", None) or {}
                    reasoning_chunk = (
                        getattr(delta, "reasoning_content", None)
                        or extra.get("reasoning_content")
                        or extra.get("thought")
                    )
                    if reasoning_chunk and isinstance(reasoning_chunk, str):
                        accumulated_reasoning.append(reasoning_chunk)

                if getattr(delta, "tool_calls", None):
                    for tc_chunk in delta.tool_calls:
                        idx = getattr(tc_chunk, "index", 0)
                        if idx not in tool_calls_builder:
                            tool_calls_builder[idx] = {
                                "id": getattr(tc_chunk, "id", None) or str(uuid.uuid4()),
                                "name": "",
                                "arguments": "",
                                "thought_signature": getattr(tc_chunk, "thought_signature", None) or (tc_chunk.get("thought_signature") if isinstance(tc_chunk, dict) else None),
                            }
                        if getattr(tc_chunk, "id", None):
                            tool_calls_builder[idx]["id"] = tc_chunk.id
                        if getattr(tc_chunk, "thought_signature", None):
                            tool_calls_builder[idx]["thought_signature"] = tc_chunk.thought_signature
                        elif isinstance(tc_chunk, dict) and tc_chunk.get("thought_signature"):
                            tool_calls_builder[idx]["thought_signature"] = tc_chunk["thought_signature"]
                        func = getattr(tc_chunk, "function", None)
                        if func:
                            if getattr(func, "name", None):
                                tool_calls_builder[idx]["name"] += func.name
                            if getattr(func, "arguments", None):
                                tool_calls_builder[idx]["arguments"] += func.arguments
                            if getattr(func, "thought_signature", None):
                                tool_calls_builder[idx]["thought_signature"] = func.thought_signature
                            elif isinstance(func, dict) and func.get("thought_signature"):
                                tool_calls_builder[idx]["thought_signature"] = func["thought_signature"]

            # NOTE: an empty stream used to trigger a silent second, non-streaming request (double
            # billing, and it hid the real finish reason). Now the empty message is returned with its
            # finish_reason so the ReAct engine can react (continue / retry / nudge) explicitly.

            final_tool_calls: List[ToolCall] = []
            for idx in sorted(tool_calls_builder.keys()):
                entry = tool_calls_builder[idx]
                args_dict = {}
                if entry["arguments"]:
                    try:
                        args_dict = json.loads(entry["arguments"])
                    except Exception:
                        args_dict = {"raw_arguments": entry["arguments"]}
                tc_obj = ToolCall(
                    id=entry["id"],
                    name=entry["name"],
                    arguments=args_dict,
                    thought_signature=entry.get("thought_signature"),
                )
                final_tool_calls.append(tc_obj)
                yield StreamEvent(event_type=StreamEventType.TOOL_CALL_DETECTED, tool_call=tc_obj)

            final_text = "".join(accumulated_content) if accumulated_content else None
            if not final_text and not final_tool_calls and accumulated_reasoning:
                final_text = "".join(accumulated_reasoning)
            final_msg = Message(
                role="assistant",
                content=final_text,
                model=model_name,
                provider=self.provider_name,
                tool_calls=final_tool_calls if final_tool_calls else None,
                usage=usage,
                finish_reason=stream_finish_reason,
            )
            yield StreamEvent(
                event_type=StreamEventType.TURN_COMPLETE,
                message=final_msg,
                usage=usage,
            )

        max_retries = kwargs.get("max_retries", self.max_retries)
        initial_delay = kwargs.get("initial_delay", self.initial_delay)
        backoff_factor = kwargs.get("backoff_factor", self.backoff_factor)
        max_delay = kwargs.get("max_delay", self.max_delay)
        jitter = kwargs.get("jitter", True)

        async for item in execute_stream_with_retry(
            _stream_call,
            provider_name=self.provider_name,
            max_retries=max_retries,
            initial_delay=initial_delay,
            backoff_factor=backoff_factor,
            max_delay=max_delay,
            jitter=jitter,
        ):
            yield item
