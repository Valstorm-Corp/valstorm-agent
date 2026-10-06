"""Gemini provider adapter implementing BaseProvider."""

from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union
import re
import uuid

from core.models import Message, StreamEvent, StreamEventType, ToolCall, UsageMetadata, collapse_repeating_text
from core.retry import execute_stream_with_retry, execute_with_retry
from providers.base import BaseProvider, extract_and_resolve_images


def _normalize_gemini_finish(raw: Any) -> Optional[str]:
    """Maps Gemini FinishReason enums/strings to the runtime's normalized vocabulary."""
    if raw is None:
        return None
    name = getattr(raw, "name", None)
    if not isinstance(name, str):
        name = raw if isinstance(raw, str) else None
    if not name:
        return None
    name = name.upper().split(".")[-1]
    if name in ("STOP", "FINISH_REASON_UNSPECIFIED"):
        return "stop"
    if name == "MAX_TOKENS":
        return "length"
    if "MALFORMED" in name or name in ("UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS"):
        return "malformed_tool_call"
    if name in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"):
        return "safety"
    return name.lower()


_GEMINI_RE = re.compile(r"^(google/)?gemini-[a-z0-9.\-]+$", re.IGNORECASE)

VERTEX_MODEL_MAP: Dict[str, str] = {
    "gemini-flash-latest": "gemini-3.8-flash",
    "gemini-flash-lite-latest": "gemini-2.5-flash-lite",
    "gemini-pro-latest": "gemini-2.5-pro",
}

def is_gemini_model(model_name: Optional[str]) -> bool:
    """True if the model name belongs to the Gemini family."""
    if not model_name:
        return False
    return bool(_GEMINI_RE.match(model_name.lower()))

def _as_str(val: Any) -> Optional[str]:
    return val if isinstance(val, str) and val else None


def _as_int(val: Any) -> Optional[int]:
    return val if isinstance(val, int) and not isinstance(val, bool) else None


def _error_status(exc: BaseException) -> Optional[int]:
    for attr in ("code", "status_code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    resp = getattr(exc, "response", None)
    val = getattr(resp, "status_code", None)
    return val if isinstance(val, int) else None


class GeminiProvider(BaseProvider):
    """Provider adapter for Google Gemini models via google-genai SDK.

    Backends (``backend`` arg, or VALSTORM_GEMINI_BACKEND env):
      - "aistudio": Gemini Developer API with an API key.
      - "vertex":   Vertex AI / Gemini Enterprise with Google credentials.
      - "valstorm": Valstorm API Gemini pass-through (/v1/ai/gemini/...). The request is
                    forwarded to Google in Gemini's native format, so thought signatures,
                    text + tool-call parts, ids and streaming survive unchanged. Falls back to
                    the OpenAI-compatible gateway automatically if the pass-through route is
                    not deployed yet.
      - None/"auto": legacy auto-detection (Vertex when Google credentials are present).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gemini-flash-latest",
        client: Optional[Any] = None,
        backend: Optional[str] = None,
        valstorm_base_url: Optional[str] = None,
        compat_provider: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key=api_key, default_model=default_model, **kwargs)
        self._client = client
        self.backend = (backend or "").strip().lower() or None
        self.valstorm_base_url = valstorm_base_url.rstrip("/") if valstorm_base_url else None
        self._compat_provider = compat_provider
        self._passthrough_unavailable = False
        self._thinking_disabled = False

    @property
    def provider_name(self) -> str:
        return "valstorm" if self.backend == "valstorm" else "gemini"

    @staticmethod
    def is_enterprise_mode() -> bool:
        """Determines whether Gemini Enterprise / Vertex AI mode is active.

        VALSTORM_GEMINI_BACKEND=aistudio|vertex overrides auto-detection explicitly. Without it,
        the presence of Google credentials (including ~/.valstorm/gcp/valstorm-gemini-enterprise-sa.json)
        selects Vertex, which silently ignores any AI Studio key.
        """
        import os
        explicit = os.getenv("VALSTORM_GEMINI_BACKEND", "").strip().lower()
        if explicit in ("aistudio", "ai-studio", "developer", "api-key", "apikey"):
            return False
        if explicit in ("vertex", "enterprise", "geap"):
            return True
        if os.getenv("GOOGLE_GENAI_USE_ENTERPRISE", "").lower() in ("true", "1", "yes"):
            return True
        if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1", "yes"):
            return True
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"):
            return True
        gac = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if gac and os.path.isfile(gac):
            return True
        default_sa = os.path.expanduser("~/.valstorm/gcp/valstorm-gemini-enterprise-sa.json")
        if os.path.isfile(default_sa):
            return True
        return False

    def backend_label(self) -> str:
        """Human-readable backend actually used for requests."""
        if self.backend == "valstorm":
            return "valstorm-compat" if self._passthrough_unavailable else "valstorm-passthrough"
        if self.backend == "vertex":
            return "vertex"
        if self.backend == "aistudio":
            return "aistudio"
        return "vertex" if self.is_enterprise_mode() else "aistudio"

    # ------------------------------------------------------------------ thinking
    def _thinking_config(self, model_name: Optional[str] = None) -> Optional[Any]:
        """Explicit thinking level (VALSTORM_THINKING_LEVEL, default "medium"; "off" to omit).

        Backends apply different defaults when none is sent, which is one reason agent effort
        changed between AI Studio and Vertex. Falls back to HIGH if the SDK lacks MEDIUM, and the
        caller disables it entirely if the model rejects the setting.
        """
        import os
        from google.genai import types

        if self._thinking_disabled:
            return None
        if model_name and "lite" in str(model_name).lower():
            return None
        level = (os.getenv("VALSTORM_THINKING_LEVEL") or "medium").strip().upper()
        if level in ("", "OFF", "NONE", "DEFAULT", "AUTO"):
            return None
        for candidate in (level, "HIGH"):
            try:
                return types.ThinkingConfig(thinking_level=candidate)
            except Exception:
                continue
        return None

    @staticmethod
    def _is_thinking_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return ("thinking" in msg or "thinking_level" in msg or "thinkinglevel" in msg) and (
            _error_status(exc) in (400, None) or "invalid" in msg
        )

    def _build_config(self, system_instruction: Optional[str], gemini_tools: Optional[List[Any]], kwargs: Dict[str, Any]) -> Any:
        from google.genai import types

        config_args: Dict[str, Any] = {}
        if system_instruction:
            config_args["system_instruction"] = system_instruction
        if gemini_tools:
            config_args["tools"] = gemini_tools
        for k in ("temperature", "top_p", "top_k", "max_output_tokens"):
            if k in kwargs:
                config_args[k] = kwargs[k]
        thinking = self._thinking_config(model_name=kwargs.get("model"))
        if thinking is not None:
            config_args["thinking_config"] = thinking
        if self.backend == "valstorm":
            chat_id = (getattr(self, "_request_context", None) or {}).get("chat_id")
            if chat_id:
                token = self.api_key or ""
                auth_header = token if token.startswith("Bearer ") else f"Bearer {token}"
                config_args["http_options"] = types.HttpOptions(
                    headers={"Authorization": auth_header, "X-Valstorm-Chat-Id": str(chat_id)}
                )
        return types.GenerateContentConfig(**config_args) if config_args else None

    # ------------------------------------------------------------------ valstorm helpers
    def _get_compat_provider(self) -> Any:
        if self._compat_provider is None:
            from providers.openai import OpenAIProvider

            env_base = os.environ.get("VALSTORM_BASE_URL") or os.environ.get("VALSTORM_API_URL")
            default_gemini_base = f"{env_base.rstrip('/')}/ai/gemini" if env_base else "https://api.valstorm.com/v1/ai/gemini"
            base = self.valstorm_base_url or default_gemini_base
            compat_base = base[: -len("/gemini")] if base.endswith("/gemini") else base
            self._compat_provider = OpenAIProvider(
                api_key=self.api_key,
                default_model=self.default_model,
                base_url=compat_base,
                provider_name="valstorm",
            )
        ctx = getattr(self, "_request_context", None) or {}
        if ctx and hasattr(self._compat_provider, "set_request_context"):
            self._compat_provider.set_request_context(**ctx)
        return self._compat_provider

    async def _refresh_valstorm_token(self) -> bool:
        try:
            from tools.valstorm_client import resolve_valstorm_auth_context, refresh_valstorm_tokens_async

            _, base_url, refresh_token, auth_file = resolve_valstorm_auth_context()
            if not refresh_token:
                return False
            tokens = await refresh_valstorm_tokens_async(base_url, refresh_token, auth_file_path=auth_file)
            if tokens:
                self.api_key = tokens[0]
                self._client = None
                if self._compat_provider is not None:
                    self._compat_provider.api_key = tokens[0]
                    self._compat_provider._client = None
                return True
        except Exception:
            pass
        return False

    def _get_client(self) -> Any:
        """Get or initialize the Google GenAI async client."""
        if self._client is not None:
            return self._client
        import os
        from google import genai

        if self.backend == "valstorm":
            from google.genai import types

            env_base = os.environ.get("VALSTORM_BASE_URL") or os.environ.get("VALSTORM_API_URL")
            default_gemini_base = f"{env_base.rstrip('/')}/ai/gemini" if env_base else "https://api.valstorm.com/v1/ai/gemini"
            base = self.valstorm_base_url or default_gemini_base
            token = self.api_key or ""
            auth_header = token if token.startswith("Bearer ") else f"Bearer {token}"
            self._client = genai.Client(
                vertexai=False,
                api_key="valstorm-managed",
                http_options=types.HttpOptions(base_url=base, headers={"Authorization": auth_header}),
            )
            return self._client

        use_vertex = self.backend == "vertex" or (self.backend != "aistudio" and self.is_enterprise_mode())

        if use_vertex:
            creds = None
            gac_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
            if gac_json:
                try:
                    import json
                    from google.oauth2.service_account import Credentials
                    info = json.loads(gac_json)
                    creds = Credentials.from_service_account_info(
                        info,
                        scopes=["https://www.googleapis.com/auth/cloud-platform"]
                    )
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error("Failed to load GOOGLE_APPLICATION_CREDENTIALS_JSON: %s", e)

            if not creds and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                default_sa = os.path.expanduser("~/.valstorm/gcp/valstorm-gemini-enterprise-sa.json")
                if os.path.isfile(default_sa):
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = default_sa

            project = (
                os.getenv("GOOGLE_CLOUD_PROJECT")
                or os.getenv("VALSTORM_GCP_PROJECT")
                or "valstorm-gemini"
            )
            location = (
                os.getenv("GOOGLE_CLOUD_LOCATION")
                or os.getenv("VALSTORM_GCP_LOCATION")
                or "global"
            )
            client_kwargs = {
                "vertexai": True,
                "project": project,
                "location": location,
            }
            if creds:
                client_kwargs["credentials"] = creds
            self._client = genai.Client(**client_kwargs)
        else:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _format_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Any]]:
        """Convert standard or OpenAI-format tool schemas into Gemini FunctionDeclarations."""
        if not tools:
            return None

        from google.genai import types

        function_declarations: List[types.FunctionDeclaration] = []
        for t in tools:
            # Handle OpenAI wrapper format {"type": "function", "function": {...}}
            if "function" in t and isinstance(t["function"], dict):
                f_data = t["function"]
            else:
                f_data = t

            name = f_data.get("name", "")
            description = f_data.get("description", "")
            parameters = f_data.get("parameters")

            # Clean parameters schema if needed
            decl_kwargs: Dict[str, Any] = {
                "name": name,
                "description": description,
            }
            if parameters:
                decl_kwargs["parameters"] = parameters

            function_declarations.append(types.FunctionDeclaration(**decl_kwargs))

        return [types.Tool(function_declarations=function_declarations)]

    def _format_messages(self, messages: List[Message]) -> Tuple[Optional[str], List[Any]]:
        """Convert unified Message list into system_instruction and Gemini Content objects."""
        import json
        from google.genai import types

        system_parts: List[str] = []
        for msg in messages:
            if msg.role == "system" and msg.content:
                system_parts.append(msg.content.strip())
        system_instruction = "\n\n".join(system_parts) if system_parts else None

        raw_contents: List[types.Content] = []
        unsigned_call_ids: set = set()
        unsigned_tool_names: set = set()

        for msg in messages:
            if msg.role == "system":
                continue

            parts: List[types.Part] = []

            if msg.role == "user":
                if msg.content:
                    cleaned_content = collapse_repeating_text(msg.content)
                    if cleaned_content:
                        parts.append(types.Part(text=cleaned_content))
                # Multimodal image parts
                images = extract_and_resolve_images(msg)
                for img_bytes, mime_type in images:
                    parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
                if not parts:
                    parts.append(types.Part(text="..."))
                raw_contents.append(types.Content(role="user", parts=parts))

            elif msg.role in ("assistant", "model"):
                if msg.content:
                    cleaned_content = collapse_repeating_text(msg.content)
                    if cleaned_content:
                        parts.append(types.Part(text=cleaned_content))
                if msg.tool_calls:
                    is_foreign_provider = bool(
                        (msg.provider and str(msg.provider).lower() not in ("gemini", "google", "valstorm"))
                        or (msg.model and not str(msg.model).lower().startswith(("gemini", "google/")))
                    )
                    for tc in msg.tool_calls:
                        sig = getattr(tc, "thought_signature", None)
                        if sig is not None:
                            if isinstance(sig, str):
                                import base64
                                try:
                                    sig = base64.b64decode(sig)
                                except Exception:
                                    sig = sig.encode("utf-8")
                            # If this tool name or ID was previously marked unsigned, re-enable signed mode
                            if tc.id in unsigned_call_ids:
                                unsigned_call_ids.remove(tc.id)
                            if tc.name in unsigned_tool_names:
                                unsigned_tool_names.remove(tc.name)
                            parts.append(
                                types.Part(
                                    function_call=types.FunctionCall(
                                        name=tc.name,
                                        args=tc.arguments or {},
                                        id=tc.id,
                                    ),
                                    thought_signature=sig,
                                )
                            )
                        elif is_foreign_provider:
                            # Tool call originates from a non-Gemini model (e.g. DeepSeek, OpenAI, Claude failover).
                            # Gemini 3.x strictly rejects functionCall parts without cryptographic thought_signature.
                            # Format as clean text observation so Gemini understands the historical tool execution without 400 error.
                            args_str = json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else str(tc.arguments or {})
                            parts.append(types.Part(text=f"[Executed tool `{tc.name}` with arguments: {args_str}]"))
                            if tc.id:
                                unsigned_call_ids.add(tc.id)
                            if tc.name:
                                unsigned_tool_names.add(tc.name)
                        else:
                            # Native Gemini tool call or unspecified provider (e.g. unit tests or historical messages)
                            parts.append(
                                types.Part(
                                    function_call=types.FunctionCall(
                                        name=tc.name,
                                        args=tc.arguments or {},
                                        id=tc.id,
                                    )
                                )
                            )
                if not parts:
                    parts.append(types.Part(text="..."))
                raw_contents.append(types.Content(role="model", parts=parts))

            elif msg.role == "tool":
                # In Gemini, function response is sent with role="user"
                tool_name = (
                    msg.tool_result.name
                    if msg.tool_result
                    else (getattr(msg, "name", None) or "tool")
                )
                output_val = (
                    msg.tool_result.output
                    if msg.tool_result
                    else (msg.content or "")
                )
                call_id = msg.tool_result.call_id if msg.tool_result else getattr(msg, "tool_call_id", None)

                is_unsigned = (
                    (call_id and call_id in unsigned_call_ids)
                    or (tool_name and tool_name in unsigned_tool_names)
                )

                if is_unsigned:
                    output_str = json.dumps(output_val) if isinstance(output_val, (dict, list)) else str(output_val or "")
                    parts.append(types.Part(text=f"[Historical tool result {tool_name}]: {output_str}"))
                    raw_contents.append(types.Content(role="user", parts=parts))
                else:
                    response_dict = (
                        output_val
                        if isinstance(output_val, dict)
                        else {"result": output_val}
                    )
                    parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=tool_name,
                                response=response_dict,
                                id=call_id,
                            )
                        )
                    )
                    raw_contents.append(types.Content(role="user", parts=parts))

        if not raw_contents:
            return system_instruction, [types.Content(role="user", parts=[types.Part(text="Hello")])]

        # Step 1: Merge consecutive turns of identical roles
        merged: List[types.Content] = []
        for c in raw_contents:
            if merged and merged[-1].role == c.role:
                for p in c.parts:
                    p_text = getattr(p, "text", None)
                    if p_text:
                        # Deduplicate identical text parts within the same model turn
                        existing_texts = [getattr(ep, "text", None) for ep in merged[-1].parts if getattr(ep, "text", None)]
                        if p_text in existing_texts:
                            continue
                    merged[-1].parts.append(p)
            else:
                merged.append(c)

        # Step 2: Ensure first turn is user
        if merged and merged[0].role != "user":
            merged.insert(0, types.Content(role="user", parts=[types.Part(text="[Session resumed]")]))

        # Step 3: Validate function call & response pairing
        sanitized: List[types.Content] = []
        for c in merged:
            if c.role == "user":
                new_parts: List[types.Part] = []
                prev_turn = sanitized[-1] if sanitized else None
                prev_fc_names = set()
                if prev_turn and prev_turn.role == "model":
                    for p in prev_turn.parts:
                        if getattr(p, "function_call", None):
                            prev_fc_names.add(p.function_call.name)

                for p in c.parts:
                    if getattr(p, "function_response", None):
                        fn_name = p.function_response.name
                        if fn_name in prev_fc_names:
                            new_parts.append(p)
                        else:
                            resp_data = p.function_response.response
                            resp_str = json.dumps(resp_data) if isinstance(resp_data, (dict, list)) else str(resp_data)
                            new_parts.append(types.Part(text=f"[Historical tool result {fn_name}]: {resp_str}"))
                    else:
                        new_parts.append(p)
                sanitized.append(types.Content(role="user", parts=new_parts))
            elif c.role == "model":
                sanitized.append(c)

        # Step 4: Re-merge after conversions and ensure starting with user
        final_contents: List[types.Content] = []
        for c in sanitized:
            if final_contents and final_contents[-1].role == c.role:
                for p in c.parts:
                    p_text = getattr(p, "text", None)
                    if p_text:
                        existing_texts = [getattr(ep, "text", None) for ep in final_contents[-1].parts if getattr(ep, "text", None)]
                        if p_text in existing_texts:
                            continue
                    final_contents[-1].parts.append(p)
            else:
                final_contents.append(c)

        if final_contents and final_contents[0].role != "user":
            final_contents.insert(0, types.Content(role="user", parts=[types.Part(text="[Session resumed]")]))

        return system_instruction, final_contents

    def _parse_response(
        self, response: Any, model_name: str
    ) -> Tuple[Message, UsageMetadata]:
        """Extract content, function calls, and usage metadata from Gemini response."""
        # 1. Extract usage metadata
        usage = UsageMetadata()
        raw_usage = getattr(response, "usage_metadata", None)
        if raw_usage is not None:
            usage.prompt_tokens = getattr(raw_usage, "prompt_token_count", 0) or 0
            usage.completion_tokens = (
                getattr(raw_usage, "candidates_token_count", 0) or 0
            )
            usage.total_tokens = getattr(raw_usage, "total_token_count", 0) or (
                usage.prompt_tokens + usage.completion_tokens
            )
            usage.cached_tokens = getattr(raw_usage, "cached_content_token_count", None)

        # 2. Extract text and function calls
        tool_calls: List[ToolCall] = []
        text_parts: List[str] = []

        candidates = getattr(response, "candidates", None)
        if candidates and len(candidates) > 0:
            candidate = candidates[0]
            content_obj = getattr(candidate, "content", None)
            if content_obj and getattr(content_obj, "parts", None):
                for part in content_obj.parts:
                    # Text part
                    p_text = getattr(part, "text", None)
                    if p_text:
                        text_parts.append(p_text)
                    # Function call part
                    p_fc = getattr(part, "function_call", None)
                    if p_fc:
                        fc_id = getattr(p_fc, "id", None) or str(uuid.uuid4())
                        fc_name = getattr(p_fc, "name", "")
                        fc_args = getattr(p_fc, "args", {})
                        p_sig = getattr(part, "thought_signature", None)
                        tool_calls.append(
                            ToolCall(
                                id=fc_id,
                                name=fc_name,
                                arguments=dict(fc_args) if isinstance(fc_args, dict) else {},
                                thought_signature=p_sig,
                            )
                        )
        elif hasattr(response, "text") and response.text:
            text_parts.append(response.text)

        # Fallback if raw function_calls helper populated but no parts
        if not tool_calls and getattr(response, "function_calls", None):
            for fc in response.function_calls:
                fc_id = getattr(fc, "id", None) or str(uuid.uuid4())
                fc_name = getattr(fc, "name", "")
                fc_args = getattr(fc, "args", {})
                tool_calls.append(
                    ToolCall(
                        id=fc_id,
                        name=fc_name,
                        arguments=dict(fc_args) if isinstance(fc_args, dict) else {},
                    )
                )

        content = "".join(text_parts).strip() if text_parts else None
        if content:
            content = collapse_repeating_text(content)

        finish_reason = None
        if candidates and len(candidates) > 0:
            finish_reason = _normalize_gemini_finish(getattr(candidates[0], "finish_reason", None))
        if raw_usage is not None:
            usage.thoughts_tokens = _as_int(getattr(raw_usage, "thoughts_token_count", None))

        msg = Message(
            role="assistant",
            content=content,
            model=model_name,
            provider=self.provider_name,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            finish_reason=finish_reason,
            model_version=_as_str(getattr(response, "model_version", None)),
        )
        return msg, usage

    async def generate_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Union[StreamEvent, Tuple[Message, UsageMetadata]]]:
        """Stream response chunks from Gemini, yielding StreamEvents and final (Message, UsageMetadata) with retry."""
        model_name = model or self.default_model or "gemini-flash-latest"
        use_vertex = self.backend == "vertex" or (self.backend != "aistudio" and self.is_enterprise_mode())
        if use_vertex:
            model_name = VERTEX_MODEL_MAP.get(model_name.lower(), model_name)

        # Intelligent routing for Valstorm Gateway backend (GEAP/Partner models)
        if self.backend == "valstorm" and not is_gemini_model(model_name):
            async for item in self._get_compat_provider().generate_stream(messages=messages, tools=tools, model=model_name, **kwargs):
                yield item
            return

        if self.backend == "valstorm" and self._passthrough_unavailable:
            async for item in self._get_compat_provider().generate_stream(messages=messages, tools=tools, model=model_name, **kwargs):
                yield item
            return

        system_instruction, contents = self._format_messages(messages)
        gemini_tools = self._format_tools(tools)

        max_retries = kwargs.get("max_retries", self.max_retries)
        initial_delay = kwargs.get("initial_delay", self.initial_delay)
        backoff_factor = kwargs.get("backoff_factor", self.backoff_factor)
        max_delay = kwargs.get("max_delay", self.max_delay)
        jitter = kwargs.get("jitter", True)

        async def _open_stream():
            """Opens the stream, handling thinking-config rejection, token refresh and missing pass-through."""
            attempts = 0
            while True:
                attempts += 1
                client = self._get_client()
                config = self._build_config(system_instruction, gemini_tools, {**kwargs, "model": model_name})
                try:
                    return await client.aio.models.generate_content_stream(
                        model=model_name,
                        contents=contents,
                        config=config,
                    )
                except Exception as exc:
                    status = _error_status(exc)
                    if attempts > 3:
                        raise
                    if not self._thinking_disabled and self._is_thinking_error(exc):
                        import logging
                        logging.getLogger(__name__).warning("Model rejected thinking_config; retrying without it: %s", exc)
                        self._thinking_disabled = True
                        continue
                    if self.backend == "valstorm" and status == 401 and await self._refresh_valstorm_token():
                        continue
                    if self.backend == "valstorm" and status in (404, 405):
                        # Pass-through route not deployed on this API yet: use the OpenAI-compatible gateway.
                        self._passthrough_unavailable = True
                        return None
                    raise

        async def _stream_call():
            stream = await _open_stream()
            if stream is None:
                async for item in self._get_compat_provider().generate_stream(messages=messages, tools=tools, model=model_name, **kwargs):
                    yield item
                return

            all_text_parts: List[str] = []
            collected_tool_calls: List[ToolCall] = []
            last_usage_metadata = None
            finish_reason: Optional[str] = None
            model_version: Optional[str] = None

            async for chunk in stream:
                if chunk.usage_metadata:
                    last_usage_metadata = chunk.usage_metadata
                if _as_str(getattr(chunk, "model_version", None)):
                    model_version = chunk.model_version

                candidates = getattr(chunk, "candidates", None)
                if candidates and len(candidates) > 0:
                    fr = getattr(candidates[0], "finish_reason", None)
                    if fr is not None:
                        finish_reason = _normalize_gemini_finish(fr)
                    content_obj = getattr(candidates[0], "content", None)
                    if content_obj and getattr(content_obj, "parts", None):
                        for part in content_obj.parts:
                            if getattr(part, "thought", None):
                                # Thought summaries (only if include_thoughts is on) are not answer text.
                                continue
                            p_text = getattr(part, "text", None)
                            if p_text:
                                all_text_parts.append(p_text)
                                yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta=p_text)

                            p_fc = getattr(part, "function_call", None)
                            if p_fc:
                                fc_id = getattr(p_fc, "id", None) or str(uuid.uuid4())
                                fc_name = getattr(p_fc, "name", "")
                                fc_args = getattr(p_fc, "args", {})
                                p_sig = getattr(part, "thought_signature", None)
                                tc = ToolCall(
                                    id=fc_id,
                                    name=fc_name,
                                    arguments=dict(fc_args) if isinstance(fc_args, dict) else {},
                                    thought_signature=p_sig,
                                )
                                collected_tool_calls.append(tc)
                                yield StreamEvent(event_type=StreamEventType.TOOL_CALL_DETECTED, tool_call=tc)

                elif hasattr(chunk, "text") and chunk.text:
                    all_text_parts.append(chunk.text)
                    yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta=chunk.text)

            usage = UsageMetadata()
            if last_usage_metadata is not None:
                usage.prompt_tokens = getattr(last_usage_metadata, "prompt_token_count", 0) or 0
                usage.completion_tokens = getattr(last_usage_metadata, "candidates_token_count", 0) or 0
                usage.total_tokens = getattr(last_usage_metadata, "total_token_count", 0) or (
                    usage.prompt_tokens + usage.completion_tokens
                )
                usage.cached_tokens = getattr(last_usage_metadata, "cached_content_token_count", None)
                usage.thoughts_tokens = _as_int(getattr(last_usage_metadata, "thoughts_token_count", None))

            final_content = "".join(all_text_parts).strip() if all_text_parts else None
            if final_content:
                final_content = collapse_repeating_text(final_content)
            msg = Message(
                role="assistant",
                content=final_content,
                model=model_name,
                provider=self.provider_name,
                tool_calls=collected_tool_calls if collected_tool_calls else None,
                usage=usage,
                finish_reason=finish_reason,
                model_version=model_version,
            )

            yield (msg, usage)

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

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Message, UsageMetadata]:
        """Generate response non-streaming from Gemini with exponential backoff retry."""
        model_name = model or self.default_model or "gemini-flash-latest"
        use_vertex = self.backend == "vertex" or (self.backend != "aistudio" and self.is_enterprise_mode())
        if use_vertex:
            model_name = VERTEX_MODEL_MAP.get(model_name.lower(), model_name)

        # Intelligent routing for Valstorm Gateway backend (GEAP/Partner models)
        if self.backend == "valstorm" and not is_gemini_model(model_name):
            return await self._get_compat_provider().generate(messages=messages, tools=tools, model=model_name, **kwargs)

        if self.backend == "valstorm" and self._passthrough_unavailable:
            return await self._get_compat_provider().generate(messages=messages, tools=tools, model=model_name, **kwargs)

        system_instruction, contents = self._format_messages(messages)
        gemini_tools = self._format_tools(tools)

        max_retries = kwargs.get("max_retries", self.max_retries)
        initial_delay = kwargs.get("initial_delay", self.initial_delay)
        backoff_factor = kwargs.get("backoff_factor", self.backoff_factor)
        max_delay = kwargs.get("max_delay", self.max_delay)
        jitter = kwargs.get("jitter", True)

        async def _call():
            attempts = 0
            while True:
                attempts += 1
                client = self._get_client()
                config = self._build_config(system_instruction, gemini_tools, {**kwargs, "model": model_name})
                try:
                    response = await client.aio.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config,
                    )
                    return self._parse_response(response, model_name)
                except Exception as exc:
                    status = _error_status(exc)
                    if attempts > 3:
                        raise
                    if not self._thinking_disabled and self._is_thinking_error(exc):
                        self._thinking_disabled = True
                        continue
                    if self.backend == "valstorm" and status == 401 and await self._refresh_valstorm_token():
                        continue
                    if self.backend == "valstorm" and status in (404, 405):
                        self._passthrough_unavailable = True
                        return await self._get_compat_provider().generate(messages=messages, tools=tools, model=model_name, **kwargs)
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
