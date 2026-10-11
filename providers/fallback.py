"""Multi-provider fallback cascade engine with automatic error recovery and failover.

Prioritizes Gemini ($0.00 credit pool), then DigitalOcean Serverless Inference
(DeepSeek, Kimi, GPT-oss-120b, Llama, Mistral under ONE payment gateway),
with direct proprietary providers as emergency safety nets.
"""

import asyncio
import inspect
import logging
import os
import re
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, Union

from core.keystore import KeyStore, looks_like_placeholder_key, mask_key
from core.models import Message, StreamEvent, StreamEventType, ToolCall, ToolResult, UsageMetadata
from core.retry import extract_status_code, is_retryable_error
from providers.base import BaseProvider

logger = logging.getLogger("vsagent.fallback")


def is_provider_failover_candidate(exc: BaseException) -> bool:
    """Determines whether an exception is a candidate for cascading to an alternative provider tier.

    Transient errors (503, 429, timeouts) and provider-specific errors (e.g. 400 thought_signature,
    401 expired key, 404 model removed) trigger failover to the next provider.
    Pure internal application bugs (ValueError, TypeError, etc.) are fatal and not cascaded.
    """
    if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
        return False

    if is_retryable_error(exc):
        return True

    code = extract_status_code(exc)
    if code is not None:
        # Any HTTP/API response from an external provider endpoint is a candidate for failover
        return True

    exc_str = (str(exc) + " " + repr(exc)).lower()
    if any(p in exc_str for p in ("thought_signature", "thought signature", "quota", "rate limit", "unavailable")):
        return True

    exc_type = type(exc).__name__
    if any(k in exc_type for k in ("API", "ClientError", "ServerError", "HTTP", "GenAI")):
        return True

    # Pure Python built-in exceptions indicate monorepo code or schema bugs; raise immediately
    if isinstance(exc, (ValueError, TypeError, AttributeError, KeyError, IndexError, AssertionError, NotImplementedError)):
        return False

    return False

# NOTE: model IDs for provider "do" must match DigitalOcean's catalog (GET https://inference.do-ai.run/v1/models)
# exactly, e.g. "openai-gpt-oss-120b" (not "openai/gpt-oss-120b"). Run `vsagent doctor` to verify them.

# Default Tier 2 (Reasoner / Monorepo Dev / QA) Cascade Hierarchy
DEFAULT_REASONER_CASCADE_SPECS = [
    {"provider": "aistudio", "model": "gemini-flash-latest", "label": "Gemini (AI Studio)"},
    {"provider": "do", "model": "openai-gpt-oss-120b", "label": "DO GPT-oss-120b"},
    {"provider": "do", "model": "kimi-k2.6", "label": "DO Kimi K2.6"},
    {"provider": "do", "model": "deepseek-v4-pro", "label": "DO DeepSeek V4 Pro"},
    {"provider": "deepseek", "model": "deepseek-chat", "label": "DeepSeek Direct"},
    {"provider": "kimi", "model": "moonshot-v1-128k", "label": "Kimi Direct"},
    {"provider": "openai", "model": "gpt-4o", "label": "OpenAI Direct (Emergency)"},
    {"provider": "anthropic", "model": "claude-3-7-sonnet-20250219", "label": "Anthropic Direct (Emergency)"},
]

# Default Tier 1 (Heavy Thinker / Architecture / RFC) Cascade Hierarchy
DEFAULT_THINKER_CASCADE_SPECS = [
    {"provider": "aistudio", "model": "gemini-pro-latest", "label": "Gemini 3.1 Pro (AI Studio)"},
    {"provider": "do", "model": "anthropic-claude-opus-5", "label": "DO Claude Opus 5"},
    {"provider": "do", "model": "openai-gpt-5.6-sol", "label": "DO GPT-5.6 Sol"},
    {"provider": "anthropic", "model": "claude-3-opus-20240229", "label": "Anthropic Direct (Emergency)"},
    {"provider": "openai", "model": "o3-mini", "label": "OpenAI Direct (Emergency)"},
]

# Default Tier 3 (Worker Bee / Subagent / CUD / Grep) Cascade Hierarchy
DEFAULT_WORKER_CASCADE_SPECS = [
    {"provider": "aistudio", "model": "gemini-flash-lite-latest", "label": "Gemini 3.5 Flash-Lite (AI Studio)"},
    {"provider": "do", "model": "deepseek-4-flash", "label": "DO DeepSeek V4 Flash"},
    {"provider": "do", "model": "openai-gpt-oss-120b", "label": "DO GPT-oss-120b"},
    {"provider": "do", "model": "openai-gpt-5-nano", "label": "DO GPT-5 Nano"},
    {"provider": "do", "model": "llama-4-maverick", "label": "DO Llama 4 Maverick"},
    {"provider": "deepseek", "model": "deepseek-chat", "label": "DeepSeek Direct"},
]


_AISTUDIO_NAMES = ("aistudio", "ai-studio")
_GEMINI_FAMILY = ("gemini", "google", "aistudio", "ai-studio", "vertex", "geap")

_PERMANENT_FAILURE_STATUS = (401, 403, 404)
_PERMANENT_FAILURE_TYPES = ("AuthenticationError", "PermissionDeniedError", "NotFoundError")


class InjectedFaultError(Exception):
    """Raised when VALSTORM_FAULT_INJECT simulates a failure on a provider tier."""

    def __init__(self, message: str, status_code: int = 429):
        super().__init__(message)
        self.status_code = status_code


def _check_injected_fault(index: int, tier: "FallbackTier") -> Optional[Tuple[int, str]]:
    """Checks VALSTORM_FAULT_INJECT for simulation rules like 'primary:429' or 'tier1:429'."""
    raw = os.environ.get("VALSTORM_FAULT_INJECT", "").strip()
    if not raw:
        return None
    for rule in raw.split(","):
        rule = rule.strip()
        if not rule or ":" not in rule:
            continue
        target, code_str = rule.split(":", 1)
        target = target.strip().lower()
        try:
            code = int(code_str.strip())
        except ValueError:
            code = 429

        matched = False
        if target in ("primary", "0") and index == 0:
            matched = True
        elif target == f"tier{index + 1}":
            matched = True
        elif target in (tier.provider_name.lower(), tier.model.lower()):
            matched = True
        elif target in tier.label.lower():
            matched = True

        if matched:
            return code, f"rule '{rule}' matched"
    return None


def permanent_tier_failure_reason(exc: BaseException) -> Optional[str]:
    """Returns a short reason when an error means *this tier cannot work until config changes*.

    401/403 (bad or unauthorised key) and 404 (unknown model) do not heal on their own, so retrying the tier
    on every turn only burns a hop. Transient errors (429/5xx/529/timeouts) return None.
    """
    code = extract_status_code(exc)
    if code in _PERMANENT_FAILURE_STATUS:
        return f"HTTP {code}"
    name = type(exc).__name__
    if name in _PERMANENT_FAILURE_TYPES:
        return name
    return None


def _is_output_item(item: Any) -> bool:
    """True once a stream item has put real model output (text or a tool call) in front of the consumer."""
    if isinstance(item, StreamEvent):
        if item.event_type == StreamEventType.TEXT_CHUNK:
            return bool(item.delta)
        return item.event_type == StreamEventType.TOOL_CALL_DETECTED
    return True  # (Message, usage) tuples / bare Messages are complete results


class FallbackTier:
    """Represents an instantiated provider and target model in the fallback chain."""

    def __init__(
        self,
        provider: BaseProvider,
        model: str,
        label: str,
        provider_name: str,
        key_source: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model
        self.label = label
        self.provider_name = provider_name
        # Where this tier's credential came from (e.g. "env:DIGITALOCEAN_AI_KEY" or a .env path); never the key itself.
        self.key_source = key_source

    def __repr__(self) -> str:
        return f"<FallbackTier label='{self.label}' provider='{self.provider_name}' model='{self.model}'>"


class ChainedFallbackProvider(BaseProvider):
    """Executes LLM turns across an ordered priority chain with automatic 503/429 failover."""

    def __init__(
        self,
        tiers: List[FallbackTier],
        default_model: Optional[str] = None,
        on_failover_callback: Optional[Callable[[FallbackTier, FallbackTier, BaseException], None]] = None,
        cooldown_sec: float = 60.0,
        retries_before_failover: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if not tiers:
            raise ValueError("ChainedFallbackProvider requires at least one active FallbackTier.")
        primary_model = default_model or tiers[0].model
        super().__init__(default_model=primary_model, **kwargs)
        self.tiers = tiers
        self.on_failover_callback = on_failover_callback
        self.cooldown_sec = cooldown_sec
        # Providers retry transient errors (429/5xx/529) with exponential backoff on their own (3 retries by default,
        # which adds up to tens of seconds on an overloaded tier). While another tier is still available, a tier only
        # gets this many retries before we move on; the last live tier keeps the provider's full retry budget.
        # Override with VALSTORM_RETRIES_BEFORE_FAILOVER (0 = fail over immediately).
        if retries_before_failover is None:
            env_value = os.environ.get("VALSTORM_RETRIES_BEFORE_FAILOVER", "").strip()
            retries_before_failover = int(env_value) if env_value.isdigit() else 1
        self.retries_before_failover = max(0, retries_before_failover)
        self._tier_cooldowns: Dict[int, float] = {}
        # Tiers disabled for the rest of this process (bad key / unknown model). Never includes tier 0.
        self._dead_tiers: Dict[int, str] = {}
        self._current_tier_index: int = 0

    def set_request_context(self, **context: Any) -> None:
        super().set_request_context(**context)
        for tier in self.tiers:
            setter = getattr(tier.provider, "set_request_context", None)
            if callable(setter):
                setter(**context)

    def reset_active_tier(self) -> None:
        """Resets tier index to primary (0) and clears all failure cooldowns and disabled tiers."""
        self._current_tier_index = 0
        self._tier_cooldowns.clear()
        self._dead_tiers.clear()

    @property
    def active_tier(self) -> FallbackTier:
        if 0 <= self._current_tier_index < len(self.tiers):
            return self.tiers[self._current_tier_index]
        return self.tiers[0]

    @property
    def provider_name(self) -> str:
        return self.active_tier.provider_name

    @property
    def default_model(self) -> str:
        if self.tiers and self.active_tier:
            return self.active_tier.model
        return getattr(self, "_default_model", "gemini-flash-latest")

    @default_model.setter
    def default_model(self, value: Optional[str]) -> None:
        self._default_model = value

    async def generate_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Union[StreamEvent, Tuple[Message, Optional[UsageMetadata]], Message]]:
        """Attempts generation through the tier chain, catching provider errors and failing over."""
        collected_errors: List[str] = []

        now = time.time()
        start_index = 0
        # If no explicit model override was requested, skip tiers currently in failure cooldown
        if not model:
            while start_index < len(self.tiers) - 1 and (
                self._tier_cooldowns.get(start_index, 0) > now or start_index in self._dead_tiers
            ):
                start_index += 1
            if start_index > 0:
                logger.info(
                    f"Tier 1..{start_index} in active failure cooldown ({self.cooldown_sec}s); "
                    f"starting turn at Tier {start_index + 1}: {self.tiers[start_index].label}"
                )

        for index in range(start_index, len(self.tiers)):
            tier = self.tiers[index]
            if index in self._dead_tiers:
                collected_errors.append(
                    f"Tier {index + 1} ({tier.label}) skipped: disabled for this session ({self._dead_tiers[index]})"
                )
                continue
            self._current_tier_index = index
            emitted_output = False  # becomes True once this tier has put text/tool calls in front of the consumer

            # Cap provider-level retries while there is somewhere else to go (unless the caller set it explicitly).
            call_kwargs = dict(kwargs)
            has_later_live_tier = any(i not in self._dead_tiers for i in range(index + 1, len(self.tiers)))
            if has_later_live_tier and "max_retries" not in call_kwargs:
                call_kwargs["max_retries"] = self.retries_before_failover
            target_model = model if (index == 0 and model) else tier.model
            logger.info(f"Attempting turn via Tier {index + 1}/{len(self.tiers)}: {tier.label} ({target_model})")

            try:
                fault = _check_injected_fault(index, tier)
                if fault is not None:
                    fault_code, fault_desc = fault
                    raise InjectedFaultError(
                        f"Injected fault ({fault_desc}) simulating HTTP {fault_code} on {tier.label}",
                        status_code=fault_code,
                    )

                # Check if provider supports streaming
                if hasattr(tier.provider, "generate_stream"):
                    stream = tier.provider.generate_stream(
                        messages=messages,
                        tools=tools,
                        model=target_model,
                        **call_kwargs,
                    )
                    if inspect.isawaitable(stream):
                        stream = await stream

                    if hasattr(stream, "__aiter__"):
                        async for item in stream:
                            if _is_output_item(item):
                                emitted_output = True
                            if isinstance(item, StreamEvent) and item.event_type == StreamEventType.TURN_COMPLETE:
                                item.metadata.setdefault("effective_tier", index + 1)
                                item.metadata.setdefault("effective_label", tier.label)
                                item.metadata.setdefault("effective_provider", tier.provider_name)
                                item.metadata.setdefault("effective_model", target_model)
                            yield item
                        self._tier_cooldowns.pop(index, None)
                        return  # Success!
                    elif hasattr(stream, "__iter__"):
                        for item in stream:
                            if _is_output_item(item):
                                emitted_output = True
                            if isinstance(item, StreamEvent) and item.event_type == StreamEventType.TURN_COMPLETE:
                                item.metadata.setdefault("effective_tier", index + 1)
                                item.metadata.setdefault("effective_label", tier.label)
                                item.metadata.setdefault("effective_provider", tier.provider_name)
                                item.metadata.setdefault("effective_model", target_model)
                            yield item
                        self._tier_cooldowns.pop(index, None)
                        return  # Success!
                    else:
                        yield stream
                        self._tier_cooldowns.pop(index, None)
                        return  # Success!

                else:
                    # Non-streaming provider execution
                    if inspect.iscoroutinefunction(tier.provider.generate):
                        res = await tier.provider.generate(
                            messages=messages,
                            tools=tools,
                            model=target_model,
                            **call_kwargs,
                        )
                    else:
                        res = tier.provider.generate(
                            messages=messages,
                            tools=tools,
                            model=target_model,
                            **call_kwargs,
                        )
                        if inspect.isawaitable(res):
                            res = await res

                    if isinstance(res, tuple):
                        final_msg, final_usage = res[0], res[1]
                    elif isinstance(res, Message):
                        final_msg, final_usage = res, getattr(res, "usage", None)
                    else:
                        final_msg, final_usage = None, None

                    if final_msg:
                        if not final_msg.provider:
                            final_msg.provider = tier.provider_name
                        if not final_msg.model:
                            final_msg.model = target_model
                        if final_msg.content:
                            yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta=final_msg.content)
                        if final_msg.tool_calls:
                            for tc in final_msg.tool_calls:
                                yield StreamEvent(event_type=StreamEventType.TOOL_CALL_DETECTED, tool_call=tc)
                        yield StreamEvent(
                            event_type=StreamEventType.TURN_COMPLETE,
                            message=final_msg,
                            usage=final_usage,
                            metadata={
                                "effective_tier": index + 1,
                                "effective_label": tier.label,
                                "effective_provider": tier.provider_name,
                                "effective_model": target_model,
                            },
                        )
                    else:
                        yield res
                    self._tier_cooldowns.pop(index, None)
                    return  # Success!

            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                can_failover = is_provider_failover_candidate(exc)
                err_summary = f"Tier {index + 1} ({tier.label} / {target_model}) failed: {type(exc).__name__}: {str(exc)}"
                logger.warning(err_summary)
                collected_errors.append(err_summary)

                # Set cooldown for this failed tier so subsequent tool steps in this run don't retry it
                self._tier_cooldowns[index] = time.time() + self.cooldown_sec

                # Bad key / unknown model: this tier cannot work until config changes. Disable it for the rest of
                # the process instead of retrying it every turn. Tier 0 is exempt (its auth is refreshed in-provider
                # and re-login fixes it) and keeps using the normal cooldown.
                dead_reason = permanent_tier_failure_reason(exc) if index > 0 else None
                if dead_reason and index not in self._dead_tiers:
                    self._dead_tiers[index] = dead_reason
                    logger.warning(
                        f"Disabling tier {index + 1} ({tier.label} / {target_model}) for this session: {dead_reason}. "
                        f"Key source: {tier.key_source or 'unknown'}. Fix the key/model and restart (or run `vsagent doctor`)."
                    )

                if emitted_output:
                    # Text or tool calls already reached the consumer. Cascading would replay the whole turn on another
                    # model and duplicate that output (or re-run tool calls), so surface the error instead.
                    logger.warning(
                        f"Tier {index + 1} ({tier.label}) failed AFTER emitting output; not failing over to avoid duplicate output."
                    )
                    raise

                next_index = next(
                    (i for i in range(index + 1, len(self.tiers)) if i not in self._dead_tiers),
                    None,
                )
                has_next_tier = next_index is not None

                if can_failover and has_next_tier:
                    next_tier = self.tiers[next_index]
                    self._current_tier_index = next_index
                    logger.info(f"Cascading from {tier.label} to {next_tier.label} due to {type(exc).__name__}: {exc}")

                    if self.on_failover_callback:
                        try:
                            self.on_failover_callback(tier, next_tier, exc)
                        except Exception:
                            pass

                    # Clean error preview without massive raw dumps
                    err_clean = str(exc)
                    if len(err_clean) > 120:
                        err_clean = err_clean[:120] + "..."

                    # Emit visual notice chunk into stream
                    failover_msg = (
                        f"\n\033[93m⚠️ [Provider Failover]: {tier.label} returned {type(exc).__name__} ({err_clean}). "
                        f"Cascading to {next_tier.label} ({next_tier.model})...\033[0m\n"
                    )
                    yield StreamEvent(
                        event_type=StreamEventType.TEXT_CHUNK,
                        delta=failover_msg,
                        metadata={
                            "failover": True,
                            "failed_tier": tier.label,
                            "failed_model": target_model,
                            "next_tier": next_tier.label,
                            "next_model": next_tier.model,
                            "error": str(exc),
                        },
                    )
                    continue  # Move to next tier in cascade

                # If last tier exhausted
                raise exc

        raise RuntimeError(f"All {len(self.tiers)} fallback tiers exhausted.\n" + "\n".join(collected_errors))

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Message, Optional[UsageMetadata]]:
        """Non-streaming generation wrapper with failover."""
        final_msg = None
        final_usage = None

        async for item in self.generate_stream(messages=messages, tools=tools, model=model, **kwargs):
            if isinstance(item, tuple) and len(item) >= 2:
                final_msg, final_usage = item[0], item[1]
            elif isinstance(item, Message):
                final_msg = item
            elif isinstance(item, StreamEvent) and item.event_type == StreamEventType.TURN_COMPLETE and item.message:
                final_msg = item.message
                final_usage = item.usage

        if final_msg is not None:
            return final_msg, final_usage
        raise RuntimeError("ChainedFallbackProvider did not yield a valid Message response.")


def infer_cascade_tier(
    model_name: Optional[str] = None,
    profile_cfg: Optional[Dict[str, Any]] = None,
) -> str:
    """Infers the appropriate fallback cascade tier (reasoner, thinker, worker)."""
    if profile_cfg and profile_cfg.get("model_tier"):
        raw_tier = str(profile_cfg["model_tier"]).lower()
        if "1" in raw_tier or "think" in raw_tier or "arch" in raw_tier:
            return "thinker"
        if "3" in raw_tier or "work" in raw_tier or "micro" in raw_tier:
            return "worker"
        return "reasoner"

    if model_name:
        # Match whole tokens, not substrings: "gemini" contains "mini", which used to push every Gemini model
        # (including gemini-pro-*) into the worker cascade.
        tokens = {t for t in re.split(r"[^a-z0-9]+", model_name.lower()) if t}
        if tokens & {"lite", "nano", "mini"}:
            return "worker"
        if tokens & {"opus", "o1", "o3"} or ("pro" in tokens and "flash" not in tokens):
            return "thinker"

    return "reasoner"


def _resolve_key_with_source(ks: Any, provider: str) -> Tuple[Optional[str], Optional[str]]:
    """Resolves a key and where it came from, tolerating KeyStore look-alikes without source tracking."""
    # Resolve the key through get_api_key (the long-standing public entry point, which callers and tests patch), then ask
    # for the source separately and only trust it when it agrees with the key we actually use.
    key = ks.get_api_key(provider)
    if not key:
        return None, None
    source: Optional[str] = None
    getter = getattr(ks, "get_api_key_with_source", None)
    if callable(getter):
        try:
            res = getter(provider)
        except Exception:
            res = None
        if isinstance(res, tuple) and len(res) == 2 and res[0] == key:
            source = res[1]
    return key, source


def build_fallback_chain(
    tier: str = "reasoner",
    primary_provider: Optional[str] = None,
    primary_model: Optional[str] = None,
    keystore: Optional[KeyStore] = None,
    custom_specs: Optional[List[Dict[str, Any]]] = None,
    enable_fallback: bool = True,
    **kwargs: Any,
) -> ChainedFallbackProvider:
    """Factory creating an active ChainedFallbackProvider rooted at your requested primary provider/model.

    If primary_provider is supplied, Tier 1 is always the requested primary model.
    Secondary and subsequent tiers form an automatic failover safety net based on available keys
    (DigitalOcean, Gemini, OpenAI, Anthropic) without requiring 'fallback' to be specified as provider.
    """
    from providers import resolve_provider_instance

    ks = keystore or KeyStore()
    t_norm = tier.strip().lower()

    # Check env override
    disable_env = os.environ.get("VALSTORM_DISABLE_FALLBACK", "").strip().lower() in ("1", "true", "yes")
    effective_enable_fallback = enable_fallback and not disable_env

    if custom_specs:
        specs = custom_specs
    elif t_norm in ("thinker", "architect", "tier1"):
        specs = DEFAULT_THINKER_CASCADE_SPECS
    elif t_norm in ("worker", "micro", "tier3"):
        specs = DEFAULT_WORKER_CASCADE_SPECS
    else:
        specs = DEFAULT_REASONER_CASCADE_SPECS

    active_tiers: List[FallbackTier] = []

    # 1. If primary_provider is explicitly requested (e.g. 'gemini', 'anthropic', 'openai', 'do')
    p_norm = (primary_provider or "").strip().lower()
    tier1_model = primary_model
    explicit_key = kwargs.pop("api_key", None)
    if p_norm and p_norm not in ("fallback", "cascade", "chained"):
        primary_key = explicit_key
        primary_source: Optional[str] = "explicit argument" if explicit_key else None
        if not primary_key and p_norm == "valstorm":
            try:
                from tools.valstorm_client import resolve_valstorm_credentials
                primary_key, _ = resolve_valstorm_credentials()
                if primary_key:
                    primary_source = "valstorm login"
            except Exception:
                pass
        if not primary_key:
            primary_key, primary_source = _resolve_key_with_source(ks, p_norm)
        if not primary_key and p_norm in ("do", "digitalocean"):
            for alias in ("digitalocean", "do"):
                primary_key, primary_source = _resolve_key_with_source(ks, alias)
                if primary_key:
                    break

        try:
            primary_inst, resolved_model, resolved_p_name = resolve_provider_instance(
                provider_name=p_norm,
                api_key=primary_key,
                model=primary_model,
                **kwargs,
            )
            tier1_model = resolved_model
            active_tiers.append(
                FallbackTier(
                    provider=primary_inst,
                    model=resolved_model,
                    label=f"{p_norm.title()} Primary ({resolved_model})",
                    provider_name=resolved_p_name,
                    key_source=primary_source,
                )
            )
        except Exception as err:
            logger.warning(f"Could not initialize primary provider '{p_norm}': {err}")

    # 2. Append cascade backup tiers if fallback is enabled
    if effective_enable_fallback:
        for spec in specs:
            spec_p = spec["provider"].lower()
            spec_m = spec["model"].lower()

            # Skip if same provider family as primary (e.g. don't failover from Gemini to Gemini)
            if p_norm and spec_p == p_norm:
                continue
            # The AI Studio tier exists to get *off* GEAP/Vertex quota. It is redundant only when the primary is
            # itself a Gemini provider already running on AI Studio.
            if spec_p in _AISTUDIO_NAMES and p_norm in _GEMINI_FAMILY:
                primary_backend = active_tiers[0].provider.backend_label() if active_tiers and hasattr(active_tiers[0].provider, "backend_label") else ""
                if p_norm in _AISTUDIO_NAMES or primary_backend == "aistudio":
                    continue
            if p_norm in ("do", "digitalocean") and spec_p in ("do", "digitalocean"):
                continue

            target_model = spec["model"]
            label = spec.get("label", f"{spec_p}/{target_model}")

            key_provider = "gemini" if spec_p in _AISTUDIO_NAMES else spec_p
            api_key, key_source = _resolve_key_with_source(ks, key_provider)
            if not api_key and spec_p in ("do", "digitalocean"):
                for alias in ("digitalocean", "do"):
                    api_key, key_source = _resolve_key_with_source(ks, alias)
                    if api_key:
                        break

            if not api_key:
                logger.debug(f"Skipping tier '{label}' - no API key configured for '{spec_p}'.")
                continue

            # A placeholder (e.g. OPENAI_API_KEY=local-vsagent exported for some local tool) can only ever produce a
            # 401 hop mid-failover. Skip the tier up front and say why.
            if spec_p not in ("ollama", "vllm") and looks_like_placeholder_key(api_key):
                logger.warning(
                    f"Skipping tier '{label}': the key from {key_source or 'an unknown source'} "
                    f"({mask_key(api_key)}) looks like a placeholder, not a real credential."
                )
                continue

            try:
                inst, resolved_model, resolved_p_name = resolve_provider_instance(
                    provider_name=spec_p,
                    api_key=api_key,
                    model=target_model,
                    **kwargs,
                )
                active_tiers.append(
                    FallbackTier(
                        provider=inst,
                        model=resolved_model,
                        label=label,
                        provider_name=resolved_p_name,
                        key_source=key_source,
                    )
                )
            except Exception as err:
                logger.warning(f"Could not initialize provider for tier '{label}': {err}")

    # 3. If no tiers could be initialized (e.g. no keys or primary failed to construct)
    if not active_tiers:
        # Fallback to default gemini even without verified key so error message indicates missing key
        default_inst, d_model, d_p = resolve_provider_instance("gemini", api_key=None, model="gemini-flash-latest")
        active_tiers.append(
            FallbackTier(
                provider=default_inst,
                model=d_model,
                label="Gemini Default (Key Required)",
                provider_name=d_p,
            )
        )

    for position, t in enumerate(active_tiers, 1):
        logger.info(
            f"Fallback chain tier {position}: {t.label} [{t.provider_name}/{t.model}] "
            f"key={mask_key(getattr(t.provider, 'api_key', None))} source={t.key_source or 'n/a'}"
        )

    return ChainedFallbackProvider(tiers=active_tiers, default_model=active_tiers[0].model, **kwargs)
