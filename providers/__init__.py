"""Providers package for agent runtime.

Supports native Gemini, Anthropic Claude, DigitalOcean Serverless Inference,
and standard OpenAI-compatible endpoints (DeepSeek, Kimi/Moonshot, Groq, Mistral, Together, Ollama, vLLM).
"""

import os
from typing import Any, Dict, Optional, Tuple

from providers.base import BaseProvider
from providers.gemini import GeminiProvider
from providers.openai import OpenAIProvider
from providers.anthropic import AnthropicProvider
from providers.fallback import (
    ChainedFallbackProvider,
    FallbackTier,
    build_fallback_chain,
    infer_cascade_tier,
    DEFAULT_REASONER_CASCADE_SPECS,
    DEFAULT_THINKER_CASCADE_SPECS,
    DEFAULT_WORKER_CASCADE_SPECS,
)
from core.retry import (
    compute_backoff_delay,
    execute_stream_with_retry,
    execute_with_retry,
    is_retryable_error,
)

OPENAI_COMPATIBLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "valstorm": {
        "base_url": "https://api.valstorm.com/v1/ai",
        "default_model": "gemini-flash-latest",
        "env_key": "VALSTORM_API_KEY",
    },
    # DigitalOcean Serverless Inference (Unified Gateway)
    "digitalocean": {
        "base_url": "https://inference.do-ai.run/v1",
        "default_model": "deepseek-v4-pro",
        "env_key": "DIGITALOCEAN_AI_KEY",
    },
    "do": {
        "base_url": "https://inference.do-ai.run/v1",
        "default_model": "deepseek-v4-pro",
        "env_key": "DIGITALOCEAN_AI_KEY",
    },
    "do-deepseek": {
        "base_url": "https://inference.do-ai.run/v1",
        "default_model": "deepseek-v4-pro",
        "env_key": "DIGITALOCEAN_AI_KEY",
    },
    "do-flash": {
        "base_url": "https://inference.do-ai.run/v1",
        "default_model": "deepseek-4-flash",
        "env_key": "DIGITALOCEAN_AI_KEY",
    },
    "do-kimi": {
        "base_url": "https://inference.do-ai.run/v1",
        "default_model": "kimi-k2.6",
        "env_key": "DIGITALOCEAN_AI_KEY",
    },
    "do-oss": {
        "base_url": "https://inference.do-ai.run/v1",
        "default_model": "openai-gpt-oss-120b",
        "env_key": "DIGITALOCEAN_AI_KEY",
    },
    "do-llama": {
        "base_url": "https://inference.do-ai.run/v1",
        "default_model": "llama-4-maverick",
        "env_key": "DIGITALOCEAN_AI_KEY",
    },
    # Direct Vendor Endpoints
    "openai": {
        "base_url": None,
        "default_model": "gpt-4o",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "moonshot-v1-128k",
    },
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "moonshot-v1-128k",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "default_model": "codestral-latest",
    },
    "codestral": {
        "base_url": "https://codestral.mistral.ai/v1",
        "default_model": "codestral-latest",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "default_model": "deepseek-ai/DeepSeek-V3",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "deepseek/deepseek-chat",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "default_model": "deepseek-ai/DeepSeek-V3",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "default_model": "qwen2.5-coder:32b",
        "default_key": "ollama",
    },
    "vllm": {
        "base_url": "http://localhost:8000/v1",
        "default_model": "default",
        "default_key": "vllm",
    },
}


def resolve_provider_instance(
    provider_name: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[BaseProvider, str, str]:
    """Factory function resolving provider instance, model name, and normalized provider key."""
    p_norm = provider_name.strip().lower()

    if p_norm in ("fallback", "cascade", "chained"):
        tier = kwargs.pop("tier", "reasoner")
        chain = build_fallback_chain(tier=tier, **kwargs)
        return chain, (model or chain.active_tier.model), "fallback"

    if p_norm in ("gemini", "google", "aistudio", "ai-studio", "vertex", "geap"):
        chosen_model = model or "gemini-flash-latest"
        backend = None
        if p_norm in ("aistudio", "ai-studio"):
            backend = "aistudio"
        elif p_norm in ("vertex", "geap"):
            backend = "vertex"
        return GeminiProvider(api_key=api_key, default_model=chosen_model, backend=backend, **kwargs), chosen_model, "gemini"

    if p_norm in ("anthropic", "claude", "anthropic-geap"):
        chosen_model = model or "claude-3-7-sonnet-20250219"

        # GEAP/Vertex AI detection for Anthropic
        use_vertex_ai = os.getenv("ANTHROPIC_USE_VERTEX_AI", "").lower() in ("true", "1", "yes")
        region = os.getenv("ANTHROPIC_GEAP_REGION")
        project_id = os.getenv("ANTHROPIC_GEAP_PROJECT_ID")

        return AnthropicProvider(
            api_key=api_key,
            default_model=chosen_model,
            use_vertex_ai=use_vertex_ai,
            region=region,
            project_id=project_id,
            **kwargs
        ), chosen_model, "anthropic" if not use_vertex_ai else "anthropic-geap"

    if p_norm == "valstorm":
        target_base_url = base_url or os.environ.get("VALSTORM_AI_BASE_URL")
        target_token = api_key
        if not target_token or not target_base_url:
            try:
                from tools.valstorm_client import resolve_valstorm_credentials
                t_token, t_base = resolve_valstorm_credentials()
                if not target_token:
                    target_token = t_token
                if not target_base_url and t_base:
                    target_base_url = f"{t_base.rstrip('/')}/ai"
            except Exception:
                pass
        if target_base_url:
            tb = target_base_url.rstrip("/")
            if tb.endswith("/ai"):
                tb = tb[:-3].rstrip("/")
            if not tb.endswith("/v1") and "/v1" not in tb:
                tb = f"{tb}/v1"
            target_base_url = f"{tb}/ai"
        else:
            env_base = os.environ.get("VALSTORM_BASE_URL") or os.environ.get("VALSTORM_API_URL")
            if env_base:
                tb = env_base.rstrip("/")
                if tb.endswith("/ai"):
                    tb = tb[:-3].rstrip("/")
                if not tb.endswith("/v1") and "/v1" not in tb:
                    tb = f"{tb}/v1"
                target_base_url = f"{tb}/ai"
            else:
                target_base_url = "https://api.valstorm.com/v1/ai"
        chosen_model = model or "gemini-flash-latest"
        compat = OpenAIProvider(
            api_key=target_token or "valstorm_managed",
            default_model=chosen_model,
            base_url=target_base_url,
            provider_name="valstorm",
            **kwargs,
        )
        # Gemini models go through the API's native Gemini pass-through (/v1/ai/gemini) so the
        # request reaches Google in Gemini's own format (thought signatures, text + tool-call parts,
        # ids, real streaming). Translating to OpenAI format and back was lossy and is what degraded
        # agent persistence. Falls back to the OpenAI-compatible route automatically if the
        # pass-through isn't deployed. Disable with VALSTORM_GEMINI_PASSTHROUGH=0.
        passthrough_on = os.environ.get("VALSTORM_GEMINI_PASSTHROUGH", "1").strip().lower() not in ("0", "false", "no", "off")
        if passthrough_on and chosen_model.lower().startswith(("gemini", "google/gemini")):
            provider = GeminiProvider(
                api_key=target_token or "valstorm_managed",
                default_model=chosen_model,
                backend="valstorm",
                valstorm_base_url=f"{target_base_url.rstrip('/')}/gemini",
                compat_provider=compat,
                **kwargs,
            )
            return provider, chosen_model, "valstorm"
        return compat, chosen_model, "valstorm"

    # OpenAI or OpenAI-compatible presets (including DigitalOcean Serverless)
    preset = OPENAI_COMPATIBLE_PRESETS.get(p_norm, {})
    do_env_base = os.environ.get("DIGITALOCEAN_BASE_URL") or os.environ.get("DO_BASE_URL") if ("do" in p_norm or "digitalocean" in p_norm) else None
    target_base_url = base_url or do_env_base or preset.get("base_url")
    chosen_model = model or preset.get("default_model") or "gpt-4o"
    resolved_key = api_key or (preset.get("default_key") if not api_key and p_norm in ("ollama", "vllm") else api_key)

    provider = OpenAIProvider(
        api_key=resolved_key,
        default_model=chosen_model,
        base_url=target_base_url,
        provider_name=p_norm,
        **kwargs,
    )
    return provider, chosen_model, p_norm


__all__ = [
    "BaseProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "ChainedFallbackProvider",
    "FallbackTier",
    "build_fallback_chain",
    "infer_cascade_tier",
    "DEFAULT_REASONER_CASCADE_SPECS",
    "DEFAULT_THINKER_CASCADE_SPECS",
    "DEFAULT_WORKER_CASCADE_SPECS",
    "OPENAI_COMPATIBLE_PRESETS",
    "resolve_provider_instance",
    "is_retryable_error",
    "compute_backoff_delay",
    "execute_with_retry",
    "execute_stream_with_retry",
]
