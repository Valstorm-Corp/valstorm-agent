"""Shared Open Valstorm remote cross-plugin AI tool discovery with TTL caching.

Used by both `server.py` (top-level sessions) and `tools/delegation.py` (delegated
subagents) so that specialist subagents spawned by the Chief of Staff can execute
scoped Open Valstorm plugin tools (`quickbooks_*`, `stripe_*`, `meta_*`, `azure_*`, etc.).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from core.tools import ToolRegistry, infer_plugin_from_tool_name

logger = logging.getLogger(__name__)

# Short-lived in-process cache: (clean_base_url, token_fingerprint) -> (fetched_at_monotonic, tool_defs)
_REMOTE_TOOLS_CACHE: Dict[Tuple[str, str], Tuple[float, List[Dict[str, Any]]]] = {}
_REMOTE_TOOLS_TTL_SEC = 120.0


def clear_remote_plugin_tools_cache() -> None:
    """Clear the cached remote plugin tool catalog (primarily for tests)."""
    _REMOTE_TOOLS_CACHE.clear()


def _normalize_base_url(valstorm_base_url: Optional[str]) -> Optional[str]:
    effective_base = (
        valstorm_base_url
        or os.environ.get("VALSTORM_BASE_URL")
        or os.environ.get("VALSTORM_API_URL")
    )
    if not effective_base:
        return None
    clean_base = effective_base.rstrip("/")
    if not clean_base.endswith("/v1") and "/v1" not in clean_base:
        clean_base = f"{clean_base}/v1"
    return clean_base


def _token_cache_key(clean_base: str, valstorm_token: str) -> Tuple[str, str]:
    raw = valstorm_token.strip()
    if raw.lower().startswith("bearer "):
        raw = raw.split(" ", 1)[1].strip()
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return (clean_base, digest)


def make_remote_plugin_tool(
    tool_name: str,
    tool_desc: str,
    tool_params: Dict[str, Any],
    base_url: str,
    valstorm_token: str,
    plugin_name: Optional[str] = None,
) -> Callable:
    """Create an async runtime tool wrapper that proxies execution to `POST /v1/ai/tools/{tool_name}/execute`."""
    clean_base = base_url.rstrip("/")
    if not clean_base.endswith("/v1") and "/v1" not in clean_base:
        clean_base = f"{clean_base}/v1"
    auth_hdr = valstorm_token if valstorm_token.startswith("Bearer ") else f"Bearer {valstorm_token}"

    async def _remote_plugin_tool_runner(**kwargs: Any) -> str:
        exec_url = f"{clean_base}/ai/tools/execute"
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                exec_url,
                headers={"Authorization": auth_hdr, "Content-Type": "application/json"},
                json={"tool_name": tool_name, "arguments": kwargs},
            )
            resp.raise_for_status()
            data = resp.json()
            result = data.get("result", data) if isinstance(data, dict) else data
            return json.dumps(result, indent=2, default=str) if not isinstance(result, str) else result

    _remote_plugin_tool_runner.__name__ = tool_name
    _remote_plugin_tool_runner._is_tool = True  # type: ignore[attr-defined]
    _remote_plugin_tool_runner._tool_name = tool_name  # type: ignore[attr-defined]
    _remote_plugin_tool_runner._tool_description = tool_desc  # type: ignore[attr-defined]
    _remote_plugin_tool_runner._tool_plugin = plugin_name or infer_plugin_from_tool_name(tool_name)  # type: ignore[attr-defined]
    _remote_plugin_tool_runner._tool_schema = {  # type: ignore[attr-defined]
        "name": tool_name,
        "description": tool_desc,
        "parameters": tool_params or {"type": "object", "properties": {}},
    }
    return _remote_plugin_tool_runner


def _register_tool_defs(
    registry: ToolRegistry,
    remote_tools: List[Dict[str, Any]],
    clean_base: str,
    valstorm_token: str,
) -> List[str]:
    discovered_names: List[str] = []
    for t_def in remote_tools:
        if not isinstance(t_def, dict):
            continue
        t_name = str(t_def.get("name") or "").strip()
        if not t_name:
            continue
        discovered_names.append(t_name)
        existing = registry.get(t_name)
        if existing is not None:
            # Backfill plugin metadata if not already tagged on local native tool
            if not getattr(existing, "_tool_plugin", None):
                p_name = t_def.get("plugin") or infer_plugin_from_tool_name(t_name)
                if p_name:
                    existing._tool_plugin = p_name  # type: ignore[attr-defined]
            continue
        t_desc = str(t_def.get("description") or f"Tool: {t_name}")
        t_params = (
            t_def.get("parameters")
            if isinstance(t_def.get("parameters"), dict)
            else {"type": "object", "properties": {}}
        )
        p_name = t_def.get("plugin")
        runner = make_remote_plugin_tool(
            tool_name=t_name,
            tool_desc=t_desc,
            tool_params=t_params,
            base_url=clean_base,
            valstorm_token=valstorm_token,
            plugin_name=str(p_name) if p_name else None,
        )
        registry.register(runner)
    return discovered_names


async def discover_and_register_remote_plugin_tools_async(
    registry: ToolRegistry,
    valstorm_token: Optional[str],
    valstorm_base_url: Optional[str],
) -> List[str]:
    """Async discovery of cross-plugin AI tools from Open Valstorm (`GET /v1/ai/tools`)."""
    clean_base = _normalize_base_url(valstorm_base_url)
    if not valstorm_token or not clean_base:
        return []

    cache_key = _token_cache_key(clean_base, valstorm_token)
    now = time.monotonic()
    cached = _REMOTE_TOOLS_CACHE.get(cache_key)
    if cached and (now - cached[0]) < _REMOTE_TOOLS_TTL_SEC:
        return _register_tool_defs(registry, cached[1], clean_base, valstorm_token)

    tools_url = f"{clean_base}/ai/tools"
    auth_hdr = valstorm_token if valstorm_token.startswith("Bearer ") else f"Bearer {valstorm_token}"

    try:
        async with httpx.AsyncClient(timeout=2.5) as client:
            resp = await client.get(tools_url, headers={"Authorization": auth_hdr})
            if resp.status_code != 200:
                return []
            data = resp.json()
            remote_tools = data.get("tools", []) if isinstance(data, dict) else []
            if isinstance(remote_tools, list):
                _REMOTE_TOOLS_CACHE[cache_key] = (now, remote_tools)
                return _register_tool_defs(registry, remote_tools, clean_base, valstorm_token)
    except Exception as exc:
        logger.debug(f"Cross-plugin tool discovery from {tools_url} skipped: {exc}")
    return []


def discover_and_register_remote_plugin_tools_sync(
    registry: ToolRegistry,
    valstorm_token: Optional[str],
    valstorm_base_url: Optional[str],
) -> List[str]:
    """Sync discovery of cross-plugin AI tools for synchronous subagent registry setup."""
    clean_base = _normalize_base_url(valstorm_base_url)
    if not valstorm_token or not clean_base:
        return []

    cache_key = _token_cache_key(clean_base, valstorm_token)
    now = time.monotonic()
    cached = _REMOTE_TOOLS_CACHE.get(cache_key)
    if cached and (now - cached[0]) < _REMOTE_TOOLS_TTL_SEC:
        return _register_tool_defs(registry, cached[1], clean_base, valstorm_token)

    tools_url = f"{clean_base}/ai/tools"
    auth_hdr = valstorm_token if valstorm_token.startswith("Bearer ") else f"Bearer {valstorm_token}"

    try:
        with httpx.Client(timeout=2.5) as client:
            resp = client.get(tools_url, headers={"Authorization": auth_hdr})
            if resp.status_code != 200:
                return []
            data = resp.json()
            remote_tools = data.get("tools", []) if isinstance(data, dict) else []
            if isinstance(remote_tools, list):
                _REMOTE_TOOLS_CACHE[cache_key] = (now, remote_tools)
                return _register_tool_defs(registry, remote_tools, clean_base, valstorm_token)
    except Exception as exc:
        logger.debug(f"Sync cross-plugin tool discovery from {tools_url} skipped: {exc}")
    return []
