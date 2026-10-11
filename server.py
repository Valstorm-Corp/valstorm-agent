"""Local HTTP / SSE Gateway Server for Valstorm Agent Runtime.

Implements standard /v1/runs and /v1/runs/{run_id}/events SSE protocol
matching Valstorm Desktop Electron IPC bridge. Runs on port 8643 by default.
"""

import asyncio
from contextlib import asynccontextmanager
import contextvars
from datetime import datetime, timezone
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import httpx
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

_current_run_event_queue: contextvars.ContextVar[Optional[asyncio.Queue]] = contextvars.ContextVar(
    "current_run_event_queue", default=None
)

def get_current_event_queue() -> Optional[asyncio.Queue]:
    return _current_run_event_queue.get()

def set_current_event_queue(q: Optional[asyncio.Queue]) -> contextvars.Token:
    return _current_run_event_queue.set(q)

from core.context import WorkspaceContextManager, load_profile
from core.keystore import KeyStore
from core.memory import MemoryStore
from core.models import Message, SessionState, StreamEvent, StreamEventType
from core.react import ReActEngine
from core.storage import SessionStore
from core.tools import get_default_registry, ToolRegistry
from core.sandbox import BaseSandbox, HostSandbox, set_current_sandbox
from core.docker_sandbox import DockerSandbox
from core.cloud_sandbox import CloudMicroVMSandbox
from core.sandbox_pool import (
    get_global_sandbox_pool,
    init_global_sandbox_pool,
    shutdown_global_sandbox_pool,
)
from providers.gemini import GeminiProvider
from providers.openai import OpenAIProvider
from providers.anthropic import AnthropicProvider
from tools.developer_tools import register_developer_tools
from tools.memory_tool import register_memory_tools
from tools.valstorm_client import (
    ValstormApiClient,
    resolve_valstorm_auth_context,
    set_current_valstorm_auth,
)
from tools.valstorm_tools import register_valstorm_tools
from tools import register_tier1_tools

# Configure environment variables so Valstorm Agent identifies as Valstorm Agent Runtime
os.environ["AI_AGENT"] = "valstorm-agent"
os.environ["VALSTORM_AGENT"] = "true"
os.environ["VALSTORM_AGENT_PORT"] = os.environ.get("VALSTORM_AGENT_PORT", "8650")
os.environ["VALSTORM_AGENT_MODE"] = os.environ.get("VALSTORM_AGENT_MODE", "host")
os.environ["VALSTORM_DEFAULT_SANDBOX"] = os.environ.get("VALSTORM_DEFAULT_SANDBOX", "host")
os.environ["VALSTORM_MAX_ITERATIONS"] = os.environ.get("VALSTORM_MAX_ITERATIONS", "250")
for _h_key in ["HERMES_AGENT", "HERMES_HOME", "HERMES_SESSION_ID", "HERMES_INTERACTIVE", "HERMES_QUIET", "HERMES_REAL_HOME"]:
    os.environ.pop(_h_key, None)

logger = logging.getLogger("vsagent.server")


def setup_server_logging(
    log_level_str: Optional[str] = None,
    log_file_path: Optional[str] = None,
    mode: Optional[str] = None,
    port: Optional[int] = None,
) -> Path:
    """Configures console and persistent file logging for the vsagent server."""
    level_name = (log_level_str or os.environ.get("VALSTORM_LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    chosen_mode = mode or os.environ.get("VALSTORM_AGENT_MODE", "host")
    chosen_port = port or int(os.environ.get("VALSTORM_AGENT_PORT", 8650))

    if log_file_path:
        log_path = Path(log_file_path).expanduser().resolve()
    else:
        env_file = os.environ.get("VALSTORM_LOG_FILE")
        if env_file:
            log_path = Path(env_file).expanduser().resolve()
        else:
            log_dir = Path.home() / ".valstorm" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"vsagent_{chosen_mode}_{chosen_port}.log"

    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger("vsagent")
    root_logger.setLevel(level)
    root_logger.propagate = False

    # Avoid duplicate handlers
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console stream
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    # File stream
    try:
        fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)
    except Exception as e:
        root_logger.warning(f"Could not initialize file handler at {log_path}: {e}")

    root_logger.info(
        f"Logging initialized. Mode={chosen_mode}, Port={chosen_port}, Level={level_name}, LogFile={log_path}"
    )
    return log_path


# Initialize default logger on module load
setup_server_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages server startup and shutdown lifecycle including the warm sandbox pool."""
    if os.environ.get("VALSTORM_DISABLE_SANDBOX_POOL", "false").lower() != "true":
        try:
            import docker
            client = docker.from_env()
            if client.ping():
                pool_size = int(os.environ.get("VALSTORM_SANDBOX_POOL_SIZE", "2"))
                await init_global_sandbox_pool(target_size=pool_size)
        except Exception:
            pass

    yield

    await shutdown_global_sandbox_pool()


app = FastAPI(
    title="Valstorm Agent Local Gateway",
    description="HTTP / SSE Agent Runtime Server for Valstorm Desktop and CLI integrations",
    version="2.3.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Active in-flight run queues: run_id -> asyncio.Queue
active_run_event_queues: Dict[str, asyncio.Queue] = {}
active_run_tasks: Dict[str, asyncio.Task] = {}


class RunRequestPayload(BaseModel):
    input: str
    images: Optional[List[str]] = None
    run_id: Optional[str] = None
    chat_id: Optional[str] = None
    agent_id: Optional[str] = None
    profile: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    session_id: Optional[str] = None
    instructions: Optional[str] = None
    system_prompt: Optional[str] = None
    max_iterations: Optional[int] = None
    execution_environment: Optional[str] = "host"
    valstorm_token: Optional[str] = None
    valstorm_base_url: Optional[str] = None
    user_context: Optional[Dict[str, Any]] = None
    user_message_id: Optional[str] = None


class RunResponse(BaseModel):
    run_id: str
    status: str = "running"
    session_id: Optional[str] = None


class MemoryRequestPayload(BaseModel):
    content: str
    target: Optional[str] = "auto"


class MemoryRemovePayload(BaseModel):
    old_text: str
    target: Optional[str] = "all"


class OpenAIChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Any]]] = ""
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    model_config = {"extra": "ignore"}


class OpenAIChatCompletionsRequest(BaseModel):
    model: str = "vsagent-developer"
    messages: List[OpenAIChatMessage] = Field(default_factory=list)
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stream_options: Optional[Dict[str, Any]] = None

    model_config = {"extra": "ignore"}


def _extract_text_and_images(content: Union[str, List[Any], None]) -> tuple[str, List[str]]:
    if not content:
        return "", []
    if isinstance(content, str):
        return content, []
    if isinstance(content, list):
        text_parts: List[str] = []
        image_urls: List[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                p_type = part.get("type")
                if p_type == "text":
                    text_parts.append(part.get("text", ""))
                elif p_type == "image_url":
                    img_info = part.get("image_url", {})
                    url = img_info.get("url") if isinstance(img_info, dict) else str(img_info)
                    if url:
                        image_urls.append(url)
                elif "content" in part:
                    text_parts.append(str(part["content"]))
        return "\n".join(text_parts), image_urls
    return str(content), []


def resolve_model_and_provider(
    raw_model: Optional[str],
    raw_provider: Optional[str],
    profile_slug: Optional[str] = None,
) -> tuple[str, str]:
    """Resolves profile names or model strings to valid LLM model strings and providers."""
    prof_cfg = None
    target_slug = profile_slug

    raw = (raw_model or "").strip()
    if not target_slug:
        if raw.startswith("profile:"):
            target_slug = raw.split(":", 1)[1].strip().lower()
        elif raw.startswith("vsagent-"):
            target_slug = raw.split("vsagent-", 1)[1].strip().lower()
        elif raw.lower() in ("chief-of-staff", "orchestrator", "developer", "researcher", "backend-tester", "writer", "valstorm-assistant", "slack-agent", "default", "vsagent"):
            target_slug = "developer" if raw.lower() == "vsagent" else raw.lower()

    if target_slug:
        prof_cfg = load_profile(target_slug)

    model_str = None
    if raw and not raw.startswith("profile:") and not raw.startswith("vsagent-") and raw.lower() not in ("chief-of-staff", "orchestrator", "developer", "researcher", "backend-tester", "writer", "valstorm-assistant", "slack-agent", "default", "vsagent"):
        model_str = raw
    elif prof_cfg and prof_cfg.get("model"):
        model_str = prof_cfg["model"]
    else:
        model_str = "gemini-flash-latest"

    provider_str = (raw_provider or (prof_cfg.get("provider") if prof_cfg else None) or "valstorm").strip().lower()
    if provider_str in ("fallback", "cascade", "chained"):
        provider_str = "fallback"
    elif provider_str in ("valstorm", "hosted", "managed"):
        provider_str = "valstorm"
    elif provider_str in ("digitalocean", "do", "do-deepseek", "do-flash", "do-kimi", "do-oss", "do-llama"):
        provider_str = provider_str
    elif "gemini" in model_str.lower() or "google" in model_str.lower():
        provider_str = "gemini"
    elif "deepseek" in model_str.lower() and provider_str not in ("digitalocean", "do", "do-deepseek"):
        provider_str = "deepseek"
    elif ("kimi" in model_str.lower() or "moonshot" in model_str.lower()) and provider_str not in ("digitalocean", "do", "do-kimi"):
        provider_str = "kimi"
    elif "gpt" in model_str.lower() or "o1" in model_str.lower() or "o3" in model_str.lower() or "openai" in provider_str:
        if provider_str not in ("digitalocean", "do", "do-oss"):
            provider_str = "openai"
    elif "claude" in model_str.lower() or "anthropic" in provider_str:
        if provider_str not in ("digitalocean", "do"):
            provider_str = "anthropic"

    return model_str, provider_str


def _get_provider_instance(
    provider_name: Optional[str],
    model_name: Optional[str],
    profile_slug: Optional[str] = None,
    valstorm_token: Optional[str] = None,
    valstorm_base_url: Optional[str] = None,
):
    keystore = KeyStore()
    model_str, p_name = resolve_model_and_provider(model_name, provider_name, profile_slug=profile_slug)

    # Auto-route uncredentialed clients to Valstorm managed provider:
    # If provider resolved to gemini/google/aistudio but user has no personal Google AI Studio key,
    # route seamlessly to 'valstorm' so Valstorm's AI gateway handles authenticated token metering.
    has_direct_gemini_key = bool(
        keystore.get_api_key("gemini")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if (p_name in ("gemini", "google", "aistudio") or not p_name) and not has_direct_gemini_key:
        p_name = "valstorm"

    prof_cfg = load_profile(profile_slug) if profile_slug else None
    from providers import build_fallback_chain, infer_cascade_tier

    cascade_tier = infer_cascade_tier(model_name=model_str, profile_cfg=prof_cfg)
    clean_valstorm_base = None
    if valstorm_base_url:
        base = valstorm_base_url.rstrip("/")
        if base.endswith("/ai"):
            base = base[:-3].rstrip("/")
        if not base.endswith("/v1") and "/v1" not in base:
            base = f"{base}/v1"
        clean_valstorm_base = f"{base}/ai"

    chain = build_fallback_chain(
        tier=cascade_tier,
        primary_provider=p_name,
        primary_model=model_str,
        keystore=keystore,
        api_key=valstorm_token if (p_name == "valstorm" and valstorm_token) else None,
        base_url=clean_valstorm_base if (p_name == "valstorm" and clean_valstorm_base) else None,
    )
    resolved_provider = p_name
    resolved_model = model_str
    return chain, resolved_model, resolved_provider


def _build_server_tool_registry(
    valstorm_token: Optional[str] = None,
    override_base_url: Optional[str] = None,
    valstorm_env: Optional[str] = None,
    memory_store: Optional[MemoryStore] = None,
    session_store: Optional[SessionStore] = None,
) -> ToolRegistry:
    registry = get_default_registry()
    register_developer_tools(registry)
    register_tier1_tools(registry)
    register_memory_tools(registry, memory_store=memory_store, session_store=session_store)

    token, base_url, refresh_token, auth_file_path = resolve_valstorm_auth_context(
        override_token=valstorm_token,
        override_base_url=override_base_url,
        env=valstorm_env
    )
    if token:
        try:
            from tools.valstorm_platform_client import RemotePlatformContext
            from tools.execute_code import register_execute_code_tools

            client = ValstormApiClient(token=token, base_url=base_url, refresh_token=refresh_token, auth_file_path=auth_file_path)
            platform = RemotePlatformContext(client=client)
            register_valstorm_tools(registry=registry, client=client)
            register_execute_code_tools(registry=registry, client=client, platform=platform)
        except Exception as err:
            logger.warning(f"Failed to register platform context tools: {err}")
    return registry


_MAX_SAFE_INTEGER = 9_007_199_254_740_991


def _safe_token_count(value: object) -> int:
    """Return provider-reported token counts only when safely representable."""
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value if 0 <= value <= _MAX_SAFE_INTEGER else 0


def _optional_non_empty_string(value: object) -> Optional[str]:
    return value if isinstance(value, str) and value.strip() else None


async def _sync_turn_to_valstorm(
    chat_id: Optional[str],
    run_id: str,
    output_text: str,
    status: str,
    input_tokens: int,
    output_tokens: int,
    model: Optional[str],
    provider: Optional[str],
    valstorm_token: Optional[str],
    valstorm_base_url: Optional[str],
    session_id: Optional[str] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    subagents: Optional[List[Dict[str, Any]]] = None,
    error: Optional[str] = None,
    message_id: Optional[str] = None,
    cached_input_tokens: Optional[int] = None,
    user_text: Optional[str] = None,
    user_message_id: Optional[str] = None,
):
    """Directly synchronizes terminal turn execution state, messages, and token telemetry to Valstorm backend.
    
    This ensures that even if UI tabs close, WebSocket connections drop, or the client navigates away,
    the background runtime daemon autonomously persists the assistant message and updates token counters.
    """
    effective_base_url = valstorm_base_url or os.environ.get("VALSTORM_BASE_URL") or os.environ.get("VALSTORM_API_URL")
    if not chat_id or not valstorm_token or not effective_base_url:
        logger.debug(
            f"[{run_id}] Direct Valstorm sync skipped (chat_id={chat_id}, has_token={bool(valstorm_token)}, has_url={bool(effective_base_url)})"
        )
        return

    base = effective_base_url.rstrip("/")
    if not base.endswith("/v1") and "/v1" not in base:
        base = f"{base}/v1"
    endpoint = f"{base}/ai/chat/{chat_id}/desktop-sync"

    payload: Dict[str, Any] = {
        "text": output_text or "",
        "status": status,
        "role": "assistant",
        "session_id": session_id or chat_id,
        "run_id": run_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": model,
        "provider": provider,
        "device_pid": os.getpid(),
    }
    if user_text:
        payload["user_text"] = user_text
    if user_message_id:
        payload["user_message_id"] = user_message_id
    if error:
        payload["error"] = error[:4000]
    if message_id:
        payload["message_id"] = message_id
    if cached_input_tokens:
        payload["cached_input_tokens"] = int(min(cached_input_tokens, input_tokens or cached_input_tokens))
    if tool_calls:
        payload["tool_calls"] = tool_calls
    if subagents:
        payload["subagent_results"] = subagents
        payload["child_syncs"] = subagents

    auth_header = valstorm_token if valstorm_token.startswith("Bearer ") else f"Bearer {valstorm_token}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": auth_header,
    }

    logger.info(f"[{run_id}] Direct runtime daemon syncing to Valstorm: {endpoint} (status={status}, tokens={input_tokens}+{output_tokens})")
    for attempt in range(1, 4):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.post(endpoint, json=payload, headers=headers)
                if res.status_code in (200, 201):
                    logger.info(f"[{run_id}] Autonomous Valstorm sync succeeded on attempt {attempt}")
                    return
                else:
                    logger.warning(
                        f"[{run_id}] Autonomous Valstorm sync attempt {attempt} returned HTTP {res.status_code}: {res.text[:200]}"
                    )
        except Exception as ex:
            logger.warning(f"[{run_id}] Autonomous Valstorm sync attempt {attempt} failed: {ex}")
        if attempt < 3:
            await asyncio.sleep(1.0 * attempt)


async def _hydrate_session_from_cloud_if_needed(
    session: SessionState,
    session_id: str,
    valstorm_token: Optional[str],
    valstorm_base_url: Optional[str],
    session_store: SessionStore,
):
    """Hydrates missing conversation turns (e.g. /help support replies or multi-device messages) from Valstorm cloud."""
    if not session_id or not isinstance(session_id, str) or not valstorm_token or not valstorm_base_url:
        return
    if not (session_id.startswith("aich_") or session_id.startswith("chat_")):
        return

    base = valstorm_base_url.rstrip("/")
    if not base.endswith("/v1") and "/v1" not in base:
        base = f"{base}/v1"
    endpoint = f"{base}/query"

    headers = {
        "Authorization": f"Bearer {valstorm_token}",
        "Content-Type": "application/json",
    }
    sql_query = f"SELECT * FROM ai_chat_message WHERE ai_chat = '{session_id}' ORDER BY created_date ASC"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.post(endpoint, json={"query": sql_query}, headers=headers)
            if res.status_code == 200:
                data = res.json()
                records = data.get("data") if isinstance(data, dict) and "data" in data else data
                if isinstance(records, list) and len(records) > 0:
                    existing_msg_ids = {m.id for m in session.messages if m.id}
                    # Build index of existing message contents by role to prevent duplicate echos
                    existing_contents_by_role: Dict[str, set] = {}
                    for m in session.messages:
                        c_text = (m.content or "").strip()
                        if c_text:
                            existing_contents_by_role.setdefault(m.role, set()).add(c_text)

                    has_new = False
                    for r in records:
                        r_id = r.get("id")
                        body = (r.get("body") or r.get("content") or "").strip()
                        role = r.get("role") or "assistant"
                        if role == "ai":
                            role = "assistant"

                        # 1. Skip if message ID is already present
                        if r_id and r_id in existing_msg_ids:
                            continue

                        # 2. Skip if adjacent last message has identical role and body
                        if session.messages:
                            last_msg = session.messages[-1]
                            if last_msg.role == role and (last_msg.content or "").strip() == body:
                                continue

                        # 3. Skip if non-trivial message content already exists for this role
                        if body and len(body) > 20 and body in existing_contents_by_role.get(role, set()):
                            continue

                        tool_calls = None
                        if r.get("tool_calls"):
                            try:
                                tc_raw = json.loads(r["tool_calls"]) if isinstance(r["tool_calls"], str) else r["tool_calls"]
                                if isinstance(tc_raw, list):
                                    tool_calls = [
                                        ToolCall(
                                            id=tc.get("id") or str(uuid.uuid4()),
                                            name=tc.get("tool_name") or tc.get("name") or "tool",
                                            arguments=tc.get("args") or tc.get("arguments") or {},
                                        )
                                        for tc in tc_raw if isinstance(tc, dict)
                                    ]
                            except Exception:
                                pass

                        msg = Message(
                            id=r_id,
                            role=role,
                            content=body,
                            tool_calls=tool_calls,
                            model=r.get("model"),
                            provider=r.get("provider"),
                        )
                        session.add_message(msg)
                        existing_msg_ids.add(r_id)
                        if body:
                            existing_contents_by_role.setdefault(role, set()).add(body)
                        has_new = True

                    if has_new:
                        session_store.save_session(session)
                        logger.info(f"Hydrated cloud messages into session {session_id}")
    except Exception as ex:
        logger.debug(f"Cloud session hydration skipped for {session_id}: {ex}")


async def _execute_agent_run(
    run_id: str,
    payload: RunRequestPayload,
    queue: asyncio.Queue,
    valstorm_token: Optional[str] = None,
    valstorm_base_url: Optional[str] = None,
):
    """Executes ReActEngine in background and streams SSE events to the queue."""
    full_output = ""
    input_tokens = 0
    output_tokens = 0
    cached_input_tokens = 0
    tool_calls_map: Dict[str, Dict[str, Any]] = {}
    subagents_list: List[Dict[str, Any]] = []

    # Resolve effective chat ID for direct background cloud sync
    effective_chat_id = payload.chat_id
    if not effective_chat_id and payload.session_id and payload.session_id.startswith("aich_"):
        effective_chat_id = payload.session_id

    # Instantiate requested execution sandbox
    sandbox: BaseSandbox
    from_pool = False
    pool = get_global_sandbox_pool()
    default_env = os.environ.get("VALSTORM_DEFAULT_SANDBOX", "host").strip().lower()
    exec_env = (payload.execution_environment or default_env or "host").strip().lower()

    logger.info(
        f"[{run_id}] Run initiated | prompt='{payload.input[:80]}' | env={exec_env} | profile={payload.profile} | model={payload.model} | chat_id={effective_chat_id}"
    )

    t_sb_start = time.perf_counter()
    if exec_env in ("cloud", "e2b", "serverless"):
        try:
            sandbox = CloudMicroVMSandbox()
            await sandbox.start()
        except (ValueError, RuntimeError) as e:
            logger.warning(
                f"[{run_id}] Cloud microVM sandbox unavailable ({e}). Falling back to HostSandbox."
            )
            sandbox = HostSandbox()
            await sandbox.start()
    elif exec_env in ("docker", "container"):
        if pool is not None:
            sandbox = await pool.acquire()
            from_pool = True
        else:
            sandbox = DockerSandbox()
            await sandbox.start()
    else:
        sandbox = HostSandbox()
        await sandbox.start()

    sb_latency_ms = (time.perf_counter() - t_sb_start) * 1000
    logger.info(
        f"[{run_id}] Sandbox ready: type={exec_env} ({sandbox.__class__.__name__}) in {sb_latency_ms:.1f}ms (from_pool={from_pool})"
    )

    token = set_current_sandbox(sandbox)
    auth_tok = set_current_valstorm_auth(valstorm_token, valstorm_base_url)
    q_tok = set_current_event_queue(queue)

    try:
        from tools.delegation import SubagentRegistry
        SubagentRegistry.get_instance().clear()
    except Exception:
        pass

    try:
        session_store = SessionStore()
        memory_store = MemoryStore()

        # Resolve profile configuration: defaults to 'developer' if host sandbox, otherwise 'chief-of-staff'
        profile_slug = payload.profile or payload.agent_id
        raw_m = (payload.model or "").strip()
        if not profile_slug:
            if raw_m.startswith("profile:"):
                profile_slug = raw_m.split(":", 1)[1].strip().lower()
            elif raw_m.lower() in ("chief-of-staff", "orchestrator", "developer", "architect", "slack-agent", "default"):
                profile_slug = raw_m.lower()
            elif exec_env in ("host", "device"):
                profile_slug = "software-engineer"
            else:
                profile_slug = "chief-of-staff"

        profile_cfg = load_profile(profile_slug) if profile_slug else load_profile("software-engineer" if exec_env in ("host", "device") else "chief-of-staff")

        provider_inst, target_model, resolved_provider = _get_provider_instance(
            payload.provider,
            payload.model,
            profile_slug,
            valstorm_token,
            valstorm_base_url,
        )
        full_tools = _build_server_tool_registry(
            valstorm_token=valstorm_token,
            override_base_url=valstorm_base_url,
            memory_store=memory_store,
            session_store=session_store,
        )

        if profile_cfg and profile_cfg.get("allowed_tools") and hasattr(full_tools, "filter_by_whitelist"):
            tools = full_tools.filter_by_whitelist(profile_cfg["allowed_tools"])
        else:
            tools = full_tools

        engine = ReActEngine(provider=provider_inst, tools=tools)

        # Load or init session
        session_id = payload.session_id or effective_chat_id or f"aich_{uuid.uuid4().hex[:16]}"
        if not effective_chat_id and session_id.startswith("aich_"):
            effective_chat_id = session_id

        session = session_store.load_session(session_id)
        if not session:
            session = SessionState(
                session_id=session_id,
                active_model=target_model,
                active_provider=resolved_provider,
            )
            sys_prompt = payload.system_prompt or payload.instructions or WorkspaceContextManager(memory_store=memory_store).build_system_prompt(profile=profile_cfg, user_context=payload.user_context)
            if (payload.system_prompt or payload.instructions) and payload.user_context:
                now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")
                ctx_lines = ["\n\nSystem Context:", f"- Current Date & Time: {now_str}"]
                u_name = payload.user_context.get("user_name")
                u_email = payload.user_context.get("user_email")
                if u_name or u_email:
                    ctx_lines.append(f"- Current User: {u_name or ''} ({u_email or ''})".strip())
                if payload.user_context.get("organization_id"):
                    ctx_lines.append(f"- Organization: {payload.user_context.get('organization_name') or ''} [{payload.user_context.get('organization_id')}]".strip())
                if payload.user_context.get("ui_context"):
                    ctx_lines.append(f"- Active UI Context: {payload.user_context.get('ui_context')}")
                sys_prompt += "\n" + "\n".join(ctx_lines)

            session.add_message(Message(role="system", content=sys_prompt, model=target_model, provider=resolved_provider))

        # Hydrate any missing cloud messages (e.g. from /help, mobile or web) before executing turn
        await _hydrate_session_from_cloud_if_needed(
            session=session,
            session_id=session_id,
            valstorm_token=valstorm_token,
            valstorm_base_url=valstorm_base_url,
            session_store=session_store,
        )

        if effective_chat_id:
            # Metered billing attributes each Valstorm gateway call to this ai_chat (X-Valstorm-Chat-Id)
            session.metadata["valstorm_chat_id"] = effective_chat_id

        effective_input = payload.input
        if effective_input and effective_input.strip().startswith("/memorize"):
            raw_fact = effective_input.strip()[len("/memorize"):].strip()
            if raw_fact:
                lower = raw_fact.lower()
                target_cat = "user" if any(lower.startswith(p) for p in ["i ", "my ", "me ", "user ", "i'm ", "i prefer"]) else "memory"
                mem_res = memory_store.add_fact(target=target_cat, content=raw_fact)
                logger.info(f"[{run_id}] /memorize executed directly: {mem_res}")
                effective_input = (
                    f"[SYSTEM DIRECTIVE: The user executed `/memorize`. The following fact has been successfully committed to persistent {target_cat} memory:\n"
                    f'"{raw_fact}"\n'
                    f"Acknowledge to the user that this fact has been saved to your persistent long-term memory, confirm how you will apply it, and do not call the memory_manage tool since it has already been saved.]"
                )
            else:
                effective_input = (
                    "[SYSTEM DIRECTIVE: The user typed `/memorize` without providing any text. "
                    "Briefly explain how to use `/memorize <fact or rule>` to commit facts to persistent memory, and provide 2-3 helpful examples.]"
                )

        async for event in engine.run_turn_stream(
            session=session,
            user_input=effective_input,
            images=payload.images,
            model=target_model,
            max_iterations=payload.max_iterations,
        ):
            if event.event_type == StreamEventType.TEXT_CHUNK and event.delta:
                if event.metadata.get("failover"):
                    await queue.put({
                        "event": "provider.failover",
                        "run_id": run_id,
                        "failed_tier": event.metadata.get("failed_tier"),
                        "target_tier": event.metadata.get("target_tier"),
                        "target_model": event.metadata.get("target_model"),
                        "reason": event.metadata.get("reason"),
                    })
                    continue
                full_output += event.delta
                await queue.put({
                    "event": "message.delta",
                    "run_id": run_id,
                    "delta": event.delta,
                })

            elif event.event_type == StreamEventType.TOOL_CALL_DETECTED and event.tool_call:
                tc = event.tool_call
                args_preview = json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else str(tc.arguments)
                step_id = f"step_{int(time.time() * 1000)}"
                tool_calls_map[step_id] = {
                    "step_id": step_id,
                    "run_id": run_id,
                    "tool_name": tc.name,
                    "args": args_preview,
                    "status": "running",
                }
                logger.info(f"[{run_id}] Tool call start: {tc.name}({args_preview[:150]})")
                await queue.put({
                    "event": "tool.started",
                    "run_id": run_id,
                    "step_id": step_id,
                    "tool": tc.name,
                    "preview": args_preview,
                    "timestamp": int(time.time() * 1000),
                })

            elif event.event_type == StreamEventType.TOOL_EXECUTION_RESULT and event.tool_result:
                tr = event.tool_result
                last_step = list(tool_calls_map.keys())[-1] if tool_calls_map else f"step_{int(time.time() * 1000)}"
                if last_step in tool_calls_map:
                    tool_calls_map[last_step]["status"] = "failed" if tr.is_error else "completed"
                    tool_calls_map[last_step]["duration"] = tr.duration_sec
                    tool_calls_map[last_step]["result"] = tr.output

                logger.info(
                    f"[{run_id}] Tool call complete: {tr.name} in {tr.duration_sec:.3f}s (error={tr.is_error})"
                )
                await queue.put({
                    "event": "tool.completed",
                    "run_id": run_id,
                    "step_id": last_step,
                    "tool": tr.name,
                    "duration": tr.duration_sec,
                    "error": tr.is_error,
                    "result": tr.output,
                })

            elif event.event_type == StreamEventType.LLM_CALL_COMPLETE:
                # Persist progress after every model call so a crash mid-turn doesn't lose work.
                try:
                    session_store.save_session(session)
                except Exception as save_err:
                    logger.debug(f"[{run_id}] Mid-turn session save failed: {save_err}")

            elif event.event_type == StreamEventType.TURN_COMPLETE and event.message:
                usage = getattr(event.message, "usage", None)
                input_tokens = _safe_token_count(getattr(usage, "prompt_tokens", None))
                output_tokens = _safe_token_count(getattr(usage, "completion_tokens", None))
                cached_input_tokens = _safe_token_count(getattr(usage, "cached_tokens", None))
                logger.info(
                    f"[{run_id}] Turn complete | tokens: {input_tokens}in / {output_tokens}out"
                )
                session_store.save_session(session)

        # Collect all subagent sessions executed during this run
        try:
            from tools.delegation import SubagentRegistry
            subagent_tasks = SubagentRegistry.get_instance().list_tasks()
            for st in subagent_tasks:
                sub_in_tok = _safe_token_count(getattr(getattr(st, "child_session", None), "total_prompt_tokens", 0))
                sub_out_tok = _safe_token_count(getattr(getattr(st, "child_session", None), "total_completion_tokens", 0))
                subagents_list.append({
                    "run_id": st.task_id,
                    "parent_run_id": run_id,
                    "agent_role": st.profile,
                    "role": "assistant",
                    "status": st.status,
                    "text": st.outcome,
                    "summary": st.outcome,
                    "goal": st.goal,
                    "input_tokens": sub_in_tok,
                    "output_tokens": sub_out_tok,
                    "model": st.model_name,
                    "provider": st.provider_name,
                    "tool_calls": [{"name": t, "status": "completed"} for t in st.tools_executed],
                    "duration_sec": getattr(st, "duration_sec", 0),
                })
        except Exception as sub_err:
            logger.warning(f"[{run_id}] Subagent task extraction warning: {sub_err}")

        # Emit completion
        logger.info(
            f"[{run_id}] Run finished successfully | total tokens: {input_tokens + output_tokens}"
        )
        resolved_model_str = _optional_non_empty_string(target_model)
        resolved_prov_str = _optional_non_empty_string(resolved_provider)
        tool_calls_list = list(tool_calls_map.values()) if tool_calls_map else None

        completed_event: Dict[str, Any] = {
            "event": "run.completed",
            "run_id": run_id,
            "output": full_output,
            "status": "completed",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": resolved_model_str,
            "provider": resolved_prov_str,
        }
        if tool_calls_list:
            completed_event["tool_calls"] = tool_calls_list
        if subagents_list:
            completed_event["subagent_results"] = subagents_list

        await queue.put(completed_event)

        # Autonomous direct sync from runtime daemon to Valstorm cloud
        last_asst_msg_id = None
        for m in reversed(session.messages):
            if m.role == "assistant" and m.id:
                last_asst_msg_id = m.id
                break

        await _sync_turn_to_valstorm(
            chat_id=effective_chat_id,
            run_id=run_id,
            output_text=full_output,
            status="completed",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=resolved_model_str,
            provider=resolved_prov_str,
            valstorm_token=valstorm_token,
            valstorm_base_url=valstorm_base_url,
            session_id=session_id,
            tool_calls=tool_calls_list,
            subagents=subagents_list if subagents_list else None,
            message_id=last_asst_msg_id,
            cached_input_tokens=cached_input_tokens or None,
            user_text=payload.input,
            user_message_id=getattr(payload, "user_message_id", None),
        )

        # Smart Context & Memory: Asynchronously extract and reconcile persistent facts (Zero latency overhead)
        try:
            from core.memory_extractor import extract_and_commit_turn_memory
            asyncio.create_task(
                extract_and_commit_turn_memory(
                    user_input=payload.input,
                    assistant_output=full_output,
                    memory_store=memory_store,
                    provider=provider_inst,
                    model=target_model,
                )
            )
        except Exception as mem_err:
            logger.debug(f"[{run_id}] Smart context extraction launch skipped: {mem_err}")

    except asyncio.CancelledError:
        logger.warning(f"[{run_id}] Run was CANCELLED by user/client")
        cancel_card = "\n\n\033[93m⚠️ [Agent Run Cancelled by User]\033[0m\n"
        full_output += cancel_card
        await queue.put({
            "event": "message.delta",
            "run_id": run_id,
            "delta": cancel_card,
        })
        await queue.put({
            "event": "run.cancelled",
            "run_id": run_id,
            "status": "cancelled",
            "output": full_output,
        })
        await _sync_turn_to_valstorm(
            chat_id=effective_chat_id,
            run_id=run_id,
            output_text=full_output,
            status="cancelled",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=_optional_non_empty_string(target_model),
            provider=_optional_non_empty_string(resolved_provider),
            valstorm_token=valstorm_token,
            valstorm_base_url=valstorm_base_url,
            session_id=payload.session_id or effective_chat_id,
            user_text=payload.input,
            user_message_id=getattr(payload, "user_message_id", None),
        )
    except Exception as e:
        logger.error(f"[{run_id}] Run failed with exception: {e}", exc_info=True)
        err_card = f"\n\n\033[91m❌ [Agent Run Terminated]: {type(e).__name__}: {str(e)}\033[0m\n"
        full_output += err_card
        await queue.put({
            "event": "message.delta",
            "run_id": run_id,
            "delta": err_card,
        })
        await queue.put({
            "event": "run.failed",
            "run_id": run_id,
            "error": str(e),
        })
        await _sync_turn_to_valstorm(
            chat_id=effective_chat_id,
            run_id=run_id,
            output_text=full_output,
            status="error",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=_optional_non_empty_string(target_model),
            provider=_optional_non_empty_string(resolved_provider),
            valstorm_token=valstorm_token,
            valstorm_base_url=valstorm_base_url,
            session_id=payload.session_id or effective_chat_id,
            error=str(e),
            user_text=payload.input,
            user_message_id=getattr(payload, "user_message_id", None),
        )
    finally:
        try:
            if from_pool and pool is not None:
                await pool.release(sandbox)
            else:
                await sandbox.close()
        except Exception:
            pass
        set_current_sandbox(None)
        await queue.put(None)  # Sentinel to close SSE stream


def _resolve_active_run_id(target_id: str) -> Optional[str]:
    """Resolves an in-flight run ID with exact, prefix, or timestamp-prefix tolerance."""
    if not target_id:
        return None
    if target_id in active_run_event_queues:
        return target_id

    # Prefix or timestamp match (e.g. run_1787177896434 matching run_1787177896438_e21f71)
    parts = target_id.split("_")
    ts_prefix = parts[1][:10] if len(parts) > 1 and len(parts[1]) >= 10 else None

    for r_id in list(active_run_event_queues.keys()):
        if r_id.startswith(target_id) or target_id.startswith(r_id):
            return r_id
        if ts_prefix and ts_prefix in r_id:
            return r_id

    # If only 1 run active, return it as single-session fallback
    if len(active_run_event_queues) == 1:
        return next(iter(active_run_event_queues.keys()))

    return None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from core.native_bridge import is_native_available
    return {
        "status": "ok",
        "engine": "valstorm-agent",
        "port": int(os.environ.get("VALSTORM_AGENT_PORT", "8650")),
        "mode": os.environ.get("VALSTORM_AGENT_MODE", "host"),
        "default_sandbox": os.environ.get("VALSTORM_DEFAULT_SANDBOX", "host"),
        "native_acceleration": is_native_available(),
    }


@app.get("/v1/memories")
async def get_memories_endpoint(target: str = "all"):
    """Returns persistent declarative memory facts."""
    mem_store = MemoryStore()
    return mem_store.get_facts(target=target)


@app.post("/v1/memories")
async def add_memory_endpoint(payload: MemoryRequestPayload):
    """Directly commits a fact to agent persistent memory."""
    mem_store = MemoryStore()
    content = payload.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Content cannot be empty.")

    target = (payload.target or "auto").strip().lower()
    if target == "auto":
        lower = content.lower()
        if any(lower.startswith(p) for p in ["i ", "my ", "me ", "user ", "i'm ", "i prefer"]):
            target = "user"
        else:
            target = "memory"

    result = mem_store.add_fact(target=target, content=content)
    return {
        "status": "success",
        "target": target,
        "content": content,
        "message": result,
    }


@app.delete("/v1/memories")
async def remove_memory_endpoint(payload: MemoryRemovePayload):
    """Removes a declarative memory fact containing old_text."""
    mem_store = MemoryStore()
    target = payload.target or "all"
    removed = mem_store.remove_fact(target=target, old_text=payload.old_text)
    return {
        "status": "success",
        "removed": removed,
        "old_text": payload.old_text,
    }


@app.post("/v1/runs", response_model=RunResponse)
async def create_run(
    payload: RunRequestPayload,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_valstorm_token: Optional[str] = Header(None, alias="x-valstorm-token"),
    x_valstorm_api_url: Optional[str] = Header(None, alias="x-valstorm-api-url"),
):
    """Creates a new agent execution run and queues it for SSE streaming."""
    run_id = (payload.run_id and payload.run_id.strip()) or f"run_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
    queue = asyncio.Queue()
    active_run_event_queues[run_id] = queue

    # Resolve Valstorm platform token & API URL
    v_token = x_valstorm_token or getattr(payload, "valstorm_token", None)
    if not v_token and payload.user_context and isinstance(payload.user_context, dict):
        v_token = payload.user_context.get("access_token")
    if not v_token and authorization and authorization.startswith("Bearer "):
        v_token = authorization.split("Bearer ", 1)[1].strip()

    v_base_url = x_valstorm_api_url or getattr(payload, "valstorm_base_url", None)
    if not v_base_url and payload.user_context and isinstance(payload.user_context, dict):
        v_base_url = payload.user_context.get("sync_url") or payload.user_context.get("api_url")
    if not v_base_url:
        v_base_url = os.environ.get("VALSTORM_BASE_URL") or os.environ.get("VALSTORM_API_URL")

    task = asyncio.create_task(
        _execute_agent_run(
            run_id,
            payload,
            queue,
            valstorm_token=v_token,
            valstorm_base_url=v_base_url,
        )
    )
    active_run_tasks[run_id] = task

    return RunResponse(run_id=run_id, status="running", session_id=payload.session_id)


@app.get("/v1/runs/{run_id}/events")
async def stream_run_events(run_id: str):
    """Server-Sent Events (SSE) stream for real-time typewriter chunks and tool badges."""
    resolved_id = _resolve_active_run_id(run_id)
    if not resolved_id:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found or already closed.")

    queue = active_run_event_queues.get(resolved_id)
    if not queue:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found or already closed.")

    async def sse_generator():
        try:
            while True:
                item = await queue.get()
                if item is None:
                    # Stream completed
                    yield f"data: [DONE]\n\n"
                    break
                data_str = json.dumps(item)
                yield f"data: {data_str}\n\n"
        finally:
            active_run_event_queues.pop(resolved_id, None)
            active_run_tasks.pop(resolved_id, None)

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/v1/runs/{run_id}")
async def cancel_run(run_id: str):
    """Cancels an in-flight agent run."""
    resolved_id = _resolve_active_run_id(run_id)

    task = active_run_tasks.get(resolved_id) if resolved_id else None
    queue = active_run_event_queues.get(resolved_id) if resolved_id else None

    if not task and not queue:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found or already completed.")

    if task and not task.done():
        task.cancel()

    return {"status": "cancelling", "run_id": resolved_id or run_id}


@app.get("/v1/models")
@app.get("/models")
async def list_models_endpoint():
    """OpenAI-compatible models discovery endpoint listing available agent profiles."""
    from core.context import list_available_profiles

    models_data = [
        {
            "id": "vsagent",
            "object": "model",
            "created": 1700000000,
            "owned_by": "valstorm",
            "permission": [],
            "root": "vsagent",
            "parent": None,
        }
    ]

    try:
        profiles = list_available_profiles()
        for prof in profiles:
            slug = prof.get("api_name") or prof.get("slug")
            if slug:
                models_data.append({
                    "id": f"vsagent-{slug}",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "valstorm",
                    "permission": [],
                    "root": f"vsagent-{slug}",
                    "parent": None,
                })
    except Exception as ex:
        logger.warning(f"Error discovering profiles for /v1/models: {ex}")
        for fallback_slug in ["developer", "architect", "researcher", "writer", "backend-tester"]:
            models_data.append({
                "id": f"vsagent-{fallback_slug}",
                "object": "model",
                "created": 1700000000,
                "owned_by": "valstorm",
                "permission": [],
                "root": f"vsagent-{fallback_slug}",
                "parent": None,
            })

    return {
        "object": "list",
        "data": models_data,
    }


@app.get("/v1/models/{model_id}")
@app.get("/models/{model_id}")
async def get_model_endpoint(model_id: str):
    """OpenAI-compatible model detail endpoint."""
    return {
        "id": model_id,
        "object": "model",
        "created": 1700000000,
        "owned_by": "valstorm",
        "permission": [],
        "root": model_id,
        "parent": None,
    }


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions_endpoint(
    payload: OpenAIChatCompletionsRequest,
    authorization: Optional[str] = Header(None),
    x_valstorm_token: Optional[str] = Header(None, alias="x-valstorm-token"),
    x_valstorm_api_url: Optional[str] = Header(None, alias="x-valstorm-api-url"),
):
    """OpenAI-compatible chat completions endpoint for Zed, IDEs, and HTTP clients."""
    system_parts: List[str] = []
    chat_history: List[tuple[str, str]] = []
    last_user_idx = -1

    for idx in range(len(payload.messages) - 1, -1, -1):
        if payload.messages[idx].role.lower() == "user":
            last_user_idx = idx
            break

    if last_user_idx == -1 and payload.messages:
        last_user_idx = len(payload.messages) - 1

    user_input = ""
    images: List[str] = []

    for idx, m in enumerate(payload.messages):
        role = m.role.lower()
        text, imgs = _extract_text_and_images(m.content)
        if role == "system":
            if text.strip():
                system_parts.append(text.strip())
        elif idx == last_user_idx:
            user_input = text
            images.extend(imgs)
        else:
            if role in ("user", "assistant", "model", "tool"):
                chat_history.append(("assistant" if role == "model" else role, text))

    if not user_input.strip():
        user_input = "Hello"

    raw_m = (payload.model or "vsagent-developer").strip()
    profile_slug = "developer"
    if raw_m.startswith("vsagent-"):
        profile_slug = raw_m[len("vsagent-"):].strip().lower()
    elif raw_m.startswith("profile:"):
        profile_slug = raw_m[len("profile:"):].strip().lower()
    elif raw_m.lower() in ("chief-of-staff", "orchestrator", "developer", "architect", "slack-agent", "writer", "backend-tester", "researcher", "marketer"):
        profile_slug = raw_m.lower()
    elif raw_m.lower() == "vsagent":
        profile_slug = "developer"

    profile_cfg = load_profile(profile_slug)
    v_token = x_valstorm_token
    if not v_token and authorization and authorization.startswith("Bearer "):
        bearer_val = authorization.split("Bearer ", 1)[1].strip()
        if bearer_val and bearer_val.lower() not in ("dummy", "none", "null", "undefined", "test", "zed"):
            v_token = bearer_val

    provider_inst, target_model, resolved_provider = _get_provider_instance(
        None, None, profile_slug,
        v_token,
        x_valstorm_api_url,
    )

    sandbox: BaseSandbox = HostSandbox()
    await sandbox.start()
    set_current_sandbox(sandbox)

    set_current_valstorm_auth(v_token, x_valstorm_api_url)

    session_store = SessionStore()
    memory_store = MemoryStore()

    full_tools = _build_server_tool_registry(
        valstorm_token=v_token,
        override_base_url=x_valstorm_api_url,
        memory_store=memory_store,
        session_store=session_store,
    )
    if profile_cfg and profile_cfg.get("allowed_tools") and hasattr(full_tools, "filter_by_whitelist"):
        tools = full_tools.filter_by_whitelist(profile_cfg["allowed_tools"])
    else:
        tools = full_tools

    engine = ReActEngine(provider=provider_inst, tools=tools)

    session_id = f"aich_{uuid.uuid4().hex[:16]}"
    session = SessionState(
        session_id=session_id,
        active_model=target_model,
        active_provider=resolved_provider,
    )

    base_sys_prompt = WorkspaceContextManager(memory_store=memory_store).build_system_prompt(profile=profile_cfg)
    if system_parts:
        base_sys_prompt += "\n\nClient Editor Context & Guidelines:\n" + "\n\n".join(system_parts)

    session.add_message(Message(role="system", content=base_sys_prompt, model=target_model, provider=resolved_provider))

    for h_role, h_text in chat_history:
        if h_text.strip():
            session.add_message(Message(role=h_role, content=h_text, model=target_model, provider=resolved_provider))

    created_ts = int(time.time())
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    model_name = payload.model

    if payload.stream:
        async def sse_chat_generator():
            input_tokens = 0
            output_tokens = 0
            try:
                async for event in engine.run_turn_stream(
                    session=session,
                    user_input=user_input,
                    images=images or None,
                    model=target_model,
                ):
                    if event.event_type == StreamEventType.TEXT_CHUNK and event.delta:
                        if event.metadata.get("failover"):
                            continue
                        chunk_data = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": event.delta},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"

                    elif event.event_type == StreamEventType.TURN_COMPLETE and event.message:
                        usage = getattr(event.message, "usage", None)
                        if usage:
                            input_tokens = _safe_token_count(getattr(usage, "prompt_tokens", 0))
                            output_tokens = _safe_token_count(getattr(usage, "completion_tokens", 0))

                stop_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                if payload.stream_options and payload.stream_options.get("include_usage"):
                    stop_chunk["usage"] = {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    }
                yield f"data: {json.dumps(stop_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as ex:
                logger.error(f"Error in chat completion streaming: {ex}", exc_info=True)
                err_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": f"\n\n⚠️ [vsagent error: {str(ex)}]\n"},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(err_chunk)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            sse_chat_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming response
    full_content = ""
    input_tokens = 0
    output_tokens = 0

    async for event in engine.run_turn_stream(
        session=session,
        user_input=user_input,
        images=images or None,
        model=target_model,
    ):
        if event.event_type == StreamEventType.TEXT_CHUNK and event.delta:
            if not event.metadata.get("failover"):
                full_content += event.delta
        elif event.event_type == StreamEventType.TURN_COMPLETE and event.message:
            usage = getattr(event.message, "usage", None)
            if usage:
                input_tokens = _safe_token_count(getattr(usage, "prompt_tokens", 0))
                output_tokens = _safe_token_count(getattr(usage, "completion_tokens", 0))

    return {
        "id": chunk_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


@app.get("/api/tags")
async def ollama_tags_endpoint():
    """Ollama-compatible model discovery endpoint."""
    from core.context import list_available_profiles

    models_list = []
    try:
        profiles = list_available_profiles()
        for prof in profiles:
            slug = prof.get("api_name") or prof.get("slug")
            if slug:
                models_list.append({
                    "name": f"vsagent-{slug}",
                    "model": f"vsagent-{slug}",
                    "modified_at": datetime.now(timezone.utc).isoformat(),
                    "size": 0,
                    "digest": f"vsagent-{slug}",
                    "details": {
                        "format": "gguf",
                        "family": "vsagent",
                        "families": ["vsagent"],
                        "parameter_size": "latest",
                        "quantization_level": "none",
                    },
                })
    except Exception as ex:
        logger.warning(f"Error discovering profiles for /api/tags: {ex}")

    if not models_list:
        for fallback_slug in ["developer", "architect", "researcher"]:
            models_list.append({
                "name": f"vsagent-{fallback_slug}",
                "model": f"vsagent-{fallback_slug}",
                "modified_at": datetime.now(timezone.utc).isoformat(),
                "size": 0,
                "digest": f"vsagent-{fallback_slug}",
                "details": {
                    "format": "gguf",
                    "family": "vsagent",
                    "families": ["vsagent"],
                    "parameter_size": "latest",
                    "quantization_level": "none",
                },
            })

    return {"models": models_list}


@app.get("/api/version")
async def ollama_version_endpoint():
    """Ollama-compatible version endpoint."""
    return {"version": "0.5.7"}


@app.post("/api/show")
async def ollama_show_endpoint(request: Request):
    """Ollama-compatible model details endpoint."""
    return {
        "license": "",
        "modelfile": "",
        "parameters": "",
        "template": "",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "vsagent",
            "families": ["vsagent"],
            "parameter_size": "latest",
            "quantization_level": "none",
        },
    }


@app.post("/api/chat")
async def ollama_chat_endpoint(request: Request):
    """Ollama-compatible chat endpoint streaming NDJSON chunks."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    model_name = body.get("model", "vsagent-developer")
    messages = body.get("messages", [])
    stream = body.get("stream", True)

    openai_messages = []
    for m in messages:
        openai_messages.append(
            OpenAIChatMessage(
                role=m.get("role", "user"),
                content=m.get("content", ""),
            )
        )

    openai_payload = OpenAIChatCompletionsRequest(
        model=model_name,
        messages=openai_messages,
        stream=stream,
    )

    if stream:
        async def ndjson_generator():
            created_at = datetime.now(timezone.utc).isoformat()
            system_parts: List[str] = []
            chat_history: List[tuple[str, str]] = []
            last_user_idx = -1

            for idx in range(len(openai_payload.messages) - 1, -1, -1):
                if openai_payload.messages[idx].role.lower() == "user":
                    last_user_idx = idx
                    break

            if last_user_idx == -1 and openai_payload.messages:
                last_user_idx = len(openai_payload.messages) - 1

            user_input = ""
            for idx, m in enumerate(openai_payload.messages):
                role = m.role.lower()
                text, _ = _extract_text_and_images(m.content)
                if role == "system":
                    if text.strip():
                        system_parts.append(text.strip())
                elif idx == last_user_idx:
                    user_input = text
                else:
                    if role in ("user", "assistant", "model", "tool"):
                        chat_history.append(("assistant" if role == "model" else role, text))

            if not user_input.strip():
                user_input = "Hello"

            raw_m = model_name.strip()
            profile_slug = "developer"
            if raw_m.startswith("vsagent-"):
                profile_slug = raw_m[len("vsagent-"):].strip().lower()
            elif raw_m.startswith("profile:"):
                profile_slug = raw_m[len("profile:"):].strip().lower()
            elif raw_m.lower() in ("chief-of-staff", "orchestrator", "developer", "architect", "slack-agent", "writer", "backend-tester", "researcher", "marketer"):
                profile_slug = raw_m.lower()

            profile_cfg = load_profile(profile_slug)
            provider_inst, target_model, resolved_provider = _get_provider_instance(None, None, profile_slug)

            sandbox: BaseSandbox = HostSandbox()
            await sandbox.start()
            set_current_sandbox(sandbox)

            session_store = SessionStore()
            memory_store = MemoryStore()

            full_tools = _build_server_tool_registry(
                memory_store=memory_store,
                session_store=session_store,
            )
            if profile_cfg and profile_cfg.get("allowed_tools") and hasattr(full_tools, "filter_by_whitelist"):
                tools = full_tools.filter_by_whitelist(profile_cfg["allowed_tools"])
            else:
                tools = full_tools

            engine = ReActEngine(provider=provider_inst, tools=tools)

            session = SessionState(
                session_id=f"aich_{uuid.uuid4().hex[:16]}",
                active_model=target_model,
                active_provider=resolved_provider,
            )

            base_sys_prompt = WorkspaceContextManager(memory_store=memory_store).build_system_prompt(profile=profile_cfg)
            if system_parts:
                base_sys_prompt += "\n\nClient Editor Context & Guidelines:\n" + "\n\n".join(system_parts)

            session.add_message(Message(role="system", content=base_sys_prompt, model=target_model, provider=resolved_provider))

            for h_role, h_text in chat_history:
                if h_text.strip():
                    session.add_message(Message(role=h_role, content=h_text, model=target_model, provider=resolved_provider))

            input_tokens = 0
            output_tokens = 0

            try:
                async for event in engine.run_turn_stream(
                    session=session,
                    user_input=user_input,
                    model=target_model,
                ):
                    if event.event_type == StreamEventType.TEXT_CHUNK and event.delta:
                        if event.metadata.get("failover"):
                            continue
                        chunk_obj = {
                            "model": model_name,
                            "created_at": created_at,
                            "message": {
                                "role": "assistant",
                                "content": event.delta,
                            },
                            "done": False,
                        }
                        yield json.dumps(chunk_obj) + "\n"

                    elif event.event_type == StreamEventType.TURN_COMPLETE and event.message:
                        usage = getattr(event.message, "usage", None)
                        if usage:
                            input_tokens = _safe_token_count(getattr(usage, "prompt_tokens", 0))
                            output_tokens = _safe_token_count(getattr(usage, "completion_tokens", 0))

                done_obj = {
                    "model": model_name,
                    "created_at": created_at,
                    "message": {
                        "role": "assistant",
                        "content": "",
                    },
                    "done": True,
                    "total_duration": 1_000_000_000,
                    "load_duration": 1_000_000,
                    "prompt_eval_count": input_tokens,
                    "eval_count": output_tokens,
                }
                yield json.dumps(done_obj) + "\n"

            except Exception as ex:
                logger.error(f"Error in Ollama chat streaming: {ex}", exc_info=True)
                err_obj = {
                    "model": model_name,
                    "created_at": created_at,
                    "message": {
                        "role": "assistant",
                        "content": f"\n\n⚠️ [vsagent error: {str(ex)}]\n",
                    },
                    "done": True,
                }
                yield json.dumps(err_obj) + "\n"

        return StreamingResponse(
            ndjson_generator(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # Non-streaming
    resp = await chat_completions_endpoint(openai_payload)
    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = resp.get("usage", {})
    return {
        "model": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "message": {
            "role": "assistant",
            "content": content,
        },
        "done": True,
        "total_duration": 1_000_000_000,
        "prompt_eval_count": usage.get("prompt_tokens", 0),
        "eval_count": usage.get("completion_tokens", 0),
    }


def _save_active_port(port: int):
    """Saves active gateway port to ~/.valstorm/agent_gateway.json."""
    try:
        p = Path.home() / ".valstorm" / "agent_gateway.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"port": port, "engine": "valstorm-agent", "updated_at": int(time.time())}, f, indent=2)
    except Exception:
        pass


def start_server(
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    mode: Optional[str] = None,
    default_sandbox: Optional[str] = None,
):
    """Entrypoint to run gateway server with uvicorn."""
    import argparse
    import socket

    parser = argparse.ArgumentParser(description="Valstorm Agent Local Gateway Server")
    parser.add_argument("--port", type=int, default=None, help="Port to bind (default: 8650 for host, 8660 for cloud)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--mode", type=str, default=None, choices=["host", "cloud", "docker"], help="Runtime execution mode")
    parser.add_argument("--default-sandbox", type=str, default=None, choices=["host", "docker", "cloud"], help="Default sandbox type")
    args, _ = parser.parse_known_args()

    chosen_mode = (args.mode or mode or os.environ.get("VALSTORM_AGENT_MODE", "host")).strip().lower()
    os.environ["VALSTORM_AGENT_MODE"] = chosen_mode

    chosen_sandbox = (args.default_sandbox or default_sandbox or os.environ.get("VALSTORM_DEFAULT_SANDBOX", "cloud" if chosen_mode == "cloud" else "host")).strip().lower()
    os.environ["VALSTORM_DEFAULT_SANDBOX"] = chosen_sandbox

    default_port_for_mode = 8660 if chosen_mode == "cloud" else 8650
    env_port = os.environ.get("VALSTORM_AGENT_PORT")
    chosen_port = args.port or port or (int(env_port) if env_port else default_port_for_mode)
    chosen_host = args.host or host

    # Fallback to next port if busy
    for p in range(chosen_port, chosen_port + 10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((chosen_host, p)) != 0:
                chosen_port = p
                break

    os.environ["VALSTORM_AGENT_PORT"] = str(chosen_port)
    _save_active_port(chosen_port)

    print(f"\n⚡ Starting Valstorm Agent Gateway on http://{chosen_host}:{chosen_port} (Mode: {chosen_mode}, Default Sandbox: {chosen_sandbox})")
    print(f"   • /v1/runs (POST)")
    print(f"   • /v1/runs/{{id}}/events (GET SSE)")
    print(f"   • /health (GET)\n")
    uvicorn.run(app, host=chosen_host, port=chosen_port, log_level="info")


if __name__ == "__main__":
    start_server()
