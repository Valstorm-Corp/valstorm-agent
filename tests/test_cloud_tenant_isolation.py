"""TDD Suite: Cloud Multi-Tenant Isolation for Valstorm Agent Runtime (vsagent).

Verifies strict tenant isolation across shared cloud pods:
1. SessionStore & session_search FTS5 isolation per (org_id, user_id) in Cloud mode.
2. ContextVar-scoped SubagentRegistry preventing cross-tenant clear/bleed under concurrent runs.
3. Strict _resolve_active_run_id (no single-run or timestamp-prefix hijack in Cloud mode) + token ownership checks.
4. Auth-scoped /v1/memories, /v1/chat/completions, and /api/chat endpoints in Cloud mode.
5. Cloud credential fallback lock (no pod-level ~/.valstorm/auth*.json or VALSTORM_TOKEN leakage) + PAT org decoding.
6. Non-colliding SHA-256 fallback cache keys in _TENANT_SKILLS_CACHE and _TENANT_PROFILES_CACHE.
7. K8s manifests explicitly declaring VALSTORM_AGENT_MODE=cloud.
"""

import asyncio
import base64
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

import server
from core.context import _TENANT_PROFILES_CACHE, _get_cached_cloud_profiles, invalidate_tenant_profiles_cache
from core.memory import MemoryStore
from core.models import Message, SessionState, StreamEvent, StreamEventType, UsageMetadata
from core.storage import SessionStore
from tools.delegation import SubagentRegistry, SubagentTaskSession
from tools.memory_tool import create_memory_tools
from tools.skill_tool import _TENANT_SKILLS_CACHE, _get_cached_cloud_skills, invalidate_tenant_skills_cache
from tools.valstorm_client import (
    get_current_tenant_context,
    resolve_valstorm_auth_context,
    set_current_valstorm_auth,
)


def _make_jwt(org_id: str | None = None, user_id: str | None = None) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).decode().rstrip("=")
    claims = {"exp": 4102444800}
    if org_id is not None:
        claims["organization_id"] = org_id
    if user_id is not None:
        claims["user_id"] = user_id
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"{header}.{payload}.sig"


def _make_pat(org_id: str, secret: str = "secret123456") -> str:
    encoded_org = base64.urlsafe_b64encode(org_id.encode()).decode().rstrip("=")
    return f"pat_{encoded_org}_{secret}"


def _sse_events(response_text: str):
    return [
        json.loads(line.removeprefix("data: "))
        for line in response_text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


# ---------------------------------------------------------------------------
# 1. SessionStore & session_search Multi-Tenant Isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_store_and_search_isolated_per_tenant_in_cloud_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VALSTORM_AGENT_MODE", "cloud")

    # Tenant A saves a session with confidential content
    set_current_valstorm_auth(
        _make_jwt("org_alpha", "user_a1"),
        "https://api.valstorm.com/v1",
        is_cloud_mode=True,
    )
    try:
        store_a = SessionStore()
        assert "tenants" in str(store_a.db_path)
        assert "org_alpha" in str(store_a.db_path)
        assert "user_a1" in str(store_a.db_path)

        sess_a = SessionState(session_id="aich_alpha_001", active_model="gemini-flash", active_provider="valstorm")
        sess_a.add_message(Message(role="user", content="Project Apollo acquisition price is $42M"))
        store_a.save_session(sess_a)
        assert len(store_a.search_sessions("Apollo")) == 1
    finally:
        set_current_valstorm_auth(None, None)

    # Tenant B on the same pod MUST NOT see Tenant A's sessions or FTS5 results
    set_current_valstorm_auth(
        _make_jwt("org_beta", "user_b1"),
        "https://api.valstorm.com/v1",
        is_cloud_mode=True,
    )
    try:
        store_b = SessionStore()
        assert "org_beta" in str(store_b.db_path)
        assert store_b.load_session("aich_alpha_001") is None
        assert store_b.list_sessions() == []
        assert store_b.search_sessions("Apollo") == []

        tools_b = {t.__name__: t for t in create_memory_tools(session_store=store_b)}
        search_out = await tools_b["session_search"](query="Apollo", limit=5)
        assert "No past sessions found" in search_out
        assert "42M" not in search_out
    finally:
        set_current_valstorm_auth(None, None)

    # Same org, different user MUST ALSO be isolated
    set_current_valstorm_auth(
        _make_jwt("org_alpha", "user_a2"),
        "https://api.valstorm.com/v1",
        is_cloud_mode=True,
    )
    try:
        store_a2 = SessionStore()
        assert "user_a2" in str(store_a2.db_path)
        assert store_a2.load_session("aich_alpha_001") is None
        assert store_a2.search_sessions("Apollo") == []
    finally:
        set_current_valstorm_auth(None, None)


def test_default_sandbox_cloud_enables_tenant_isolation_for_storage_and_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("VALSTORM_AGENT_MODE", raising=False)
    monkeypatch.setenv("VALSTORM_DEFAULT_SANDBOX", "cloud")

    set_current_valstorm_auth(_make_jwt("org_sandbox", "user_s1"), "https://api.valstorm.com/v1")
    try:
        sess_store = SessionStore()
        mem_store = MemoryStore()
        assert "tenants/org_sandbox/user_s1/agent_state.db" in str(sess_store.db_path)
        assert "tenants/org_sandbox/user_s1/memories.json" in str(mem_store.file_path)
    finally:
        set_current_valstorm_auth(None, None)


def test_session_store_host_mode_uses_default_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VALSTORM_AGENT_MODE", "host")
    monkeypatch.setenv("VALSTORM_DEFAULT_SANDBOX", "host")
    set_current_valstorm_auth(None, None)

    store = SessionStore()
    assert store.db_path == (tmp_path / ".valstorm" / "agent_state.db").resolve()


# ---------------------------------------------------------------------------
# 2. Concurrent SubagentRegistry Isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_runs_have_isolated_subagent_registries(tmp_path, monkeypatch):
    """Two concurrent tenant runs must never clear or read each other's SubagentRegistry tasks."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VALSTORM_AGENT_MODE", "cloud")

    a_registered = asyncio.Event()
    b_started = asyncio.Event()
    b_registered = asyncio.Event()

    sync_calls = []

    async def fake_sync(**kwargs):
        sync_calls.append(kwargs)

    async def fake_run_turn_stream(self, **kwargs):
        ctx = get_current_tenant_context()
        assert ctx is not None
        org = ctx.org_id

        child_sess = SessionState(
            session_id=f"aich_sub_{org}",
            active_model="gemini-pro-latest",
            active_provider="gemini",
        )
        child_sess.add_message(
            Message(
                role="assistant",
                content=f"Secret subagent output for {org}",
                usage=UsageMetadata(prompt_tokens=100, completion_tokens=50),
            )
        )
        task_sess = SubagentTaskSession(
            task_id=f"subtask_{org}",
            child_session_id=f"aich_sub_{org}",
            profile="developer",
            profile_name="Developer",
            goal=f"Confidential goal for {org}",
            context=None,
            model_name="gemini-pro-latest",
            provider_name="gemini",
            scoped_tools=[],
            child_session=child_sess,
        )
        task_sess.status = "completed"
        task_sess.outcome = f"Outcome for {org}"

        if org == "org_tenant_A":
            SubagentRegistry.get_instance().register(task_sess)
            a_registered.set()
            # Wait until Tenant B has started its run and registered its own subagent
            await asyncio.wait_for(b_registered.wait(), timeout=5.0)
        else:
            b_started.set()
            await asyncio.wait_for(a_registered.wait(), timeout=5.0)
            SubagentRegistry.get_instance().register(task_sess)
            b_registered.set()

        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta=f"Done {org}")
        yield StreamEvent(
            event_type=StreamEventType.TURN_COMPLETE,
            message=Message(
                role="assistant",
                content=f"Done {org}",
                usage=UsageMetadata(prompt_tokens=10, completion_tokens=5),
            ),
        )

    monkeypatch.setattr(server, "_get_provider_instance", lambda *_: (object(), "gemini-flash-latest", "gemini"))
    monkeypatch.setattr(server, "_build_server_tool_registry", lambda **_: object())
    monkeypatch.setattr(server, "_sync_turn_to_valstorm", fake_sync)
    monkeypatch.setattr(server.ReActEngine, "run_turn_stream", fake_run_turn_stream)

    jwt_a = _make_jwt("org_tenant_A", "user_A1")
    jwt_b = _make_jwt("org_tenant_B", "user_B1")

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res_a, res_b = await asyncio.gather(
            ac.post(
                "/v1/runs",
                headers={"Authorization": f"Bearer {jwt_a}"},
                json={"input": "Run A", "session_id": "aich_run_A", "sandbox_type": "cloud"},
            ),
            ac.post(
                "/v1/runs",
                headers={"Authorization": f"Bearer {jwt_b}"},
                json={"input": "Run B", "session_id": "aich_run_B", "sandbox_type": "cloud"},
            ),
        )
        assert res_a.status_code == 200
        assert res_b.status_code == 200
        run_a = res_a.json()["run_id"]
        run_b = res_b.json()["run_id"]

        ev_a_resp, ev_b_resp = await asyncio.gather(
            ac.get(f"/v1/runs/{run_a}/events", headers={"Authorization": f"Bearer {jwt_a}"}),
            ac.get(f"/v1/runs/{run_b}/events", headers={"Authorization": f"Bearer {jwt_b}"}),
        )
        assert ev_a_resp.status_code == 200
        assert ev_b_resp.status_code == 200

    comp_a = [e for e in _sse_events(ev_a_resp.text) if e["event"] == "run.completed"][0]
    comp_b = [e for e in _sse_events(ev_b_resp.text) if e["event"] == "run.completed"][0]

    assert [s["run_id"] for s in comp_a.get("subagent_results", [])] == ["subtask_org_tenant_A"]
    assert [s["run_id"] for s in comp_b.get("subagent_results", [])] == ["subtask_org_tenant_B"]


# ---------------------------------------------------------------------------
# 3. Strict _resolve_active_run_id & Tenant Ownership Checks in Cloud Mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cloud_run_events_and_cancel_block_hijack_and_cross_tenant_access(monkeypatch):
    monkeypatch.setenv("VALSTORM_AGENT_MODE", "cloud")

    # Clear active maps cleanly
    server.active_run_event_queues.clear()
    server.active_run_tasks.clear()
    server.active_chat_runs.clear()
    if hasattr(server, "active_run_tenants"):
        server.active_run_tenants.clear()

    gate = asyncio.Event()

    async def slow_run_turn_stream(self, **kwargs):
        await asyncio.wait_for(gate.wait(), timeout=5.0)
        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="ok")
        yield StreamEvent(
            event_type=StreamEventType.TURN_COMPLETE,
            message=Message(role="assistant", content="ok", usage=UsageMetadata(prompt_tokens=1, completion_tokens=1)),
        )

    monkeypatch.setattr(server, "_get_provider_instance", lambda *_: (object(), "gemini-flash-latest", "gemini"))
    monkeypatch.setattr(server, "_build_server_tool_registry", lambda **_: object())
    monkeypatch.setattr(server.ReActEngine, "run_turn_stream", slow_run_turn_stream)

    jwt_a = _make_jwt("org_tenant_A", "user_A1")
    jwt_b = _make_jwt("org_tenant_B", "user_B1")

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        created = await ac.post(
            "/v1/runs",
            headers={"Authorization": f"Bearer {jwt_b}"},
            json={
                "input": "Long task for B",
                "run_id": "run_1795377999123_bbbbbb",
                "session_id": "aich_b_only",
                "sandbox_type": "cloud",
            },
        )
        assert created.status_code == 200
        assert len(server.active_run_event_queues) == 1

        # 1) Non-existent run_id (even with same 10-digit timestamp prefix & only 1 active run) -> 404!
        stale_poll = await ac.get("/v1/runs/run_1795377999999_aaaaaa/events")
        assert stale_poll.status_code == 404

        stale_cancel = await ac.delete("/v1/runs/run_1795377999999_aaaaaa")
        assert stale_cancel.status_code == 404

        # 2) Exact run_id of Tenant B called with Tenant A's token -> 403!
        cross_poll = await ac.get(
            "/v1/runs/run_1795377999123_bbbbbb/events",
            headers={"Authorization": f"Bearer {jwt_a}"},
        )
        assert cross_poll.status_code == 403

        cross_cancel = await ac.delete(
            "/v1/runs/run_1795377999123_bbbbbb",
            headers={"Authorization": f"Bearer {jwt_a}"},
        )
        assert cross_cancel.status_code == 403

        # 3) Release Tenant B and confirm exact run_id works cleanly (including open-api no-header SSE proxy)
        gate.set()
        valid_poll = await ac.get("/v1/runs/run_1795377999123_bbbbbb/events")
        assert valid_poll.status_code == 200


# ---------------------------------------------------------------------------
# 4. /v1/memories, /v1/chat/completions, and /api/chat Auth & Isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memories_endpoint_requires_auth_and_isolates_tenants_in_cloud_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VALSTORM_AGENT_MODE", "cloud")

    jwt_a = _make_jwt("org_mem_A", "user_mA")
    jwt_b = _make_jwt("org_mem_B", "user_mB")

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Unauthenticated calls rejected in Cloud mode
        assert (await ac.get("/v1/memories")).status_code == 401
        assert (await ac.post("/v1/memories", json={"content": "No auth fact"})).status_code == 401
        assert (await ac.request("DELETE", "/v1/memories", json={"old_text": "No auth"})).status_code == 401

        # Tenant A writes a secret fact
        post_a = await ac.post(
            "/v1/memories",
            headers={"Authorization": f"Bearer {jwt_a}"},
            json={"content": "Tenant A database password policy is strict.", "target": "memory"},
        )
        assert post_a.status_code == 200

        # Tenant A sees its own fact
        get_a = await ac.get("/v1/memories", headers={"Authorization": f"Bearer {jwt_a}"})
        assert get_a.status_code == 200
        assert any("Tenant A database" in f for f in get_a.json().get("memory", []))

        # Tenant B sees ZERO facts from Tenant A
        get_b = await ac.get("/v1/memories", headers={"Authorization": f"Bearer {jwt_b}"})
        assert get_b.status_code == 200
        assert get_b.json().get("memory", []) == []
        assert get_b.json().get("user", []) == []


@pytest.mark.asyncio
async def test_chat_completions_and_ollama_chat_require_auth_in_cloud_mode(monkeypatch):
    monkeypatch.setenv("VALSTORM_AGENT_MODE", "cloud")

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r1 = await ac.post(
            "/v1/chat/completions",
            json={"model": "vsagent", "messages": [{"role": "user", "content": "hi"}], "stream": False},
        )
        assert r1.status_code == 401

        r2 = await ac.post(
            "/api/chat",
            json={"model": "vsagent-developer", "messages": [{"role": "user", "content": "hi"}], "stream": False},
        )
        assert r2.status_code == 401


# ---------------------------------------------------------------------------
# 5. Cloud Credential Fallback Lock & PAT Org Extraction
# ---------------------------------------------------------------------------


def test_cloud_mode_never_falls_back_to_local_auth_files_or_env_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VALSTORM_AGENT_MODE", "cloud")
    monkeypatch.setenv("VALSTORM_TOKEN", "leaked_pod_env_token_xyz")

    auth_dir = tmp_path / ".valstorm"
    auth_dir.mkdir(parents=True, exist_ok=True)
    (auth_dir / "auth_prod_default.json").write_text(
        json.dumps({"access_token": "leaked_disk_token_abc"}),
        encoding="utf-8",
    )

    set_current_valstorm_auth(None, None)
    token, _, _, _ = resolve_valstorm_auth_context()
    assert token == ""


def test_pat_token_populates_org_id_in_tenant_context():
    pat = _make_pat("org_pat_tenant_999")
    set_current_valstorm_auth(pat, "https://api.valstorm.com/v1", is_cloud_mode=True)
    try:
        ctx = get_current_tenant_context()
        assert ctx is not None
        assert ctx.org_id == "org_pat_tenant_999"
    finally:
        set_current_valstorm_auth(None, None)


# ---------------------------------------------------------------------------
# 6. Non-Colliding Cache Keys for Skills & Profiles When org_id Missing
# ---------------------------------------------------------------------------


def test_skill_and_profile_caches_do_not_collide_on_jwt_header_prefix():
    invalidate_tenant_skills_cache()
    invalidate_tenant_profiles_cache()

    # Two distinct JWTs with identical first 16 chars ("eyJhbGciOiJIUzI1") and no org_id claim
    jwt_1 = _make_jwt(org_id=None, user_id="user_one")
    jwt_2 = _make_jwt(org_id=None, user_id="user_two")
    assert jwt_1[:16] == jwt_2[:16]

    def fake_query_skills(token, base_url):
        if token == jwt_1:
            return [{"id": "aisk_1", "api_name": "secret-skill-1", "name": "Secret 1", "body": "Only for 1"}]
        return [{"id": "aisk_2", "api_name": "secret-skill-2", "name": "Secret 2", "body": "Only for 2"}]

    def fake_query_agents(token, base_url):
        if token == jwt_1:
            return [{"id": "aiag_1", "api_name": "agent-1", "name": "Agent 1", "system_prompt": "Only 1"}]
        return [{"id": "aiag_2", "api_name": "agent-2", "name": "Agent 2", "system_prompt": "Only 2"}]

    with patch("tools.skill_tool._query_tenant_ai_skills", side_effect=fake_query_skills), patch(
        "core.context._query_tenant_ai_agents", side_effect=fake_query_agents
    ):
        set_current_valstorm_auth(jwt_1, "https://api.valstorm.com/v1", is_cloud_mode=True)
        try:
            skills_1, _ = _get_cached_cloud_skills()
            profs_1, _ = _get_cached_cloud_profiles()
        finally:
            set_current_valstorm_auth(None, None)

        set_current_valstorm_auth(jwt_2, "https://api.valstorm.com/v1", is_cloud_mode=True)
        try:
            skills_2, _ = _get_cached_cloud_skills()
            profs_2, _ = _get_cached_cloud_profiles()
        finally:
            set_current_valstorm_auth(None, None)

    assert [s["slug"] for s in skills_1] == ["secret-skill-1"]
    assert [s["slug"] for s in skills_2] == ["secret-skill-2"]
    assert [p["api_name"] for p in profs_1] == ["agent-1"]
    assert [p["api_name"] for p in profs_2] == ["agent-2"]


# ---------------------------------------------------------------------------
# 7. Kubernetes Manifests Explicitly Set VALSTORM_AGENT_MODE=cloud
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel_path",
    [
        "k8s/main/open/valstorm-agent-runtime.yaml",
        "k8s/main/valstorm-agent-runtime.yaml",
        "k8s/dev/open/valstorm-agent-runtime-dev.yaml",
        "k8s/dev/valstorm-agent-runtime-dev.yaml",
    ],
)
def test_k8s_agent_runtime_manifests_set_cloud_agent_mode(rel_path):
    repo_root = Path(__file__).resolve().parents[3]
    manifest = (repo_root / rel_path).read_text(encoding="utf-8")
    assert "VALSTORM_AGENT_MODE" in manifest
    assert 'value: "cloud"' in manifest


# ---------------------------------------------------------------------------
# 8. Per-Run ContextVar ProcessRegistry Isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_registry_isolated_per_task_via_contextvar():
    from tools.process_manager import (
        ProcessRegistry,
        ProcessSession,
        reset_current_process_registry,
        set_current_process_registry,
    )

    class _DummyProc:
        pid = 12345
        returncode = None
        stdout = None
        stderr = None
        stdin = None

    gate_a = asyncio.Event()
    gate_b = asyncio.Event()

    async def worker(label: str, wait_ev: asyncio.Event, signal_ev: asyncio.Event):
        tok = set_current_process_registry(ProcessRegistry())
        try:
            sess = ProcessSession(session_id=f"proc_{label}", command=f"echo {label}", process=_DummyProc())
            ProcessRegistry.get_instance().register(sess)
            signal_ev.set()
            await asyncio.wait_for(wait_ev.wait(), timeout=3.0)
            return [p.session_id for p in ProcessRegistry.get_instance().list_all()]
        finally:
            reset_current_process_registry(tok)

    seen_a, seen_b = await asyncio.gather(
        worker("tenant_a", gate_b, gate_a),
        worker("tenant_b", gate_a, gate_b),
    )
    assert seen_a == ["proc_tenant_a"]
    assert seen_b == ["proc_tenant_b"]


# ---------------------------------------------------------------------------
# 9. Cloud Mode Withholds Host Shell/FS/Code/Process Tools Without MicroVM
# ---------------------------------------------------------------------------


def test_cloud_mode_withholds_host_tools_in_server_and_subagent_registries(monkeypatch):
    from core.sandbox import HostSandbox, set_current_sandbox
    from tools.delegation import _build_base_subagent_tool_registry

    monkeypatch.setenv("VALSTORM_AGENT_MODE", "cloud")

    forbidden_host_tools = {
        "terminal_exec",
        "read_file",
        "write_file",
        "patch_file",
        "search_files",
        "read_local_file",
        "code_outline",
        "find_symbols",
        "package_blast_radius",
        "process_manage",
        "execute_code",
        "computer_use",
    }

    jwt_a = _make_jwt("org_cloud_tools", "user_ct1")
    set_current_valstorm_auth(jwt_a, "https://api.valstorm.com/v1", is_cloud_mode=True)
    set_current_sandbox(HostSandbox())
    try:
        server_reg = server._build_server_tool_registry(
            valstorm_token=jwt_a,
            override_base_url="https://api.valstorm.com/v1",
            allow_local_host_tools=False,
        )
        server_tool_names = set(server_reg.list_tools())
        assert forbidden_host_tools.isdisjoint(server_tool_names), (
            f"Host tools leaked in cloud mode: {forbidden_host_tools & server_tool_names}"
        )
        # Safe platform/memory/web/delegation tools must remain available
        assert "calculator" in server_tool_names
        assert "memory_manage" in server_tool_names
        assert "session_search" in server_tool_names
        assert "valstorm_sql_query" in server_tool_names

        subagent_reg = _build_base_subagent_tool_registry()
        subagent_tool_names = set(subagent_reg.list_tools())
        assert forbidden_host_tools.isdisjoint(subagent_tool_names), (
            f"Host tools leaked to subagent in cloud mode: {forbidden_host_tools & subagent_tool_names}"
        )
        assert "calculator" in subagent_tool_names
        assert "memory_manage" in subagent_tool_names
    finally:
        set_current_sandbox(None)
        set_current_valstorm_auth(None, None)


@pytest.mark.asyncio
async def test_cloud_pod_ignores_host_execution_environment_and_isolates_fallback_dir(tmp_path, monkeypatch):
    """On a Cloud pod, a request asking for execution_environment='host' must still enforce cloud isolation."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VALSTORM_AGENT_MODE", "cloud")

    captured = {}

    def fake_build_registry(**kwargs):
        captured["allow_local_host_tools"] = kwargs.get("allow_local_host_tools")
        from core.sandbox import get_current_sandbox
        sb = get_current_sandbox()
        captured["sandbox_base_dir"] = getattr(sb, "base_dir", None)
        return server.get_default_registry()

    async def fake_run_turn_stream(self, **kwargs):
        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="ok")
        yield StreamEvent(
            event_type=StreamEventType.TURN_COMPLETE,
            message=Message(role="assistant", content="ok", usage=UsageMetadata(prompt_tokens=1, completion_tokens=1)),
        )

    # Simulate CloudMicroVMSandbox failing (e.g. no E2B_API_KEY on K8s pod)
    class _FailingCloudSandbox:
        def __init__(self, *args, **kwargs):
            raise ValueError("E2B_API_KEY not configured")

    monkeypatch.setattr(server, "CloudMicroVMSandbox", _FailingCloudSandbox)
    monkeypatch.setattr(server, "_get_provider_instance", lambda *_: (object(), "gemini-flash-latest", "gemini"))
    monkeypatch.setattr(server, "_build_server_tool_registry", fake_build_registry)
    monkeypatch.setattr(server.ReActEngine, "run_turn_stream", fake_run_turn_stream)

    jwt_a = _make_jwt("org_cloud_force", "user_cf1")
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post(
            "/v1/runs",
            headers={"Authorization": f"Bearer {jwt_a}"},
            json={
                "input": "hello",
                "session_id": "aich_cloud_force",
                "execution_environment": "host",
            },
        )
        assert res.status_code == 200
        run_id = res.json()["run_id"]
        ev = await ac.get(f"/v1/runs/{run_id}/events", headers={"Authorization": f"Bearer {jwt_a}"})
        assert ev.status_code == 200

    assert captured.get("allow_local_host_tools") is False
    assert captured.get("sandbox_base_dir") is not None
    assert "vsagent_run_" in str(captured["sandbox_base_dir"])


# ---------------------------------------------------------------------------
# 10. POST /v1/runs Requires Authentication in Cloud Mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_run_requires_auth_in_cloud_mode(monkeypatch):
    monkeypatch.setenv("VALSTORM_AGENT_MODE", "cloud")

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post(
            "/v1/runs",
            json={"input": "Unauthenticated cloud run attempt", "session_id": "aich_noauth"},
        )
        assert res.status_code == 401
