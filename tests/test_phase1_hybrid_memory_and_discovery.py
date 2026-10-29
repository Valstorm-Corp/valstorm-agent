"""Phase 1 TDD Suite: Unified Tenant-Scoped Memory & Hybrid Skill/Profile Discovery.

Verifies:
1. TenantRuntimeContext propagation via set_current_valstorm_auth / get_current_tenant_context
   (including automatic JWT org_id and user_id extraction).
2. Multi-tenant MemoryStore isolation in Cloud mode (zero cross-tenant file bleed across orgs)
   while preserving ~/.valstorm/memories.json behavior in Host mode.
3. Asynchronous Cloud Memory reconciliation (reconcile_fact_async) with `supersedes` forwarding
   to POST /v1/ai/memory/fact and automatic background extraction via extract_and_commit_turn_memory.
4. Deduplicated system prompt formatting when Turn 0 working_memory is already injected.
5. Hybrid Skill Discovery (_index_all_skills, skill_list, skill_view) merging local SKILL.md files
   with tenant MongoDB `ai_skill` records and supporting TTL cache invalidation.
6. Hybrid Profile Loading (load_profile, list_available_profiles) resolving custom tenant `ai_agent`
   records, translating `ai_skills` record IDs (`aisk_...`) into `attached_skill_slugs`, normalizing
   `vaults` into `knowledge_vaults`, and executing `delegate_task` with custom cloud profiles.
"""

import base64
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from core.context import (
    WorkspaceContextManager,
    format_attached_skills_context,
    invalidate_tenant_profiles_cache,
    list_available_profiles,
    load_profile,
)
from core.memory import MemoryStore
from core.memory_extractor import extract_and_commit_turn_memory
from core.models import Message, UsageMetadata
from tools.delegation import delegate_task, set_subagent_provider_override
from tools.memory_tool import create_memory_tools
from tools.skill_tool import (
    _index_all_skills,
    get_skill_id_to_slug_map,
    invalidate_tenant_skills_cache,
    skill_list,
    skill_view,
)
from tools.valstorm_client import (
    get_current_tenant_context,
    get_current_valstorm_auth,
    set_current_valstorm_auth,
)


def _make_test_jwt(org_id: str, user_id: str) -> str:
    """Builds an unsigned test JWT containing organization_id and user_id claims."""
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).decode().rstrip("=")
    payload = base64.urlsafe_b64encode(
        json.dumps({"organization_id": org_id, "user_id": user_id, "exp": 4102444800}).encode()
    ).decode().rstrip("=")
    return f"{header}.{payload}.sig"


def test_tenant_runtime_context_from_jwt_and_explicit_args():
    """Verify set_current_valstorm_auth populates TenantRuntimeContext and decodes JWT claims."""
    jwt_tok = _make_test_jwt("org_alpha_111", "user_alice_222")
    tok = set_current_valstorm_auth(
        jwt_tok,
        "https://api.valstorm.com/v1",
        is_cloud_mode=True,
        has_working_memory_injected=True,
    )
    try:
        auth_tuple = get_current_valstorm_auth()
        assert auth_tuple == (jwt_tok, "https://api.valstorm.com/v1")

        ctx = get_current_tenant_context()
        assert ctx is not None
        assert ctx.org_id == "org_alpha_111"
        assert ctx.user_id == "user_alice_222"
        assert ctx.is_cloud_mode is True
        assert ctx.has_working_memory_injected is True
    finally:
        set_current_valstorm_auth(None, None)
        assert get_current_tenant_context() is None


def test_memory_store_cloud_multi_tenant_isolation(tmp_path, monkeypatch):
    """Verify Cloud mode isolates MemoryStore files per (org_id, user_id) with zero cross-tenant bleed."""
    monkeypatch.setenv("HOME", str(tmp_path))

    # Tenant A in Cloud mode
    set_current_valstorm_auth(
        _make_test_jwt("org_tenant_A", "user_A1"),
        "https://api.valstorm.com/v1",
        is_cloud_mode=True,
    )
    try:
        store_a = MemoryStore()
        store_a.add_fact("memory", "Tenant A secret internal API port is 9001.")
        store_a.add_fact("user", "Alice prefers concise bullet summaries.")
        assert "org_tenant_A" in str(store_a.file_path)
        assert "user_A1" in str(store_a.file_path)
    finally:
        set_current_valstorm_auth(None, None)

    # Tenant B in Cloud mode on the same pod/filesystem
    set_current_valstorm_auth(
        _make_test_jwt("org_tenant_B", "user_B1"),
        "https://api.valstorm.com/v1",
        is_cloud_mode=True,
    )
    try:
        store_b = MemoryStore()
        facts_b = store_b.get_facts()
        assert facts_b["memory"] == []
        assert facts_b["user"] == []
        store_b.add_fact("memory", "Tenant B uses Snowflake warehouse.")
        assert "Tenant A" not in json.dumps(store_b.get_facts())
    finally:
        set_current_valstorm_auth(None, None)


def test_memory_store_skips_duplicate_prompt_injection_when_working_memory_present(tmp_path):
    """Verify MemoryStore.format_for_system_prompt avoids duplicating facts already in Turn 0 working_memory."""
    mem_file = tmp_path / "cloud_mem.json"
    store = MemoryStore(file_path=mem_file)
    store.add_fact("user", "Prefers concise executive summaries.")
    store.add_fact("memory", "Always use pnpm in the monorepo.")

    set_current_valstorm_auth(
        _make_test_jwt("org_1", "user_1"),
        "https://api.valstorm.com/v1",
        is_cloud_mode=True,
        has_working_memory_injected=True,
    )
    try:
        ctx_mgr = WorkspaceContextManager(workdir=str(tmp_path), memory_store=store)
        user_ctx = {
            "user_id": "user_1",
            "organization_id": "org_1",
            "working_memory": {
                "formatted_prompt_block": (
                    "# Current User Working Context (Authoritative)\n"
                    "- Prefers concise executive summaries.\n"
                )
            },
        }
        sys_prompt = ctx_mgr.build_system_prompt(user_context=user_ctx)
        assert "# Current User Working Context (Authoritative)" in sys_prompt
        # The duplicate user fact should not be injected twice
        assert sys_prompt.count("Prefers concise executive summaries.") == 1
        # Distinct environment note is still preserved
        assert "Always use pnpm in the monorepo." in sys_prompt
    finally:
        set_current_valstorm_auth(None, None)


@pytest.mark.asyncio
async def test_extract_and_commit_turn_memory_syncs_to_cloud_with_supersedes(tmp_path):
    """Verify background SmartMemory extraction reconciles locally and syncs to /v1/ai/memory/fact with supersedes."""
    mem_file = tmp_path / "memories.json"
    store = MemoryStore(file_path=mem_file)
    store.add_fact("memory", "We use Twilio for transactional SMS.")

    mock_provider = MagicMock()
    mock_provider.chat_complete = AsyncMock(
        return_value=Message(
            role="assistant",
            content=json.dumps([
                {
                    "category": "memory",
                    "fact": "Transactional SMS provider is AWS SNS (Twilio is deprecated).",
                    "supersedes": "Twilio",
                }
            ]),
        )
    )

    captured_requests = []

    async def mock_post(url, headers=None, json=None, **kwargs):
        captured_requests.append({"url": url, "headers": headers, "json": json})
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"fact_id": "fact_999", "fact": json.get("fact")}
        return resp

    set_current_valstorm_auth(
        _make_test_jwt("org_sync_1", "user_sync_1"),
        "https://api.valstorm.com/v1",
        is_cloud_mode=False,
    )
    try:
        with patch("httpx.AsyncClient.post", side_effect=mock_post):
            results = await extract_and_commit_turn_memory(
                user_input="Actually, we switched from Twilio to AWS SNS for transactional SMS",
                assistant_output="Got it, I will use AWS SNS for transactional SMS.",
                memory_store=store,
                provider=mock_provider,
                chat_id="aich_test_chat_123",
            )

        assert len(results) == 1
        assert results[0]["action"] == "superseded"
        assert len(captured_requests) == 1
        req = captured_requests[0]
        assert req["url"].endswith("/ai/memory/fact")
        assert req["json"]["fact"] == "Transactional SMS provider is AWS SNS (Twilio is deprecated)."
        assert req["json"]["supersedes"] == "Twilio"
        assert req["json"]["source_chat_id"] == "aich_test_chat_123"
        assert req["json"]["category"] == "environment_note"
    finally:
        set_current_valstorm_auth(None, None)


@pytest.mark.asyncio
async def test_memory_manage_tool_syncs_both_user_and_environment_facts_to_cloud(tmp_path):
    """Verify memory_manage routes both target='user' and target='memory' to cloud when authenticated."""
    mem_file = tmp_path / "memories.json"
    store = MemoryStore(file_path=mem_file)
    memory_manage_fn, _ = create_memory_tools(memory_store=store)

    posted_payloads = []

    async def mock_post(url, headers=None, json=None, **kwargs):
        posted_payloads.append(json)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"status": "ok"}
        return resp

    set_current_valstorm_auth(
        _make_test_jwt("org_tool_1", "user_tool_1"),
        "https://api.valstorm.com/v1",
        is_cloud_mode=True,
    )
    try:
        with patch("httpx.AsyncClient.post", side_effect=mock_post):
            res_user = await memory_manage_fn(action="add", target="user", content="I prefer dark mode charts.")
            res_mem = await memory_manage_fn(action="add", target="memory", content="Our fiscal year starts in February.")

        assert "Saved" in res_user or "Successfully added" in res_user
        assert "Saved" in res_mem or "Successfully added" in res_mem
        assert len(posted_payloads) == 2
        assert posted_payloads[0]["category"] == "user_fact"
        assert posted_payloads[1]["category"] == "environment_note"
    finally:
        set_current_valstorm_auth(None, None)


@pytest.mark.asyncio
async def test_hybrid_skill_discovery_merges_disk_and_cloud_skills(tmp_path, monkeypatch):
    """Verify skill_list, skill_view, and get_skill_id_to_slug_map merge local disk and cloud ai_skill records."""
    skills_dir = tmp_path / "skills"
    tdd_dir = skills_dir / "software-development" / "test-driven-development"
    tdd_dir.mkdir(parents=True)
    (tdd_dir / "SKILL.md").write_text(
        "---\nname: test-driven-development\ndescription: Local TDD guide\nversion: 1.0.0\n---\n# Local TDD Body\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VALSTORM_SKILLS_DIR", str(skills_dir))
    invalidate_tenant_skills_cache()

    cloud_skill_records = [
        {
            "id": "aisk_cloud_erp_01",
            "name": "NetSuite ERP Sync Protocol",
            "api_name": "netsuite-erp-sync",
            "category": "enterprise-integrations",
            "description": "Procedure for syncing invoices and GL accounts with NetSuite.",
            "body": (
                "---\n"
                "name: NetSuite ERP Sync Protocol\n"
                "version: 2.1.0\n"
                "tags: [netsuite, erp, finance]\n"
                "---\n\n"
                "# NetSuite Sync SOP\n"
                "Always verify subsidiary ID before posting journal entries."
            ),
            "is_active": True,
        },
        {
            "id": "aisk_tdd_override_02",
            "name": "Test Driven Development (Tenant Customized)",
            "api_name": "test-driven-development",
            "category": "software-development",
            "description": "Tenant customized TDD workflow.",
            "body": "# Tenant Cloud TDD Body\nUse custom tenant test runner.",
            "is_active": True,
        },
    ]

    def mock_fetch_cloud_skills(*args, **kwargs):
        return cloud_skill_records

    set_current_valstorm_auth(
        _make_test_jwt("org_skills_1", "user_skills_1"),
        "https://api.valstorm.com/v1",
        is_cloud_mode=True,
    )
    try:
        with patch("tools.skill_tool._query_tenant_ai_skills", side_effect=mock_fetch_cloud_skills):
            # 1. Verify skill_list shows both cloud-only skill and merged skill
            list_out = await skill_list()
            assert "enterprise-integrations/netsuite-erp-sync" in list_out
            assert "software-development/test-driven-development" in list_out

            # 2. Verify skill_view renders cloud-only skill without needing a local file on disk
            view_cloud = await skill_view("netsuite-erp-sync")
            assert "=== SKILL: enterprise-integrations/netsuite-erp-sync (v2.1.0) ===" in view_cloud
            assert "Always verify subsidiary ID before posting journal entries." in view_cloud

            # 3. Verify in Cloud mode, tenant DB skill overrides static container disk skill
            view_tdd = await skill_view("test-driven-development")
            assert "Tenant Cloud TDD Body" in view_tdd

            # 4. Verify ID -> slug map resolves aisk_... IDs for profile attachment
            id_map = get_skill_id_to_slug_map()
            assert id_map["aisk_cloud_erp_01"] == "netsuite-erp-sync"
            assert id_map["aisk_tdd_override_02"] == "test-driven-development"
    finally:
        invalidate_tenant_skills_cache()
        set_current_valstorm_auth(None, None)


@pytest.mark.asyncio
async def test_hybrid_profile_loading_and_subagent_delegation_with_cloud_agent(tmp_path, monkeypatch):
    """Verify load_profile and delegate_task resolve custom cloud ai_agent records and hydrate ai_skills IDs."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(parents=True)
    monkeypatch.setenv("VALSTORM_SKILLS_DIR", str(skills_dir))
    monkeypatch.setenv("VALSTORM_PROFILES_DIR", str(tmp_path / "empty_profiles"))
    invalidate_tenant_skills_cache()
    invalidate_tenant_profiles_cache()

    cloud_skill_records = [
        {
            "id": "aisk_revops_99",
            "name": "RevOps Pipeline Hygiene",
            "api_name": "revops-pipeline-hygiene",
            "category": "agency-operations",
            "description": "Audit deal stage velocity and close-date accuracy.",
            "body": "# RevOps Pipeline Hygiene\nCheck close dates weekly.",
            "is_active": True,
        }
    ]

    cloud_agent_records = [
        {
            "id": "aia_custom_revops_01",
            "name": "RevOps Specialist",
            "api_name": "revops-specialist",
            "description": "Audits CRM pipeline hygiene and revenue forecasts.",
            "provider": "Valstorm",
            "model": "gemini-flash-latest",
            "model_tier": "tier_2",
            "reasoning_effort": "medium",
            "system_prompt": "You are the RevOps Specialist for Acme Corp.",
            "allowed_tools": ["valstorm_sql_query", "skill_view", "skill_list", "calculator"],
            "ai_skills": ["aisk_revops_99"],
            "tag": ["RevOps", "Forecasting"],
            "vaults": ["vaul_revops_playbook_01"],
            "scoped_paths": ["apps/api/**"],
            "is_active": True,
        }
    ]

    set_current_valstorm_auth(
        _make_test_jwt("org_revops_1", "user_revops_1"),
        "https://api.valstorm.com/v1",
        is_cloud_mode=True,
    )
    try:
        with (
            patch("tools.skill_tool._query_tenant_ai_skills", return_value=cloud_skill_records),
            patch("core.context._query_tenant_ai_agents", return_value=cloud_agent_records),
        ):
            # 1. Load by api_name slug
            prof = load_profile("revops-specialist")
            assert prof["name"] == "RevOps Specialist"
            assert prof["api_name"] == "revops-specialist"
            assert prof["provider"] == "valstorm"
            assert prof["attached_skill_slugs"] == ["revops-pipeline-hygiene"]
            assert prof["knowledge_vaults"] == ["vaul_revops_playbook_01"]

            # 2. Load by record ID (aia_...)
            prof_by_id = load_profile("aia_custom_revops_01")
            assert prof_by_id["api_name"] == "revops-specialist"

            # 3. Verify list_available_profiles includes the custom cloud agent
            all_profs = list_available_profiles()
            slugs = [p.get("api_name") for p in all_profs]
            assert "revops-specialist" in slugs

            # 4. Verify WorkspaceContextManager injects both resolved skill and knowledge vault scope
            ctx_mgr = WorkspaceContextManager(workdir=str(tmp_path))
            prompt = ctx_mgr.build_system_prompt(profile=prof)
            assert "You are the RevOps Specialist for Acme Corp." in prompt
            assert "revops-pipeline-hygiene" in prompt
            assert "Audit deal stage velocity" in prompt
            assert "vaul_revops_playbook_01" in prompt

            # 5. Verify delegate_task can spawn the cloud-defined specialist subagent
            class MockSubagentProvider:
                provider_name = "valstorm"

                async def generate(self, messages, tools=None, model=None, **kwargs):
                    return (
                        Message(role="assistant", content="RevOps pipeline audit completed: 100% compliant."),
                        UsageMetadata(prompt_tokens=20, completion_tokens=10, total_tokens=30),
                    )

            set_subagent_provider_override(MockSubagentProvider())
            try:
                del_res = await delegate_task(
                    profile="revops-specialist",
                    goal="Audit Q4 deal close dates",
                )
                assert "PROFILE: REVOPS-SPECIALIST" in del_res
                assert "RevOps pipeline audit completed: 100% compliant." in del_res
            finally:
                set_subagent_provider_override(None)
    finally:
        invalidate_tenant_skills_cache()
        invalidate_tenant_profiles_cache()
        set_current_valstorm_auth(None, None)
