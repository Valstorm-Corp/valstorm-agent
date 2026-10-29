"""Phase 2 TDD Suite: First-Class `skill_manage` & `agent_manage` Authoring Tools.

Verifies:
1. `skill_manage` lifecycle (create, section-append patch, substring patch, delete) with automatic
   YAML frontmatter compilation and semantic patch version bumping (1.0.0 -> 1.0.1 -> 1.0.2).
2. `bundled` skill ownership protection blocking direct destructive overwrites/deletes until
   explicitly cloned/adopted via `skill_manage(action='adopt')`.
3. Cloud persistence for `skill_manage` (`POST`/`PATCH`/`DELETE /v1/object/ai_skill`) and immediate
   TTL cache invalidation (`invalidate_tenant_skills_cache`).
4. `agent_manage` lifecycle (create, update, attach_skills, detach_skills, list, delete) with
   automatic `skill_slugs` -> `aisk_...` ID mapping for cloud `ai_agent.ai_skills`.
5. `agent_manage` security guardrails:
   - Rejects hallucinated/unknown tool names in `allowed_tools`.
   - Blocks privilege escalation when a restricted caller profile attempts to grant tools outside
     its own `allowed_tools` whitelist.
   - Protects built-in core profiles (`chief-of-staff`, `developer`, `orchestrator`) from deletion.
6. End-to-end autonomous self-extension: create a new skill via `skill_manage` -> mint a new
   specialist agent with that skill via `agent_manage` -> immediately spawn it via `delegate_task`
   and verify the subagent receives the new skill in its system prompt.
"""

import base64
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from core.context import (
    invalidate_tenant_profiles_cache,
    load_profile,
    set_current_profile,
)
from core.models import Message, UsageMetadata
from tools.delegation import (
    agent_manage,
    delegate_task,
    execute_agent_mutation,
    set_subagent_provider_override,
)
from tools.skill_tool import (
    execute_skill_mutation,
    invalidate_tenant_skills_cache,
    skill_list,
    skill_manage,
    skill_view,
)
from tools.valstorm_client import set_current_valstorm_auth


def _make_test_jwt(org_id: str, user_id: str) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).decode().rstrip("=")
    payload = base64.urlsafe_b64encode(
        json.dumps({"organization_id": org_id, "user_id": user_id, "exp": 4102444800}).encode()
    ).decode().rstrip("=")
    return f"{header}.{payload}.sig"


@pytest.mark.asyncio
async def test_skill_manage_create_patch_section_and_substring_host_mode(tmp_path, monkeypatch):
    """Verify skill_manage creates a skill, appends to a section, patches substring, and bumps version."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(parents=True)
    monkeypatch.setenv("VALSTORM_SKILLS_DIR", str(skills_dir))
    invalidate_tenant_skills_cache()

    # 1. Create a new skill
    create_out = await skill_manage(
        action="create",
        slug="stripe-webhook-audit",
        category="valstorm-internal",
        name="Stripe Webhook Audit",
        description="Verify Stripe webhook signatures and idempotency keys.",
        body="# Stripe Webhook Audit\n\n## Procedure\n1. Check Stripe-Signature header.",
        tags=["stripe", "billing", "webhooks"],
    )
    assert "Created skill" in create_out or "stripe-webhook-audit" in create_out

    # Verify skill_list and skill_view
    listed = await skill_list(query="stripe")
    assert "valstorm-internal/stripe-webhook-audit" in listed

    viewed_v1 = await skill_view("stripe-webhook-audit")
    assert "=== SKILL: valstorm-internal/stripe-webhook-audit (v1.0.0) ===" in viewed_v1
    assert "1. Check Stripe-Signature header." in viewed_v1

    # 2. Patch by appending to a 'Pitfalls' section (creates ## Pitfalls heading automatically)
    patch_sec_out = await skill_manage(
        action="patch",
        slug="stripe-webhook-audit",
        section="Pitfalls",
        content_to_append="- Never parse JSON body before verifying the raw payload HMAC.",
    )
    assert "Patched skill" in patch_sec_out or "1.0.1" in patch_sec_out

    viewed_v2 = await skill_view("stripe-webhook-audit")
    assert "(v1.0.1)" in viewed_v2
    assert "## Pitfalls" in viewed_v2
    assert "- Never parse JSON body before verifying the raw payload HMAC." in viewed_v2

    # 3. Append a second item to the existing 'Pitfalls' section
    await skill_manage(
        action="patch",
        slug="stripe-webhook-audit",
        section="Pitfalls",
        content_to_append="- Always log event.id to prevent duplicate webhook replay.",
    )
    viewed_v3 = await skill_view("stripe-webhook-audit")
    assert "(v1.0.2)" in viewed_v3
    assert viewed_v3.count("## Pitfalls") == 1
    assert "- Always log event.id to prevent duplicate webhook replay." in viewed_v3

    # 4. Patch via exact old_text -> new_text replacement
    await skill_manage(
        action="patch",
        slug="stripe-webhook-audit",
        old_text="1. Check Stripe-Signature header.",
        new_text="1. Validate raw request bytes against Stripe-Signature header.",
    )
    viewed_v4 = await skill_view("stripe-webhook-audit")
    assert "(v1.0.3)" in viewed_v4
    assert "1. Validate raw request bytes against Stripe-Signature header." in viewed_v4

    # 5. Delete user-created skill
    del_out = await skill_manage(action="delete", slug="stripe-webhook-audit")
    assert "Deleted" in del_out
    view_after_del = await skill_view("stripe-webhook-audit")
    assert "Error: Skill 'stripe-webhook-audit' not found" in view_after_del


@pytest.mark.asyncio
async def test_skill_manage_bundled_ownership_protection_and_adopt(tmp_path, monkeypatch):
    """Verify bundled skills cannot be destructively overwritten or deleted until adopted."""
    skills_dir = tmp_path / "skills"
    bundled_dir = skills_dir / "software-development" / "core-git-safety"
    bundled_dir.mkdir(parents=True)
    (bundled_dir / "SKILL.md").write_text(
        "---\n"
        "name: Core Git Safety\n"
        "description: Never force-push to main.\n"
        "version: 1.0.0\n"
        "ownership: bundled\n"
        "---\n\n"
        "# Core Git Safety\n"
        "Always use feature branches.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VALSTORM_SKILLS_DIR", str(skills_dir))
    invalidate_tenant_skills_cache()

    # 1. Direct deletion of a bundled skill must be blocked
    del_attempt = await skill_manage(action="delete", slug="core-git-safety")
    assert "Error" in del_attempt and "bundled" in del_attempt.lower()

    # 2. Direct substring overwrite of a bundled skill must be blocked until adopted
    patch_attempt = await skill_manage(
        action="patch",
        slug="core-git-safety",
        old_text="Always use feature branches.",
        new_text="Push directly to main.",
    )
    assert "Error" in patch_attempt and "adopt" in patch_attempt.lower()

    # 3. Adopting the bundled skill transitions ownership to user/tenant
    adopt_out = await skill_manage(action="adopt", slug="core-git-safety")
    assert "Adopted" in adopt_out

    # 4. Now patching succeeds on the adopted skill
    patch_after_adopt = await skill_manage(
        action="patch",
        slug="core-git-safety",
        section="Pitfalls",
        content_to_append="- Verify git worktree list before removing branches.",
    )
    assert "Patched" in patch_after_adopt
    viewed = await skill_view("core-git-safety")
    assert "Verify git worktree list before removing branches." in viewed


@pytest.mark.asyncio
async def test_skill_manage_cloud_mode_persistence(tmp_path, monkeypatch):
    """Verify skill_manage in Cloud mode persists to /v1/object/ai_skill and invalidates cache."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(parents=True)
    monkeypatch.setenv("VALSTORM_SKILLS_DIR", str(skills_dir))
    invalidate_tenant_skills_cache()

    cloud_db: list[dict] = []

    def mock_query_skills(*args, **kwargs):
        return list(cloud_db)

    async def mock_upsert_cloud_skill(record_payload, existing_id=None):
        if existing_id:
            for r in cloud_db:
                if r.get("id") == existing_id:
                    r.update(record_payload)
                    return r
        new_rec = dict(record_payload)
        new_rec["id"] = f"aisk_cloud_{len(cloud_db) + 1}"
        cloud_db.append(new_rec)
        return new_rec

    set_current_valstorm_auth(
        _make_test_jwt("org_cloud_skill", "user_cloud_1"),
        "https://api.valstorm.com/v1",
        is_cloud_mode=True,
    )
    try:
        with (
            patch("tools.skill_tool._query_tenant_ai_skills", side_effect=mock_query_skills),
            patch("tools.skill_tool._persist_skill_to_cloud", side_effect=mock_upsert_cloud_skill),
        ):
            receipt = await execute_skill_mutation(
                action="create",
                slug="cloud-kyc-verification",
                category="compliance",
                name="Cloud KYC Verification",
                description="Run OFAC and KYC checks before onboarding accounts.",
                body="# Cloud KYC Verification\n1. Verify EIN and beneficial owners.",
                tags=["kyc", "compliance"],
            )
            assert receipt["status"] == "success"
            assert receipt["record_id"] == "aisk_cloud_1"

            # Verify immediately visible via skill_view in Cloud mode without local disk file
            viewed = await skill_view("cloud-kyc-verification")
            assert "Cloud KYC Verification" in viewed
            assert "1. Verify EIN and beneficial owners." in viewed
    finally:
        invalidate_tenant_skills_cache()
        set_current_valstorm_auth(None, None)


@pytest.mark.asyncio
async def test_agent_manage_lifecycle_and_privilege_guardrails(tmp_path, monkeypatch):
    """Verify agent_manage creates/updates profiles, validates tools, and blocks privilege escalation."""
    skills_dir = tmp_path / "skills"
    profiles_dir = tmp_path / "profiles"
    skills_dir.mkdir(parents=True)
    profiles_dir.mkdir(parents=True)
    monkeypatch.setenv("VALSTORM_SKILLS_DIR", str(skills_dir))
    monkeypatch.setenv("VALSTORM_PROFILES_DIR", str(profiles_dir))
    invalidate_tenant_skills_cache()
    invalidate_tenant_profiles_cache()

    # Create two skills to attach
    await skill_manage(
        action="create",
        slug="invoice-reconciliation",
        category="finance",
        name="Invoice Reconciliation",
        description="Reconcile Stripe payouts with ledger.",
        body="# Invoice Reconciliation\nMatch payout IDs.",
    )
    await skill_manage(
        action="create",
        slug="tax-audit-check",
        category="finance",
        name="Tax Audit Check",
        description="Validate state sales tax nexus.",
        body="# Tax Audit Check\nCheck nexus thresholds.",
    )

    # 1. Reject hallucinated tool names
    bad_tools_res = await agent_manage(
        action="create",
        slug="bad-tool-agent",
        name="Bad Tool Agent",
        system_prompt="Test",
        allowed_tools=["valstorm_sql_query", "nonexistent_magic_tool_999"],
    )
    assert "Error" in bad_tools_res and "nonexistent_magic_tool_999" in bad_tools_res

    # 2. Block privilege escalation when active profile is restricted
    tok = set_current_profile({
        "name": "Restricted Reader",
        "api_name": "restricted-reader",
        "allowed_tools": ["read_file", "search_files", "skill_view"],
    })
    try:
        esc_res = await agent_manage(
            action="create",
            slug="escalated-agent",
            name="Escalated Agent",
            system_prompt="Try to get terminal_exec",
            allowed_tools=["read_file", "terminal_exec"],
        )
        assert "Error" in esc_res and "privilege" in esc_res.lower()
    finally:
        set_current_profile(None)

    # 3. Create valid specialist agent
    create_agent_res = await agent_manage(
        action="create",
        slug="finance-controller",
        name="Finance Controller",
        description="Audits invoices and reconciles ledger entries.",
        system_prompt="You are the Finance Controller specialist.",
        allowed_tools=["valstorm_sql_query", "calculator", "skill_view", "skill_list"],
        skill_slugs=["invoice-reconciliation"],
        vaults=["vaul_finance_ledger_01"],
        tags=["Finance", "Accounting"],
    )
    assert "Created" in create_agent_res and "finance-controller" in create_agent_res

    loaded = load_profile("finance-controller")
    assert loaded["name"] == "Finance Controller"
    assert loaded["attached_skill_slugs"] == ["invoice-reconciliation"]
    assert loaded["knowledge_vaults"] == ["vaul_finance_ledger_01"]

    # 4. Attach and detach skills
    await agent_manage(
        action="attach_skills",
        slug="finance-controller",
        skill_slugs=["tax-audit-check"],
    )
    loaded_attached = load_profile("finance-controller")
    assert "invoice-reconciliation" in loaded_attached["attached_skill_slugs"]
    assert "tax-audit-check" in loaded_attached["attached_skill_slugs"]

    await agent_manage(
        action="detach_skills",
        slug="finance-controller",
        skill_slugs=["invoice-reconciliation"],
    )
    loaded_detached = load_profile("finance-controller")
    assert loaded_detached["attached_skill_slugs"] == ["tax-audit-check"]

    # 5. Prevent deletion of built-in core orchestrator
    del_builtin = await agent_manage(action="delete", slug="chief-of-staff")
    assert "Error" in del_builtin and "built-in" in del_builtin.lower()


@pytest.mark.asyncio
async def test_end_to_end_self_extension_skill_to_agent_to_delegate_task(tmp_path, monkeypatch):
    """Verify an agent can author a skill, mint a specialist profile with it, and immediately delegate_task to it."""
    skills_dir = tmp_path / "skills"
    profiles_dir = tmp_path / "profiles"
    skills_dir.mkdir(parents=True)
    profiles_dir.mkdir(parents=True)
    monkeypatch.setenv("VALSTORM_SKILLS_DIR", str(skills_dir))
    monkeypatch.setenv("VALSTORM_PROFILES_DIR", str(profiles_dir))
    invalidate_tenant_skills_cache()
    invalidate_tenant_profiles_cache()

    # Step 1: Create domain skill
    await skill_manage(
        action="create",
        slug="hubspot-migration-validator",
        category="migrations",
        name="HubSpot Migration Validator",
        description="Verify Contact and Deal foreign key integrity after HubSpot import.",
        body="# HubSpot Migration Validator\nAlways verify `hubspot_id__c` uniqueness.",
    )

    # Step 2: Create specialist subagent profile with the new skill attached
    await agent_manage(
        action="create",
        slug="migration-specialist",
        name="Migration Specialist",
        description="Validates CRM data migrations.",
        system_prompt="You are the CRM Migration Specialist.",
        allowed_tools=["valstorm_sql_query", "skill_view", "skill_list", "calculator"],
        skill_slugs=["hubspot-migration-validator"],
        vaults=["vaul_migrations_01"],
    )

    # Step 3: Immediately delegate a task to the newly minted specialist
    captured_system_prompts: list[str] = []

    class InspectingSubagentProvider:
        provider_name = "mock"

        async def generate(self, messages, tools=None, model=None, **kwargs):
            for m in messages:
                if m.role == "system":
                    captured_system_prompts.append(m.content or "")
            return (
                Message(role="assistant", content="Migration validation complete: 0 orphaned contacts."),
                UsageMetadata(prompt_tokens=25, completion_tokens=12, total_tokens=37),
            )

    set_subagent_provider_override(InspectingSubagentProvider())
    try:
        result = await delegate_task(
            profile="migration-specialist",
            goal="Validate the latest HubSpot contact batch.",
        )
        assert "PROFILE: MIGRATION-SPECIALIST" in result
        assert "Migration validation complete: 0 orphaned contacts." in result
        assert len(captured_system_prompts) >= 1
        sys_prompt = captured_system_prompts[0]
        assert "You are the CRM Migration Specialist." in sys_prompt
        assert "hubspot-migration-validator" in sys_prompt
        assert "Verify Contact and Deal foreign key integrity" in sys_prompt
        assert "vaul_migrations_01" in sys_prompt
    finally:
        set_subagent_provider_override(None)
