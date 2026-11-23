"""Tests for scoped Open Valstorm plugin tool isolation, alias resolution, and delegated subagents."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.context import BUILTIN_PROFILES, PROFILE_ALIASES, load_profile
from core.models import Message, UsageMetadata
from core.tools import ToolRegistry, matches_tool_pattern
from tools.remote_plugins import (
    clear_remote_plugin_tools_cache,
    make_remote_plugin_tool,
)


def _build_synthetic_full_registry() -> ToolRegistry:
    """Builds a registry populated with core tools + representative Open Valstorm plugin tools."""
    reg = ToolRegistry()

    core_names = [
        "delegate_task",
        "subagent_manage",
        "valstorm_sql_query",
        "valstorm_record_cud",
        "valstorm_record_merge",
        "valstorm_schema_inspect",
        "valstorm_generate_dashboard",
        "valstorm_send_sms",
        "valstorm_create_drip_cadence",
        "valstorm_vfs_search",
        "valstorm_vfs_write_file",
        "platform_methods_search",
        "platform_invoke",
        "terminal_exec",
        "read_file",
        "write_file",
        "patch_file",
        "search_files",
        "generate_image",
        "generate_video",
        "analyze_image",
        "confirmation_required",
        "clarify",
        "skill_list",
        "skill_view",
        "memory_manage",
        "web_search",
        "web_scrape",
        "computer_use",
    ]
    for name in core_names:
        def _mk(n: str):
            def _fn(**kwargs):
                return f"ok:{n}"
            _fn.__name__ = n
            return _fn
        reg.register(_mk(name), name=name, description=f"Core {name}")

    plugin_samples = [
        ("quickbooks_list_invoices", "quickbooks"),
        ("quickbooks_delete_invoice", "quickbooks"),
        ("stripe_list_subscriptions", "stripe"),
        ("billing_get_overview", "billing"),
        ("meta_create_ad_campaign", "valstorm-plugin-meta"),
        ("linkedin_publish_post", "valstorm-plugin-linkedin"),
        ("marketing_publish_social", "marketing"),
        ("sendgrid_send_email", "valstorm-plugin-sendgrid"),
        ("list_ads_campaigns", "google-marketing"),
        ("google_youtube_get_analytics", "google-marketing"),
        ("azure_deploy_valstorm_stack", "azure"),
        ("cloudflare_delete_dns_record", "cloudflare"),
        ("digitalocean_scale_app", "digitalocean"),
        ("gcp_bigquery_run_query", "valstorm-plugin-gcp"),
        ("gcp_verify_connection", "valstorm-plugin-gcp"),
        ("github_create_pull_request", "github"),
        ("salesforce_delete_records", "salesforce"),
        ("hubspot_search_objects", "hubspot"),
        ("telnyx_send_sms", "valstorm-plugin-telnyx"),
        ("twilio_send_sms", "twilio"),
        ("deepgram_transcribe_vfs_file", "valstorm-plugin-deepgram"),
        ("booking_get_availability", "valstorm-plugin-booking"),
        ("google_workspace_send_email", "valstorm-plugin-google-workspace"),
        ("microsoft_create_teams_meeting", "microsoft"),
        ("slack_post_message", "slack"),
        ("email_send", "email"),
    ]
    for tool_name, plugin_name in plugin_samples:
        fn = make_remote_plugin_tool(
            tool_name=tool_name,
            tool_desc=f"Plugin tool {tool_name}",
            tool_params={"type": "object", "properties": {}},
            base_url="https://api.valstorm.com/v1",
            valstorm_token="pat_test_token",
            plugin_name=plugin_name,
        )
        reg.register(fn)

    return reg


def test_plugin_pattern_matching_normalizes_valstorm_plugin_prefix():
    assert matches_tool_pattern("linkedin_publish_post", "plugin:linkedin", "valstorm-plugin-linkedin")
    assert matches_tool_pattern("linkedin_publish_post", "plugin:valstorm-plugin-linkedin", "valstorm-plugin-linkedin")
    assert matches_tool_pattern("meta_create_ad_campaign", "plugin:meta", "valstorm-plugin-meta")
    assert matches_tool_pattern("list_ads_campaigns", "plugin:google-marketing", "google-marketing")
    assert matches_tool_pattern("list_ads_campaigns", "plugin:google-marketing", None)
    assert matches_tool_pattern("google_ads_optimize_bids_and_budgets", "plugin:google-marketing", None)
    assert matches_tool_pattern("google_gsc_inspect_url", "plugin:google-marketing", None)
    assert matches_tool_pattern("google_ga4_verify_mp_event", "plugin:google-marketing", None)
    assert matches_tool_pattern("google_gtm_list_tags", "plugin:google-marketing", None)
    assert matches_tool_pattern("gcp_bigquery_run_query", "gcp_bigquery_*", "valstorm-plugin-gcp")
    assert not matches_tool_pattern("azure_deploy_valstorm_stack", "plugin:quickbooks", "azure")


def test_top_level_fleet_strict_plugin_isolation():
    full = _build_synthetic_full_registry()

    cos = set(full.filter_by_whitelist(BUILTIN_PROFILES["chief-of-staff"]["allowed_tools"]).list_tools())
    assert "delegate_task" in cos
    assert "valstorm_generate_dashboard" in cos
    assert "quickbooks_delete_invoice" not in cos
    assert "azure_deploy_valstorm_stack" not in cos
    assert "cloudflare_delete_dns_record" not in cos
    assert "salesforce_delete_records" not in cos
    assert "platform_invoke" not in cos

    finance = set(full.filter_by_whitelist(BUILTIN_PROFILES["finance"]["allowed_tools"]).list_tools())
    assert {"quickbooks_list_invoices", "quickbooks_delete_invoice", "stripe_list_subscriptions", "billing_get_overview"} <= finance
    assert "azure_deploy_valstorm_stack" not in finance
    assert "meta_create_ad_campaign" not in finance
    assert "salesforce_delete_records" not in finance

    marketer = set(full.filter_by_whitelist(BUILTIN_PROFILES["marketer"]["allowed_tools"]).list_tools())
    assert {
        "meta_create_ad_campaign",
        "linkedin_publish_post",
        "marketing_publish_social",
        "sendgrid_send_email",
        "list_ads_campaigns",
        "google_youtube_get_analytics",
        "email_send",
        "generate_image",
    } <= marketer
    assert "quickbooks_delete_invoice" not in marketer
    assert "azure_deploy_valstorm_stack" not in marketer
    assert "cloudflare_delete_dns_record" not in marketer

    revops = set(full.filter_by_whitelist(BUILTIN_PROFILES["revops"]["allowed_tools"]).list_tools())
    assert {
        "salesforce_delete_records",
        "hubspot_search_objects",
        "telnyx_send_sms",
        "twilio_send_sms",
        "deepgram_transcribe_vfs_file",
        "booking_get_availability",
        "valstorm_send_sms",
    } <= revops
    assert "azure_deploy_valstorm_stack" not in revops
    assert "quickbooks_delete_invoice" not in revops

    devops = set(full.filter_by_whitelist(BUILTIN_PROFILES["devops"]["allowed_tools"]).list_tools())
    assert {
        "azure_deploy_valstorm_stack",
        "cloudflare_delete_dns_record",
        "digitalocean_scale_app",
        "gcp_bigquery_run_query",
        "github_create_pull_request",
    } <= devops
    assert "quickbooks_delete_invoice" not in devops
    assert "meta_create_ad_campaign" not in devops

    comms = set(full.filter_by_whitelist(BUILTIN_PROFILES["workspace-comms"]["allowed_tools"]).list_tools())
    assert {
        "google_workspace_send_email",
        "microsoft_create_teams_meeting",
        "slack_post_message",
        "email_send",
        "booking_get_availability",
    } <= comms
    assert "quickbooks_delete_invoice" not in comms
    assert "azure_deploy_valstorm_stack" not in comms

    analyst = set(full.filter_by_whitelist(BUILTIN_PROFILES["data-analyst"]["allowed_tools"]).list_tools())
    assert {"gcp_bigquery_run_query", "gcp_verify_connection", "valstorm_sql_query", "platform_invoke"} <= analyst
    assert "valstorm_record_cud" not in analyst
    assert "azure_deploy_valstorm_stack" not in analyst

    se = set(full.filter_by_whitelist(BUILTIN_PROFILES["software-engineer"]["allowed_tools"]).list_tools())
    assert {"github_create_pull_request", "terminal_exec", "read_file", "patch_file"} <= se
    assert "valstorm_send_sms" not in se
    assert "valstorm_create_drip_cadence" not in se
    assert "quickbooks_list_invoices" not in se


def test_filter_by_whitelist_negated_exclusions():
    full = _build_synthetic_full_registry()
    scoped = set(full.filter_by_whitelist(["plugin:quickbooks", "!*_delete_*"]).list_tools())
    assert "quickbooks_list_invoices" in scoped
    assert "quickbooks_delete_invoice" not in scoped


def test_legacy_profile_aliases_resolve_cleanly(tmp_path, monkeypatch):
    monkeypatch.setenv("VALSTORM_PROFILES_DIR", str(tmp_path))
    expected_mappings = {
        "developer": "valstorm-developer",
        "growth-marketer": "marketer",
        "marketing-specialist": "marketer",
        "finance-ops": "finance",
        "revops-crm": "revops",
        "cloud-infra-devops": "devops",
        "slack-agent": "workspace-comms",
        "orchestrator": "chief-of-staff",
        "browser-scraper": "researcher",
        "archivist": "researcher",
    }
    for legacy, target in expected_mappings.items():
        assert PROFILE_ALIASES[legacy] == target
        loaded = load_profile(legacy)
        assert loaded["api_name"] == target


@pytest.mark.asyncio
async def test_delegated_subagent_receives_only_scoped_remote_plugin_tools(tmp_path, monkeypatch):
    """When Chief of Staff delegates to 'finance', the child receives QuickBooks/Stripe/Billing tools and NO Azure/Meta tools."""
    from tools.delegation import (
        SubagentRegistry,
        delegate_task,
        set_current_subagent_registry,
        set_subagent_provider_override,
    )

    clear_remote_plugin_tools_cache()
    monkeypatch.setenv("VALSTORM_PROFILES_DIR", str(tmp_path))

    captured_tool_names: list[str] = []

    class _CapturingProvider:
        provider_name = "mock"

        async def generate(self, messages, tools=None, model=None, **kwargs):
            for t in (tools or []):
                name = t.get("name") or (t.get("function") or {}).get("name")
                if name:
                    captured_tool_names.append(name)
            return (
                Message(role="assistant", content="Finance audit complete."),
                UsageMetadata(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

    remote_catalog = [
        {"name": "quickbooks_list_invoices", "plugin": "quickbooks", "description": "List QB invoices", "parameters": {"type": "object", "properties": {}}},
        {"name": "stripe_list_customers", "plugin": "stripe", "description": "List Stripe customers", "parameters": {"type": "object", "properties": {}}},
        {"name": "billing_get_overview", "plugin": "billing", "description": "Billing overview", "parameters": {"type": "object", "properties": {}}},
        {"name": "azure_deploy_valstorm_stack", "plugin": "azure", "description": "Deploy Azure stack", "parameters": {"type": "object", "properties": {}}},
        {"name": "meta_create_ad_campaign", "plugin": "valstorm-plugin-meta", "description": "Create Meta ad", "parameters": {"type": "object", "properties": {}}},
    ]

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"tools": remote_catalog}

    fake_client = MagicMock()
    fake_client.__enter__.return_value = fake_client
    fake_client.__exit__.return_value = None
    fake_client.get.return_value = fake_resp

    tok = set_current_subagent_registry(SubagentRegistry())
    set_subagent_provider_override(_CapturingProvider())
    try:
        with patch("tools.valstorm_client.resolve_valstorm_credentials", return_value=("pat_test_123", "https://api.valstorm.com/v1")), \
             patch("tools.remote_plugins.httpx.Client", return_value=fake_client):
            result = await delegate_task(
                profile="finance",
                goal="Audit open QuickBooks AR and Stripe subscriptions.",
                background=False,
            )
            assert "Finance audit complete." in result
            assert "quickbooks_list_invoices" in captured_tool_names
            assert "stripe_list_customers" in captured_tool_names
            assert "billing_get_overview" in captured_tool_names
            assert "azure_deploy_valstorm_stack" not in captured_tool_names
            assert "meta_create_ad_campaign" not in captured_tool_names
    finally:
        set_subagent_provider_override(None)
        clear_remote_plugin_tools_cache()
        from tools.delegation import _current_subagent_registry
        _current_subagent_registry.reset(tok)
