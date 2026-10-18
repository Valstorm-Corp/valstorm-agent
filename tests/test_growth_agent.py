"""Unit and integration tests for the Multi-Horizon ReAct Growth Agent and SRE/Growth Worker."""

import json
from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient

from core.growth_agent import (
    GrowthActionProposal,
    GrowthAgentReviewResult,
    build_deterministic_growth_review,
    build_growth_investigation_registry,
    classify_ga4_page_vertical,
    classify_google_ads_campaign,
    classify_instantly_campaign,
    classify_meta_ads_campaign,
    format_growth_review_slack_blocks,
    run_growth_react_review,
    select_primary_ads_campaign,
)
from sre_worker import app


def test_classify_google_ads_campaign_verticals():
    """Verify Enterprise Greenfield, CT Local General, and CT Contractors classify distinctly."""
    gf = classify_google_ads_campaign("[BOFU] 30-Day Greenfield Sprint ($25k-$40k)")
    assert gf["vertical"] == "Enterprise Greenfield"
    assert gf["short_code"] == "BOFU_GREENFIELD"

    ct_gen = classify_google_ads_campaign("[BOFU] CT Local General Services (0.8s Speed & Ad Audit)")
    assert ct_gen["vertical"] == "CT Local General Services"
    assert ct_gen["short_code"] == "BOFU_CT_GENERAL"

    ct_con = classify_google_ads_campaign("[BOFU] Contractors & Home Services (Roofing, HVAC, Trades)")
    assert ct_con["vertical"] == "CT Contractors & Trades"
    assert ct_con["short_code"] == "BOFU_CT_CONTRACTORS"


def test_classify_meta_ads_campaign_verticals_and_roles():
    """Verify Meta ABM Air Cover and BOFU Conversion Engines classify into their verticals and roles."""
    abm = classify_meta_ads_campaign("[ABM AIR COVER] CT 1,018 Licensed Contractors ($8/day Reach)", "OUTCOME_AWARENESS")
    assert abm["vertical"] == "CT Contractors & Trades"
    assert abm["short_code"] == "META_ABM_AIR_COVER"
    assert "ABM" in abm["funnel_stage"]

    ct_con = classify_meta_ads_campaign("[BOFU] CT Contractors & Trades Conversion Engine", "OUTCOME_LEADS")
    assert ct_con["vertical"] == "CT Contractors & Trades"
    assert ct_con["short_code"] == "META_BOFU_CT_CONTRACTORS"
    assert "BOFU" in ct_con["funnel_stage"]

    ct_gen = classify_meta_ads_campaign("[BOFU] CT Local General Services Conversion Engine", "OUTCOME_LEADS")
    assert ct_gen["vertical"] == "CT Local General Services"
    assert ct_gen["short_code"] == "META_BOFU_CT_GENERAL"


def test_select_primary_ads_campaign_does_not_overwrite_with_paused():
    """Regression test: adding paused [BOFU] CT campaigns must not overwrite the active Greenfield campaign."""
    campaigns = [
        {
            "id": "24336205138",
            "name": "[BOFU] 30-Day Greenfield Sprint ($25k-$40k)",
            "status": "ENABLED",
            "cost_usd": 647.96,
            "clicks": 50,
            "impressions": 1724,
            "ctr_pct": 2.90,
            "avg_cpc_usd": 12.96,
            "conversions": 0.0,
        },
        {
            "id": "24357654402",
            "name": "[BOFU] CT Local General Services (0.8s Speed & Ad Audit)",
            "status": "PAUSED",
            "cost_usd": 0.0,
            "clicks": 0,
            "impressions": 0,
            "ctr_pct": 0.0,
            "avg_cpc_usd": 0.0,
            "conversions": 0.0,
        },
        {
            "id": "24357673347",
            "name": "[BOFU] Contractors & Home Services (Roofing, HVAC, Trades)",
            "status": "PAUSED",
            "cost_usd": 0.0,
            "clicks": 0,
            "impressions": 0,
            "ctr_pct": 0.0,
            "avg_cpc_usd": 0.0,
            "conversions": 0.0,
        },
    ]

    primary = select_primary_ads_campaign(campaigns)
    assert primary["id"] == "24336205138"
    assert primary["status"] == "ENABLED"
    assert primary["cost_usd"] == 647.96
    assert primary["clicks"] == 50
    assert primary["active_campaigns_count"] == 1
    assert primary["paused_campaigns_count"] == 2


def test_classify_instantly_campaign_angles_and_ct_tiers():
    """Verify CT Trades Tier 1A/1B/1C/2 and Enterprise Sprint campaigns get distinct angle codes."""
    t1a = classify_instantly_campaign("[CT Trades] Tier 1A - Bleeding on Ads")
    assert t1a["angle"] == "CT_TIER_1A"
    assert t1a["vertical"] == "CT Contractors & Trades"

    t2 = classify_instantly_campaign("[CT Trades] Tier 2 - Tracking & RevOps Gap")
    assert t2["angle"] == "CT_TIER_2"
    assert t2["vertical"] == "CT Contractors & Trades"

    hire = classify_instantly_campaign("[Valstorm] [Sprint] - HIRE_ARB - TechOps - 2026-10")
    assert hire["angle"] == "HIRE_ARB"
    assert hire["vertical"] == "Enterprise Greenfield"


def test_classify_ga4_page_vertical():
    """Verify GA4 landing paths map to their respective verticals and commercial intent."""
    c_page = classify_ga4_page_vertical("/ad-conversion-audit/contractors")
    assert c_page["vertical"] == "CT Contractors & Trades"
    assert c_page["is_high_intent"] is True

    g_page = classify_ga4_page_vertical("/ad-conversion-audit")
    assert g_page["vertical"] == "CT Local General Services"
    assert g_page["is_high_intent"] is True

    e_page = classify_ga4_page_vertical("/book-architecture-audit")
    assert e_page["vertical"] == "Enterprise Greenfield"
    assert e_page["is_high_intent"] is True


@pytest.mark.asyncio
async def test_growth_investigation_tools_and_deterministic_review():
    """Verify the ReAct Growth Agent tools and deterministic review builder."""
    ads_data = {
        "total_spend_usd": 647.96,
        "total_clicks": 50,
        "total_impressions": 1724,
        "waste_spend_usd": 63.77,
        "waste_clicks": 5,
        "campaigns": [
            {
                "id": "1",
                "name": "[BOFU] 30-Day Greenfield Sprint ($25k-$40k)",
                "vertical": "Enterprise Greenfield",
                "status": "ENABLED",
                "cost_usd": 647.96,
                "clicks": 50,
                "avg_cpc_usd": 12.96,
            },
            {
                "id": "2",
                "name": "[BOFU] CT Local General Services (0.8s Speed & Ad Audit)",
                "vertical": "CT Local General Services",
                "status": "PAUSED",
                "cost_usd": 0.0,
                "clicks": 0,
                "avg_cpc_usd": 0.0,
            },
        ],
        "search_terms": [
            {"search_term": "filemaker software", "clicks": 1, "cost_usd": 17.15, "is_waste": True},
        ],
        "horizons": {
            "yesterday": {"spend_usd": 0.0, "clicks": 0},
            "7d": {"spend_usd": 587.67, "clicks": 41},
            "30d": {"spend_usd": 647.96, "clicks": 50},
            "all_time": {"spend_usd": 647.96, "clicks": 50},
        },
    }
    ga4_data = {
        "active_users": 625,
        "high_intent_page_views": 300,
        "top_pages": [
            {"path": "/ad-conversion-audit/contractors", "pageviews": 59, "users": 49, "avg_duration_sec": 14.3, "vertical": "CT Contractors & Trades"},
        ],
        "funnel_events": {"qualify_lead": {"count": 0}, "form_start": {"count": 0}},
        "horizons": {
            "yesterday": {"active_users": 94, "high_intent_views": 73},
            "7d": {"active_users": 625, "high_intent_views": 300},
            "30d": {"active_users": 1076, "high_intent_views": 452},
            "all_time": {"active_users": 5500, "high_intent_views": 576},
        },
    }
    gsc_data = {"queries": [], "horizons": {}}
    instantly_data = {
        "active_campaigns_count": 3,
        "paused_campaigns_count": 4,
        "total_sent": 540,
        "total_replies": 3,
        "inboxes_active": 35,
        "inboxes_total": 35,
        "campaigns": [
            {"name": "[CT Trades] Tier 1A - Bleeding on Ads", "angle": "CT_TIER_1A", "vertical": "CT Contractors & Trades", "status_code": 0, "leads_count": 0},
        ],
    }

    proposals_sink = []
    trace_sink = []
    meta_data = {
        "ad_account_id": "act_1022302937529342",
        "active_campaigns_count": 0,
        "paused_campaigns_count": 3,
        "total_adsets_count": 4,
        "total_ads_count": 10,
        "configured_daily_budget_usd": 73.0,
        "total_spend_usd": 0.0,
        "total_clicks": 0,
        "total_leads": 0,
        "campaigns": [
            {
                "id": "120250981486420465",
                "name": "[ABM AIR COVER] CT 1,018 Licensed Contractors ($8/day Reach)",
                "vertical": "CT Contractors & Trades",
                "funnel_stage": "ABM Air Cover (Reach)",
                "status": "PAUSED",
                "daily_budget_usd": 8.0,
                "adsets_count": 1,
                "ads_count": 2,
                "cost_usd": 0.0,
                "clicks": 0,
                "leads": 0,
            }
        ],
        "custom_audiences": [
            {"id": "ca_1", "name": "VS - CT Licensed Contractors 1,018 (Value-Weighted)", "subtype": "CUSTOM"},
        ],
        "horizons": {
            "yesterday": {"spend_usd": 0.0, "clicks": 0, "leads": 0},
            "7d": {"spend_usd": 0.0, "clicks": 0, "leads": 0},
            "30d": {"spend_usd": 0.0, "clicks": 0, "leads": 0},
            "all_time": {"spend_usd": 0.0, "clicks": 0, "leads": 0},
        },
    }
    ga4_data["traffic_sources"] = [
        {"source": "meta", "medium": "paid_social", "users": 38, "sessions": 38, "engaged_sessions": 38, "avg_duration_sec": 11.5},
    ]

    registry = build_growth_investigation_registry(
        ads_data=ads_data,
        ga4_data=ga4_data,
        gsc_data=gsc_data,
        instantly_data=instantly_data,
        proposals_sink=proposals_sink,
        trace_sink=trace_sink,
        meta_data=meta_data,
    )

    res_meta = await registry.execute_async("inspect_meta_ads_campaigns_and_audiences", {"vertical": "all"})
    assert not res_meta.is_error
    parsed_meta = json.loads(res_meta.output)
    assert parsed_meta["paused_campaigns_count"] == 3
    assert parsed_meta["total_ads_count"] == 10
    assert parsed_meta["configured_daily_budget_usd"] == 73.0

    res_ads = await registry.execute_async("inspect_ads_campaigns_and_horizons", {"vertical": "all"})
    assert not res_ads.is_error
    parsed_ads = json.loads(res_ads.output)
    assert parsed_ads["active_campaigns_count"] == 1
    assert parsed_ads["paused_campaigns_count"] == 1

    res_prop = await registry.execute_async(
        "propose_growth_action",
        {
            "action_id": "A1",
            "category": "google_ads",
            "vertical": "Enterprise Greenfield",
            "priority": "P1",
            "title": "Add FileMaker negative keyword",
            "rationale": "$17.15 wasted on filemaker software",
            "suggested_execution": "Add phrase negative 'filemaker'",
        },
    )
    assert not res_prop.is_error
    assert len(proposals_sink) == 1
    assert proposals_sink[0].action_id == "A1"

    review = build_deterministic_growth_review(
        ads_data=ads_data,
        ga4_data=ga4_data,
        gsc_data=gsc_data,
        instantly_data=instantly_data,
        meta_data=meta_data,
        existing_proposals=proposals_sink,
        tool_trace=trace_sink,
    )
    assert isinstance(review, GrowthAgentReviewResult)
    assert any("Meta Ad Review Crawler" in a for a in review.anomalies_and_bottlenecks)
    assert len(review.proposals) == 1
    blocks = format_growth_review_slack_blocks(review)
    assert blocks[0]["type"] == "header"
    assert any("[A1]" in json.dumps(b) for b in blocks)


def test_sre_worker_health_and_slack_growth_ticket():
    """Verify sre_worker endpoints including Slack @Valstorm Bot create ticket A1."""
    client = TestClient(app)

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    h_data = health_resp.json()
    assert h_data["status"] == "ok"
    assert "valstorm:queue:growth" in h_data["queues"]

    mock_growth_ctx = {
        "agent_review": {
            "proposals": [
                {
                    "action_id": "A1",
                    "category": "google_ads",
                    "vertical": "Enterprise Greenfield",
                    "priority": "P1",
                    "title": "Negate FileMaker Queries",
                    "rationale": "$17.15 waste on filemaker",
                    "suggested_execution": "Add negative keyword 'filemaker'",
                }
            ]
        }
    }

    with (
        patch("sre_worker._load_cached_growth_context", new=AsyncMock(return_value=mock_growth_ctx)),
        patch(
            "sre_worker.create_bug_ticket_in_db",
            new=AsyncMock(
                return_value={
                    "id": "tick_growth123",
                    "number": 412,
                    "name": "B-412 [Growth A1] Negate FileMaker Queries",
                    "url": "valstorm://record/ticket/tick_growth123/standard",
                }
            ),
        ),
        patch("sre_worker._post_slack_thread_reply", new=AsyncMock()),
    ):
        slack_resp = client.post(
            "/v1/sre/slack-events",
            json={
                "event": {
                    "type": "app_mention",
                    "text": "<@U12345> create ticket A1",
                    "channel": "C0C8WBKV8PJ",
                    "thread_ts": "1729260000.000100",
                    "user": "U99999",
                }
            },
        )
        assert slack_resp.status_code == 200
        s_data = slack_resp.json()
        assert s_data["ok"] is True
        assert s_data["ticket_id"] == "tick_growth123"
        assert s_data["action_id"] == "A1"
