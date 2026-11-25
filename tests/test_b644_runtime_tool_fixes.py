"""Comprehensive unit tests for B-644:
- Surface API JSON `detail` on HTTP 4xx/5xx responses
- `valstorm_mongo_query` removed 100%
- `valstorm_analytics_compute` wired to POST /v2/analytics/query
- `records_delete` sends comma-separated IDs (batch delete)
- `schema_get` TTL caching + invalidation & zero-network `valstorm_sql_query`
- `tool_healer` supports Open Valstorm `properties` schema key
- `create_list_filter` / `generate_report` / `generate_dashboard` schema alignment
"""

import json
import httpx
import pytest
from unittest.mock import AsyncMock

from core.context import BUILTIN_PROFILES
from core.react import SPECULATIVE_SAFE_TOOLS
from core.tool_healer import heal_record_payload_s1, heal_sql_query_s1
from tools.valstorm_client import ValstormApiClient
from tools.valstorm_tools import create_valstorm_tools


@pytest.mark.asyncio
async def test_http_400_surfaces_api_json_detail():
    """ValstormApiClient._request_with_retry must preserve the API's JSON detail body on 400s."""
    detail_text = (
        "Column(s) do not exist on object 'dashboard': app_id. "
        "Did you mean: app_id -> (dashboard has no app_id column; use id, name, layout, layout_items)?. "
        "Available columns on 'dashboard': [created_date, id, layout, layout_items, name, owner]"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": detail_text}, request=request)

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://api-service/v1",
    )
    client = ValstormApiClient(token="tok_test", base_url="http://api-service/v1", client=http_client)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.sql_query("SELECT id, name, app_id FROM dashboard LIMIT 10")

    err_str = str(exc_info.value)
    assert "400" in err_str
    assert "app_id" in err_str
    assert "Available columns on 'dashboard'" in err_str


@pytest.mark.asyncio
async def test_analytics_compute_routes_to_v2_analytics_query():
    """analytics_compute must call POST /v2/analytics/query instead of removed /v1/analytics/report."""
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append((request.method, str(request.url), json.loads(request.content.decode())))
        return httpx.Response(
            200,
            json={"rows": [{"status": "Active", "count": 42}], "meta": {"engine": "v2"}},
            request=request,
        )

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://api-service/v1",
    )
    client = ValstormApiClient(token="tok_test", base_url="http://api-service/v1", client=http_client)

    # 1. Bar chart with WHERE clause + x/y fields (as passed by valstorm_analytics_compute tool)
    res = await client.analytics_compute(
        query="SELECT id, status, loan_amount FROM lead WHERE created_date = 'this_year'",
        chart_type="Bar",
        x=[{"key": "status", "groupByDateTime": None}],
        y=[{"key": "loan_amount", "aggregate": "Sum"}],
        aggregate="sum",
        bypass_cache=True,
    )
    assert res["meta"]["engine"] == "v2"
    method, url, body = captured[-1]
    assert method == "POST"
    assert url == "http://api-service/v2/analytics/query"
    assert body["source"] == "lead"
    assert body["dimensions"] == [{"field": "status"}]
    assert body["measures"][0]["agg"] == "sum"
    assert body["measures"][0]["field"] == "loan_amount"
    assert body["bypass_cache"] is True
    assert "WHERE created_date = 'this_year'" in body["base_sql"]

    # 2. Pie chart passes dimension via `field`
    await client.analytics_compute(
        query="SELECT id, lead_source FROM lead",
        chart_type="Pie",
        field=[{"key": "lead_source", "aggregate": "Count"}],
        aggregate="count",
    )
    _, url2, body2 = captured[-1]
    assert url2 == "http://api-service/v2/analytics/query"
    assert body2["source"] == "lead"
    assert body2["dimensions"] == [{"field": "lead_source"}]
    assert body2["measures"][0]["agg"] == "count"

    # 3. Cumulative Line chart sets window='cumulative'
    await client.analytics_compute(
        query="SELECT id, created_date, loan_amount FROM loan",
        chart_type="Line",
        x=[{"key": "created_date", "groupByDateTime": "Month"}],
        y=[{"key": "loan_amount", "aggregate": "Cumulative"}],
        aggregate="cumulative",
        group_by_date_time="Month",
    )
    _, _, body3 = captured[-1]
    assert body3["dimensions"] == [{"field": "created_date", "time_grain": "month", "fill_gaps": True}]
    assert body3["measures"][0]["window"] == "cumulative"
    assert body3["measures"][0]["agg"] == "sum"


@pytest.mark.asyncio
async def test_records_delete_sends_comma_separated_ids():
    """records_delete must send comma-separated ids so all IDs are deleted, not only the last one."""
    captured_urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        return httpx.Response(204, request=request)

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://api-service/v1",
    )
    client = ValstormApiClient(token="tok_test", base_url="http://api-service/v1", client=http_client)

    res = await client.records_delete("dashboard", ["dash_1", "dash_2", "dash_3", "dash_4"])
    assert res["status"] == "success"
    assert res["deleted_count"] == 4
    assert len(captured_urls) == 1
    assert "ids=dash_1%2Cdash_2%2Cdash_3%2Cdash_4" in captured_urls[0]

    # Empty list should not make a network call
    empty_res = await client.records_delete("dashboard", [])
    assert empty_res["deleted_count"] == 0
    assert len(captured_urls) == 1


@pytest.mark.asyncio
async def test_schema_get_caching_and_zero_network_sql_healing():
    """schema_get caches responses, invalidates on writes, and valstorm_sql_query does not fetch /schema."""
    call_log = []

    def handler(request: httpx.Request) -> httpx.Response:
        call_log.append((request.method, request.url.path))
        if request.url.path == "/v1/schema":
            return httpx.Response(
                200,
                json={
                    "file": {
                        "id": "obj_file",
                        "api_name": "file",
                        "properties": {"name": {"type": "string"}, "vaults": {"type": "list"}},
                    }
                },
                request=request,
            )
        if request.url.path == "/v1/schema" and request.method == "POST":
            return httpx.Response(200, json={"id": "obj_new"}, request=request)
        if request.url.path == "/v1/query":
            body = json.loads(request.content.decode())
            return httpx.Response(200, json={"records": [{"executed_query": body["query"]}]}, request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://api-service/v1",
    )
    client = ValstormApiClient(token="tok_test", base_url="http://api-service/v1", client=http_client)
    tools = {t.__name__: t for t in create_valstorm_tools(client=client)}

    # 1. Cold cache: valstorm_sql_query makes ONLY 1 HTTP call (POST /v1/query), NO GET /v1/schema
    await tools["valstorm_sql_query"](query="SELECT id, name FROM file LIMIT 5")
    assert call_log == [("POST", "/v1/query")]

    # 2. Warm cache via schema_get: second call hits cache (no second GET /v1/schema)
    await client.schema_get()
    await client.schema_get()
    await client.schema_get("file")
    assert call_log.count(("GET", "/v1/schema")) == 1
    assert call_log.count(("GET", "/v1/schema/file")) == 0

    # 3. With warm cache, valstorm_sql_query heals `filename` -> `name` using Open Valstorm `properties`
    res_json = await tools["valstorm_sql_query"](query="SELECT id, filename FROM file LIMIT 5")
    parsed = json.loads(res_json)
    assert "SELECT id, name FROM file LIMIT 5" == parsed[0]["executed_query"]
    assert call_log.count(("GET", "/v1/schema")) == 1

    # 4. Schema mutation invalidates cache
    await client.schema_create({"name": "CustomObj"})
    await client.schema_get()
    assert call_log.count(("GET", "/v1/schema")) == 2


def test_tool_healer_uses_open_valstorm_properties_key():
    """heal_record_payload_s1 and heal_sql_query_s1 must read `properties` (not only legacy `fields`)."""
    schemas = {
        "contact": {
            "api_name": "contact",
            "properties": {
                "email": {"type": "string"},
                "phone": {"type": "string"},
                "owner": {"type": "string", "format": "lookup"},
            },
        }
    }
    healed_rec, notes = heal_record_payload_s1(
        {"email_address": "a@b.com", "owner": {"id": "user_123", "name": "Jane"}},
        schemas["contact"],
    )
    assert healed_rec == {"email": "a@b.com", "owner": "user_123"}
    assert any("email_address" in n for n in notes)

    healed_sql, sql_notes = heal_sql_query_s1("SELECT id, email_address FROM contacts LIMIT 10", schemas)
    assert "FROM contact" in healed_sql
    assert "email" in healed_sql
    assert "email_address" not in healed_sql
    assert len(sql_notes) >= 2


@pytest.mark.asyncio
async def test_create_list_filter_and_report_dashboard_schema_alignment():
    """create_list_filter writes real list_filter fields; report/dashboard omit non-existent `app`."""
    client = ValstormApiClient(token="tok_test", base_url="http://api-service/v1")
    client.schema_get = AsyncMock(
        return_value={
            "id": "obj_lead",
            "api_name": "lead",
            "properties": {"name": {"type": "string"}, "status": {"type": "string"}},
        }
    )
    captured_creates = []

    async def fake_create(api_name, records):
        captured_creates.append((api_name, records))
        return [{**records[0], "id": f"{api_name[:4]}_1"}]

    client.records_create = AsyncMock(side_effect=fake_create)

    await client.create_list_filter(
        name="Active Leads",
        object_api_name="lead",
        sql_query="SELECT * FROM lead WHERE status = 'Active'",
        display_fields=["name", "status"],
        is_pinned=True,
        is_default=False,
        app_id="app_123",
    )
    lf_api, lf_recs = captured_creates[-1]
    assert lf_api == "list_filter"
    lf = lf_recs[0]
    assert lf["list_fields"] == ["name", "status"]
    assert lf["table_columns"] == [{"field": "name"}, {"field": "status"}]
    assert lf["settings"] == {"pinned": True, "is_default": False, "app_id": "app_123"}
    for forbidden in ("display_fields", "pinned", "is_default", "app"):
        assert forbidden not in lf

    await client.generate_report(
        name="Leads by Status",
        object_api_name="lead",
        sql_query="SELECT id, status FROM lead",
        app_id="app_123",
    )
    rep_api, rep_recs = captured_creates[-1]
    assert rep_api == "report"
    assert "app" not in rep_recs[0]
    assert "app_id" not in rep_recs[0]


def test_valstorm_mongo_query_is_completely_removed():
    """valstorm_mongo_query must not exist in tools, client, profiles, or read-only sets."""
    client = ValstormApiClient(token="tok_test", base_url="http://api-service/v1")
    assert not hasattr(client, "mongo_query")

    tools = {t.__name__: t for t in create_valstorm_tools(client=client)}
    assert "valstorm_mongo_query" not in tools
    assert "valstorm_mongo_query" not in SPECULATIVE_SAFE_TOOLS

    for slug, prof in BUILTIN_PROFILES.items():
        assert "valstorm_mongo_query" not in (prof.get("allowed_tools") or []), f"Found in {slug}"
        assert "valstorm_mongo_query" not in (prof.get("system_prompt") or ""), f"Found in {slug} prompt"
