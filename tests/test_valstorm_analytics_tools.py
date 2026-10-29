import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from tools.valstorm_client import ValstormApiClient
from tools.valstorm_tools import create_valstorm_tools


@pytest.fixture
def mock_valstorm_client():
    client = MagicMock(spec=ValstormApiClient)
    client.schema_get = AsyncMock(return_value={
        "id": "obj_lead_123",
        "api_name": "lead",
        "properties": {
            "name": {"type": "string"},
            "status": {"type": "string"},
            "loan_amount": {"type": "number"},
            "owner": {"format": "lookup"},
            "created_date": {"format": "date-time"},
        }
    })
    client.records_create = AsyncMock(return_value=[{
        "id": "lifi_test_123",
        "name": "High Value Leads (30d)",
        "query": "SELECT * FROM lead WHERE created_date = last_30_days",
    }])
    client.create_list_filter = AsyncMock(return_value=[{
        "id": "lifi_test_123",
        "name": "High Value Leads (30d)",
    }])
    client.generate_report = AsyncMock(return_value=[{
        "id": "repo_test_456",
        "name": "Revenue by Lead Source",
    }])
    client.generate_dashboard = AsyncMock(return_value={
        "id": "dash_test_789",
        "name": "Lead & Sales Velocity",
        "url": "/view/dashboard/record/dash_test_789/standard",
        "widgets_count": 2,
        "reports_created": [{"id": "rep_1", "name": "New Leads"}],
    })
    client.analytics_compute = AsyncMock(return_value={
        "chartType": "Bar",
        "calculated_data": [{"label": "Website", "value": 150000}],
        "summary": {"total": 150000}
    })
    return client


@pytest.mark.asyncio
async def test_valstorm_create_list_filter_tool(mock_valstorm_client):
    """Verify that valstorm_create_list_filter invokes the client and parses fields."""
    tools = create_valstorm_tools(client=mock_valstorm_client)
    tool_map = {t.__name__: t for t in tools}
    create_filter_fn = tool_map["valstorm_create_list_filter"]

    res_str = await create_filter_fn(
        name="High Value Leads (30d)",
        object_api_name="lead",
        sql_query="SELECT * FROM lead WHERE created_date = last_30_days",
        display_fields=["name", "status", "loan_amount"],
        is_pinned=True,
    )
    res = json.loads(res_str)
    assert res[0]["id"] == "lifi_test_123"
    mock_valstorm_client.create_list_filter.assert_called_once_with(
        name="High Value Leads (30d)",
        object_api_name="lead",
        sql_query="SELECT * FROM lead WHERE created_date = last_30_days",
        display_fields=["name", "status", "loan_amount"],
        is_pinned=True,
        is_default=False,
        app_id=None,
    )


@pytest.mark.asyncio
async def test_valstorm_generate_report_tool(mock_valstorm_client):
    """Verify that valstorm_generate_report formats chart settings and delegates to client."""
    tools = create_valstorm_tools(client=mock_valstorm_client)
    tool_map = {t.__name__: t for t in tools}
    generate_report_fn = tool_map["valstorm_generate_report"]

    res_str = await generate_report_fn(
        name="Revenue by Lead Source",
        object_api_name="deal",
        sql_query="SELECT lead_source, amount FROM deal WHERE stage = 'Closed Won'",
        chart_type="Bar",
        x_field="lead_source",
        y_field="amount",
        aggregation="Sum",
    )
    res = json.loads(res_str)
    assert res[0]["id"] == "repo_test_456"
    mock_valstorm_client.generate_report.assert_called_once_with(
        name="Revenue by Lead Source",
        object_api_name="deal",
        sql_query="SELECT lead_source, amount FROM deal WHERE stage = 'Closed Won'",
        chart_type="Bar",
        x_field="lead_source",
        y_field="amount",
        aggregation="Sum",
        group_by_time=None,
        app_id=None,
    )


@pytest.mark.asyncio
async def test_valstorm_analytics_compute_tool(mock_valstorm_client):
    """Verify that valstorm_analytics_compute computes statistical aggregation via API."""
    tools = create_valstorm_tools(client=mock_valstorm_client)
    tool_map = {t.__name__: t for t in tools}
    analytics_compute_fn = tool_map["valstorm_analytics_compute"]

    res_str = await analytics_compute_fn(
        sql_query="SELECT lead_source, amount FROM deal",
        chart_type="Bar",
        x_field="lead_source",
        y_field="amount",
        aggregate="Sum",
    )
    res = json.loads(res_str)
    assert res["summary"]["total"] == 150000
    mock_valstorm_client.analytics_compute.assert_called_once()


@pytest.mark.asyncio
async def test_valstorm_generate_dashboard_tool(mock_valstorm_client):
    """Verify that valstorm_generate_dashboard parses widgets and delegates to client."""
    tools = create_valstorm_tools(client=mock_valstorm_client)
    tool_map = {t.__name__: t for t in tools}
    generate_dash_fn = tool_map["valstorm_generate_dashboard"]

    widgets = [
        {
            "title": "New Leads (30d)",
            "object_api_name": "lead",
            "sql_query": "SELECT id, name FROM lead WHERE created_date = 'last_n_days:30'",
            "chart_type": "Metric",
            "y_field": "name",
            "aggregation": "Count",
            "size": "kpi",
        },
        {
            "title": "Leads by Source",
            "object_api_name": "lead",
            "sql_query": "SELECT lead_source, name FROM lead WHERE created_date = 'this_year'",
            "chart_type": "Bar",
            "x_field": "lead_source",
            "y_field": "name",
            "aggregation": "Count",
            "size": "half",
        },
    ]

    res_str = await generate_dash_fn(
        name="Lead & Sales Velocity",
        widgets=json.dumps(widgets),
    )
    res = json.loads(res_str)
    assert res["id"] == "dash_test_789"
    assert res["url"] == "/view/dashboard/record/dash_test_789/standard"
    mock_valstorm_client.generate_dashboard.assert_called_once_with(
        name="Lead & Sales Velocity",
        widgets=widgets,
        app_id=None,
        dashboard_id=None,
    )


@pytest.mark.asyncio
async def test_valstorm_client_generate_dashboard_grid_packing():
    """Verify 12-column responsive grid packing and schema/query wiring in ValstormApiClient.generate_dashboard."""
    client = ValstormApiClient(token="test_token", base_url="http://localhost:8010/v1")
    client.schema_get = AsyncMock(return_value={"id": "obj_MBZTa8B7SfCFsWQ8", "api_name": "lead"})

    created_reports = []
    created_dashboards = []

    async def fake_records_create(api_name, records):
        if api_name == "report":
            rep_id = f"rep_mock_{len(created_reports) + 1}"
            rec = {**records[0], "id": rep_id}
            created_reports.append(rec)
            return [rec]
        if api_name == "dashboard":
            dash = {**records[0], "id": "dash_packed_123"}
            created_dashboards.append(dash)
            return [dash]
        return []

    client.records_create = AsyncMock(side_effect=fake_records_create)

    widgets = [
        {"title": "KPI 1", "object_api_name": "lead", "sql_query": "SELECT * FROM lead", "chart_type": "Metric", "y_field": "name"},
        {"title": "KPI 2", "object_api_name": "lead", "sql_query": "SELECT * FROM lead", "chart_type": "Metric", "y_field": "name"},
        {"title": "KPI 3", "object_api_name": "lead", "sql_query": "SELECT * FROM lead", "chart_type": "Metric", "y_field": "name"},
        {"title": "KPI 4", "object_api_name": "lead", "sql_query": "SELECT * FROM lead", "chart_type": "Metric", "y_field": "name"},
        {"title": "Leads by Source", "object_api_name": "lead", "sql_query": "SELECT * FROM lead", "chart_type": "Bar", "x_field": "lead_source", "y_field": "name"},
        {"title": "Leads Trend", "object_api_name": "lead", "sql_query": "SELECT * FROM lead", "chart_type": "Line", "x_field": "created_date", "y_field": "name", "group_by_time": "Month"},
    ]

    result = await client.generate_dashboard(name="Executive Lead Flow", widgets=widgets)

    assert result["id"] == "dash_packed_123"
    assert result["url"] == "/view/dashboard/record/dash_packed_123/standard"
    assert result["widgets_count"] == 6
    assert len(created_reports) == 6
    # Schema should only be fetched once due to per-call schema caching
    client.schema_get.assert_called_once_with("lead")

    dash_payload = created_dashboards[0]
    layout = dash_payload["layout"]
    # First 4 KPI cards should pack across row y=0 with x=0, 3, 6, 9 and w=3, h=3
    assert [(item["x"], item["y"], item["w"], item["h"]) for item in layout[:4]] == [
        (0, 0, 3, 3),
        (3, 0, 3, 3),
        (6, 0, 3, 3),
        (9, 0, 3, 3),
    ]
    # Next 2 standard charts should pack across row y=3 with x=0, 6 and w=6, h=6
    assert [(item["x"], item["y"], item["w"], item["h"]) for item in layout[4:6]] == [
        (0, 3, 6, 6),
        (6, 3, 6, 6),
    ]

    # Verify Dashboard.tsx compatibility fields on layout_items
    first_comp = dash_payload["layout_items"][0]["component"]
    assert first_comp["schema"] == "obj_MBZTa8B7SfCFsWQ8"
    assert first_comp["object"] == "obj_MBZTa8B7SfCFsWQ8"
    assert first_comp["query"]["sql_query"] == "SELECT * FROM lead"
    assert first_comp["settings"]["labels"] is True
    assert first_comp["settings"]["title"] == "KPI 1"


def test_chief_of_staff_and_developer_profiles_include_dashboard_tool():
    """Verify Chief of Staff and Valstorm Developer profiles include valstorm_generate_dashboard."""
    from core.context import BUILTIN_PROFILES, load_profile

    for slug in ("chief-of-staff", "developer", "valstorm-developer"):
        prof = load_profile(slug)
        assert "valstorm_generate_dashboard" in prof["allowed_tools"], f"Missing in {slug}"
        assert "valstorm_generate_dashboard" in prof["system_prompt"], f"Prompt missing in {slug}"

    assert "valstorm_generate_dashboard" in BUILTIN_PROFILES["chief-of-staff"]["allowed_tools"]
    assert "valstorm_generate_dashboard" in BUILTIN_PROFILES["developer"]["allowed_tools"]

