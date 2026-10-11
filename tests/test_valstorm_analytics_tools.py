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
