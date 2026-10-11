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
    })
    client.create_scheduled_task = AsyncMock(return_value=[{
        "id": "item_sched_123",
        "name": "Weekly Report Task",
    }])
    client.create_drip_cadence = AsyncMock(return_value=[{
        "id": "drip_def_123",
        "name": "14-Day Inbound Nurture",
    }])
    client.enroll_in_drip = AsyncMock(return_value=[{
        "id": "drip_enr_123",
        "status": "Active",
    }])
    return client


@pytest.mark.asyncio
async def test_valstorm_create_scheduled_task_tool(mock_valstorm_client):
    """Verify valstorm_create_scheduled_task delegates to client for one-off and cron tasks."""
    tools = create_valstorm_tools(client=mock_valstorm_client)
    tool_map = {t.__name__: t for t in tools}
    create_task_fn = tool_map["valstorm_create_scheduled_task"]

    # Test One-off
    res_str = await create_task_fn(
        name="Lead 1042 Reminder",
        target_type="function",
        target_id="fun_send_sms",
        run_at_utc="2026-09-15T14:00:00Z",
        payload_data={"lead_id": "lead_1042"},
    )
    res = json.loads(res_str)
    assert res[0]["id"] == "item_sched_123"
    mock_valstorm_client.create_scheduled_task.assert_called_once_with(
        name="Lead 1042 Reminder",
        target_type="function",
        target_id="fun_send_sms",
        run_at_utc="2026-09-15T14:00:00Z",
        cron_expression=None,
        payload_data={"lead_id": "lead_1042"},
    )


@pytest.mark.asyncio
async def test_valstorm_create_drip_cadence_tool(mock_valstorm_client):
    """Verify valstorm_create_drip_cadence constructs the drip blueprint."""
    tools = create_valstorm_tools(client=mock_valstorm_client)
    tool_map = {t.__name__: t for t in tools}
    create_drip_fn = tool_map["valstorm_create_drip_cadence"]

    steps = [
        {"step": 1, "delay_minutes": 0},
        {"step": 2, "delay_minutes": 2880},
    ]
    exit_conds = {"status": ["Closed Won", "Meeting Booked"]}

    res_str = await create_drip_fn(
        name="14-Day Inbound Nurture",
        target_schema="lead",
        steps=steps,
        exit_conditions=exit_conds,
        target_type="automation",
        target_id="wf_lead_drip",
    )
    res = json.loads(res_str)
    assert res[0]["id"] == "drip_def_123"
    mock_valstorm_client.create_drip_cadence.assert_called_once_with(
        name="14-Day Inbound Nurture",
        target_schema="lead",
        steps=steps,
        exit_conditions=exit_conds,
        target_type="automation",
        target_id="wf_lead_drip",
        app_id=None,
    )


@pytest.mark.asyncio
async def test_valstorm_enroll_in_drip_tool(mock_valstorm_client):
    """Verify valstorm_enroll_in_drip enrolls a record into an active cadence."""
    tools = create_valstorm_tools(client=mock_valstorm_client)
    tool_map = {t.__name__: t for t in tools}
    enroll_fn = tool_map["valstorm_enroll_in_drip"]

    res_str = await enroll_fn(
        drip_definition_id="drip_def_123",
        target_record_id="lead_492019",
        target_schema="lead",
        start_date_time_utc="2026-09-12T19:00:00Z",
    )
    res = json.loads(res_str)
    assert res[0]["id"] == "drip_enr_123"
    mock_valstorm_client.enroll_in_drip.assert_called_once_with(
        drip_definition_id="drip_def_123",
        target_record_id="lead_492019",
        target_schema="lead",
        start_date_time_utc="2026-09-12T19:00:00Z",
    )
