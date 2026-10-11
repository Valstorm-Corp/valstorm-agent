"""Unit Tests for Valstorm Function & Observability Tools (Phase 2 & 3)."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from core.tools import ToolRegistry
from tools.valstorm_tools import create_valstorm_tools
from tools.valstorm_client import ValstormApiClient


@pytest.fixture
def mock_valstorm_client():
    client = MagicMock(spec=ValstormApiClient)
    client.function_list = AsyncMock(return_value=[
        {"id": "fun_001", "name": "calculate_mrr", "description": "MRR metric calculator"}
    ])
    client.function_call = AsyncMock(return_value={
        "status": "success", "data": {"mrr": 12500.0}
    })
    client.function_validate = AsyncMock(return_value={
        "status": "valid", "inputs_tested": {"acc": "123"}
    })
    client.get_execution_logs = AsyncMock(return_value=[
        {"id": "log_001", "name": "Function: calculate_mrr", "status": "Success", "duration_ms": 25.0}
    ])
    client.get_log_detail = AsyncMock(return_value={
        "id": "log_001", "status": "Success", "payload": {"inputs": {"acc": "123"}, "output": {"mrr": 12500.0}}
    })
    return client


@pytest.mark.asyncio
async def test_valstorm_function_list_tool(mock_valstorm_client):
    tools = create_valstorm_tools(client=mock_valstorm_client)
    tool_map = {t._tool_name: t for t in tools}

    fn_list_tool = tool_map["valstorm_function_list"]
    res = await fn_list_tool()
    data = json.loads(res)
    assert len(data) == 1
    assert data[0]["name"] == "calculate_mrr"
    mock_valstorm_client.function_list.assert_awaited_once()


@pytest.mark.asyncio
async def test_valstorm_function_call_tool(mock_valstorm_client):
    tools = create_valstorm_tools(client=mock_valstorm_client)
    tool_map = {t._tool_name: t for t in tools}

    fn_call_tool = tool_map["valstorm_function_call"]
    res = await fn_call_tool(function_name="calculate_mrr", inputs=json.dumps({"account_id": "acc_001"}))
    data = json.loads(res)
    assert data["status"] == "success"
    assert data["data"]["mrr"] == 12500.0
    mock_valstorm_client.function_call.assert_awaited_once_with(
        function_name="calculate_mrr", function_id=None, inputs={"account_id": "acc_001"}
    )


@pytest.mark.asyncio
async def test_valstorm_validate_function_tool(mock_valstorm_client):
    tools = create_valstorm_tools(client=mock_valstorm_client)
    tool_map = {t._tool_name: t for t in tools}

    validate_tool = tool_map["valstorm_validate_function"]
    sample_code = "async def execute(platform, **kwargs): return True"
    res = await validate_tool(code=sample_code, function_name="test_fn", inputs={"x": 1})
    data = json.loads(res)
    assert data["status"] == "valid"
    mock_valstorm_client.function_validate.assert_awaited_once_with(
        code=sample_code, function_name="test_fn", inputs={"x": 1}
    )


@pytest.mark.asyncio
async def test_valstorm_log_tools(mock_valstorm_client):
    tools = create_valstorm_tools(client=mock_valstorm_client)
    tool_map = {t._tool_name: t for t in tools}

    # Test get_execution_logs
    get_logs_tool = tool_map["valstorm_get_execution_logs"]
    res_logs = await get_logs_tool(status="Success", limit=5)
    logs_data = json.loads(res_logs)
    assert len(logs_data) == 1
    assert logs_data[0]["id"] == "log_001"

    # Test get_log_detail
    get_detail_tool = tool_map["valstorm_get_log_detail"]
    res_detail = await get_detail_tool(log_id="log_001")
    detail_data = json.loads(res_detail)
    assert detail_data["id"] == "log_001"
    assert detail_data["payload"]["output"]["mrr"] == 12500.0
