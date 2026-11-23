"""Tests for platform_methods_search and platform_invoke in Valstorm Agent Runtime."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from core.context import BUILTIN_PROFILES
from core.tools import ToolRegistry
from server import _discover_and_register_remote_plugin_tools
from tools.valstorm_client import ValstormApiClient
from tools.valstorm_tools import create_valstorm_tools, register_valstorm_tools


class _StubApiClient:
    def __init__(self) -> None:
        self.list_calls: list[dict[str, Any]] = []
        self.invoke_calls: list[dict[str, Any]] = []
        self.list_response: dict[str, Any] = {
            "methods": [
                {
                    "target": "utils.phone_formatter",
                    "path": "/v1/platform/invoke/utils/phone_formatter",
                    "scope": "utils:phone_formatter",
                    "tags": ["utils"],
                    "description": "Formats US phone numbers to E.164.",
                    "signature": "(phone: str)",
                    "parameters": [{"name": "phone", "required": True, "type": "str"}],
                }
            ],
            "count": 1,
            "total_matched": 1,
            "total_available": 859,
            "namespaces": {"utils": 7, "quickbooks": 52, "stripe": 39},
        }
        self.invoke_response: Any = {"phone_number": "+18015550199", "friendly_number": "(801) 555-0199"}

    async def list_platform_methods(
        self,
        query: str | None = None,
        namespace: str | None = None,
        limit: int | None = 25,
    ) -> dict[str, Any]:
        self.list_calls.append({"query": query, "namespace": namespace, "limit": limit})
        return self.list_response

    async def invoke_platform_method(self, target: str, *args: Any, **kwargs: Any) -> Any:
        self.invoke_calls.append({"target": target, "args": list(args), "kwargs": dict(kwargs)})
        return self.invoke_response


def _get_tool(stub: _StubApiClient, name: str):
    tools = {t._tool_name: t for t in create_valstorm_tools(client=stub)}
    assert name in tools, f"Tool '{name}' was not registered by create_valstorm_tools"
    return tools[name]


@pytest.mark.asyncio
async def test_platform_methods_search_passes_server_filters():
    stub = _StubApiClient()
    search_tool = _get_tool(stub, "platform_methods_search")

    raw = await search_tool(query="phone_formatter", namespace="utils", limit=10)
    parsed = json.loads(raw)

    assert stub.list_calls == [{"query": "phone_formatter", "namespace": "utils", "limit": 10}]
    assert parsed["count"] == 1
    assert parsed["total_available"] == 859
    assert parsed["methods"][0]["target"] == "utils.phone_formatter"


@pytest.mark.asyncio
async def test_platform_methods_search_client_side_fallback_for_legacy_api():
    stub = _StubApiClient()
    # Simulate an older Open Valstorm API that returns the full unfiltered list without `total_available`
    stub.list_response = {
        "methods": [
            {
                "target": "quickbooks.list_invoices",
                "scope": "quickbooks:list_invoices",
                "tags": ["quickbooks"],
                "description": "Lists QuickBooks Online invoices",
            },
            {
                "target": "stripe.list_subscriptions",
                "scope": "stripe:list_subscriptions",
                "tags": ["stripe"],
                "description": "Lists Stripe subscriptions",
            },
            {
                "target": "quickbooks.create_customer",
                "scope": "quickbooks:create_customer",
                "tags": ["quickbooks"],
                "description": "Creates a QuickBooks customer",
            },
        ],
        "count": 3,
    }
    search_tool = _get_tool(stub, "platform_methods_search")

    raw = await search_tool(query="invoice", namespace="quickbooks", limit=5)
    parsed = json.loads(raw)

    assert parsed["count"] == 1
    assert parsed["total_matched"] == 1
    assert parsed["total_available"] == 3
    assert parsed["namespaces"] == {"quickbooks": 2, "stripe": 1}
    assert parsed["methods"][0]["target"] == "quickbooks.list_invoices"


@pytest.mark.asyncio
async def test_platform_invoke_supports_dict_and_json_string_arguments():
    stub = _StubApiClient()
    invoke_tool = _get_tool(stub, "platform_invoke")

    # 1. Dict kwargs
    out1 = await invoke_tool(target="utils.phone_formatter", kwargs={"phone": "8015550199"})
    assert json.loads(out1)["phone_number"] == "+18015550199"
    assert stub.invoke_calls[-1] == {
        "target": "utils.phone_formatter",
        "args": [],
        "kwargs": {"phone": "8015550199"},
    }

    # 2. JSON-stringified kwargs & args
    out2 = await invoke_tool(
        target="utils/phone_formatter",
        kwargs='{"default_country": "US"}',
        args='["8015550199"]',
    )
    assert json.loads(out2)["phone_number"] == "+18015550199"
    assert stub.invoke_calls[-1] == {
        "target": "utils/phone_formatter",
        "args": ["8015550199"],
        "kwargs": {"default_country": "US"},
    }

    # 3. Validation errors return helpful error messages
    err_empty = await invoke_tool(target="   ")
    assert "Error:" in err_empty
    err_bad_json = await invoke_tool(target="utils.phone_formatter", kwargs="{bad")
    assert "Error:" in err_bad_json
    err_wrong_type = await invoke_tool(target="utils.phone_formatter", args='{"not": "a list"}')
    assert "Error:" in err_wrong_type


@pytest.mark.asyncio
async def test_valstorm_client_list_platform_methods_http_params():
    client = ValstormApiClient(token="pat_test", base_url="https://api.valstorm.com/v1")
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"methods": [], "count": 0, "total_available": 0}
    client._request_with_retry = AsyncMock(return_value=fake_resp)

    res = await client.list_platform_methods(query="invoice", namespace="quickbooks", limit=15)
    assert res["count"] == 0
    client._request_with_retry.assert_awaited_once_with(
        "GET",
        "/platform/methods",
        params={"q": "invoice", "namespace": "quickbooks", "limit": 15},
    )


def test_builtin_profiles_include_platform_rpc_tools():
    # platform_methods_search / platform_invoke are explicitly gated to technical RPC specialists
    rpc_enabled = ["valstorm-developer", "data-analyst"]
    for slug in rpc_enabled:
        allowed = BUILTIN_PROFILES[slug]["allowed_tools"]
        assert "platform_methods_search" in allowed, f"Missing platform_methods_search in {slug}"
        assert "platform_invoke" in allowed, f"Missing platform_invoke in {slug}"

    # Scoped non-RPC personas must NOT receive the universal platform_invoke escape hatch
    for slug in ("chief-of-staff", "marketer", "finance", "workspace-comms", "researcher", "architect"):
        allowed = BUILTIN_PROFILES[slug]["allowed_tools"]
        assert "platform_invoke" not in allowed, f"Unexpected platform_invoke escape hatch on {slug}"


@pytest.mark.asyncio
async def test_remote_plugin_discovery_deduplicates_native_platform_rpc_tools():
    stub = _StubApiClient()
    registry = ToolRegistry()
    register_valstorm_tools(registry, client=stub)

    native_search = registry.get("platform_methods_search")
    native_invoke = registry.get("platform_invoke")
    assert native_search is not None
    assert native_invoke is not None

    fake_http_resp = MagicMock()
    fake_http_resp.status_code = 200
    fake_http_resp.json.return_value = {
        "tools": [
            {"name": "platform_methods_search", "description": "remote copy", "parameters": {"type": "object", "properties": {}}},
            {"name": "platform_invoke", "description": "remote copy", "parameters": {"type": "object", "properties": {}}},
            {"name": "some_new_plugin_tool", "description": "brand new", "parameters": {"type": "object", "properties": {}}},
        ]
    }

    mock_client_instance = AsyncMock()
    mock_client_instance.__aenter__.return_value = mock_client_instance
    mock_client_instance.__aexit__.return_value = None
    mock_client_instance.get.return_value = fake_http_resp

    with patch("server.httpx.AsyncClient", return_value=mock_client_instance):
        discovered = await _discover_and_register_remote_plugin_tools(
            registry=registry,
            valstorm_token="jwt_test",
            valstorm_base_url="https://api.valstorm.com/v1",
        )

    assert "platform_methods_search" in discovered
    assert "platform_invoke" in discovered
    assert "some_new_plugin_tool" in discovered
    # Native tools (which hit /v1/platform/invoke directly) must be preserved, not overwritten
    assert registry.get("platform_methods_search") is native_search
    assert registry.get("platform_invoke") is native_invoke
    assert registry.get("some_new_plugin_tool") is not None
