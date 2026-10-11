"""Unit tests for Slack Integration Tools and RemoteSlackContext."""

import json
import pytest
import httpx
from unittest.mock import AsyncMock

from core.tools import ToolRegistry
from tools.valstorm_client import ValstormApiClient
from tools.slack_tools import create_slack_tools, register_slack_tools
from tools.valstorm_platform_client import RemotePlatformContext
from tools.valstorm_tools import register_valstorm_tools


@pytest.fixture
def mock_slack_api_client():
    mock_responses = {
        ("POST", "/slack/chat/post"): httpx.Response(
            status_code=200,
            json={"ok": True, "channel": "C01234567", "ts": "1712345678.123456", "message": {"text": "Hello world"}},
            request=httpx.Request("POST", "http://localhost:8000/slack/chat/post"),
        ),
        ("GET", "/slack/channels"): httpx.Response(
            status_code=200,
            json={"ok": True, "channels": [{"id": "C01234567", "name": "general", "is_private": False, "num_members": 15}]},
            request=httpx.Request("GET", "http://localhost:8000/slack/channels"),
        ),
        ("GET", "/slack/chat/history"): httpx.Response(
            status_code=200,
            json={"ok": True, "messages": [{"ts": "1712345678.123456", "user": "U111", "text": "Deploy complete"}]},
            request=httpx.Request("GET", "http://localhost:8000/slack/chat/history"),
        ),
        ("GET", "/slack/users"): httpx.Response(
            status_code=200,
            json={"ok": True, "members": [{"id": "U111", "name": "alice", "real_name": "Alice Smith"}]},
            request=httpx.Request("GET", "http://localhost:8000/slack/users"),
        ),
        ("GET", "/slack/users/U111/profile"): httpx.Response(
            status_code=200,
            json={"ok": True, "profile": {"real_name": "Alice Smith", "email": "alice@example.com", "title": "VP Marketing"}},
            request=httpx.Request("GET", "http://localhost:8000/slack/users/U111/profile"),
        ),
        ("POST", "/slack/reactions/add"): httpx.Response(
            status_code=200,
            json={"ok": True},
            request=httpx.Request("POST", "http://localhost:8000/slack/reactions/add"),
        ),
        ("POST", "/slack/chat/update"): httpx.Response(
            status_code=200,
            json={"ok": True, "channel": "C01234567", "ts": "1712345678.123456", "text": "Updated message"},
            request=httpx.Request("POST", "http://localhost:8000/slack/chat/update"),
        ),
        ("POST", "/slack/chat/delete"): httpx.Response(
            status_code=200,
            json={"ok": True, "channel": "C01234567", "ts": "1712345678.123456"},
            request=httpx.Request("POST", "http://localhost:8000/slack/chat/delete"),
        ),
        ("GET", "/slack/auth/status"): httpx.Response(
            status_code=200,
            json={"status": "connected", "team_id": "T0123", "team_name": "Acme Org", "bot_user_id": "B0123"},
            request=httpx.Request("GET", "http://localhost:8000/slack/auth/status"),
        ),
    }

    async def mock_handle_request(request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        if key in mock_responses:
            return mock_responses[key]
        return httpx.Response(status_code=200, json={"ok": True}, request=request)

    mock_transport = AsyncMock()
    mock_transport.handle_async_request = AsyncMock(side_effect=mock_handle_request)

    http_client = httpx.AsyncClient(transport=mock_transport, base_url="http://localhost:8000")
    return ValstormApiClient(token="mock_token", client=http_client)


@pytest.mark.asyncio
async def test_slack_tools_registration(mock_slack_api_client):
    registry = ToolRegistry()
    register_slack_tools(registry=registry, client=mock_slack_api_client)

    tool_names = registry.list_tools()
    assert "slack_post_message" in tool_names
    assert "slack_list_channels" in tool_names
    assert "slack_get_channel_history" in tool_names
    assert "slack_list_users" in tool_names
    assert "slack_get_user_profile" in tool_names
    assert "slack_add_reaction" in tool_names
    assert "slack_update_message" in tool_names
    assert "slack_delete_message" in tool_names
    assert "slack_get_auth_status" in tool_names

    schemas = registry.get_schemas()
    assert len(schemas) == 9
    schema_map = {s["name"]: s for s in schemas}
    assert "channel" in schema_map["slack_post_message"]["parameters"]["properties"]
    assert "types" in schema_map["slack_list_channels"]["parameters"]["properties"]


@pytest.mark.asyncio
async def test_slack_tools_execution(mock_slack_api_client):
    registry = ToolRegistry()
    register_slack_tools(registry=registry, client=mock_slack_api_client)

    # 1. Post Message
    post_res = await registry.execute_async(
        "slack_post_message",
        {"channel": "C01234567", "text": "Hello world", "as_user": False}
    )
    assert not post_res.is_error
    data = json.loads(post_res.output)
    assert data.get("ok") is True
    assert data.get("ts") == "1712345678.123456"

    # 2. Post Message with JSON string blocks
    blocks_json = json.dumps([{"type": "section", "text": {"type": "mrkdwn", "text": "*Bold*"}}])
    post_blocks_res = await registry.execute_async(
        "slack_post_message",
        {"channel": "C01234567", "blocks": blocks_json}
    )
    assert not post_blocks_res.is_error

    # 3. List Channels
    channels_res = await registry.execute_async("slack_list_channels", {})
    assert not channels_res.is_error
    assert "general" in channels_res.output

    # 4. Channel History
    history_res = await registry.execute_async("slack_get_channel_history", {"channel": "C01234567"})
    assert not history_res.is_error
    assert "Deploy complete" in history_res.output

    # 5. List Users
    users_res = await registry.execute_async("slack_list_users", {})
    assert not users_res.is_error
    assert "Alice Smith" in users_res.output

    # 6. User Profile
    prof_res = await registry.execute_async("slack_get_user_profile", {"user_id": "U111"})
    assert not prof_res.is_error
    assert "alice@example.com" in prof_res.output

    # 7. Add Reaction
    react_res = await registry.execute_async(
        "slack_add_reaction",
        {"channel": "C01234567", "timestamp": "1712345678.123456", "name": ":eyes:"}
    )
    assert not react_res.is_error

    # 8. Update Message
    update_res = await registry.execute_async(
        "slack_update_message",
        {"channel": "C01234567", "ts": "1712345678.123456", "text": "Updated message"}
    )
    assert not update_res.is_error
    assert "Updated message" in update_res.output

    # 9. Delete Message
    delete_res = await registry.execute_async(
        "slack_delete_message",
        {"channel": "C01234567", "ts": "1712345678.123456"}
    )
    assert not delete_res.is_error

    # 10. Auth Status
    status_res = await registry.execute_async("slack_get_auth_status", {})
    assert not status_res.is_error
    assert "Acme Org" in status_res.output


@pytest.mark.asyncio
async def test_remote_platform_context_slack(mock_slack_api_client):
    platform = RemotePlatformContext(client=mock_slack_api_client)

    # Test direct context methods
    res = platform.slack.post_message(channel="C01234567", text="Testing context")
    assert res.get("ok") is True

    channels = platform.slack.list_channels()
    assert channels.get("ok") is True

    history = platform.slack.get_channel_history(channel="C01234567")
    assert history.get("ok") is True

    # Test nested integrations.slack
    assert platform.integrations.slack.list_users().get("ok") is True
    assert platform.integrations.slack.get_auth_status().get("team_id") == "T0123"


@pytest.mark.asyncio
async def test_valstorm_tools_includes_slack(mock_slack_api_client):
    registry = ToolRegistry()
    register_valstorm_tools(registry=registry, client=mock_slack_api_client)

    tool_names = registry.list_tools()
    assert "valstorm_sql_query" in tool_names
    assert "slack_post_message" in tool_names
    assert "slack_list_channels" in tool_names
