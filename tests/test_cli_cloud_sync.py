import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from core.models import Message, SessionState, UsageMetadata, ToolCall
from cli.helpers import sync_cli_turn_to_valstorm, stream_and_render_turn


@pytest.mark.asyncio
async def test_sync_cli_turn_to_valstorm_success():
    session = SessionState(active_model="gemini-flash-latest", active_provider="valstorm")
    user_msg = Message(role="user", content="Hello world from CLI")
    session.add_message(user_msg)

    asst_msg = Message(
        role="assistant",
        content="Hello back from Assistant!",
        model="gemini-flash-latest",
        provider="valstorm",
        usage=UsageMetadata(prompt_tokens=12, completion_tokens=8, total_tokens=20),
        tool_calls=[ToolCall(id="call_1", name="calculator", arguments={"expression": "2+2"})],
    )
    session.add_message(asst_msg)

    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("tools.valstorm_client.resolve_valstorm_credentials", return_value=("fake_token", "http://localhost:8010/v1")), \
         patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
        success = await sync_cli_turn_to_valstorm(
            session=session,
            user_prompt="Hello world from CLI",
            final_message=asst_msg,
            valstorm_env="local",
        )

    assert success is True
    mock_post.assert_awaited_once()
    url = mock_post.await_args.args[0]
    assert url == f"http://localhost:8010/v1/ai/chat/{session.session_id}/desktop-sync"
    payload = mock_post.await_args.kwargs["json"]
    assert payload["text"] == "Hello back from Assistant!"
    assert payload["status"] == "completed"
    assert payload["user_text"] == "Hello world from CLI"
    assert payload["user_message_id"] == user_msg.id
    assert payload["message_id"] == asst_msg.id
    assert payload["input_tokens"] == 12
    assert payload["output_tokens"] == 8
    assert payload["model"] == "gemini-flash-latest"
    assert payload["provider"] == "valstorm"
    assert len(payload["tool_calls"]) == 1
    assert payload["tool_calls"][0]["name"] == "calculator"


@pytest.mark.asyncio
async def test_sync_cli_turn_with_turn_tools_and_aggregated_tokens():
    session = SessionState(active_model="gemini-flash-latest", active_provider="valstorm")
    user_msg = Message(role="user", content="Read file test")
    session.add_message(user_msg)

    final_msg = Message(
        role="assistant",
        content="File has been read and verified.",
        model="gemini-flash-latest",
        provider="valstorm",
        usage=UsageMetadata(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        tool_calls=None,  # Final message has no tool calls
    )
    session.add_message(final_msg)

    tool_calls_executed = [
        {"name": "read_file", "tool_name": "read_file", "args": {"path": "a.txt"}, "stdout": "hello", "status": "completed"},
        {"name": "patch_file", "tool_name": "patch_file", "args": {"path": "a.txt"}, "stdout": "ok", "status": "completed"},
    ]

    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("tools.valstorm_client.resolve_valstorm_credentials", return_value=("fake_token", "http://localhost:8010/v1")), \
         patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
        success = await sync_cli_turn_to_valstorm(
            session=session,
            user_prompt="Read file test",
            final_message=final_msg,
            valstorm_env="local",
            tool_calls_executed=tool_calls_executed,
            turn_input_tokens=5000,
            turn_output_tokens=1200,
        )

    assert success is True
    mock_post.assert_awaited_once()
    payload = mock_post.await_args.kwargs["json"]
    assert payload["input_tokens"] == 5000
    assert payload["output_tokens"] == 1200
    assert len(payload["tool_calls"]) == 2
    assert payload["tool_calls"][0]["name"] == "read_file"
    assert payload["tool_calls"][1]["name"] == "patch_file"


@pytest.mark.asyncio
async def test_sync_cli_turn_to_valstorm_skips_when_no_credentials():
    session = SessionState(active_model="gemini-flash-latest", active_provider="valstorm")
    asst_msg = Message(role="assistant", content="No creds")

    with patch("tools.valstorm_client.resolve_valstorm_credentials", return_value=(None, None)):
        success = await sync_cli_turn_to_valstorm(
            session=session,
            user_prompt="test",
            final_message=asst_msg,
        )

    assert success is False


@pytest.mark.asyncio
async def test_sync_cli_turn_to_valstorm_handles_network_error_gracefully():
    session = SessionState(active_model="gemini-flash-latest", active_provider="valstorm")
    asst_msg = Message(role="assistant", content="error test")

    with patch("tools.valstorm_client.resolve_valstorm_credentials", return_value=("fake_token", "http://localhost:8010/v1")), \
         patch("httpx.AsyncClient.post", side_effect=Exception("Connection refused")):
        success = await sync_cli_turn_to_valstorm(
            session=session,
            user_prompt="test",
            final_message=asst_msg,
        )

    assert success is False
