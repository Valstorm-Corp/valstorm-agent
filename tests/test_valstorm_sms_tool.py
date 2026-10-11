import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from tools.valstorm_client import ValstormApiClient
from tools.valstorm_tools import create_valstorm_tools
from core.context import BUILTIN_PROFILES


@pytest.fixture
def mock_valstorm_client():
    client = MagicMock(spec=ValstormApiClient)
    client.send_sms = AsyncMock(return_value={
        "id": "twme_mock_999",
        "sid": "IM_twilio_sid_999",
        "conversation_id": "twco_mock_convo_999",
        "body": "Hello from Chief of Staff!",
        "status": "sent",
    })
    return client


@pytest.mark.asyncio
async def test_valstorm_send_sms_tool_with_contact(mock_valstorm_client):
    """Verify valstorm_send_sms sends text to contact via client."""
    tools = create_valstorm_tools(client=mock_valstorm_client)
    tool_map = {t.__name__: t for t in tools}
    assert "valstorm_send_sms" in tool_map

    send_sms_fn = tool_map["valstorm_send_sms"]
    res_str = await send_sms_fn(
        contact_id="con_12345",
        message="Hi Jane, loved our chat earlier!",
    )
    res = json.loads(res_str)
    assert res["id"] == "twme_mock_999"
    assert res["body"] == "Hello from Chief of Staff!"
    mock_valstorm_client.send_sms.assert_called_once_with(
        message="Hi Jane, loved our chat earlier!",
        to_phone=None,
        contact_id="con_12345",
        conversation_id=None,
        from_number=None,
        file_id=None,
        is_group=False,
    )


@pytest.mark.asyncio
async def test_valstorm_send_sms_tool_validation(mock_valstorm_client):
    """Verify valstorm_send_sms validates empty message and missing destinations."""
    tools = create_valstorm_tools(client=mock_valstorm_client)
    tool_map = {t.__name__: t for t in tools}
    send_sms_fn = tool_map["valstorm_send_sms"]

    # Empty message
    err1 = await send_sms_fn(contact_id="con_123", message="")
    assert "Error: message text cannot be empty" in err1

    # Missing destination
    err2 = await send_sms_fn(message="Hello world")
    assert "Error: You must provide at least one of to_phone, contact_id, or conversation_id" in err2


def test_agent_profiles_include_valstorm_send_sms():
    """Verify Chief of Staff, Marketer, and Developer profiles include valstorm_send_sms."""
    cos_tools = BUILTIN_PROFILES["chief-of-staff"]["allowed_tools"]
    marketer_tools = BUILTIN_PROFILES["marketer"]["allowed_tools"]
    dev_tools = BUILTIN_PROFILES["developer"]["allowed_tools"]

    assert "valstorm_send_sms" in cos_tools
    assert "valstorm_send_sms" in marketer_tools
    assert "valstorm_send_sms" in dev_tools

    # Verify attached skills
    assert "twilio-conversations-sms" in BUILTIN_PROFILES["chief-of-staff"]["attached_skill_slugs"]
    assert "sms-marketing-and-conversational-outreach" in BUILTIN_PROFILES["chief-of-staff"]["attached_skill_slugs"]
    assert "twilio-conversations-sms" in BUILTIN_PROFILES["marketer"]["attached_skill_slugs"]
    assert "sms-marketing-and-conversational-outreach" in BUILTIN_PROFILES["marketer"]["attached_skill_slugs"]
