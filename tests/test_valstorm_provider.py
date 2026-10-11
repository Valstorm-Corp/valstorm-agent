"""Unit tests for Valstorm Managed Provider and preset resolution."""

import os
import pytest
from providers import OPENAI_COMPATIBLE_PRESETS, resolve_provider_instance
from cli.helpers import ensure_provider_key
import cli.helpers


def test_valstorm_preset_configuration():
    assert "valstorm" in OPENAI_COMPATIBLE_PRESETS
    preset = OPENAI_COMPATIBLE_PRESETS["valstorm"]
    assert preset["default_model"] == "gemini-flash-latest"
    assert "api.valstorm.com" in preset["base_url"]


def test_valstorm_provider_resolution(monkeypatch):
    monkeypatch.setenv("VALSTORM_AI_BASE_URL", "http://test-server/v1/ai")
    provider, model, p_name = resolve_provider_instance("valstorm", api_key="test_tok_123")

    assert p_name == "valstorm"
    assert model == "gemini-flash-latest"
    # Gemini models use the native Gemini pass-through on the Valstorm API...
    from providers.gemini import GeminiProvider
    assert isinstance(provider, GeminiProvider)
    assert provider.backend == "valstorm"
    assert provider.valstorm_base_url == "http://test-server/v1/ai/gemini"
    assert provider.api_key == "test_tok_123"
    # ...with the OpenAI-compatible gateway kept as an automatic fallback.
    assert provider._compat_provider.base_url == "http://test-server/v1/ai"
    assert provider._compat_provider.api_key == "test_tok_123"


def test_valstorm_provider_resolution_passthrough_disabled(monkeypatch):
    monkeypatch.setenv("VALSTORM_AI_BASE_URL", "http://test-server/v1/ai")
    monkeypatch.setenv("VALSTORM_GEMINI_PASSTHROUGH", "0")
    provider, model, p_name = resolve_provider_instance("valstorm", api_key="test_tok_123")
    assert p_name == "valstorm"
    assert provider.base_url == "http://test-server/v1/ai"
    assert provider.api_key == "test_tok_123"


def test_valstorm_provider_non_gemini_model_uses_openai_gateway(monkeypatch):
    monkeypatch.setenv("VALSTORM_AI_BASE_URL", "http://test-server/v1/ai")
    provider, model, _ = resolve_provider_instance("valstorm", api_key="tok", model="claude-opus-5-5")
    assert model == "claude-opus-5-5"
    assert provider.base_url == "http://test-server/v1/ai"


def test_valstorm_key_auto_resolution(monkeypatch):
    monkeypatch.delenv("VALSTORM_API_KEY", raising=False)
    monkeypatch.delenv("VALSTORM_TOKEN", raising=False)
    monkeypatch.setattr(cli.helpers.KeyStore, "get_api_key", lambda self, prov, override_key=None: None)
    monkeypatch.setattr(
        cli.helpers,
        "resolve_valstorm_credentials",
        lambda *args, **kwargs: ("mock_valstorm_pat_abc", "https://api.valstorm.com/v1"),
    )
    key = ensure_provider_key("valstorm", interactive=False)
    assert key == "mock_valstorm_pat_abc"


def test_server_resolves_valstorm_provider():
    from server import resolve_model_and_provider
    model, prov = resolve_model_and_provider("gemini-flash-latest", "valstorm")
    assert prov == "valstorm"
    assert model == "gemini-flash-latest"

    # Also when raw_provider is 'managed' or 'hosted'
    model2, prov2 = resolve_model_and_provider("gemini-flash-latest", "managed")
    assert prov2 == "valstorm"


def test_openai_provider_preserves_thought_signature():
    from providers.openai import OpenAIProvider
    from core.models import Message, ToolCall

    provider = OpenAIProvider(api_key="mock", base_url="http://mock", provider_name="valstorm")

    # 1. Formatting assistant messages with signed tool calls
    tc = ToolCall(
        id="call_abc",
        name="calculator",
        arguments={"expr": "1+1"},
        thought_signature=b"test_sig_bytes",
    )
    msg = Message(role="assistant", tool_calls=[tc])
    formatted = provider._format_messages([msg])
    assert len(formatted) == 1
    assert "tool_calls" in formatted[0]
    tc_out = formatted[0]["tool_calls"][0]
    import base64
    assert tc_out["thought_signature"] == base64.b64encode(b"test_sig_bytes").decode("ascii")


@pytest.mark.asyncio
async def test_valstorm_provider_auto_refresh_on_401(monkeypatch):
    from providers.openai import OpenAIProvider
    from core.models import Message
    from unittest.mock import AsyncMock, MagicMock
    import openai

    provider = OpenAIProvider(api_key="expired_tok", base_url="https://api.valstorm.com/v1/ai", provider_name="valstorm")

    # Mock chat completion responses: 1st raises 401 AuthenticationError, 2nd succeeds
    auth_err = openai.AuthenticationError(
        message="Error code: 401 - {'detail': 'Signature has expired.'}",
        response=MagicMock(status_code=401),
        body={"detail": "Signature has expired."},
    )

    success_resp = MagicMock()
    success_resp.choices = [
        MagicMock(message=MagicMock(content="Hello after auto-refresh!", tool_calls=None))
    ]
    success_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    mock_create = AsyncMock(side_effect=[auth_err, success_resp])

    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create
    provider._client = mock_client

    # Mock refresh logic
    refresh_called = False
    async def mock_refresh():
        nonlocal refresh_called
        refresh_called = True
        provider.api_key = "refreshed_access_token_xyz"
        # Reset provider client to mock_client again with 2nd response ready
        provider._client = mock_client
        return True

    monkeypatch.setattr(provider, "_refresh_valstorm_auth", mock_refresh)

    msg, usage = await provider.generate([Message(role="user", content="Hello")])

    assert refresh_called is True
    assert msg.content == "Hello after auto-refresh!"
    assert provider.api_key == "refreshed_access_token_xyz"


