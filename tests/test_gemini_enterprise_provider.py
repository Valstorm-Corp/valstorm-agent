import os
import pytest
from unittest.mock import MagicMock, patch
from providers.gemini import GeminiProvider
from core.models import Message


def test_gemini_provider_default_model():
    provider = GeminiProvider()
    assert provider.default_model == "gemini-flash-latest"


def test_gemini_enterprise_mode_detection(monkeypatch):
    monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    with patch("os.path.isfile", return_value=False):
        assert GeminiProvider.is_enterprise_mode() is False

    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "true")
    assert GeminiProvider.is_enterprise_mode() is True

    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "false")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    assert GeminiProvider.is_enterprise_mode() is True


def test_gemini_provider_client_initialization_enterprise(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "valstorm-gemini")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")

    with patch("google.genai.Client") as mock_client_cls:
        provider = GeminiProvider()
        client = provider._get_client()
        mock_client_cls.assert_called_once_with(
            vertexai=True,
            project="valstorm-gemini",
            location="global",
        )


@pytest.mark.asyncio
async def test_gemini_provider_generate_with_default_model(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "true")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_candidate = MagicMock()
    mock_part = MagicMock()
    mock_part.text = "Hello from enterprise flash"
    mock_part.function_call = None
    mock_candidate.content.parts = [mock_part]
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata.prompt_token_count = 12
    mock_response.usage_metadata.candidates_token_count = 6
    mock_response.usage_metadata.total_token_count = 18

    # Async generate_content
    async def mock_gen(*args, **kwargs):
        return mock_response

    mock_client.aio.models.generate_content = mock_gen

    provider = GeminiProvider(client=mock_client)
    msg, usage = await provider.generate(
        messages=[Message(role="user", content="Hello")],
        model="gemini-flash-latest",
    )

    assert msg.content == "Hello from enterprise flash"
    assert msg.model == "gemini-flash-latest"
    assert usage.total_tokens == 18
