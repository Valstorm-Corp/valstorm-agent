"""Unit tests for retry logic and exponential backoff in LLM providers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest

from core.models import Message, StreamEvent, StreamEventType, UsageMetadata
from core.retry import (
    compute_backoff_delay,
    execute_stream_with_retry,
    execute_with_retry,
    extract_status_code,
    is_retryable_error,
)
from providers.gemini import GeminiProvider
from providers.openai import OpenAIProvider
from providers.anthropic import AnthropicProvider


class TestRetryErrorClassification:
    """Tests for is_retryable_error and status code extraction."""

    def test_503_unavailable_error_matching_prompt(self):
        """Test the exact error reported by user is classified as retryable."""
        err_msg = (
            "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently "
            "experiencing high demand. Spikes in demand are usually temporary. Please try again later.', "
            "'status': 'UNAVAILABLE'}}"
        )
        exc = RuntimeError(err_msg)
        assert is_retryable_error(exc) is True

    def test_429_rate_limit_and_resource_exhausted(self):
        exc1 = Exception("429 RESOURCE_EXHAUSTED: Rate limit exceeded")
        assert is_retryable_error(exc1) is True

        exc2 = Exception("Rate limit reached. Please try again in 5s.")
        assert is_retryable_error(exc2) is True

    def test_server_and_gateway_errors(self):
        assert is_retryable_error(Exception("500 Internal Server Error")) is True
        assert is_retryable_error(Exception("502 Bad Gateway")) is True
        assert is_retryable_error(Exception("504 Gateway Timeout")) is True
        assert is_retryable_error(Exception("529 Overloaded")) is True

    def test_network_and_timeout_errors(self):
        assert is_retryable_error(TimeoutError("Connection timed out")) is True
        assert is_retryable_error(ConnectionError("Connection reset by peer")) is True
        assert is_retryable_error(ConnectionRefusedError("Connection refused")) is True

    def test_status_code_attribute(self):
        class CustomApiError(Exception):
            def __init__(self, code):
                self.status_code = code

        assert is_retryable_error(CustomApiError(503)) is True
        assert is_retryable_error(CustomApiError(429)) is True
        assert is_retryable_error(CustomApiError(500)) is True
        assert is_retryable_error(CustomApiError(401)) is False
        assert is_retryable_error(CustomApiError(403)) is False
        assert is_retryable_error(CustomApiError(404)) is False

    def test_non_retryable_errors(self):
        assert is_retryable_error(ValueError("Invalid argument 'temperature'")) is False
        assert is_retryable_error(Exception("401 Unauthorized: Invalid API Key")) is False
        assert is_retryable_error(Exception("403 Forbidden: Access denied")) is False
        assert is_retryable_error(Exception("404 Not Found: Model does not exist")) is False

    def test_cancellation_and_interrupts_never_retried(self):
        assert is_retryable_error(KeyboardInterrupt()) is False
        assert is_retryable_error(asyncio.CancelledError()) is False
        assert is_retryable_error(SystemExit()) is False


class TestBackoffCalculation:
    """Tests for compute_backoff_delay."""

    def test_exponential_progression_no_jitter(self):
        d0 = compute_backoff_delay(attempt=0, initial_delay=1.0, backoff_factor=2.0, jitter=False)
        d1 = compute_backoff_delay(attempt=1, initial_delay=1.0, backoff_factor=2.0, jitter=False)
        d2 = compute_backoff_delay(attempt=2, initial_delay=1.0, backoff_factor=2.0, jitter=False)
        d3 = compute_backoff_delay(attempt=3, initial_delay=1.0, backoff_factor=2.0, jitter=False)

        assert d0 == 1.0
        assert d1 == 2.0
        assert d2 == 4.0
        assert d3 == 8.0

    def test_max_delay_cap(self):
        d = compute_backoff_delay(attempt=10, initial_delay=1.0, backoff_factor=2.0, max_delay=15.0, jitter=False)
        assert d == 15.0

    def test_jitter_adds_bounded_randomness(self):
        for _ in range(10):
            d = compute_backoff_delay(attempt=0, initial_delay=1.0, backoff_factor=2.0, jitter=True)
            assert 1.0 <= d <= 1.5


class TestExecuteWithRetry:
    """Tests for async execute_with_retry wrapper."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        calls = 0

        async def _call():
            nonlocal calls
            calls += 1
            return "success"

        res = await execute_with_retry(_call, max_retries=3, initial_delay=0.01, jitter=False)
        assert res == "success"
        assert calls == 1

    @pytest.mark.asyncio
    async def test_retries_transient_failure_then_succeeds(self):
        calls = 0
        retries_recorded = []

        def on_retry(att, exc, delay):
            retries_recorded.append((att, str(exc), delay))

        async def _call():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RuntimeError("503 UNAVAILABLE. High demand spike")
            return "recovered"

        res = await execute_with_retry(
            _call,
            max_retries=3,
            initial_delay=0.001,
            backoff_factor=1.5,
            jitter=False,
            on_retry=on_retry,
        )
        assert res == "recovered"
        assert calls == 3
        assert len(retries_recorded) == 2
        assert retries_recorded[0][0] == 1
        assert retries_recorded[1][0] == 2

    @pytest.mark.asyncio
    async def test_exhausts_retries_and_raises(self):
        calls = 0

        async def _call():
            nonlocal calls
            calls += 1
            raise RuntimeError("503 UNAVAILABLE. Persistent high demand")

        with pytest.raises(RuntimeError) as exc_info:
            await execute_with_retry(_call, max_retries=2, initial_delay=0.001, jitter=False)

        assert "503 UNAVAILABLE" in str(exc_info.value)
        # Attempt 0 (initial) + 2 retries = 3 calls total
        assert calls == 3

    @pytest.mark.asyncio
    async def test_non_retryable_fails_immediately_without_retry(self):
        calls = 0

        async def _call():
            nonlocal calls
            calls += 1
            raise ValueError("401 Unauthorized: Invalid API Key")

        with pytest.raises(ValueError) as exc_info:
            await execute_with_retry(_call, max_retries=3, initial_delay=0.001, jitter=False)

        assert "401 Unauthorized" in str(exc_info.value)
        assert calls == 1


class TestExecuteStreamWithRetry:
    """Tests for execute_stream_with_retry async generator wrapper."""

    @pytest.mark.asyncio
    async def test_stream_succeeds_on_first_try(self):
        calls = 0

        async def _stream_gen():
            nonlocal calls
            calls += 1
            yield "chunk 1"
            yield "chunk 2"

        results = []
        async for chunk in execute_stream_with_retry(_stream_gen, max_retries=3, initial_delay=0.001, jitter=False):
            results.append(chunk)

        assert results == ["chunk 1", "chunk 2"]
        assert calls == 1

    @pytest.mark.asyncio
    async def test_stream_retries_transient_failure_then_succeeds(self):
        calls = 0

        async def _stream_gen():
            nonlocal calls
            calls += 1
            if calls < 2:
                raise RuntimeError("503 UNAVAILABLE: Spikes in demand")
            yield "chunk recovered 1"
            yield "chunk recovered 2"

        results = []
        async for chunk in execute_stream_with_retry(_stream_gen, max_retries=3, initial_delay=0.001, jitter=False):
            results.append(chunk)

        assert results == ["chunk recovered 1", "chunk recovered 2"]
        assert calls == 2


class TestProviderRetries:
    """Integration tests verifying providers retry on 503/429 errors."""

    @pytest.mark.asyncio
    async def test_gemini_generate_retries_on_503(self):
        mock_client = MagicMock()
        mock_generate = AsyncMock()

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 10
        mock_usage.candidates_token_count = 5
        mock_usage.total_token_count = 15
        mock_usage.cached_content_token_count = 0

        mock_part = MagicMock()
        mock_part.text = "Gemini recovered after 503"
        mock_part.function_call = None

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        mock_response = MagicMock()
        mock_response.usage_metadata = mock_usage
        mock_response.function_calls = None
        mock_response.candidates = [mock_candidate]

        # Fail with 503 on 1st call, succeed on 2nd
        mock_generate.side_effect = [
            RuntimeError("503 UNAVAILABLE. High demand"),
            mock_response,
        ]
        mock_client.aio.models.generate_content = mock_generate

        provider = GeminiProvider(api_key="fake-key", client=mock_client, initial_delay=0.001, jitter=False)
        messages = [Message(role="user", content="Hello")]

        msg, usage = await provider.generate(messages)
        assert msg.content == "Gemini recovered after 503"
        assert usage.total_tokens == 15
        assert mock_generate.await_count == 2

    @pytest.mark.asyncio
    async def test_gemini_stream_retries_on_503(self):
        mock_client = MagicMock()
        mock_stream_func = AsyncMock()

        class MockChunk:
            def __init__(self, text):
                self.text = text
                self.candidates = []
                self.usage_metadata = None

        async def successful_stream(*args, **kwargs):
            yield MockChunk("Stream ")
            yield MockChunk("recovered!")

        call_count = 0

        async def stream_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("503 UNAVAILABLE. Spikes in demand are usually temporary.")
            return successful_stream()

        mock_stream_func.side_effect = stream_side_effect
        mock_client.aio.models.generate_content_stream = mock_stream_func

        provider = GeminiProvider(api_key="fake-key", client=mock_client, initial_delay=0.001, jitter=False)
        messages = [Message(role="user", content="Hello stream")]

        events = []
        async for item in provider.generate_stream(messages):
            events.append(item)

        text_events = [e.delta for e in events if isinstance(e, StreamEvent) and e.event_type == StreamEventType.TEXT_CHUNK]
        assert "".join(text_events) == "Stream recovered!"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_openai_generate_retries_on_429(self):
        mock_client = MagicMock()
        mock_completions = AsyncMock()

        mock_choice = MagicMock()
        mock_choice.message.content = "OpenAI response after rate limit"
        mock_choice.message.tool_calls = None

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.usage.prompt_tokens_details = None

        mock_completions.side_effect = [
            RuntimeError("429 Rate limit exceeded. Please try again later."),
            mock_response,
        ]
        mock_client.chat.completions.create = mock_completions

        provider = OpenAIProvider(api_key="fake-key", client=mock_client, initial_delay=0.001, jitter=False)
        messages = [Message(role="user", content="Hello")]

        msg, usage = await provider.generate(messages)
        assert msg.content == "OpenAI response after rate limit"
        assert mock_completions.await_count == 2
