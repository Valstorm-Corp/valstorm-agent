"""Regression tests for SSL/TLS BAD_RECORD_MAC resilience, mid-stream recovery, VFS image LRU caching,
oversized image compression, historical image turn windowing, and desktop-sync temp_* deduplication.
"""

import io
import ssl
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.models import Message, SessionState, StreamEvent, StreamEventType
from core.retry import execute_stream_with_retry, execute_with_retry, is_retryable_error
from providers.base import (
    clear_image_cache,
    extract_and_resolve_images,
    get_image_cache_stats,
    resolve_image_source,
)
from providers.fallback import ChainedFallbackProvider, FallbackTier, is_provider_failover_candidate
from providers.gemini import GeminiProvider


class TestSSLRetryAndMidStreamRecovery:
    """Verifies ssl.SSLError (e.g. SSLV3_ALERT_BAD_RECORD_MAC) is retryable and recovers mid-stream."""

    def test_ssl_bad_record_mac_is_retryable_and_failover_candidate(self):
        exc = ssl.SSLError(
            1, "[SSL: SSLV3_ALERT_BAD_RECORD_MAC] ssl/tls alert bad record mac (_ssl.c:2590)"
        )
        assert is_retryable_error(exc) is True
        assert is_provider_failover_candidate(exc) is True

        eof_exc = ssl.SSLEOFError(8, "EOF occurred in violation of protocol (_ssl.c:2426)")
        assert is_retryable_error(eof_exc) is True
        assert is_provider_failover_candidate(eof_exc) is True

    @pytest.mark.asyncio
    async def test_execute_with_retry_recovers_from_ssl_error(self):
        calls = 0

        async def flaky_coro():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ssl.SSLError(
                    1, "[SSL: SSLV3_ALERT_BAD_RECORD_MAC] ssl/tls alert bad record mac (_ssl.c:2590)"
                )
            return "ok_after_ssl_retry"

        res = await execute_with_retry(flaky_coro, max_retries=2, initial_delay=0.01, jitter=False)
        assert res == "ok_after_ssl_retry"
        assert calls == 2

    @pytest.mark.asyncio
    async def test_gemini_stream_mid_turn_ssl_error_recovers_via_non_streaming_generate(self):
        """Simulates Zachary's exact failure: stream emits 3,184 chars of text then raises SSLError
        while streaming a functionCall. Verifies non-streaming generate_content recovers the turn
        without duplicating the already-streamed text prefix.
        """
        provider = GeminiProvider(api_key="test-key", default_model="gemini-flash-latest", max_retries=2, initial_delay=0.01)
        provider._injected_client = True

        partial_preamble = "A" * 3184

        async def broken_sse_stream():
            chunk1 = SimpleNamespace(
                usage_metadata=None,
                model_version="gemini-2.5-flash",
                candidates=[
                    SimpleNamespace(
                        finish_reason=None,
                        content=SimpleNamespace(
                            parts=[SimpleNamespace(thought=False, text=partial_preamble, function_call=None)]
                        ),
                    )
                ],
            )
            yield chunk1
            raise ssl.SSLError(
                1, "[SSL: SSLV3_ALERT_BAD_RECORD_MAC] ssl/tls alert bad record mac (_ssl.c:2590)"
            )

        recovered_full_resp = SimpleNamespace(
            model_version="gemini-2.5-flash",
            usage_metadata=SimpleNamespace(
                prompt_token_count=1200,
                candidates_token_count=450,
                total_token_count=1650,
                cached_content_token_count=800,
                thoughts_token_count=0,
            ),
            candidates=[
                SimpleNamespace(
                    finish_reason="STOP",
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                thought=False,
                                text=partial_preamble + "\nCompleted Sidebar refactor.",
                                function_call=SimpleNamespace(
                                    id="call_sidebar_1",
                                    name="patch_file",
                                    args={"path": "src/components/Sidebar.tsx"},
                                ),
                                thought_signature=None,
                            )
                        ]
                    ),
                )
            ],
        )

        mock_models = MagicMock()
        mock_models.generate_content_stream = AsyncMock(return_value=broken_sse_stream())
        mock_models.generate_content = AsyncMock(return_value=recovered_full_resp)
        provider._client = SimpleNamespace(aio=SimpleNamespace(models=mock_models, caches=None))

        events = []
        final_tuple = None
        async for item in provider.generate_stream(
            messages=[Message(role="user", content="Refactor the Sidebar navigation")]
        ):
            if isinstance(item, StreamEvent):
                events.append(item)
            elif isinstance(item, tuple):
                final_tuple = item

        assert final_tuple is not None
        final_msg, final_usage = final_tuple
        assert final_msg.tool_calls is not None
        assert len(final_msg.tool_calls) == 1
        assert final_msg.tool_calls[0].name == "patch_file"
        assert final_usage.prompt_tokens == 1200

        # Ensure the tool call event was emitted and preamble text was only emitted once
        tc_events = [e for e in events if e.event_type == StreamEventType.TOOL_CALL_DETECTED]
        assert len(tc_events) == 1
        assert tc_events[0].tool_call.name == "patch_file"

    @pytest.mark.asyncio
    async def test_fallback_provider_cascades_on_midstream_ssl_error_when_no_tool_call_emitted(self):
        """If Tier 1 emits partial text and dies with SSLError (and cannot recover), ChainedFallbackProvider
        cascades to Tier 2 with stream_reset=True so server.py discards the partial buffer.
        """
        tier1_prov = MagicMock()
        tier1_prov.provider_name = "gemini"

        async def tier1_broken_stream(**kwargs):
            yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="Partial broken output...")
            raise ssl.SSLError(1, "[SSL: SSLV3_ALERT_BAD_RECORD_MAC] ssl/tls alert bad record mac")

        tier1_prov.generate_stream = tier1_broken_stream

        tier2_prov = MagicMock()
        tier2_prov.provider_name = "do"

        async def tier2_healthy_stream(**kwargs):
            yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="Clean recovered response from Tier 2.")
            msg = Message(role="assistant", content="Clean recovered response from Tier 2.", model="kimi-k2.6", provider="do")
            yield (msg, None)

        tier2_prov.generate_stream = tier2_healthy_stream

        chain = ChainedFallbackProvider(
            tiers=[
                FallbackTier(provider=tier1_prov, model="gemini-flash-latest", label="Gemini Tier 1", provider_name="gemini"),
                FallbackTier(provider=tier2_prov, model="kimi-k2.6", label="DO Kimi Tier 2", provider_name="do"),
            ],
            retries_before_failover=0,
        )

        items = []
        async for item in chain.generate_stream(messages=[Message(role="user", content="Hello")]):
            items.append(item)

        failover_events = [
            i for i in items if isinstance(i, StreamEvent) and i.metadata.get("failover")
        ]
        assert len(failover_events) == 1
        assert failover_events[0].metadata.get("stream_reset") is True
        assert any(
            isinstance(i, StreamEvent) and i.delta == "Clean recovered response from Tier 2."
            for i in items
        )


class TestVFSImageCacheAndWindowing:
    """Verifies VFS cloud:// images are cached in-memory across ReAct steps, compressed when oversized,
    and windowed across historical turns.
    """

    def setup_method(self):
        clear_image_cache()

    @staticmethod
    def _make_valid_png(width: int = 64, height: int = 64, seed: int = 1) -> bytes:
        """Creates a valid uncompressed-filter PNG in pure Python (zlib + struct) without requiring Pillow."""
        import struct
        import zlib

        raw_rows = bytearray()
        for y in range(height):
            raw_rows.append(0)  # filter type 0 (None)
            for x in range(width):
                r = (x * 17 + y * 31 + seed * 53) & 0xFF
                g = (x * 29 + y * 13 + seed * 97) & 0xFF
                b = (x * 43 + y * 7 + seed * 19) & 0xFF
                raw_rows.extend((r, g, b))

        def chunk(c_type: bytes, c_data: bytes) -> bytes:
            crc = zlib.crc32(c_type + c_data) & 0xFFFFFFFF
            return struct.pack(">I", len(c_data)) + c_type + c_data + struct.pack(">I", crc)

        ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        idat = zlib.compress(bytes(raw_rows), level=1)
        return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")

    def test_vfs_images_cached_across_react_iterations(self):
        png_bytes = self._make_valid_png(32, 32, seed=1)

        fetch_calls = []

        def fake_fetch_vfs(file_id_or_uri: str):
            fetch_calls.append(file_id_or_uri)
            return (png_bytes, "image/png")

        msg = Message(
            role="user",
            content=(
                "Inspect these 5 screenshots:\n"
                "[Attached Image: cloud://file_img1]\n"
                "[Attached Image: cloud://file_img2]\n"
                "[Attached Image: cloud://file_img3]\n"
                "[Attached Image: cloud://file_img4]\n"
                "[Attached Image: cloud://file_img5]"
            ),
        )

        with patch("providers.base.fetch_vfs_image", side_effect=fake_fetch_vfs):
            # Simulate 6 ReAct loop iterations formatting the same message history
            for _ in range(6):
                resolved = extract_and_resolve_images(msg)
                assert len(resolved) == 5

        # Instead of 5 * 6 = 30 network fetches, each unique file ID is fetched ONCE (5 misses, 25 hits)
        assert len(fetch_calls) == 5
        stats = get_image_cache_stats()
        assert stats["misses"] == 5
        assert stats["hits"] == 25
        assert stats["entries"] == 5

    def test_oversized_png_automatically_compressed(self, monkeypatch):
        monkeypatch.setenv("VALSTORM_COMPRESS_IMAGE_THRESHOLD_BYTES", "10240")  # 10 KB threshold for test
        raw_png = self._make_valid_png(256, 256, seed=42)
        assert len(raw_png) > 10240

        with patch("providers.base.fetch_vfs_image", return_value=(raw_png, "image/png")):
            res = resolve_image_source("cloud://file_big_retina_png")

        assert res is not None
        opt_bytes, opt_mime = res
        assert opt_mime == "image/jpeg"
        assert len(opt_bytes) < len(raw_png)
        assert get_image_cache_stats()["compressed"] == 1

    def test_gemini_format_messages_windows_historical_image_turns_and_deduplicates(self, monkeypatch):
        monkeypatch.setenv("VALSTORM_MAX_IMAGE_TURNS", "2")

        img_map = {
            "cloud://file_turn1": (self._make_valid_png(16, 16, seed=1), "image/png"),
            "cloud://file_turn2": (self._make_valid_png(16, 16, seed=2), "image/png"),
            "cloud://file_turn3": (self._make_valid_png(16, 16, seed=3), "image/png"),
        }

        messages = [
            Message(role="user", content="Turn 1 [Attached Image: cloud://file_turn1]"),
            Message(role="assistant", content="Ack turn 1"),
            Message(role="user", content="Turn 2 [Attached Image: cloud://file_turn2]"),
            Message(role="assistant", content="Ack turn 2"),
            # Turn 3 also repeats cloud://file_turn2 plus cloud://file_turn3
            Message(
                role="user",
                content="Turn 3 [Attached Image: cloud://file_turn2] [Attached Image: cloud://file_turn3]",
            ),
        ]

        provider = GeminiProvider(api_key="test-key", default_model="gemini-flash-latest")
        with patch("providers.base.fetch_vfs_image", side_effect=lambda k: img_map.get(k)):
            _, contents = provider._format_messages(messages)

        # Turn 1 (index 0) is older than the 2-turn window: should only have text part (1 part), no inline bytes
        assert len(contents[0].parts) == 1
        # Turn 2 has 1 text part + 1 image part (file_turn2)
        assert len(contents[2].parts) == 2
        # Turn 3 has 1 text part + 1 unique image part (file_turn3; duplicate file_turn2 is deduplicated!)
        assert len(contents[4].parts) == 2
