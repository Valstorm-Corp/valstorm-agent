"""Comprehensive test suite for DigitalOcean Serverless Inference, DeepSeek, Kimi, and ChainedFallbackProvider."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.keystore import KeyStore
from core.models import Message, SessionState, StreamEvent, StreamEventType, ToolCall, ToolResult
from core.react import ReActEngine
from core.tools import ToolRegistry
from providers import (
    ChainedFallbackProvider,
    FallbackTier,
    OpenAIProvider,
    build_fallback_chain,
    resolve_provider_instance,
)
from providers.base import BaseProvider


def test_resolve_digitalocean_presets():
    """Verify all DigitalOcean and model family presets resolve correctly."""
    # 1. Base DigitalOcean
    p1, m1, name1 = resolve_provider_instance("digitalocean", api_key="dop_v1_test")
    assert isinstance(p1, OpenAIProvider)
    assert p1.base_url == "https://inference.do-ai.run/v1"
    assert m1 == "deepseek-v4-pro"
    assert name1 == "digitalocean"

    # 2. DO shorthand
    p2, m2, name2 = resolve_provider_instance("do", api_key="dop_v1_test", model="claude-sonnet-5")
    assert isinstance(p2, OpenAIProvider)
    assert p2.base_url == "https://inference.do-ai.run/v1"
    assert m2 == "claude-sonnet-5"
    assert name2 == "do"

    # 3. DO DeepSeek shortcut
    p3, m3, _ = resolve_provider_instance("do-deepseek", api_key="dop_v1_test")
    assert p3.base_url == "https://inference.do-ai.run/v1"
    assert m3 == "deepseek-v4-pro"

    # 4. DO Flash shortcut
    p4, m4, _ = resolve_provider_instance("do-flash", api_key="dop_v1_test")
    assert p4.base_url == "https://inference.do-ai.run/v1"
    assert m4 == "deepseek-4-flash"

    # 5. DO Kimi shortcut
    p5, m5, _ = resolve_provider_instance("do-kimi", api_key="dop_v1_test")
    assert p5.base_url == "https://inference.do-ai.run/v1"
    assert m5 == "kimi-k2.6"

    # 6. DO OSS shortcut
    p6, m6, _ = resolve_provider_instance("do-oss", api_key="dop_v1_test")
    assert p6.base_url == "https://inference.do-ai.run/v1"
    assert m6 == "openai-gpt-oss-120b"


def test_keystore_digitalocean_resolution(monkeypatch, tmp_path):
    """Test DigitalOcean key resolution from env vars and config files."""
    # Test env var resolution
    monkeypatch.setenv("DIGITALOCEAN_AI_KEY", "dop_v1_env_key_test")
    ks = KeyStore()
    assert ks.get_api_key("digitalocean") == "dop_v1_env_key_test"
    assert ks.get_api_key("do") == "dop_v1_env_key_test"

    # Test alternate env alias
    monkeypatch.delenv("DIGITALOCEAN_AI_KEY")
    monkeypatch.setenv("DO_GENAI_KEY", "dop_v1_genai_alias")  # gitleaks:allow (fake test value)
    assert ks.get_api_key("digitalocean") == "dop_v1_genai_alias"
    assert ks.get_api_key("do") == "dop_v1_genai_alias"


def test_build_fallback_chain_tier_selection(monkeypatch):
    """Test fallback chain construction across tiers."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy_fake_test_gemini")
    monkeypatch.setenv("DIGITALOCEAN_AI_KEY", "dop_v1_fake_test_do")

    # Reasoner tier
    chain_reasoner = build_fallback_chain(tier="reasoner")
    labels = [t.label for t in chain_reasoner.tiers]
    assert any("Gemini" in l for l in labels)
    assert any("DO DeepSeek" in l for l in labels)

    # Worker tier
    chain_worker = build_fallback_chain(tier="worker")
    w_labels = [t.label for t in chain_worker.tiers]
    assert any("Gemini 3.5 Flash-Lite" in l for l in w_labels)
    assert any("DO DeepSeek V4 Flash" in l for l in w_labels)


class MockFailingProvider(BaseProvider):
    """Mock provider that fails with a specific error."""

    def __init__(self, error_to_raise: Exception, name: str = "mock-fail"):
        super().__init__(default_model="mock-model")
        self.error_to_raise = error_to_raise
        self._name = name
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return self._name

    async def generate(self, messages, tools=None, model=None, **kwargs):
        self.call_count += 1
        raise self.error_to_raise

    async def generate_stream(self, messages, tools=None, model=None, **kwargs):
        self.call_count += 1
        raise self.error_to_raise
        yield  # Make it an async generator


class MockSuccessProvider(BaseProvider):
    """Mock provider that succeeds with a response."""

    def __init__(self, response_text: str, name: str = "mock-success"):
        super().__init__(default_model="mock-model")
        self.response_text = response_text
        self._name = name
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return self._name

    async def generate(self, messages, tools=None, model=None, **kwargs):
        self.call_count += 1
        msg = Message(role="assistant", content=self.response_text, provider=self._name)
        return msg, None

    async def generate_stream(self, messages, tools=None, model=None, **kwargs):
        self.call_count += 1
        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta=self.response_text)
        yield StreamEvent(
            event_type=StreamEventType.TURN_COMPLETE,
            message=Message(role="assistant", content=self.response_text, provider=self._name),
        )


@pytest.mark.asyncio
async def test_chained_fallback_failover_on_503():
    """Verify ChainedFallbackProvider catches 503 and cascades to next tier."""
    # Tier 1 fails with 503 (High Demand / Overloaded)
    tier1_prov = MockFailingProvider(
        error_to_raise=RuntimeError("503 Service Unavailable: High demand spikes in region"),
        name="gemini-mock",
    )
    # Tier 2 succeeds
    tier2_prov = MockSuccessProvider(
        response_text="Hello from DigitalOcean DeepSeek V4 Pro!",
        name="do-deepseek-mock",
    )

    tiers = [
        FallbackTier(provider=tier1_prov, model="gemini-flash-latest", label="Gemini Primary", provider_name="gemini"),
        FallbackTier(provider=tier2_prov, model="deepseek-v4-pro", label="DO DeepSeek V4 Pro", provider_name="do"),
    ]

    fallback_provider = ChainedFallbackProvider(tiers=tiers)
    events: list[StreamEvent] = []

    async for item in fallback_provider.generate_stream(
        messages=[Message(role="user", content="Hello")]
    ):
        if isinstance(item, StreamEvent):
            events.append(item)

    # Verify failover notice was emitted
    text_outputs = [e.delta for e in events if e.delta]
    combined_text = "".join(text_outputs)
    assert "Provider Failover" in combined_text
    assert "Gemini Primary" in combined_text
    assert "DO DeepSeek V4 Pro" in combined_text
    assert "Hello from DigitalOcean DeepSeek V4 Pro!" in combined_text

    # Verify turn complete message was delivered from Tier 2
    turn_complete_event = next(e for e in events if e.event_type == StreamEventType.TURN_COMPLETE)
    assert turn_complete_event.message.content == "Hello from DigitalOcean DeepSeek V4 Pro!"
    assert turn_complete_event.message.provider == "do-deepseek-mock"


@pytest.mark.asyncio
async def test_react_loop_with_chained_fallback():
    """Verify ReActEngine seamless execution through ChainedFallbackProvider."""
    tier1_prov = MockFailingProvider(
        error_to_raise=RuntimeError("429 Resource Exhausted: Rate limit reached"),
        name="gemini-mock",
    )
    tier2_prov = MockSuccessProvider(
        response_text="Task completed accurately by failover model.",
        name="do-deepseek-mock",
    )

    tiers = [
        FallbackTier(provider=tier1_prov, model="gemini-flash-latest", label="Gemini", provider_name="gemini"),
        FallbackTier(provider=tier2_prov, model="deepseek-v4-pro", label="DO DeepSeek", provider_name="do"),
    ]

    fallback_provider = ChainedFallbackProvider(tiers=tiers)
    tools = ToolRegistry()
    engine = ReActEngine(provider=fallback_provider, tools=tools)

    session = SessionState(
        session_id="test_fallback_session",
        active_model="gemini-flash-latest",
        active_provider="fallback",
    )
    res_msg = await engine.run_turn(session=session, user_input="Do a simple task")

    assert res_msg.content == "Task completed accurately by failover model."
    assert len(session.messages) >= 2
    assert session.messages[-1].content == "Task completed accurately by failover model."


@pytest.mark.asyncio
async def test_chained_fallback_non_retryable_error():
    """Verify fatal non-retryable errors are raised immediately without masking."""
    fatal_prov = MockFailingProvider(
        error_to_raise=ValueError("Fatal: Invalid JSON schema parameter 'xyz'"),
        name="fatal-mock",
    )
    backup_prov = MockSuccessProvider(response_text="Should not reach here", name="backup")

    tiers = [
        FallbackTier(provider=fatal_prov, model="m1", label="Tier1", provider_name="p1"),
        FallbackTier(provider=backup_prov, model="m2", label="Tier2", provider_name="p2"),
    ]
    fallback_provider = ChainedFallbackProvider(tiers=tiers)

    with pytest.raises(ValueError, match="Fatal: Invalid JSON schema parameter"):
        async for _ in fallback_provider.generate_stream(messages=[Message(role="user", content="Test")]):
            pass


def test_build_fallback_chain_with_primary_gemini(monkeypatch):
    """Verify specifying primary Gemini automatically builds DO failover tiers underneath."""
    # Pin the primary to Vertex so the result does not depend on whether this machine has Google credentials.
    monkeypatch.setenv("VALSTORM_GEMINI_BACKEND", "vertex")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy_fake_test_gemini")
    monkeypatch.setenv("DIGITALOCEAN_AI_KEY", "dop_v1_fake_test_do")

    chain = build_fallback_chain(
        primary_provider="gemini",
        primary_model="gemini-3.7-flash",
    )
    assert len(chain.tiers) >= 2
    assert chain.tiers[0].provider_name == "gemini"
    assert chain.tiers[0].model == "gemini-3.7-flash"
    assert "Gemini Primary" in chain.tiers[0].label

    # Backups must not duplicate the primary's backend. The one allowed Gemini backup is the explicit AI Studio tier
    # (separate quota from Vertex/GEAP); everything else must be a different provider.
    backups = chain.tiers[1:]
    gemini_backups = [t for t in backups if t.provider_name == "gemini"]
    assert all(t.provider.backend == "aistudio" for t in gemini_backups)
    assert len(gemini_backups) <= 1
    assert any(t.provider_name == "do" for t in backups)


def test_build_fallback_chain_with_primary_anthropic(monkeypatch):
    """Verify specifying primary Anthropic places Anthropic first, followed by Gemini and DO."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", ("sk-ant-" + "fake_test"))
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy_fake_test_gemini")
    monkeypatch.setenv("DIGITALOCEAN_AI_KEY", "dop_v1_fake_test_do")

    chain = build_fallback_chain(
        primary_provider="anthropic",
        primary_model="claude-3-7-sonnet-20250219",
    )
    assert chain.tiers[0].provider_name == "anthropic"
    assert chain.tiers[0].model == "claude-3-7-sonnet-20250219"

    backup_labels = [t.label for t in chain.tiers[1:]]
    assert any("Gemini" in l for l in backup_labels)
    assert any("DO DeepSeek" in l for l in backup_labels)


def test_build_fallback_chain_disable_flag(monkeypatch):
    """Verify fallback can be explicitly disabled via env or flag."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy_fake_test_gemini")
    monkeypatch.setenv("DIGITALOCEAN_AI_KEY", "dop_v1_fake_test_do")

    # With flag enable_fallback=False
    chain_disabled = build_fallback_chain(
        primary_provider="gemini",
        primary_model="gemini-3.7-flash",
        enable_fallback=False,
    )
    assert len(chain_disabled.tiers) == 1
    assert chain_disabled.tiers[0].provider_name == "gemini"

    # With env var
    monkeypatch.setenv("VALSTORM_DISABLE_FALLBACK", "1")
    chain_env_disabled = build_fallback_chain(
        primary_provider="gemini",
        primary_model="gemini-3.7-flash",
    )
    assert len(chain_env_disabled.tiers) == 1


def test_server_get_provider_instance_returns_resilient_chain(monkeypatch):
    """Verify server._get_provider_instance returns a resilient chain rooted at requested model."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy_fake_test_gemini")
    monkeypatch.setenv("DIGITALOCEAN_AI_KEY", "dop_v1_fake_test_do")

    import server
    inst, target_model, resolved_prov = server._get_provider_instance("gemini", "gemini-3.7-flash")

    assert isinstance(inst, ChainedFallbackProvider)
    assert target_model == "gemini-3.7-flash"
    assert resolved_prov == "gemini"
    assert inst.tiers[0].provider_name == "gemini"
    assert inst.tiers[0].model == "gemini-3.7-flash"
    assert any(t.provider_name == "do" for t in inst.tiers[1:])


def test_cli_get_provider_returns_resilient_chain(monkeypatch):
    """Verify cli.get_provider returns a resilient chain rooted at requested model."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy_fake_test_gemini")
    monkeypatch.setenv("DIGITALOCEAN_AI_KEY", "dop_v1_fake_test_do")

    from cli.helpers import get_provider
    provider = get_provider("gemini", model="gemini-flash-latest")

    assert isinstance(provider, ChainedFallbackProvider)
    assert provider.tiers[0].provider_name == "gemini"
    assert any(t.provider_name == "do" for t in provider.tiers[1:])


@pytest.mark.asyncio
async def test_auto_fallback_cascade_from_primary_gemini_to_digitalocean():
    """Verify that when a user requests Gemini as primary, 503/500 automatically fails over to DO."""
    failing_gemini = MockFailingProvider(
        error_to_raise=RuntimeError("503 UNAVAILABLE. High demand spikes. Please try again later."),
        name="gemini",
    )
    succeeding_do = MockSuccessProvider(
        response_text="Resolved seamlessly via DigitalOcean DeepSeek V4 Pro!",
        name="do",
    )

    tiers = [
        FallbackTier(provider=failing_gemini, model="gemini-3.7-flash", label="Gemini Primary (gemini-3.7-flash)", provider_name="gemini"),
        FallbackTier(provider=succeeding_do, model="deepseek-v4-pro", label="DO DeepSeek V4 Pro", provider_name="do"),
    ]

    resilient_chain = ChainedFallbackProvider(tiers=tiers)
    tools = ToolRegistry()
    engine = ReActEngine(provider=resilient_chain, tools=tools)

    session = SessionState(
        session_id="test_gemini_to_do_failover",
        active_model="gemini-3.7-flash",
        active_provider="gemini",
    )

    res_msg = await engine.run_turn(
        session=session,
        user_input="Run build task",
    )

    # Turn must succeed and be answered by DigitalOcean without raising an error
    assert res_msg.content == "Resolved seamlessly via DigitalOcean DeepSeek V4 Pro!"
    assert res_msg.provider == "do"
    assert resilient_chain.active_tier.provider_name == "do"
    assert resilient_chain.active_tier.model == "deepseek-v4-pro"


class MockClientError(Exception):
    """Simulates google.genai.errors.ClientError(400, 'Function call is missing a thought_signature')."""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = status_code
        self.status_code = status_code


@pytest.mark.asyncio
async def test_chained_fallback_failover_on_400_client_error():
    """Verify that a 400 ClientError (e.g. missing thought_signature) triggers failover rather than crashing."""
    failing_gemini_400 = MockFailingProvider(
        error_to_raise=MockClientError(
            "400 INVALID_ARGUMENT. Function call is missing a thought_signature in functionCall parts.",
            status_code=400,
        ),
        name="gemini",
    )
    succeeding_do = MockSuccessProvider(
        response_text="Recovered via DeepSeek after 400 thought_signature mismatch.",
        name="do",
    )

    tiers = [
        FallbackTier(provider=failing_gemini_400, model="gemini-3.7-flash", label="Gemini Primary", provider_name="gemini"),
        FallbackTier(provider=succeeding_do, model="deepseek-v4-pro", label="DO DeepSeek V4 Pro", provider_name="do"),
    ]

    resilient_chain = ChainedFallbackProvider(tiers=tiers)
    tools = ToolRegistry()
    engine = ReActEngine(provider=resilient_chain, tools=tools)

    session = SessionState(
        session_id="test_gemini_400_failover",
        active_model="gemini-3.7-flash",
        active_provider="gemini",
    )

    res_msg = await engine.run_turn(session=session, user_input="Perform research task")

    assert res_msg.content == "Recovered via DeepSeek after 400 thought_signature mismatch."
    assert res_msg.provider == "do"
    assert resilient_chain.active_tier.model == "deepseek-v4-pro"


@pytest.mark.asyncio
async def test_chained_fallback_cooldown_skips_failed_tier_on_next_turn():
    """Verify that after Tier 1 fails, a subsequent turn within cooldown directly starts at Tier 2."""
    tier1 = MockFailingProvider(
        error_to_raise=RuntimeError("503 UNAVAILABLE. Demand spike."),
        name="tier1-failing",
    )
    tier2 = MockSuccessProvider(
        response_text="Tier 2 success",
        name="tier2-ok",
    )

    tiers = [
        FallbackTier(provider=tier1, model="m1", label="Tier 1", provider_name="p1"),
        FallbackTier(provider=tier2, model="m2", label="Tier 2", provider_name="p2"),
    ]

    fallback = ChainedFallbackProvider(tiers=tiers, cooldown_sec=120.0)

    # First turn: Tier 1 fails, cascades to Tier 2
    events1 = []
    async for ev in fallback.generate_stream(messages=[Message(role="user", content="Turn 1")]):
        events1.append(ev)

    assert tier1.call_count == 1
    assert tier2.call_count == 1
    assert fallback.active_tier.label == "Tier 2"

    # Second turn immediately after (within cooldown): Tier 1 should be skipped directly!
    events2 = []
    async for ev in fallback.generate_stream(messages=[Message(role="user", content="Turn 2")]):
        events2.append(ev)

    # Tier 1 call count must NOT have incremented (it was skipped during cooldown!)
    assert tier1.call_count == 1
    # Tier 2 was called directly
    assert tier2.call_count == 2


def test_gemini_format_messages_converts_unsigned_tool_calls_to_text():
    """Verify tool calls without thought_signature (e.g. from DeepSeek) are rendered as text to prevent 400."""
    from providers.gemini import GeminiProvider
    provider = GeminiProvider(api_key="fake_key")

    messages = [
        Message(role="user", content="Search files"),
        Message(
            role="assistant",
            content="I will search the repo.",
            provider="do",
            tool_calls=[
                ToolCall(id="call_123", name="search_files", arguments={"pattern": "*.py"}),
            ],
        ),
        Message(
            role="tool",
            tool_result=ToolResult(call_id="call_123", name="search_files", output={"files": ["main.py"]}),
        ),
        Message(role="user", content="Next question"),
    ]

    _, contents = provider._format_messages(messages)

    # Inspect the model parts in the formatted contents
    model_contents = [c for c in contents if c.role == "model"]
    assert len(model_contents) == 1

    # Ensure NO function_call part exists on model turn (because it lacked thought_signature)
    for p in model_contents[0].parts:
        assert getattr(p, "function_call", None) is None

    # Ensure a text observation part was generated describing the execution
    model_text = " ".join(p.text for p in model_contents[0].parts if getattr(p, "text", None))
    assert "[Executed tool `search_files`" in model_text
    assert "*.py" in model_text

    # Ensure the corresponding user tool response turn was also rendered as text rather than an orphaned function_response
    user_contents = [c for c in contents if c.role == "user"]
    all_user_text = " ".join(
        p.text for c in user_contents for p in c.parts if getattr(p, "text", None)
    )
    assert "[Historical tool result search_files]" in all_user_text
    assert "main.py" in all_user_text


def test_gemini_format_messages_preserves_signed_tool_calls():
    """Verify tool calls WITH thought_signature continue to emit native function_call parts."""
    from providers.gemini import GeminiProvider
    provider = GeminiProvider(api_key="fake_key")

    messages = [
        Message(role="user", content="Search files"),
        Message(
            role="assistant",
            content="I will search.",
            tool_calls=[
                ToolCall(id="call_signed", name="search_files", arguments={"pattern": "*.ts"}, thought_signature=b"crypto_sig"),
            ],
        ),
        Message(
            role="tool",
            tool_result=ToolResult(call_id="call_signed", name="search_files", output={"files": ["app.ts"]}),
        ),
    ]

    _, contents = provider._format_messages(messages)

    model_contents = [c for c in contents if c.role == "model"]
    assert len(model_contents) == 1

    # Native function_call part MUST exist with the thought_signature attached
    fc_parts = [p for p in model_contents[0].parts if getattr(p, "function_call", None)]
    assert len(fc_parts) == 1
    assert fc_parts[0].function_call.name == "search_files"
    assert fc_parts[0].thought_signature == b"crypto_sig"
