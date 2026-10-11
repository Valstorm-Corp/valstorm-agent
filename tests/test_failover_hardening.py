"""Tests for failover hardening: cascade tier inference, DigitalOcean model IDs, key resolution and cascade behavior."""

import pytest
from typer.testing import CliRunner

from core.keystore import KeyStore, looks_like_placeholder_key
from core.models import Message, StreamEvent, StreamEventType
from providers import (
    DEFAULT_REASONER_CASCADE_SPECS,
    DEFAULT_THINKER_CASCADE_SPECS,
    DEFAULT_WORKER_CASCADE_SPECS,
    OPENAI_COMPATIBLE_PRESETS,
    ChainedFallbackProvider,
    FallbackTier,
    build_fallback_chain,
    infer_cascade_tier,
)
from providers.base import BaseProvider
from providers.fallback import permanent_tier_failure_reason

# Model IDs present in DigitalOcean's catalog (GET https://inference.do-ai.run/v1/models) on 2026-10-01.
# Only the IDs the runtime references; extend this when a new DO model is added to a cascade.
VERIFIED_DO_CATALOG_IDS = frozenset({
    "deepseek-v4-pro",
    "deepseek-4-flash",
    "kimi-k2.6",
    "openai-gpt-oss-120b",
    "openai-gpt-5-nano",
    "openai-gpt-5.6-sol",
    "anthropic-claude-opus-5",
    "llama-4-maverick",
})


# --------------------------------------------------------------------------- tier inference


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gemini-flash-latest", "reasoner"),  # "gemini" contains "mini": must NOT be treated as a mini model
        ("gemini-3.5-flash", "reasoner"),
        ("gemini-pro-latest", "thinker"),
        ("gemini-flash-lite-latest", "worker"),
        ("gpt-5-nano", "worker"),
        ("gpt-4o-mini", "worker"),
        ("o3-mini", "worker"),
        ("claude-opus-5", "thinker"),
        ("deepseek-v4-pro", "thinker"),
        ("kimi-k2.6", "reasoner"),
        (None, "reasoner"),
    ],
)
def test_infer_cascade_tier_matches_whole_tokens(model, expected):
    assert infer_cascade_tier(model_name=model) == expected


# --------------------------------------------------------------------------- DO model IDs


def test_default_cascade_do_model_ids_exist_in_do_catalog():
    for specs in (DEFAULT_REASONER_CASCADE_SPECS, DEFAULT_THINKER_CASCADE_SPECS, DEFAULT_WORKER_CASCADE_SPECS):
        for spec in specs:
            if spec["provider"] in ("do", "digitalocean"):
                assert spec["model"] in VERIFIED_DO_CATALOG_IDS, f"{spec['label']}: unknown DO model id {spec['model']!r}"


def test_do_preset_default_models_exist_in_do_catalog():
    for name, preset in OPENAI_COMPATIBLE_PRESETS.items():
        if "do-ai.run" in (preset.get("base_url") or ""):
            assert preset["default_model"] in VERIFIED_DO_CATALOG_IDS, f"preset {name!r}: {preset['default_model']!r}"


# --------------------------------------------------------------------------- key resolution


def _clear_key_env(monkeypatch):
    for names in KeyStore.PROVIDER_ENV_MAP.values():
        for name in names:
            monkeypatch.delenv(name, raising=False)
    for name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "VALSTORM_DISABLE_FALLBACK"):
        monkeypatch.delenv(name, raising=False)


def test_nearest_dotenv_wins_over_parent_dotenv(tmp_path, monkeypatch):
    _clear_key_env(monkeypatch)
    (tmp_path / ".env").write_text("DIGITALOCEAN_AI_KEY=far_parent_key_000000\n")
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("DIGITALOCEAN_AI_KEY=near_project_key_00000\n")
    monkeypatch.chdir(project)

    ks = KeyStore(config_path=tmp_path / "keys.json", env_file_path=tmp_path / "missing.env")
    key, source = ks.get_api_key_with_source("do")
    assert key == "near_project_key_00000"
    assert source.endswith("project/.env:DIGITALOCEAN_AI_KEY")


def test_explicit_env_file_wins_over_discovered_files(tmp_path, monkeypatch):
    _clear_key_env(monkeypatch)
    (tmp_path / ".env").write_text("DIGITALOCEAN_AI_KEY=discovered_key_000000\n")
    explicit = tmp_path / "explicit.env"
    explicit.write_text("DIGITALOCEAN_AI_KEY=explicit_key_0000000\n")
    monkeypatch.chdir(tmp_path)

    ks = KeyStore(config_path=tmp_path / "keys.json", env_file_path=explicit)
    assert ks.get_api_key("do") == "explicit_key_0000000"


def test_key_source_reporting(tmp_path, monkeypatch):
    _clear_key_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "keys.json"
    ks = KeyStore(config_path=config, env_file_path=tmp_path / ".env.ai.keys")

    assert ks.get_api_key_with_source("openai") == (None, None)
    assert ks.get_api_key_with_source("openai", override_key="explicit-123456") == ("explicit-123456", "override")

    (tmp_path / ".env.ai.keys").write_text("OPENAI_API_KEY=dotenv_openai_key_1\n")
    key, source = ks.get_api_key_with_source("openai")
    assert key == "dotenv_openai_key_1" and source.endswith(".env.ai.keys:OPENAI_API_KEY")

    config.write_text('{"openai": "json_openai_key_123"}')
    key, source = ks.get_api_key_with_source("openai")
    assert key == "json_openai_key_123" and source.endswith("keys.json:openai")

    monkeypatch.setenv("OPENAI_API_KEY", "env_openai_key_1234")
    assert ks.get_api_key_with_source("openai") == ("env_openai_key_1234", "env:OPENAI_API_KEY")


@pytest.mark.parametrize("value", ["local-vsagent", "changeme", "your_key_here", "short", "<token>", "${OPENAI_API_KEY}", "", None])
def test_placeholder_keys_are_detected(value):
    assert looks_like_placeholder_key(value) is True


@pytest.mark.parametrize("value", ["sk-ant-api03-abcdefghijkl", "doo_v1_" + "a" * 64, "AIzaSy_fake_test_gemini"])
def test_real_looking_keys_are_not_placeholders(value):
    assert looks_like_placeholder_key(value) is False


def test_build_chain_skips_placeholder_backup_and_records_key_source(tmp_path, monkeypatch):
    _clear_key_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "local-vsagent")
    monkeypatch.setenv("DIGITALOCEAN_AI_KEY", "dop_v1_" + "a" * 40)
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy_fake_test_gemini")
    ks = KeyStore(config_path=tmp_path / "keys.json", env_file_path=tmp_path / ".env.ai.keys")

    chain = build_fallback_chain(tier="reasoner", keystore=ks)

    assert not any("OpenAI" in t.label for t in chain.tiers)
    do_tiers = [t for t in chain.tiers if t.provider_name == "do"]
    assert do_tiers
    assert all(t.key_source == "env:DIGITALOCEAN_AI_KEY" for t in do_tiers)


# --------------------------------------------------------------------------- cascade behavior


class HttpError(Exception):
    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status_code = status
        self.code = status


class AuthenticationError(Exception):
    """Mimics openai.AuthenticationError by class name."""


class ScriptedProvider(BaseProvider):
    """Provider whose streaming behavior is scripted: optional text chunks, then optionally an error."""

    def __init__(self, name: str, text: str = "", error: Exception = None, chunks_before_error: int = 0):
        super().__init__(default_model="mock-model")
        self._name = name
        self.text = text
        self.error = error
        self.chunks_before_error = chunks_before_error
        self.call_count = 0
        self.last_kwargs = {}

    @property
    def provider_name(self) -> str:
        return self._name

    async def generate(self, messages, tools=None, model=None, **kwargs):
        raise NotImplementedError

    async def generate_stream(self, messages, tools=None, model=None, **kwargs):
        self.call_count += 1
        self.last_kwargs = dict(kwargs)
        for _ in range(self.chunks_before_error):
            yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="partial ")
        if self.error is not None:
            raise self.error
        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta=self.text)
        yield StreamEvent(
            event_type=StreamEventType.TURN_COMPLETE,
            message=Message(role="assistant", content=self.text, provider=self._name),
        )


def _tier(provider: ScriptedProvider, label: str) -> FallbackTier:
    return FallbackTier(provider=provider, model=f"{label}-model", label=label, provider_name=provider.provider_name)


async def _run_turn(chain: ChainedFallbackProvider) -> str:
    text = ""
    async for item in chain.generate_stream(messages=[Message(role="user", content="hi")]):
        if isinstance(item, StreamEvent) and item.event_type == StreamEventType.TURN_COMPLETE:
            text = item.message.content
    return text


def test_permanent_tier_failure_reason():
    assert permanent_tier_failure_reason(HttpError("x", 401)) == "HTTP 401"
    assert permanent_tier_failure_reason(HttpError("x", 404)) == "HTTP 404"
    assert permanent_tier_failure_reason(AuthenticationError("bad key")) == "AuthenticationError"
    assert permanent_tier_failure_reason(HttpError("overloaded", 529)) is None
    assert permanent_tier_failure_reason(HttpError("rate limited", 429)) is None
    assert permanent_tier_failure_reason(RuntimeError("503 UNAVAILABLE")) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 404])
async def test_backup_tier_with_bad_key_or_model_is_disabled_and_skipped(status):
    primary = ScriptedProvider("p", error=RuntimeError("503 UNAVAILABLE. High demand."))
    broken = ScriptedProvider("do", error=HttpError("Unable to authenticate you", status))
    healthy = ScriptedProvider("ok", text="served by third tier")
    chain = ChainedFallbackProvider(
        tiers=[_tier(primary, "primary"), _tier(broken, "broken"), _tier(healthy, "healthy")],
        cooldown_sec=120.0,
    )

    assert await _run_turn(chain) == "served by third tier"
    assert broken.call_count == 1
    assert chain._dead_tiers == {1: f"HTTP {status}"}

    # Next turn: primary is cooling down, the broken tier is disabled -> straight to the healthy tier.
    assert await _run_turn(chain) == "served by third tier"
    assert primary.call_count == 1
    assert broken.call_count == 1
    assert healthy.call_count == 2


@pytest.mark.asyncio
async def test_primary_auth_failure_is_not_permanently_disabled():
    primary = ScriptedProvider("p", error=HttpError("expired token", 401))
    backup = ScriptedProvider("ok", text="backup answer")
    chain = ChainedFallbackProvider(tiers=[_tier(primary, "primary"), _tier(backup, "backup")], cooldown_sec=0.0)

    assert await _run_turn(chain) == "backup answer"
    assert await _run_turn(chain) == "backup answer"
    assert primary.call_count == 2  # retried on the next turn (re-login can fix it)
    assert chain._dead_tiers == {}


@pytest.mark.asyncio
async def test_failure_after_output_does_not_fail_over_or_duplicate_text():
    primary = ScriptedProvider("p", error=RuntimeError("503 UNAVAILABLE mid-stream"), chunks_before_error=1)
    backup = ScriptedProvider("ok", text="should never run")
    chain = ChainedFallbackProvider(tiers=[_tier(primary, "primary"), _tier(backup, "backup")])

    with pytest.raises(RuntimeError, match="mid-stream"):
        await _run_turn(chain)
    assert backup.call_count == 0


@pytest.mark.asyncio
async def test_failure_before_output_still_fails_over():
    primary = ScriptedProvider("p", error=RuntimeError("503 UNAVAILABLE. High demand."))
    backup = ScriptedProvider("ok", text="recovered")
    chain = ChainedFallbackProvider(tiers=[_tier(primary, "primary"), _tier(backup, "backup")])

    assert await _run_turn(chain) == "recovered"


@pytest.mark.asyncio
async def test_all_backup_tiers_disabled_surfaces_a_clear_error():
    primary = ScriptedProvider("p", error=RuntimeError("503 UNAVAILABLE"))
    broken = ScriptedProvider("do", error=HttpError("Unable to authenticate you", 401))
    chain = ChainedFallbackProvider(tiers=[_tier(primary, "primary"), _tier(broken, "broken")], cooldown_sec=120.0)

    with pytest.raises(HttpError):  # turn 1: the last live tier's own error is raised
        await _run_turn(chain)
    with pytest.raises(RuntimeError, match="exhausted"):  # turn 2: nothing left to try
        await _run_turn(chain)


def test_reset_active_tier_clears_disabled_tiers():
    chain = ChainedFallbackProvider(tiers=[_tier(ScriptedProvider("a"), "a"), _tier(ScriptedProvider("b"), "b")])
    chain._dead_tiers[1] = "HTTP 401"
    chain.reset_active_tier()
    assert chain._dead_tiers == {}


# --------------------------------------------------------------------------- doctor command


def test_doctor_offline_reports_tiers_and_flags_missing_do_model(tmp_path, monkeypatch):
    _clear_key_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "220")  # keep rich from wrapping table cells in the captured output
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy_fake_test_gemini")
    monkeypatch.setenv("DIGITALOCEAN_AI_KEY", "dop_v1_" + "a" * 40)

    import cli.cmds.doctor_cmd as doctor

    # Catalog without the cascade's DO models -> every DO tier must be flagged, without any network access.
    monkeypatch.setattr(doctor, "_fetch_do_catalog", lambda base, key, timeout: ({"some-other-model"}, None))

    from cli.main import app

    result = CliRunner().invoke(app, ["doctor", "--offline", "--provider", "gemini", "--model", "gemini-flash-latest"])
    assert result.exit_code == 0, result.output
    assert "MISSING" in result.output
    assert "healthy" in result.output


@pytest.mark.asyncio
async def test_retry_budget_is_capped_only_while_another_tier_remains():
    primary = ScriptedProvider("p", error=RuntimeError("529 overloaded"))
    last = ScriptedProvider("ok", text="served")
    chain = ChainedFallbackProvider(tiers=[_tier(primary, "primary"), _tier(last, "last")])

    assert await _run_turn(chain) == "served"
    assert primary.last_kwargs.get("max_retries") == 1  # moves on after one retry instead of the full backoff ladder
    assert "max_retries" not in last.last_kwargs  # the last tier keeps the provider's own retry budget


@pytest.mark.asyncio
async def test_retry_budget_override_and_caller_value_win(monkeypatch):
    monkeypatch.setenv("VALSTORM_RETRIES_BEFORE_FAILOVER", "0")
    primary = ScriptedProvider("p", error=RuntimeError("529 overloaded"))
    chain = ChainedFallbackProvider(tiers=[_tier(primary, "primary"), _tier(ScriptedProvider("ok", text="x"), "last")])
    await _run_turn(chain)
    assert primary.last_kwargs["max_retries"] == 0

    explicit = ScriptedProvider("p2", error=RuntimeError("529 overloaded"))
    chain = ChainedFallbackProvider(tiers=[_tier(explicit, "primary"), _tier(ScriptedProvider("ok", text="x"), "last")])
    async for _ in chain.generate_stream(messages=[Message(role="user", content="hi")], max_retries=5):
        pass
    assert explicit.last_kwargs["max_retries"] == 5


# --------------------------------------------------------------------------- AI Studio tier (separate quota from GEAP)


def test_tier2_is_an_explicit_ai_studio_gemini_tier_when_primary_is_valstorm(tmp_path, monkeypatch):
    _clear_key_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy_fake_test_gemini")
    monkeypatch.setenv("DIGITALOCEAN_AI_KEY", "dop_v1_" + "a" * 40)
    ks = KeyStore(config_path=tmp_path / "keys.json", env_file_path=tmp_path / ".env.ai.keys")

    chain = build_fallback_chain(
        tier="reasoner", primary_provider="valstorm", primary_model="gemini-flash-latest", keystore=ks, api_key="vs-token-123456"
    )

    studio = [t for t in chain.tiers[1:] if t.provider_name == "gemini"]
    assert len(studio) == 1
    assert studio[0].provider.backend == "aistudio"  # forced: never silently Vertex, which shares GEAP's quota
    assert studio[0].provider.backend_label() == "aistudio"
    assert studio[0].key_source == "env:GEMINI_API_KEY"
    assert chain.tiers[1] is studio[0]  # right after the primary, ahead of every DO tier


def test_ai_studio_tier_is_skipped_without_a_gemini_key(tmp_path, monkeypatch):
    _clear_key_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DIGITALOCEAN_AI_KEY", "dop_v1_" + "a" * 40)
    ks = KeyStore(config_path=tmp_path / "keys.json", env_file_path=tmp_path / ".env.ai.keys")

    chain = build_fallback_chain(
        tier="reasoner", primary_provider="valstorm", primary_model="gemini-flash-latest", keystore=ks, api_key="vs-token-123456"
    )

    assert not [t for t in chain.tiers[1:] if t.provider_name == "gemini"]
    assert any(t.provider_name == "do" for t in chain.tiers)


def test_ai_studio_tier_is_skipped_when_primary_already_runs_on_ai_studio(tmp_path, monkeypatch):
    _clear_key_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy_fake_test_gemini")
    ks = KeyStore(config_path=tmp_path / "keys.json", env_file_path=tmp_path / ".env.ai.keys")

    chain = build_fallback_chain(
        tier="reasoner", primary_provider="aistudio", primary_model="gemini-flash-latest", keystore=ks
    )

    assert len([t for t in chain.tiers if t.provider_name == "gemini"]) == 1


# --------------------------------------------------------------------------- Gemini -> non-Gemini history handoff


def _gemini_origin_history():
    from core.models import ToolCall, ToolResult

    call = ToolCall(id="call_1", name="add", arguments={"a": 1, "b": 2}, thought_signature=b"sig-bytes")
    return [
        Message(role="user", content="1+2?"),
        Message(role="assistant", content=None, tool_calls=[call], provider="gemini", model="gemini-flash-latest"),
        Message(role="tool", content="3", tool_result=ToolResult(call_id="call_1", name="add", output=3)),
    ]


@pytest.mark.parametrize("name", ["do", "openai", "deepseek", "kimi"])
def test_openai_compatible_providers_do_not_forward_thought_signature(name):
    from providers.openai import OpenAIProvider

    provider = OpenAIProvider(api_key="k" * 20, base_url="http://mock", provider_name=name)
    formatted = provider._format_messages(_gemini_origin_history())

    tool_call = formatted[1]["tool_calls"][0]
    assert "thought_signature" not in tool_call
    assert tool_call["function"]["name"] == "add"
    assert formatted[2]["role"] == "tool" and formatted[2]["tool_call_id"] == "call_1"  # result still pairs with its call


def test_valstorm_gateway_still_receives_thought_signature():
    from providers.openai import OpenAIProvider

    provider = OpenAIProvider(api_key="k" * 20, base_url="http://mock", provider_name="valstorm")
    tool_call = provider._format_messages(_gemini_origin_history())[1]["tool_calls"][0]
    assert "thought_signature" in tool_call


# --------------------------------------------------------------------------- doctor --replay


def test_replay_history_is_shaped_for_the_target_family():
    import cli.cmds.doctor_cmd as doctor

    to_gemini = doctor._replay_messages(target_is_gemini=True)
    to_other = doctor._replay_messages(target_is_gemini=False)

    assert to_gemini[1].tool_calls[0].thought_signature is None and to_gemini[1].provider == "do"
    assert to_other[1].tool_calls[0].thought_signature is not None and to_other[1].provider == "gemini"
    for history in (to_gemini, to_other):
        assert [m.role for m in history] == ["user", "assistant", "tool"]
        assert history[2].tool_result.call_id == history[1].tool_calls[0].id


@pytest.mark.asyncio
async def test_replay_probe_passes_only_when_the_model_answers_from_the_tool_result():
    import cli.cmds.doctor_cmd as doctor
    from core.models import ToolCall

    class Canned(BaseProvider):
        def __init__(self, reply):
            super().__init__(api_key="k" * 20, default_model="m")
            self.reply = reply
            self.seen = None

        @property
        def provider_name(self):
            return "do"

        async def generate(self, messages, tools=None, model=None, **kwargs):
            self.seen = (messages, tools)
            return self.reply, None

        async def generate_stream(self, messages, tools=None, model=None, **kwargs):
            if False:
                yield None

    def tier_for(reply):
        return FallbackTier(provider=Canned(reply), model="m", label="canned", provider_name="do")

    ok, detail, _ = await doctor._probe_replay(tier_for(Message(role="assistant", content="The sum is 42.")), 5)
    assert ok and "42" in detail

    ok, detail, _ = await doctor._probe_replay(tier_for(Message(role="assistant", content="")), 5)
    assert not ok and "empty" in detail

    again = Message(role="assistant", content=None, tool_calls=[ToolCall(name="add", arguments={"a": 17, "b": 25})])
    ok, detail, _ = await doctor._probe_replay(tier_for(again), 5)
    assert not ok and "tool again" in detail

    ok, detail, _ = await doctor._probe_replay(tier_for(Message(role="assistant", content="I cannot say")), 5)
    assert not ok and "unexpected" in detail


@pytest.mark.asyncio
async def test_doctor_probe_and_replay_share_one_event_loop_via_run_checks():
    """Regression: probe + replay used two asyncio.run() calls, so loop-bound SDK clients failed with 'Event loop is closed'."""
    import asyncio

    import cli.cmds.doctor_cmd as doctor

    class LoopBound(BaseProvider):
        """Fails like a cached SDK client does when it is called from a different event loop than its first use."""

        def __init__(self):
            super().__init__(api_key="k" * 20, default_model="m")
            self.loop = None

        @property
        def provider_name(self):
            return "do"

        async def generate(self, messages, tools=None, model=None, **kwargs):
            running = asyncio.get_running_loop()
            if self.loop is None:
                self.loop = running
            elif self.loop is not running:
                raise RuntimeError("Event loop is closed")
            return Message(role="assistant", content="42"), None

    tier = FallbackTier(provider=LoopBound(), model="m", label="loopbound", provider_name="do")
    probes, replays = await doctor._run_checks([tier], 5, replay=True)

    assert probes[0][0] is True
    assert replays[0][0] is True, replays[0][1]


def test_doctor_runs_checks_in_a_single_asyncio_run(monkeypatch):
    import inspect

    import cli.cmds.doctor_cmd as doctor

    assert inspect.getsource(doctor.doctor_command).count("asyncio.run(") == 1


# --------------------------------------------------------------------------- fault injection & provenance


@pytest.mark.asyncio
async def test_fault_injection_primary_429(monkeypatch):
    """VALSTORM_FAULT_INJECT=primary:429 causes tier 0 to raise 429 and cascades to tier 1."""
    monkeypatch.setenv("VALSTORM_FAULT_INJECT", "primary:429")
    p0 = ScriptedProvider("p0", text="should not be reached")
    p1 = ScriptedProvider("p1", text="recovered from backup")
    chain = ChainedFallbackProvider(
        tiers=[
            FallbackTier(p0, model="m0", label="Primary", provider_name="p0"),
            FallbackTier(p1, model="m1", label="Backup", provider_name="p1"),
        ]
    )
    events = [e async for e in chain.generate_stream([Message(role="user", content="hi")])]
    assert p0.call_count == 0
    assert p1.call_count == 1
    texts = [e.delta for e in events if e.event_type == StreamEventType.TEXT_CHUNK and e.delta]
    assert "recovered from backup" in "".join(texts)

    # Check effective tier metadata on TURN_COMPLETE
    tc_event = next(e for e in events if e.event_type == StreamEventType.TURN_COMPLETE)
    assert tc_event.metadata["effective_tier"] == 2
    assert tc_event.metadata["effective_label"] == "Backup"
    assert tc_event.metadata["effective_provider"] == "p1"
    assert tc_event.metadata["effective_model"] == "m1"


@pytest.mark.asyncio
async def test_openai_provider_captures_reasoning_content_when_content_null():
    """Reasoning models returning reasoning_content with null content are properly captured."""
    from providers.openai import OpenAIProvider

    prov = OpenAIProvider(api_key="k" * 20, default_model="deepseek-v4-pro")

    # 1. Non-streaming (_parse_response)
    class MockMessage:
        content = None
        tool_calls = None
        reasoning_content = "Here is the reasoning solution."

    class MockChoice:
        message = MockMessage()
        finish_reason = "stop"

    class MockResponse:
        choices = [MockChoice()]
        usage = None

    parsed_msg, _ = prov._parse_response(MockResponse(), "deepseek-v4-pro")
    assert parsed_msg.content == "Here is the reasoning solution."

    # 2. Streaming (generate_stream)
    class MockDeltaReasoning:
        content = None
        tool_calls = None
        reasoning_content = "Streamed reasoning token."

    class MockChunkReasoning:
        choices = [type("Choice", (), {"delta": MockDeltaReasoning(), "finish_reason": None})()]
        usage = None

    class MockChunkFinish:
        choices = [type("Choice", (), {"delta": None, "finish_reason": "stop"})()]
        usage = None

    class MockStream:
        def __aiter__(self):
            return self

        def __init__(self):
            self.items = [MockChunkReasoning(), MockChunkFinish()]
            self.idx = 0

        async def __anext__(self):
            if self.idx >= len(self.items):
                raise StopAsyncIteration
            item = self.items[self.idx]
            self.idx += 1
            return item

    async def mock_create(*a, **kw):
        return MockStream()

    class MockAsyncClient:
        chat = type("Chat", (), {"completions": type("Completions", (), {"create": mock_create})()})()

    prov._client = MockAsyncClient()
    stream_events = [e async for e in prov.generate_stream([Message(role="user", content="test")])]
    tc_event = next(e for e in stream_events if e.event_type == StreamEventType.TURN_COMPLETE)
    assert tc_event.message.content == "Streamed reasoning token."
