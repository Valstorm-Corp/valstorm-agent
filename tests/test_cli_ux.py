"""Tests for Agent Runtime CLI user experience: Ctrl+C interruption, buffer clearing, steering, YOLO mode, context compression, and status telemetry."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from core.models import Message, SessionState, StreamEvent, StreamEventType, UsageMetadata, get_telemetry_badge
from core.react import ReActEngine
from cli import compress_session_context, run_interactive_repl, stream_and_render_turn
from tools.confirmation import confirmation_required


class InterruptingEngine:
    """Mock engine that simulates a KeyboardInterrupt during streaming."""

    async def run_turn_stream(self, session: SessionState, user_input: str, model: str):
        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="Initial text chunk...")
        raise KeyboardInterrupt()


class CancelledEngine:
    """Mock engine that simulates asyncio.CancelledError during streaming."""

    async def run_turn_stream(self, session: SessionState, user_input: str, model: str):
        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="Streaming...")
        raise asyncio.CancelledError()


class ApiErrorEngine:
    """Mock engine that simulates a 503 ServerError / API exception during streaming."""

    async def run_turn_stream(self, session: SessionState, user_input: str, model: str):
        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="Partial chunk...")
        raise RuntimeError("503 UNAVAILABLE: This model is currently experiencing high demand.")


class NormalEngine:
    """Mock engine that returns a normal turn completion."""

    async def run_turn_stream(self, session: SessionState, user_input: str, model: str):
        msg = Message(role="assistant", content=f"Response to: {user_input[:20]}")
        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta=msg.content)
        yield StreamEvent(event_type=StreamEventType.TURN_COMPLETE, message=msg)


@pytest.mark.asyncio
async def test_stream_and_render_turn_handles_keyboard_interrupt(capsys):
    """Verify stream_and_render_turn catches KeyboardInterrupt, returns (None, True), and notices user."""
    engine = InterruptingEngine()
    session = SessionState(active_model="gemini-3.5-flash", active_provider="gemini")

    msg, was_interrupted = await stream_and_render_turn(
        engine=engine,
        session=session,
        prompt="Test prompt",
        model="gemini-3.5-flash",
    )

    assert msg is None
    assert was_interrupted is True
    captured = capsys.readouterr()
    assert "In-progress prompt stopped" in captured.out


@pytest.mark.asyncio
async def test_stream_and_render_turn_handles_asyncio_cancelled(capsys):
    """Verify stream_and_render_turn catches asyncio.CancelledError and returns (None, True)."""
    engine = CancelledEngine()
    session = SessionState(active_model="gemini-3.5-flash", active_provider="gemini")

    msg, was_interrupted = await stream_and_render_turn(
        engine=engine,
        session=session,
        prompt="Test prompt",
        model="gemini-3.5-flash",
    )

    assert msg is None
    assert was_interrupted is True
    captured = capsys.readouterr()
    assert "In-progress prompt stopped" in captured.out


@pytest.mark.asyncio
async def test_stream_and_render_turn_handles_model_api_error(capsys):
    """Verify stream_and_render_turn catches API errors, prints the error, and does not crash."""
    engine = ApiErrorEngine()
    session = SessionState(active_model="gemini-3.5-flash", active_provider="gemini")
    session.add_message(Message(role="system", content="System instruction"))
    initial_msg_count = len(session.messages)

    msg, was_interrupted = await stream_and_render_turn(
        engine=engine,
        session=session,
        prompt="Test prompt",
        model="gemini-3.5-flash",
    )

    assert msg is None
    assert was_interrupted is False
    captured = capsys.readouterr()
    assert "[API Error]" in captured.out
    assert "503 UNAVAILABLE" in captured.out
    # Verify session messages were restored cleanly without orphaned messages
    assert len(session.messages) == initial_msg_count


@pytest.mark.asyncio
async def test_repl_recovers_after_model_api_error(capsys):
    """Verify REPL does not crash when a model API error occurs and allows subsequent prompts."""
    turn_count = 0

    class FailThenSucceedEngine:
        def __init__(self, *args, **kwargs):
            pass

        async def run_turn_stream(self, session: SessionState, user_input: str, model: str, **kwargs):
            nonlocal turn_count
            turn_count += 1
            if turn_count == 1:
                raise RuntimeError("503 UNAVAILABLE: High demand")
            else:
                msg = Message(role="assistant", content="Recovery response")
                yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta=msg.content)
                yield StreamEvent(event_type=StreamEventType.TURN_COMPLETE, message=msg)

    with patch("cli.get_provider"), \
         patch("cli.build_tool_registry"), \
         patch("cli.PromptSession") as mock_prompt_session_cls, \
         patch("cli.SessionStore") as mock_session_store_cls, \
         patch("cli.repl.ReActEngine", FailThenSucceedEngine):

        mock_prompt_session = MagicMock()
        mock_prompt_session.prompt_async = AsyncMock(side_effect=["prompt 1 (will fail)", "prompt 2 (will succeed)", "exit"])
        mock_prompt_session_cls.return_value = mock_prompt_session

        mock_session_store = MagicMock()
        mock_session_store_cls.return_value = mock_session_store

        await run_interactive_repl(
            initial_model="gemini-3.5-flash",
            initial_provider="gemini",
        )

        captured = capsys.readouterr()
        assert "[API Error]" in captured.out
        assert "503 UNAVAILABLE" in captured.out
        assert "Recovery response" in captured.out
        assert "Goodbye!" in captured.out


@pytest.mark.asyncio
async def test_repl_exits_immediately_when_not_in_a_prompt(capsys):
    """Verify when not in a prompt turn, KeyboardInterrupt at prompt exits the CLI session immediately."""
    with patch("cli.get_provider") as mock_get_provider, \
         patch("cli.build_tool_registry") as mock_build_registry, \
         patch("cli.PromptSession") as mock_prompt_session_cls, \
         patch("cli.SessionStore") as mock_session_store_cls:

        mock_prompt_session = MagicMock()
        mock_prompt_session.prompt_async = AsyncMock(side_effect=KeyboardInterrupt())
        mock_prompt_session_cls.return_value = mock_prompt_session

        mock_session_store = MagicMock()
        mock_session_store_cls.return_value = mock_session_store

        await run_interactive_repl(
            initial_model="gemini-3.5-flash",
            initial_provider="gemini",
        )

        captured = capsys.readouterr()
        assert "Exiting..." in captured.out
        mock_session_store.save_session.assert_called_once()


@pytest.mark.asyncio
async def test_ctrl_c_clears_non_empty_buffer():
    """Verify Ctrl+C clears prompt buffer when text is present without exiting the terminal."""
    with patch("cli.get_provider"), \
         patch("cli.build_tool_registry"), \
         patch("cli.PromptSession") as mock_prompt_session_cls, \
         patch("cli.SessionStore"):

        captured_kb = None

        def capture_init(*args, **kwargs):
            nonlocal captured_kb
            captured_kb = kwargs.get("key_bindings")
            mock_session = MagicMock()
            mock_session.prompt_async = AsyncMock(side_effect=EOFError())
            return mock_session

        mock_prompt_session_cls.side_effect = capture_init

        await run_interactive_repl(
            initial_model="gemini-3.5-flash",
            initial_provider="gemini",
        )

        assert captured_kb is not None

        mock_event = MagicMock()
        mock_buffer = MagicMock()
        mock_buffer.text = "Some pasted or typed text"
        mock_event.app.current_buffer = mock_buffer

        c_c_handler = None
        for binding in getattr(captured_kb, "bindings", []):
            keys_str = tuple(str(k) for k in binding.keys)
            if "c-c" in keys_str or ("ControlC",) == keys_str or "c-c" in str(binding.keys):
                c_c_handler = binding.handler
                break

        assert c_c_handler is not None
        c_c_handler(mock_event)

        assert mock_buffer.text == ""
        assert mock_buffer.cursor_position == 0
        mock_event.app.exit.assert_not_called()

        mock_buffer.text = ""
        c_c_handler(mock_event)
        mock_event.app.exit.assert_called_once()


@pytest.mark.asyncio
async def test_steer_injection_in_react_loop():
    """Verify ReActEngine.steer enqueues an out-of-band instruction and injects it into session messages."""
    msg = Message(role="assistant", content="Final response")
    usage = UsageMetadata(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    async def mock_stream(*args, **kwargs):
        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="Initial response")
        yield (msg, usage)

    mock_provider = MagicMock()
    mock_provider.provider_name = "mock"
    mock_provider.generate_stream = mock_stream

    engine = ReActEngine(provider=mock_provider, tools=MagicMock())
    engine.steer("Stop search and focus on file.py")

    session = SessionState(active_model="gemini-3.5-flash", active_provider="gemini")
    events = [e async for e in engine.run_turn_stream(session=session, user_input="Initial goal", model="gemini-3.5-flash")]

    # Verify out-of-band message was injected into session
    steer_msgs = [m for m in session.messages if "[OUT-OF-BAND USER STEERING INSTRUCTION]" in (m.content or "")]
    assert len(steer_msgs) == 1
    assert "Stop search and focus on file.py" in (steer_msgs[0].content or "")


@pytest.mark.asyncio
async def test_yolo_mode_auto_approves_confirmation():
    """Verify YOLO_MODE environment variable auto-approves confirmation_required tool calls."""
    os.environ["YOLO_MODE"] = "true"
    try:
        res = confirmation_required(action="Delete test database", details={"db": "test"})
        assert "YOLO Mode Active" in res
        assert "automatically APPROVED" in res
    finally:
        os.environ.pop("YOLO_MODE", None)


@pytest.mark.asyncio
async def test_compress_session_context():
    """Verify compress_session_context summarizes older turns and replaces them with a system summary."""
    session = SessionState(active_model="gemini-3.5-flash", active_provider="gemini")
    session.add_message(Message(role="system", content="Initial system prompt"))

    # Add 10 historical turns
    for i in range(10):
        session.add_message(Message(role="user", content=f"User turn {i}", usage=UsageMetadata(prompt_tokens=100, completion_tokens=0, total_tokens=100)))
        session.add_message(Message(role="assistant", content=f"Assistant turn {i}", usage=UsageMetadata(prompt_tokens=0, completion_tokens=50, total_tokens=50)))

    assert len(session.messages) == 21

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=(
        Message(role="assistant", content="Compressed summary of turns 0 to 7."),
        UsageMetadata(prompt_tokens=50, completion_tokens=20, total_tokens=70),
    ))

    tb, ta = await compress_session_context(session=session, provider=mock_provider, keep_last=4)

    # Initial system + 1 compressed summary system + 4 recent kept turns
    assert len(session.messages) <= 6
    assert any("[COMPRESSED HISTORICAL CONTEXT SUMMARY]" in (m.content or "") for m in session.messages)


@pytest.mark.asyncio
async def test_status_telemetry_badge():
    """Verify get_telemetry_badge formats context window usage %, token telemetry, and estimated cost."""
    session = SessionState(active_model="gemini-3.6-flash", active_provider="gemini")
    session.add_message(Message(role="user", content="Hello", usage=UsageMetadata(prompt_tokens=10000, completion_tokens=5000, total_tokens=15000)))

    badge = get_telemetry_badge(session)
    assert "gemini-3.6-flash" in badge
    assert "15,000" in badge
    assert "1,048,576 tokens" in badge
    assert "% used" in badge
    assert "Est. Session Cost" in badge


def test_server_cli_start_modes(monkeypatch):
    """Verify 'vsagent server start host' and 'vsagent server start cloud' configure ports and modes properly."""
    from cli.cmds.server_cmds import start_server
    captured_runs = []

    def fake_uvicorn_run(app_str, host, port, reload, **kwargs):
        captured_runs.append({
            "app": app_str,
            "host": host,
            "port": port,
            "reload": reload,
            "log_level": kwargs.get("log_level"),
            "mode": os.environ.get("VALSTORM_AGENT_MODE"),
            "port_env": os.environ.get("VALSTORM_AGENT_PORT"),
            "sandbox": os.environ.get("VALSTORM_DEFAULT_SANDBOX"),
        })

    monkeypatch.setattr("uvicorn.run", fake_uvicorn_run)

    # 1. vsagent server start host -> port 8650, host mode
    start_server(mode_arg="host")
    assert captured_runs[-1]["port"] == 8650
    assert captured_runs[-1]["mode"] == "host"
    assert captured_runs[-1]["sandbox"] == "host"

    # 2. vsagent server start cloud --dev -> port 8660, cloud mode, debug log level
    start_server(mode_arg="cloud", dev=True)
    assert captured_runs[-1]["port"] == 8660
    assert captured_runs[-1]["mode"] == "cloud"
    assert captured_runs[-1]["sandbox"] == "cloud"
    assert captured_runs[-1]["log_level"] == "debug"

    # 3. vsagent server start (default) -> port 8650
    start_server()
    assert captured_runs[-1]["port"] == 8650
    assert captured_runs[-1]["mode"] == "host"


@pytest.mark.asyncio
async def test_stream_and_render_turn_prints_session_info_and_resume_cmd(capsys):
    """Verify stream_and_render_turn outputs Usage Stats, Session Cumulative, Session Info, and Resume Command."""
    engine = NormalEngine()
    session = SessionState(active_model="gemini-flash-latest", active_provider="gemini")
    
    # Simulate a normal turn with usage metadata
    class UsageEngine:
        async def run_turn_stream(self, session: SessionState, user_input: str, model: str):
            msg = Message(
                role="assistant",
                content="Hello world",
                model="gemini-flash-latest",
                provider="gemini",
                usage=UsageMetadata(prompt_tokens=150, completion_tokens=25, total_tokens=175),
            )
            yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta=msg.content)
            yield StreamEvent(event_type=StreamEventType.TURN_COMPLETE, message=msg)

    msg, was_interrupted = await stream_and_render_turn(
        engine=UsageEngine(),
        session=session,
        prompt="Hi",
        model="gemini-flash-latest",
    )

    assert not was_interrupted
    captured = capsys.readouterr().out
    assert "[Usage Stats] Turn Tokens: Prompt=150 | Completion=25 | Total=175" in captured
    assert "[Session Cumulative]" in captured
    assert f"[Session Info] ID: {session.session_id} | Model: gemini-flash-latest (gemini)" in captured
    assert f"[Resume Command] vsagent chat --session {session.session_id}" in captured


def test_cli_version_flag_and_subcommand():
    """Verify that vsagent --version, -v, and vsagent version exit cleanly and show version info."""
    from typer.testing import CliRunner
    from cli.main import app

    runner = CliRunner()
    res1 = runner.invoke(app, ["--version"])
    assert res1.exit_code == 0
    assert "vsagent v" in res1.output

    res2 = runner.invoke(app, ["-v"])
    assert res2.exit_code == 0
    assert "vsagent v" in res2.output

    res3 = runner.invoke(app, ["version"])
    assert res3.exit_code == 0
    assert "vsagent v" in res3.output



