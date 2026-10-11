"""Unit tests for Phase 2: Context Compaction Engine & Low-Water Mark Pruning."""

import pytest
from core.compaction import (
    CompactionResult,
    ContextCompactor,
    estimate_message_tokens,
    estimate_session_history_tokens,
    estimate_tokens_for_text,
    get_model_context_ceiling,
)
from core.models import Message, SessionState, ToolCall, ToolResult, UsageMetadata
from core.react import ReActEngine, StreamEventType


def test_estimate_tokens():
    """Verify heuristic token estimators."""
    assert estimate_tokens_for_text("") == 0
    assert estimate_tokens_for_text("hello world") == 2
    assert estimate_tokens_for_text("a" * 100) == 25

    msg = Message(role="user", content="Test message with 28 characters.")
    toks = estimate_message_tokens(msg)
    assert toks > 0


def test_compaction_on_short_session():
    """Verify compaction skips short sessions."""
    session = SessionState(
        session_id="test_sess",
        active_model="gemini-flash-latest",
        active_provider="gemini",
    )
    session.add_message(Message(role="system", content="You are a helper."))
    session.add_message(Message(role="user", content="Hello."))
    session.add_message(Message(role="assistant", content="Hi there!"))

    compactor = ContextCompactor()
    res = compactor.compact(session)
    assert not res.compacted
    assert len(session.messages) == 3


def test_compaction_prunes_large_tool_outputs():
    """Verify that verbose historical tool outputs are pruned and truncated."""
    session = SessionState(
        session_id="test_sess_long",
        active_model="gemini-flash-latest",
        active_provider="gemini",
    )
    session.add_message(Message(role="system", content="System prompt instructions."))

    # Add 10 turns of tools and dialogue
    for i in range(10):
        session.add_message(Message(role="user", content=f"User request {i}"))
        session.add_message(
            Message(
                role="assistant",
                content=f"Calling tool {i} with extensive reasoning...",
                tool_calls=[ToolCall(name="terminal_exec", arguments={"command": f"echo {i}"})],
            )
        )
        # Giant tool output (2000 chars)
        session.add_message(
            Message(
                role="tool",
                content=f"TOOL_START_{i} " + ("large log output text " * 100) + f" TOOL_END_{i}",
            )
        )

    # Final recent turn
    session.add_message(Message(role="user", content="Latest question"))
    session.add_message(Message(role="assistant", content="Latest answer"))

    total_msgs_before = len(session.messages)
    tokens_before = estimate_session_history_tokens(session)

    compactor = ContextCompactor(keep_recent_messages=4, max_tool_chars=200)
    result = compactor.compact(session)

    assert result.compacted is True
    assert result.tokens_saved > 0
    assert result.compression_ratio < 0.6  # Reduced by over 40%

    # Verify system message preserved
    assert session.messages[0].role == "system"
    assert session.messages[0].content == "System prompt instructions."

    # Verify older tool messages contain compaction markers
    found_compacted_marker = False
    for m in session.messages[1:-4]:
        if m.role == "tool" and "Compacted Output:" in (m.content or ""):
            found_compacted_marker = True
            break
    assert found_compacted_marker is True

    # Verify recent messages intact
    assert session.messages[-1].content == "Latest answer"
    assert session.messages[-2].content == "Latest question"


def test_should_auto_compact():
    """Verify auto-compaction trigger thresholds."""
    session = SessionState(
        session_id="test_sess_auto",
        active_model="gpt-4o",  # 128k ceiling
        active_provider="openai",
    )
    compactor = ContextCompactor(auto_threshold_ratio=0.75, min_messages_to_compact=4)

    # Too few messages
    assert compactor.should_auto_compact(session) is False

    # Add messages with large content
    for _ in range(5):
        session.add_message(Message(role="user", content="x" * 80000))

    # Should trigger auto compact when threshold exceeded
    assert compactor.should_auto_compact(session, force_threshold_tokens=50000) is True


def test_compaction_hysteresis_low_water_mark_deep_clearance():
    """Verify that when context redlines, compaction drives token usage down to target low-water mark."""
    session = SessionState(
        session_id="test_sess_hysteresis",
        active_model="gpt-4o",  # 128k ceiling
        active_provider="openai",
    )
    session.add_message(Message(role="system", content="System instruction context."))

    # Generate 15 conversation turns with substantial text
    for i in range(15):
        session.add_message(Message(role="user", content=f"User prompt {i}: " + ("context information " * 50)))
        session.add_message(
            Message(
                role="assistant",
                content=f"Assistant response {i}: " + ("analysis data " * 50),
                tool_calls=[ToolCall(name="read_file", arguments={"path": f"file_{i}.py"})],
            )
        )
        session.add_message(Message(role="tool", content=f"File content line {i} " + ("data " * 100)))

    # Recent turns
    session.add_message(Message(role="user", content="What is the latest status?"))
    session.add_message(Message(role="assistant", content="Everything is ready."))

    tokens_before = estimate_session_history_tokens(session)
    assert tokens_before > 2000

    # Compact with explicit target token budget of 800 tokens
    compactor = ContextCompactor(
        auto_threshold_ratio=0.75,
        target_ratio=0.40,
        keep_recent_messages=2,
        max_tool_chars=100,
    )
    result = compactor.compact(session, target_tokens=800)

    assert result.compacted is True
    assert result.estimated_tokens_after <= 900
    assert result.tokens_saved > 7000
    assert result.compression_ratio < 0.20  # Over 80% tokens freed
    # Retains root system message, digest summary, and recent slice
    assert session.messages[0].role == "system"
    assert any("[HISTORICAL CONTEXT DIGEST" in (m.content or "") for m in session.messages)
    assert session.messages[-1].content == "Everything is ready."


@pytest.mark.asyncio
async def test_react_engine_pre_turn_compaction_runs_before_loop():
    """Verify that ReActEngine triggers pre-turn compaction before executing LLM tools."""
    class MockProvider:
        async def generate_stream(self, messages, tools, model):
            yield Message(role="assistant", content="Final answer after pre-turn compaction.", model=model)

    class MockTools:
        def get_schemas(self):
            return []

    session = SessionState(
        session_id="test_sess_pre_turn",
        active_model="gpt-4o",
        active_provider="openai",
    )
    session.add_message(Message(role="system", content="System instruction context."))

    for i in range(10):
        session.add_message(Message(role="user", content=f"Old turn {i} " + ("data " * 100)))
        session.add_message(
            Message(
                role="assistant",
                content=f"Old reply {i}",
                tool_calls=[ToolCall(name="search_files", arguments={"pattern": "test"})],
            )
        )
        session.add_message(Message(role="tool", content=f"Tool output {i} " + ("result " * 100)))

    # Set threshold low to force pre-turn compaction on high token count
    compactor = ContextCompactor(
        auto_threshold_ratio=0.01,
        target_ratio=0.005,
        keep_recent_messages=2,
        max_tool_chars=100,
    )
    engine = ReActEngine(
        provider=MockProvider(),
        tools=MockTools(),
        compactor=compactor,
    )

    events = []
    async for event in engine.run_turn_stream(
        session=session,
        user_input="Run next phase of work",
    ):
        events.append(event)

    # Verify CONTEXT_COMPACTED was emitted as pre-turn compaction
    compacted_events = [e for e in events if e.event_type == StreamEventType.CONTEXT_COMPACTED]
    assert len(compacted_events) >= 1
    assert compacted_events[0].metadata.get("phase") == "pre-turn"
    assert compacted_events[0].metadata.get("tokens_saved") > 0


def test_compaction_preserves_clean_user_turn_boundary():
    """Verify that compaction never leaves orphan tool calls or starts conversational messages on model/tool turns."""
    session = SessionState(
        session_id="test_sess_boundary",
        active_model="gemini-flash-latest",
        active_provider="gemini",
    )
    session.add_message(Message(role="system", content="System instruction context."))
    session.add_message(Message(role="user", content="Initial user prompt"))

    # Add multiple tool turns
    for i in range(8):
        cid = f"call_{i}"
        session.add_message(
            Message(
                role="assistant",
                content=f"Calling tool {i}",
                tool_calls=[ToolCall(id=cid, name="search_files", arguments={"pattern": f"p_{i}"})],
            )
        )
        session.add_message(
            Message(
                role="tool",
                content=f"Search result {i} " * 200,
                tool_result=ToolResult(call_id=cid, name="search_files", output=f"result {i}"),
            )
        )

    # Compact aggressively with keep_recent_messages=3 (which would otherwise slice midway through a tool pair)
    compactor = ContextCompactor(keep_recent_messages=3, max_tool_chars=50)
    result = compactor.compact(session, target_tokens=100)

    assert result.compacted is True
    # Find all non-system messages
    conv_msgs = [m for m in session.messages if m.role != "system"]
    assert len(conv_msgs) > 0
    # First conversational message must be a user turn (or bridge turn)
    assert conv_msgs[0].role == "user"
    # No leading orphan tool results
    assert conv_msgs[0].role != "tool"


def test_should_auto_compact_with_ground_truth_prompt_tokens():
    """Verify that should_auto_compact detects ground-truth prompt tokens from provider usage telemetry."""
    session = SessionState(
        session_id="test_sess_gt",
        active_model="gemini-flash-latest",
        active_provider="gemini",
    )
    session.add_message(Message(role="system", content="System instruction context."))

    # 10 short messages (heuristic tokens will be very small, < 1000)
    for i in range(10):
        session.add_message(Message(role="user", content=f"Short prompt {i}"))
        session.add_message(Message(role="assistant", content=f"Short answer {i}"))

    compactor = ContextCompactor()
    # Without usage metadata, short messages don't trigger compaction
    assert compactor.should_auto_compact(session) is False

    # Simulate provider attaching ground-truth usage of 260,000 prompt tokens
    session.messages[-1].usage = UsageMetadata(prompt_tokens=260_000, completion_tokens=100, total_tokens=260_100)

    # Should trigger auto-compaction because 260,000 >= 320,000 * 0.75 (240,000 threshold)
    assert compactor.should_auto_compact(session) is True


def test_storage_save_session_purges_evicted_messages(tmp_path):
    """Verify that save_session deletes evicted messages and updates compacted bodies in SQLite."""
    from core.storage import SessionStore

    db_file = tmp_path / "test_purge.db"
    store = SessionStore(db_path=db_file)

    session = SessionState(
        session_id="test_sess_purge",
        active_model="gemini-flash-latest",
        active_provider="gemini",
    )
    session.add_message(Message(role="system", content="System instruction context."))

    # Create 20 messages with large tool outputs
    for i in range(15):
        session.add_message(Message(role="user", content=f"User {i}"))
        session.add_message(
            Message(
                role="assistant",
                content=f"Calling tool {i}",
                tool_calls=[ToolCall(name="exec", arguments={"cmd": f"run {i}"})],
            )
        )
        session.add_message(
            Message(
                role="tool",
                content=f"Tool output {i} " + ("verbose log data " * 50),
            )
        )

    # Initial save: 46 messages persisted
    store.save_session(session)
    loaded_initial = store.load_session(session.session_id)
    assert len(loaded_initial.messages) == len(session.messages)

    # Run compaction to target budget of 500 tokens (evicting older messages)
    compactor = ContextCompactor(keep_recent_messages=2, max_tool_chars=100)
    c_res = compactor.compact(session, target_tokens=500)
    assert c_res.compacted is True
    compacted_msg_count = len(session.messages)
    assert compacted_msg_count < len(loaded_initial.messages)

    # Save compacted session
    store.save_session(session)

    # Load from SQLite: must have exact compacted count (evicted messages purged, not resurrected)
    loaded_after = store.load_session(session.session_id)
    assert len(loaded_after.messages) == compacted_msg_count
    # Verify compacted tool marker is persisted
    assert any("[Compacted Output:" in (m.content or m.body or "") for m in loaded_after.messages)

