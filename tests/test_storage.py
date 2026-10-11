"""Unit tests for SQLite SessionStore and FTS5 search."""

import pytest
from pathlib import Path

from core.models import Message, SessionState, ToolCall, ToolResult, UsageMetadata
from core.storage import SessionStore


def test_session_store_init_and_save_load(tmp_path):
    db_file = tmp_path / "test_agent_state.db"
    store = SessionStore(db_path=db_file)

    session = SessionState(active_model="gemini-flash-lite-latest", active_provider="gemini")
    
    # Add User Message
    msg1 = Message(role="user", content="Hello agent, what is the capital of France?")
    session.add_message(msg1)

    # Add Assistant Message with tool call
    tc = ToolCall(name="calculator", arguments={"expression": "100 + 200"})
    msg2 = Message(
        role="assistant",
        content="Let me compute this.",
        model="gemini-flash-lite-latest",
        provider="gemini",
        tool_calls=[tc],
        usage=UsageMetadata(prompt_tokens=150, completion_tokens=25, total_tokens=175),
    )
    session.add_message(msg2)

    # Add Tool Result Message
    tr = ToolResult(call_id=tc.id, name="calculator", output="Result: 300", duration_ms=12.5, payload_bytes=11)
    msg3 = Message(role="tool", content="Result: 300", tool_result=tr)
    session.add_message(msg3)

    # Save Session
    store.save_session(session)

    # Load Session
    loaded = store.load_session(session.session_id)
    assert loaded is not None
    assert loaded.session_id == session.session_id
    assert loaded.active_model == "gemini-flash-lite-latest"
    assert loaded.total_prompt_tokens == 150
    assert loaded.total_completion_tokens == 25
    assert loaded.total_tokens == 175
    assert len(loaded.messages) == 3

    # Check reconstructed messages
    assert loaded.messages[0].role == "user"
    assert loaded.messages[0].content == "Hello agent, what is the capital of France?"
    assert loaded.messages[1].tool_calls is not None
    assert loaded.messages[1].tool_calls[0].name == "calculator"
    assert loaded.messages[2].tool_result is not None
    assert loaded.messages[2].tool_result.duration_ms == 12.5


def test_session_store_list_and_delete(tmp_path):
    db_file = tmp_path / "test_agent_state.db"
    store = SessionStore(db_path=db_file)

    s1 = SessionState(active_model="gemini-3.6-flash", active_provider="gemini")
    s1.add_message(Message(role="user", content="First task about data migration"))
    store.save_session(s1)

    s2 = SessionState(active_model="gpt-4o-mini", active_provider="openai")
    s2.add_message(Message(role="user", content="Second task about visual layout"))
    store.save_session(s2)

    sessions = store.list_sessions(limit=10)
    assert len(sessions) == 2
    titles = [s["title"] for s in sessions]
    assert any("First task" in t for t in titles)
    assert any("Second task" in t for t in titles)

    # Delete session 1
    deleted = store.delete_session(s1.session_id)
    assert deleted is True

    remaining = store.list_sessions(limit=10)
    assert len(remaining) == 1
    assert remaining[0]["session_id"] == s2.session_id


def test_session_store_fts5_search(tmp_path):
    db_file = tmp_path / "test_agent_state.db"
    store = SessionStore(db_path=db_file)

    s1 = SessionState(active_model="gemini-flash-lite-latest", active_provider="gemini")
    s1.add_message(Message(role="user", content="Please inspect the contact schema in PostgreSQL"))
    s1.add_message(Message(role="assistant", content="The contact schema has email, phone, and organization fields."))
    store.save_session(s1)

    s2 = SessionState(active_model="gemini-flash-lite-latest", active_provider="gemini")
    s2.add_message(Message(role="user", content="How do I configure Vite with Tailwind CSS?"))
    store.save_session(s2)

    # Search for "PostgreSQL"
    results = store.search_sessions(query="PostgreSQL", limit=5)
    assert len(results) >= 1
    assert results[0]["session_id"] == s1.session_id
    assert "PostgreSQL" in results[0]["snippet"]

    # Search for "Tailwind"
    results_tw = store.search_sessions(query="Tailwind", limit=5)
    assert len(results_tw) >= 1
    assert results_tw[0]["session_id"] == s2.session_id
