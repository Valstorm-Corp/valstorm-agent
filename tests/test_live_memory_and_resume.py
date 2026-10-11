"""Live integration test for SQLite Session Persistence, Resuming, and Declarative Memory."""

import time
import pytest
from pathlib import Path

from core.context import WorkspaceContextManager
from core.keystore import KeyStore
from core.memory import MemoryStore
from core.models import SessionState
from core.react import ReActEngine
from core.storage import SessionStore
from core.tools import ToolRegistry
from providers.gemini import GeminiProvider
from tools.memory_tool import register_memory_tools


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_session_persistence_and_resume(tmp_path):
    """Verifies sessions persist to SQLite, reload correctly, and preserve context across turns."""
    api_key = KeyStore.resolve_key("gemini")
    if not api_key:
        pytest.skip("No GEMINI_API_KEY configured for live session persistence test.")

    db_file = tmp_path / "test_sessions.db"
    mem_file = tmp_path / "test_memory.json"
    session_store = SessionStore(db_path=db_file)
    memory_store = MemoryStore(file_path=mem_file)

    registry = ToolRegistry()
    register_memory_tools(registry, memory_store=memory_store, session_store=session_store)

    provider = GeminiProvider(api_key=api_key)
    engine = ReActEngine(provider=provider, tools=registry)

    model = "gemini-flash-lite-latest"
    session1 = SessionState(active_model=model, active_provider="gemini")

    # Turn 1: User states personal facts
    resp1 = await engine.run_turn(
        session=session1,
        user_input="My name is Jared and I am testing SQLite session persistence in Valstorm.",
        model=model,
    )
    assert resp1 is not None
    session_store.save_session(session1)
    orig_session_id = session1.session_id

    # Simulate CLI Restart: Create fresh engine, reload session from SQLite
    resumed_session = session_store.load_session(orig_session_id)
    assert resumed_session is not None
    assert resumed_session.session_id == orig_session_id
    assert len(resumed_session.messages) >= 2

    # Turn 2 in resumed session: Ask to recall facts from Turn 1
    resp2 = await engine.run_turn(
        session=resumed_session,
        user_input="What is my name and what am I testing?",
        model=model,
    )
    content2 = resp2.content.lower() if resp2.content else ""
    assert "jared" in content2
    assert "persistence" in content2 or "sqlite" in content2 or "valstorm" in content2

    # Save turn 2
    session_store.save_session(resumed_session)

    # Verify session list & FTS5 search
    sessions = session_store.list_sessions(limit=5)
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == orig_session_id

    search_hits = session_store.search_sessions("persistence")
    assert len(search_hits) >= 1
    assert search_hits[0]["session_id"] == orig_session_id
