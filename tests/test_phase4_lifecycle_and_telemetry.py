"""Phase 4 TDD Suite: Memory Lifecycle Hygiene, Token-Budgeted Injection & SSE Telemetry.

Verifies:
1. Token-budgeted and relevance-ranked `MemoryStore.format_for_system_prompt()`:
   - Enforces `max_facts_per_category` and `max_chars` caps so system prompts never bloat unboundedly.
   - Always pins critical guardrails (`CRITICAL`, `NEVER`, `ALWAYS`, `GUARDRAIL`) at the top.
   - Ranks remaining facts by query relevance (`query_hint` token overlap) + recency.
2. Real-time SSE telemetry emission (`skill.updated`, `agent.updated`, `memory.updated`) onto the
   active run event queue (`get_current_event_queue()`) when skills, agents, or memories are mutated.
3. On-demand `/learn` slash command in `server.py`:
   - Distills the current session trajectory via `curate_and_commit_turn_trajectory(..., force=True)`,
     streams a formatted confirmation card, and emits `skill.updated` + `run.completed`.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest
from httpx import ASGITransport, AsyncClient

from core.memory import MemoryStore
from core.models import Message, SessionState, ToolCall, ToolResult
import server
from server import app, set_current_event_queue
from tools.delegation import agent_manage, invalidate_tenant_profiles_cache
from tools.skill_tool import invalidate_tenant_skills_cache, skill_manage


def test_memory_store_token_budget_and_critical_guardrail_prioritization(tmp_path):
    """Verify format_for_system_prompt enforces budgets while pinning CRITICAL rules and query-relevant facts."""
    mem_file = tmp_path / "budget_memories.json"
    store = MemoryStore(file_path=mem_file)

    # 1. Oldest critical guardrail (must be pinned even after 35 newer facts are added)
    store.add_fact("memory", "CRITICAL WORKSPACE RULE: Always preserve chmod +x on scripts/*.sh.")
    store.add_fact("memory", "Old query-relevant fact: Snowflake warehouse is used for BI analytics reporting.")

    # 2. Add 35 unrelated filler facts
    for i in range(35):
        store.add_fact("memory", f"Historical filler note #{i}: service {i} runs on internal port {8100 + i}.")

    # 3. Format with tight budget (max 8 facts, max 900 chars) and a query hint matching Snowflake
    formatted = store.format_for_system_prompt(
        query_hint="Run BI analytics reporting on the warehouse",
        max_facts_per_category=8,
        max_chars=900,
    )

    assert len(formatted) <= 950
    # Critical rule is pinned at the top despite being the oldest fact
    assert "CRITICAL WORKSPACE RULE: Always preserve chmod +x on scripts/*.sh." in formatted
    # Query-relevant fact is retained despite being the second oldest fact
    assert "Snowflake warehouse is used for BI analytics reporting." in formatted
    # Early unrelated filler notes are evicted by the budget cap
    assert "Historical filler note #0:" not in formatted
    # Most recent filler notes are kept within budget
    assert "Historical filler note #34:" in formatted


@pytest.mark.asyncio
async def test_sse_telemetry_events_emitted_on_skill_agent_and_memory_mutations(tmp_path, monkeypatch):
    """Verify skill_manage, agent_manage, and MemoryStore.reconcile_fact_async emit SSE events to the active queue."""
    skills_dir = tmp_path / "skills"
    profiles_dir = tmp_path / "profiles"
    skills_dir.mkdir(parents=True)
    profiles_dir.mkdir(parents=True)
    monkeypatch.setenv("VALSTORM_SKILLS_DIR", str(skills_dir))
    monkeypatch.setenv("VALSTORM_PROFILES_DIR", str(profiles_dir))
    invalidate_tenant_skills_cache()
    invalidate_tenant_profiles_cache()

    queue: asyncio.Queue = asyncio.Queue()
    q_tok = set_current_event_queue(queue)
    try:
        # 1. Skill creation emits skill.updated
        await skill_manage(
            action="create",
            slug="sse-telemetry-skill",
            category="devops",
            name="SSE Telemetry Skill",
            description="Test skill for SSE telemetry.",
            body="# SSE Telemetry Skill\nStep 1.",
        )

        # 2. Agent creation emits agent.updated
        await agent_manage(
            action="create",
            slug="sse-telemetry-agent",
            name="SSE Telemetry Agent",
            description="Test agent for SSE telemetry.",
            system_prompt="You are SSE Telemetry Agent.",
            allowed_tools=["read_file", "skill_view"],
            skill_slugs=["sse-telemetry-skill"],
        )

        # 3. Memory reconciliation emits memory.updated
        mem_store = MemoryStore(file_path=tmp_path / "sse_mem.json")
        await mem_store.reconcile_fact_async(
            target="user",
            content="Prefers compact SSE telemetry badges in chat.",
        )

        events = []
        while not queue.empty():
            events.append(await queue.get())

        event_types = [e.get("event") for e in events]
        assert "skill.updated" in event_types
        assert "agent.updated" in event_types
        assert "memory.updated" in event_types

        skill_evt = next(e for e in events if e.get("event") == "skill.updated")
        assert skill_evt["slug"] == "sse-telemetry-skill"
        assert skill_evt["action"] == "create"
        assert skill_evt["version"] == "1.0.0"

        agent_evt = next(e for e in events if e.get("event") == "agent.updated")
        assert agent_evt["slug"] == "sse-telemetry-agent"
        assert agent_evt["action"] == "create"

        mem_evt = next(e for e in events if e.get("event") == "memory.updated")
        assert mem_evt["action"] == "added"
        assert "compact SSE telemetry badges" in mem_evt["fact"]
    finally:
        set_current_event_queue(None)


@pytest.mark.asyncio
async def test_learn_slash_command_distills_session_trajectory_on_demand(tmp_path, monkeypatch):
    """Verify /learn slash command in server.py curates a skill from the session trajectory and streams confirmation."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(parents=True)
    monkeypatch.setenv("VALSTORM_SKILLS_DIR", str(skills_dir))
    invalidate_tenant_skills_cache()

    # Build a session with prior tool debugging history
    seeded_session = SessionState(session_id="aich_learn_cmd_test")
    seeded_session.add_message(Message(role="user", content="Run the API tests"))
    seeded_session.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c1", name="terminal_exec", arguments={"command": "pytest apps/api"})],
        )
    )
    seeded_session.add_message(
        Message(
            role="tool",
            content="ModuleNotFoundError: No module named 'ai'",
            tool_result=ToolResult(call_id="c1", name="terminal_exec", output="ModuleNotFoundError: No module named 'ai'", is_error=True),
        )
    )
    seeded_session.add_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c2", name="terminal_exec", arguments={"command": "PYTHONPATH=apps/api/app pytest apps/api"})],
        )
    )
    seeded_session.add_message(
        Message(
            role="tool",
            content="4 passed",
            tool_result=ToolResult(call_id="c2", name="terminal_exec", output="4 passed", is_error=False),
        )
    )
    seeded_session.add_message(Message(role="assistant", content="All 4 tests passed with PYTHONPATH=apps/api/app."))

    class _SeededSessionStore:
        def load_session(self, session_id):
            return seeded_session

        def get_session(self, session_id):
            return seeded_session

        def save_session(self, session):
            pass

    mock_provider = MagicMock()
    mock_provider.chat_complete = AsyncMock(
        return_value=Message(
            role="assistant",
            content=json.dumps({
                "action": "create",
                "slug": "api-pytest-pythonpath",
                "category": "valstorm-internal",
                "name": "API Pytest PYTHONPATH Setup",
                "description": "Set PYTHONPATH=apps/api/app when running apps/api tests.",
                "body": "# API Pytest PYTHONPATH Setup\n\n## Pitfalls\n- Always prefix `PYTHONPATH=apps/api/app`.",
                "tags": ["pytest", "api"],
            }),
        )
    )

    monkeypatch.setattr(server, "SessionStore", _SeededSessionStore)
    monkeypatch.setattr(server, "MemoryStore", lambda: MemoryStore(file_path=tmp_path / "mem.json"))
    monkeypatch.setattr(server, "_get_provider_instance", lambda *_: (mock_provider, "gemini-flash-latest", "gemini"))
    monkeypatch.setattr(server, "_build_server_tool_registry", lambda **_: MagicMock())

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        created = await ac.post(
            "/v1/runs",
            json={
                "input": "/learn Codify the PYTHONPATH requirement for apps/api pytest",
                "session_id": "aich_learn_cmd_test",
            },
        )
        assert created.status_code == 200
        run_id = created.json()["run_id"]

        resp = await ac.get(f"/v1/runs/{run_id}/events")
        assert resp.status_code == 200
        raw_lines = [line[6:] for line in resp.text.splitlines() if line.startswith("data: ")]
        events = [json.loads(line) for line in raw_lines if line.strip() and line.strip() != "[DONE]"]

    completed = [e for e in events if e.get("event") == "run.completed"]
    assert len(completed) == 1
    assert "api-pytest-pythonpath" in completed[0]["output"]

    skill_events = [e for e in events if e.get("event") == "skill.updated"]
    assert len(skill_events) >= 1
    assert skill_events[0]["slug"] == "api-pytest-pythonpath"
