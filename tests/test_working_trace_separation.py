"""Unit tests for FI-599: separating intermediate agent working trace from final human-readable reply."""

import json

import pytest
from httpx import AsyncClient, ASGITransport

import server
from core.models import (
    ExecutionTraceEntry,
    Message,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolResult,
    UsageMetadata,
)
from server import app


class _SessionStoreStub:
    def load_session(self, session_id):
        return None

    def save_session(self, session):
        pass


def _sse_events(response_text: str):
    return [
        json.loads(line.removeprefix("data: "))
        for line in response_text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


async def _run_and_collect(ac: AsyncClient, payload_overrides=None):
    body = {
        "input": "Fix the bug and verify",
        "model": "target-model",
        "provider": "target-provider",
        "session_id": "aich_trace_test",
    }
    if payload_overrides:
        body.update(payload_overrides)
    created = await ac.post("/v1/runs", json=body)
    assert created.status_code == 200
    run_id = created.json()["run_id"]
    events = await ac.get(f"/v1/runs/{run_id}/events")
    assert events.status_code == 200
    return _sse_events(events.text)


def _patch_common(monkeypatch, fake_stream, sync_calls):
    async def fake_sync(**kwargs):
        sync_calls.append(kwargs)

    monkeypatch.setattr(server, "SessionStore", _SessionStoreStub)
    monkeypatch.setattr(server, "MemoryStore", lambda: object())
    monkeypatch.setattr(server, "_get_provider_instance", lambda *_: (object(), "resolved-model", "resolved-provider"))
    monkeypatch.setattr(server, "_build_server_tool_registry", lambda **_: object())
    monkeypatch.setattr(server, "_sync_turn_to_valstorm", fake_sync)
    monkeypatch.setattr(server.ReActEngine, "run_turn_stream", fake_stream)


@pytest.mark.asyncio
async def test_multi_iteration_react_separates_narration_from_final_reply(monkeypatch):
    """Intermediate narration must never be glued onto the final reply and must appear chronologically in trace."""
    sync_calls = []

    async def fake_stream(self, **kwargs):
        # Iteration 0: narrate + call read_file
        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="Let me check the files first.")
        tc1 = ToolCall(id="call_1", name="read_file", arguments={"path": "server.py"})
        yield StreamEvent(event_type=StreamEventType.TOOL_CALL_DETECTED, tool_call=tc1)
        yield StreamEvent(event_type=StreamEventType.LLM_CALL_COMPLETE)
        yield StreamEvent(
            event_type=StreamEventType.TOOL_EXECUTION_RESULT,
            tool_result=ToolResult(call_id="call_1", name="read_file", output="file contents", duration_ms=120.0),
        )

        # Iteration 1: narrate + parallel patch_file & terminal_exec
        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="I'm patching server.py and running tests.")
        tc2 = ToolCall(id="call_2", name="patch_file", arguments={"path": "server.py"})
        tc3 = ToolCall(id="call_3", name="terminal_exec", arguments={"command": "pytest"})
        yield StreamEvent(event_type=StreamEventType.TOOL_CALL_DETECTED, tool_call=tc2)
        yield StreamEvent(event_type=StreamEventType.TOOL_CALL_DETECTED, tool_call=tc3)
        yield StreamEvent(event_type=StreamEventType.LLM_CALL_COMPLETE)
        # Return results out of order to verify call_id matching
        yield StreamEvent(
            event_type=StreamEventType.TOOL_EXECUTION_RESULT,
            tool_result=ToolResult(call_id="call_3", name="terminal_exec", output="18 passed", duration_ms=450.0),
        )
        yield StreamEvent(
            event_type=StreamEventType.TOOL_EXECUTION_RESULT,
            tool_result=ToolResult(call_id="call_2", name="patch_file", output="patched ok", duration_ms=30.0),
        )

        # Iteration 2: final human-readable reply (no tool calls)
        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="Done — both services are updated and all 18 tests pass.")
        yield StreamEvent(event_type=StreamEventType.LLM_CALL_COMPLETE)
        yield StreamEvent(
            event_type=StreamEventType.TURN_COMPLETE,
            message=Message(
                role="assistant",
                content="Done — both services are updated and all 18 tests pass.",
                usage=UsageMetadata(prompt_tokens=500, completion_tokens=80),
            ),
        )

    _patch_common(monkeypatch, fake_stream, sync_calls)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        events = await _run_and_collect(ac)

    # 1. Verify SSE event ordering: activity.narration emitted right before tool.started
    event_types = [e["event"] for e in events]
    assert "activity.narration" in event_types
    assert "iteration.boundary" in event_types

    first_narration_idx = event_types.index("activity.narration")
    first_tool_idx = event_types.index("tool.started")
    assert first_narration_idx < first_tool_idx

    narrations = [e for e in events if e["event"] == "activity.narration"]
    assert [n["text"] for n in narrations] == [
        "Let me check the files first.",
        "I'm patching server.py and running tests.",
    ]

    # 2. Verify run.completed.output contains ONLY the final human-readable reply (no glued narration!)
    completed = next(e for e in events if e["event"] == "run.completed")
    assert completed["output"] == "Done — both services are updated and all 18 tests pass."
    assert "Let me check" not in completed["output"]
    assert "I'm patching" not in completed["output"]

    # 3. Verify chronological execution_trace and backwards-compatible tool_calls
    trace = completed["execution_trace"]
    assert [entry["kind"] for entry in trace] == [
        "narration",
        "tool_call",
        "narration",
        "tool_call",
        "tool_call",
    ]
    assert trace[0]["text"] == "Let me check the files first."
    assert trace[1]["tool_name"] == "read_file"
    assert trace[1]["result"] == "file contents"
    assert trace[2]["text"] == "I'm patching server.py and running tests."
    # Parallel tool call results must match exact call_id even when completed out of order
    assert trace[3]["tool_name"] == "patch_file"
    assert trace[3]["result"] == "patched ok"
    assert trace[4]["tool_name"] == "terminal_exec"
    assert trace[4]["result"] == "18 passed"
    assert trace[3]["step_id"] != trace[4]["step_id"]

    # Validate against Pydantic model
    for item in trace:
        ExecutionTraceEntry.model_validate(item)

    # 4. Verify synced payload persists only final reply in output_text and carries chronological trace
    assert len(sync_calls) == 1
    assert sync_calls[0]["output_text"] == "Done — both services are updated and all 18 tests pass."
    assert sync_calls[0]["execution_trace"] == trace
    assert sync_calls[0]["tool_calls"] == trace


@pytest.mark.asyncio
async def test_max_iterations_without_final_text_turn_uses_safe_fallback(monkeypatch):
    """If max iterations are reached on a tool-calling turn, body falls back safely without returning empty."""
    sync_calls = []

    async def fake_stream(self, **kwargs):
        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="Inspecting final log file.")
        tc = ToolCall(id="call_last", name="read_file", arguments={"path": "app.log"})
        yield StreamEvent(event_type=StreamEventType.TOOL_CALL_DETECTED, tool_call=tc)
        yield StreamEvent(event_type=StreamEventType.LLM_CALL_COMPLETE)
        yield StreamEvent(
            event_type=StreamEventType.TOOL_EXECUTION_RESULT,
            tool_result=ToolResult(call_id="call_last", name="read_file", output="log tail", duration_ms=50.0),
        )
        yield StreamEvent(event_type=StreamEventType.ITERATION_LIMIT_REACHED, delta="limit reached")
        yield StreamEvent(
            event_type=StreamEventType.TURN_COMPLETE,
            message=Message(
                role="assistant",
                content="Inspecting final log file.",
                tool_calls=[tc],
                usage=UsageMetadata(prompt_tokens=200, completion_tokens=40),
            ),
        )

    _patch_common(monkeypatch, fake_stream, sync_calls)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        events = await _run_and_collect(ac)

    completed = next(e for e in events if e["event"] == "run.completed")
    assert "Inspecting final log file." in completed["output"]
    assert "maximum iterations" in completed["output"]
    assert len(completed["execution_trace"]) == 2
    assert sync_calls[0]["output_text"] == completed["output"]


@pytest.mark.asyncio
async def test_retry_after_tool_less_llm_call_does_not_glue_text(monkeypatch):
    """Action-nudge retry after a tool-less LLM_CALL_COMPLETE demotes earlier text cleanly."""
    sync_calls = []

    async def fake_stream(self, **kwargs):
        # Iteration 0: model says it will act, no tools, then engine retries
        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="I will write the file now.")
        yield StreamEvent(event_type=StreamEventType.LLM_CALL_COMPLETE)
        # Iteration 1: model actually calls write_file
        tc = ToolCall(id="call_w", name="write_file", arguments={"path": "a.txt"})
        yield StreamEvent(event_type=StreamEventType.TOOL_CALL_DETECTED, tool_call=tc)
        yield StreamEvent(event_type=StreamEventType.LLM_CALL_COMPLETE)
        yield StreamEvent(
            event_type=StreamEventType.TOOL_EXECUTION_RESULT,
            tool_result=ToolResult(call_id="call_w", name="write_file", output="ok", duration_ms=10.0),
        )
        # Iteration 2: final answer
        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="Created `a.txt`.")
        yield StreamEvent(event_type=StreamEventType.LLM_CALL_COMPLETE)
        yield StreamEvent(
            event_type=StreamEventType.TURN_COMPLETE,
            message=Message(
                role="assistant",
                content="Created `a.txt`.",
                usage=UsageMetadata(prompt_tokens=100, completion_tokens=20),
            ),
        )

    _patch_common(monkeypatch, fake_stream, sync_calls)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        events = await _run_and_collect(ac)

    completed = next(e for e in events if e["event"] == "run.completed")
    assert completed["output"] == "Created `a.txt`."
    assert "I will write the file now." not in completed["output"]
    narrations = [e["text"] for e in completed["execution_trace"] if e["kind"] == "narration"]
    assert narrations == ["I will write the file now."]


@pytest.mark.asyncio
async def test_stream_reset_on_failover_clears_only_active_iteration(monkeypatch):
    """Failover stream_reset clears partial text from the failed attempt without corrupting earlier trace."""
    sync_calls = []

    async def fake_stream(self, **kwargs):
        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="Checking status.")
        tc = ToolCall(id="call_s", name="terminal_exec", arguments={"command": "git status"})
        yield StreamEvent(event_type=StreamEventType.TOOL_CALL_DETECTED, tool_call=tc)
        yield StreamEvent(event_type=StreamEventType.LLM_CALL_COMPLETE)
        yield StreamEvent(
            event_type=StreamEventType.TOOL_EXECUTION_RESULT,
            tool_result=ToolResult(call_id="call_s", name="terminal_exec", output="clean", duration_ms=15.0),
        )
        # Partial corrupted chunk before failover reset
        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="Corrupted partial...")
        yield StreamEvent(
            event_type=StreamEventType.TEXT_CHUNK,
            delta="",
            metadata={"stream_reset": True, "failover": True, "failed_tier": 1, "target_tier": 2},
        )
        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="Repository is clean.")
        yield StreamEvent(event_type=StreamEventType.LLM_CALL_COMPLETE)
        yield StreamEvent(
            event_type=StreamEventType.TURN_COMPLETE,
            message=Message(
                role="assistant",
                content="Repository is clean.",
                usage=UsageMetadata(prompt_tokens=150, completion_tokens=30),
            ),
        )

    _patch_common(monkeypatch, fake_stream, sync_calls)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        events = await _run_and_collect(ac)

    completed = next(e for e in events if e["event"] == "run.completed")
    assert completed["output"] == "Repository is clean."
    assert "Corrupted partial" not in completed["output"]
    assert any(
        e["event"] == "iteration.boundary" and e.get("reason") == "stream_reset"
        for e in events
    )
