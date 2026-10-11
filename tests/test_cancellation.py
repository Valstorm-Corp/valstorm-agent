"""Unit tests for Workstream 4C: Mid-Run Cancellation (DELETE /v1/runs/{id})."""

import asyncio
import json
import pytest
from httpx import AsyncClient, ASGITransport

import server
from core.models import Message, StreamEvent, StreamEventType
from server import app


class _SessionStoreStub:
    def load_session(self, session_id): return None
    def save_session(self, session): pass


def _sse_events(response_text: str):
    return [
        json.loads(line.removeprefix("data: "))
        for line in response_text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


@pytest.mark.asyncio
async def test_cancel_nonexistent_run_returns_404():
    """Verify that attempting to cancel an unknown run returns a 404 error."""
    server.active_run_event_queues.clear()
    server.active_run_tasks.clear()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.delete("/v1/runs/non_existent_run_99999")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_cancel_active_run_emits_cancelled_event(monkeypatch):
    """Verify that deleting an in-flight run cancels the background task and emits run.cancelled."""
    cancel_entered = asyncio.Event()

    async def slow_stream(self, **kwargs):
        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="Starting long task...")
        cancel_entered.set()
        try:
            await asyncio.sleep(10.0)  # Long running task to be aborted
            yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="Finished")
        except asyncio.CancelledError:
            raise

    monkeypatch.setattr(server, "SessionStore", _SessionStoreStub)
    monkeypatch.setattr(server, "MemoryStore", lambda: object())
    monkeypatch.setattr(server, "_get_provider_instance", lambda *_: (object(), "gemini-flash-latest", "gemini"))
    monkeypatch.setattr(server, "_build_server_tool_registry", lambda **_: object())
    monkeypatch.setattr(server.ReActEngine, "run_turn_stream", slow_stream)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        create_resp = await ac.post("/v1/runs", json={
            "input": "Run long task",
            "model": "gemini-flash-latest",
            "session_id": "aich_cancel_test",
        })
        assert create_resp.status_code == 200
        run_id = create_resp.json()["run_id"]

        # Wait until runner has started
        async def cancel_after_start():
            await cancel_entered.wait()
            del_resp = await ac.delete(f"/v1/runs/{run_id}")
            assert del_resp.status_code == 200
            assert del_resp.json()["status"] == "cancelling"

        asyncio.create_task(cancel_after_start())

        # Stream events and verify run.cancelled is received
        events_resp = await ac.get(f"/v1/runs/{run_id}/events")
        assert events_resp.status_code == 200
        events = _sse_events(events_resp.text)

        cancelled_events = [e for e in events if e.get("event") == "run.cancelled"]
        assert len(cancelled_events) == 1
        assert cancelled_events[0]["status"] == "cancelled"
        assert cancelled_events[0]["run_id"] == run_id
