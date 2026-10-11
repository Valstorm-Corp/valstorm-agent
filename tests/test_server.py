"""Unit tests for the Valstorm Agent Gateway Server (/v1/runs and SSE events)."""

import json

import pytest
from httpx import AsyncClient, ASGITransport

import server
from core.models import Message, StreamEvent, StreamEventType, UsageMetadata
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


async def _create_and_consume_run(ac: AsyncClient):
    created = await ac.post("/v1/runs", json={
        "input": "test input",
        "model": "target-model",
        "provider": "target-provider",
        "session_id": "aich_test_session",
    })
    assert created.status_code == 200
    run_id = created.json()["run_id"]
    events = await ac.get(f"/v1/runs/{run_id}/events")
    assert events.status_code == 200
    return _sse_events(events.text)


@pytest.mark.asyncio
async def test_run_completed_sse_includes_final_turn_usage_and_resolved_target(monkeypatch):
    async def fake_run_turn_stream(self, **kwargs):
        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="answer")
        yield StreamEvent(
            event_type=StreamEventType.TURN_COMPLETE,
            message=Message(
                role="assistant",
                content="answer",
                usage=UsageMetadata(prompt_tokens=123, completion_tokens=45),
            ),
        )

    monkeypatch.setattr(server, "SessionStore", _SessionStoreStub)
    monkeypatch.setattr(server, "MemoryStore", lambda: object())
    monkeypatch.setattr(server, "_get_provider_instance", lambda *_: (object(), "resolved-model", "resolved-provider"))
    monkeypatch.setattr(server, "_build_server_tool_registry", lambda **_: object())
    monkeypatch.setattr(server.ReActEngine, "run_turn_stream", fake_run_turn_stream)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        events = await _create_and_consume_run(ac)

    completed = [event for event in events if event["event"] == "run.completed"]
    assert completed == [{
        "event": "run.completed",
        "run_id": completed[0]["run_id"],
        "output": "answer",
        "status": "completed",
        "input_tokens": 123,
        "output_tokens": 45,
        "model": "resolved-model",
        "provider": "resolved-provider",
    }]


@pytest.mark.asyncio
async def test_subagent_tokens_captured_and_synced_on_run_completed(monkeypatch):
    """Verify that when subagents execute during a run, their tokens and metadata are captured in run.completed and synced."""
    from tools.delegation import SubagentRegistry, SubagentTaskSession
    from core.models import SessionState, Message, UsageMetadata, StreamEvent, StreamEventType

    sync_calls = []
    async def fake_sync(**kwargs):
        sync_calls.append(kwargs)

    async def fake_run_turn_stream(self, **kwargs):
        # Simulate spawning a subagent that completed with tokens
        child_sess = SessionState(
            session_id="aich_sub_test_123",
            active_model="gemini-pro-latest",
            active_provider="gemini",
        )
        child_sess.add_message(Message(
            role="assistant",
            content="Subagent developer finished task.",
            usage=UsageMetadata(prompt_tokens=450, completion_tokens=150),
        ))
        task_sess = SubagentTaskSession(
            task_id="subtask_dev_99",
            child_session_id="aich_sub_test_123",
            profile="developer",
            profile_name="Valstorm Developer",
            goal="Write dynamic Python function",
            context=None,
            model_name="gemini-pro-latest",
            provider_name="gemini",
            scoped_tools=["valstorm_validate_function", "valstorm_record_cud"],
            child_session=child_sess,
        )
        task_sess.status = "completed"
        task_sess.outcome = "Function created and validated."
        task_sess.tools_executed = ["valstorm_validate_function", "valstorm_record_cud"]
        SubagentRegistry.get_instance().register(task_sess)

        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="Chief summary.")
        yield StreamEvent(
            event_type=StreamEventType.TURN_COMPLETE,
            message=Message(
                role="assistant",
                content="Chief summary.",
                usage=UsageMetadata(prompt_tokens=80, completion_tokens=30),
            ),
        )

    monkeypatch.setattr(server, "SessionStore", _SessionStoreStub)
    monkeypatch.setattr(server, "MemoryStore", lambda: object())
    monkeypatch.setattr(server, "_get_provider_instance", lambda *_: (object(), "gemini-flash-latest", "gemini"))
    monkeypatch.setattr(server, "_build_server_tool_registry", lambda **_: object())
    monkeypatch.setattr(server, "_sync_turn_to_valstorm", fake_sync)
    monkeypatch.setattr(server.ReActEngine, "run_turn_stream", fake_run_turn_stream)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        events = await _create_and_consume_run(ac)

    completed = [event for event in events if event["event"] == "run.completed"]
    assert len(completed) == 1
    comp = completed[0]
    assert comp["input_tokens"] == 80
    assert comp["output_tokens"] == 30
    assert "subagent_results" in comp
    assert len(comp["subagent_results"]) == 1
    sub_res = comp["subagent_results"][0]
    assert sub_res["run_id"] == "subtask_dev_99"
    assert sub_res["agent_role"] == "developer"
    assert sub_res["input_tokens"] == 450
    assert sub_res["output_tokens"] == 150
    assert sub_res["tool_calls"] == [{"name": "valstorm_validate_function", "status": "completed"}, {"name": "valstorm_record_cud", "status": "completed"}]

    # Check that _sync_turn_to_valstorm received child_syncs with exact tokens
    assert len(sync_calls) == 1
    synced_subagents = sync_calls[0]["subagents"]
    assert len(synced_subagents) == 1
    assert synced_subagents[0]["input_tokens"] == 450
    assert synced_subagents[0]["output_tokens"] == 150


@pytest.mark.asyncio
@pytest.mark.parametrize("usage", [None, {"prompt_tokens": -1, "completion_tokens": "45"}])
async def test_run_completed_sse_uses_zero_usage_when_final_turn_usage_is_missing_or_invalid(monkeypatch, usage):
    async def fake_run_turn_stream(self, **kwargs):
        message = Message(role="assistant", content="answer")
        if usage is not None:
            message.usage = usage
        yield StreamEvent(event_type=StreamEventType.TURN_COMPLETE, message=message)

    monkeypatch.setattr(server, "SessionStore", _SessionStoreStub)
    monkeypatch.setattr(server, "MemoryStore", lambda: object())
    monkeypatch.setattr(server, "_get_provider_instance", lambda *_: (object(), "resolved-model", "resolved-provider"))
    monkeypatch.setattr(server, "_build_server_tool_registry", lambda **_: object())
    monkeypatch.setattr(server.ReActEngine, "run_turn_stream", fake_run_turn_stream)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        events = await _create_and_consume_run(ac)

    completed = next(event for event in events if event["event"] == "run.completed")
    assert completed["input_tokens"] == 0
    assert completed["output_tokens"] == 0


@pytest.mark.asyncio
async def test_server_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["engine"] == "valstorm-agent"
        assert "port" in data
        assert "mode" in data
        assert "default_sandbox" in data


@pytest.mark.asyncio
async def test_server_create_run_and_404_events(monkeypatch):
    async def fake_run_turn_stream(self, **kwargs):
        yield StreamEvent(event_type=StreamEventType.TURN_COMPLETE, message=Message(role="assistant", content="4"))

    monkeypatch.setattr(server, "SessionStore", _SessionStoreStub)
    monkeypatch.setattr(server, "MemoryStore", lambda: object())
    monkeypatch.setattr(server, "_get_provider_instance", lambda *_: (object(), "gemini-flash-lite-latest", "gemini"))
    monkeypatch.setattr(server, "_build_server_tool_registry", lambda **_: object())
    monkeypatch.setattr(server.ReActEngine, "run_turn_stream", fake_run_turn_stream)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Invalid run ID should 404
        bad_resp = await ac.get("/v1/runs/non_existent_run_12345/events")
        assert bad_resp.status_code == 404

        # 2. Create Run
        payload = {
            "input": "Calculate 2 + 2",
            "model": "gemini-flash-lite-latest",
            "provider": "gemini",
            "session_id": "aich_test_session_123",
        }
        create_resp = await ac.post("/v1/runs", json=payload)
        assert create_resp.status_code == 200
        data = create_resp.json()
        assert "run_id" in data
        assert data["status"] == "running"
        assert data["session_id"] == "aich_test_session_123"


@pytest.mark.asyncio
async def test_autonomous_direct_sync_to_valstorm(monkeypatch):
    synced_payloads = []

    class MockResponse:
        status_code = 200
        text = '{"status": "success"}'

    async def mock_post(self, url, json=None, headers=None):
        synced_payloads.append({"url": url, "json": json, "headers": headers})
        return MockResponse()

    monkeypatch.setattr("httpx.AsyncClient.post", mock_post)

    await server._sync_turn_to_valstorm(
        chat_id="aich_test_123",
        run_id="run_sync_test",
        output_text="Autonomous sync output",
        status="completed",
        input_tokens=100,
        output_tokens=50,
        model="gemini-flash-latest",
        provider="gemini",
        valstorm_token="test_token_abc",
        valstorm_base_url="https://api.valstorm.com/v1",
        session_id="aich_test_123",
    )

    assert len(synced_payloads) == 1
    assert synced_payloads[0]["url"] == "https://api.valstorm.com/v1/ai/chat/aich_test_123/desktop-sync"
    assert synced_payloads[0]["headers"]["Authorization"] == "Bearer test_token_abc"
    assert synced_payloads[0]["json"]["run_id"] == "run_sync_test"
    assert synced_payloads[0]["json"]["input_tokens"] == 100
    assert synced_payloads[0]["json"]["output_tokens"] == 50
    assert synced_payloads[0]["json"]["status"] == "completed"
    assert synced_payloads[0]["json"]["text"] == "Autonomous sync output"


@pytest.mark.asyncio
async def test_hydrate_session_from_cloud_if_needed(monkeypatch):
    from core.models import SessionState, Message

    session = SessionState(session_id="aich_help_turn_test", active_model="gemini-flash-lite-latest", active_provider="gemini")
    session.add_message(Message(role="system", content="System prompt"))

    mock_db_messages = [
        {"id": "msg_help_user", "role": "user", "body": "/help how do I create custom objects", "created_date": "2026-09-05T10:00:00Z"},
        {"id": "msg_help_asst", "role": "assistant", "body": "Here is the gold guide on creating custom objects: ...", "created_date": "2026-09-05T10:00:01Z"},
    ]

    class MockQueryResponse:
        status_code = 200
        def json(self):
            return {"data": mock_db_messages}

    async def mock_query_post(self, url, json=None, headers=None):
        return MockQueryResponse()

    monkeypatch.setattr("httpx.AsyncClient.post", mock_query_post)

    saved_sessions = []
    class MockSessionStore:
        def save_session(self, s):
            saved_sessions.append(s)

    await server._hydrate_session_from_cloud_if_needed(
        session=session,
        session_id="aich_help_turn_test",
        valstorm_token="test_jwt",
        valstorm_base_url="https://api.valstorm.com/v1",
        session_store=MockSessionStore(),
    )

    assert len(session.messages) == 3
    assert session.messages[1].role == "user"
    assert session.messages[1].content == "/help how do I create custom objects"
    assert session.messages[2].role == "assistant"
    assert "Here is the gold guide" in session.messages[2].content
    assert len(saved_sessions) == 1


@pytest.mark.asyncio
async def test_memories_endpoints(tmp_path, monkeypatch):
    from core.memory import MemoryStore

    test_mem_file = tmp_path / "test_memories.json"
    fake_store = MemoryStore(file_path=test_mem_file)
    monkeypatch.setattr(server, "MemoryStore", lambda *args, **kwargs: fake_store)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Initial list should be empty
        get_res = await ac.get("/v1/memories")
        assert get_res.status_code == 200
        assert get_res.json() == {"user": [], "memory": []}

        # 2. Add environment/system memory fact
        add_res = await ac.post("/v1/memories", json={
            "content": "Always run pytest with -v flag",
            "target": "memory",
        })
        assert add_res.status_code == 200
        data = add_res.json()
        assert data["status"] == "success"
        assert data["target"] == "memory"
        assert "Always run pytest" in data["content"]

        # 3. Add user profile fact with auto-classification
        user_add_res = await ac.post("/v1/memories", json={
            "content": "I prefer TypeScript strict mode",
            "target": "auto",
        })
        assert user_add_res.status_code == 200
        user_data = user_add_res.json()
        assert user_data["status"] == "success"
        assert user_data["target"] == "user"

        # 4. Verify memories stored
        get_all_res = await ac.get("/v1/memories")
        all_data = get_all_res.json()
        assert "Always run pytest with -v flag" in all_data["memory"]
        assert "I prefer TypeScript strict mode" in all_data["user"]

        # 5. Delete a memory fact
        del_res = await ac.request("DELETE", "/v1/memories", json={
            "old_text": "strict mode",
            "target": "user",
        })
        assert del_res.status_code == 200
        assert del_res.json()["removed"] is True

        # Verify it was removed
        get_user_res = await ac.get("/v1/memories?target=user")
        assert len(get_user_res.json()["user"]) == 0


@pytest.mark.asyncio
async def test_run_turn_memorize_command(tmp_path, monkeypatch):
    from core.memory import MemoryStore

    test_mem_file = tmp_path / "test_mem_run.json"
    fake_store = MemoryStore(file_path=test_mem_file)
    monkeypatch.setattr(server, "MemoryStore", lambda *args, **kwargs: fake_store)
    monkeypatch.setattr(server, "SessionStore", _SessionStoreStub)
    monkeypatch.setattr(server, "_get_provider_instance", lambda *_: (object(), "target-model", "target-provider"))
    monkeypatch.setattr(server, "_build_server_tool_registry", lambda **_: object())

    captured_inputs = []

    async def fake_run_turn_stream(self, session, user_input, **kwargs):
        captured_inputs.append(user_input)
        yield StreamEvent(
            event_type=StreamEventType.TURN_COMPLETE,
            message=Message(role="assistant", content="I have committed this to memory."),
        )

    monkeypatch.setattr(server.ReActEngine, "run_turn_stream", fake_run_turn_stream)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        created = await ac.post("/v1/runs", json={
            "input": "/memorize Never push directly to main branch",
            "model": "target-model",
            "provider": "target-provider",
            "session_id": "aich_mem_session_1",
        })
        assert created.status_code == 200
        run_id = created.json()["run_id"]
        events_res = await ac.get(f"/v1/runs/{run_id}/events")
        assert events_res.status_code == 200

    # Verify fact was committed directly into MemoryStore
    facts = fake_store.get_facts("memory")
    assert "Never push directly to main branch" in facts["memory"]

    # Verify system directive was injected for the model
    assert len(captured_inputs) == 1
    assert "SYSTEM DIRECTIVE" in captured_inputs[0]
    assert "Never push directly to main branch" in captured_inputs[0]


@pytest.mark.asyncio
async def test_run_resolves_profile_from_agent_id(monkeypatch):
    captured_profile = []

    def fake_load_profile(slug):
        captured_profile.append(slug)
        return {
            "name": "Software Developer",
            "api_name": "developer",
            "allowed_tools": ["terminal_exec", "read_file"],
            "system_prompt": "You are a developer",
        }

    async def fake_run_turn_stream(self, **kwargs):
        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="ok")
        yield StreamEvent(
            event_type=StreamEventType.TURN_COMPLETE,
            message=Message(role="assistant", content="ok"),
        )

    monkeypatch.setattr(server, "SessionStore", _SessionStoreStub)
    monkeypatch.setattr(server, "MemoryStore", lambda: object())
    monkeypatch.setattr(server, "load_profile", fake_load_profile)
    monkeypatch.setattr(server, "_get_provider_instance", lambda *_: (object(), "resolved-model", "resolved-provider"))
    monkeypatch.setattr(server, "_build_server_tool_registry", lambda **_: object())
    monkeypatch.setattr(server.ReActEngine, "run_turn_stream", fake_run_turn_stream)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        created = await ac.post("/v1/runs", json={
            "input": "echo hello",
            "agent_id": "developer",  # Passed from DesktopRemoteAgent
            "session_id": "aich_dev_test",
        })
        assert created.status_code == 200
        run_id = created.json()["run_id"]
        events = await ac.get(f"/v1/runs/{run_id}/events")
        assert events.status_code == 200

    assert "developer" in captured_profile


@pytest.mark.asyncio
async def test_openai_models_endpoint():
    """Verify /v1/models and /models return list of available agent profiles in OpenAI format."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res1 = await ac.get("/v1/models")
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["object"] == "list"
        model_ids = [m["id"] for m in data1["data"]]
        assert "vsagent" in model_ids
        assert "vsagent-developer" in model_ids

        res2 = await ac.get("/models")
        assert res2.status_code == 200
        assert res2.json()["object"] == "list"

        res3 = await ac.get("/v1/models/vsagent-developer")
        assert res3.status_code == 200
        assert res3.json()["id"] == "vsagent-developer"


@pytest.mark.asyncio
async def test_openai_chat_completions_non_streaming(monkeypatch):
    """Verify non-streaming POST /v1/chat/completions returns standard OpenAI JSON."""
    async def fake_run_turn_stream(self, session, user_input, **kwargs):
        assert user_input == "What is 2+2?"
        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="The answer is 4.")
        yield StreamEvent(
            event_type=StreamEventType.TURN_COMPLETE,
            message=Message(
                role="assistant",
                content="The answer is 4.",
                usage=UsageMetadata(prompt_tokens=25, completion_tokens=10),
            ),
        )

    monkeypatch.setattr(server, "SessionStore", _SessionStoreStub)
    monkeypatch.setattr(server, "MemoryStore", lambda: object())
    monkeypatch.setattr(server, "_get_provider_instance", lambda *_: (object(), "gemini-flash", "gemini"))
    monkeypatch.setattr(server, "_build_server_tool_registry", lambda **_: object())
    monkeypatch.setattr(server.ReActEngine, "run_turn_stream", fake_run_turn_stream)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/v1/chat/completions", json={
            "model": "vsagent-developer",
            "messages": [
                {"role": "system", "content": "You are a calculator."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            "stream": False,
        })
        assert res.status_code == 200
        body = res.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == "vsagent-developer"
        assert len(body["choices"]) == 1
        assert body["choices"][0]["message"]["content"] == "The answer is 4."
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"]["prompt_tokens"] == 25
        assert body["usage"]["completion_tokens"] == 10
        assert body["usage"]["total_tokens"] == 35


@pytest.mark.asyncio
async def test_openai_chat_completions_streaming(monkeypatch):
    """Verify streaming POST /v1/chat/completions yields standard OpenAI SSE chunks and [DONE]."""
    async def fake_run_turn_stream(self, session, user_input, **kwargs):
        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="Hello ")
        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="world!")
        yield StreamEvent(
            event_type=StreamEventType.TURN_COMPLETE,
            message=Message(
                role="assistant",
                content="Hello world!",
                usage=UsageMetadata(prompt_tokens=15, completion_tokens=5),
            ),
        )

    monkeypatch.setattr(server, "SessionStore", _SessionStoreStub)
    monkeypatch.setattr(server, "MemoryStore", lambda: object())
    monkeypatch.setattr(server, "_get_provider_instance", lambda *_: (object(), "gemini-flash", "gemini"))
    monkeypatch.setattr(server, "_build_server_tool_registry", lambda **_: object())
    monkeypatch.setattr(server.ReActEngine, "run_turn_stream", fake_run_turn_stream)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/v1/chat/completions", json={
            "model": "vsagent",
            "messages": [
                {"role": "user", "content": "Say hello."},
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
        })
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]

        lines = res.text.splitlines()
        data_lines = [l.removeprefix("data: ") for l in lines if l.startswith("data: ")]
        assert data_lines[-1] == "[DONE]"

        chunks = [json.loads(d) for d in data_lines[:-1]]
        deltas = [c["choices"][0]["delta"].get("content", "") for c in chunks if c.get("choices") and "content" in c["choices"][0]["delta"]]
        assert "".join(deltas) == "Hello world!"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert chunks[-1]["usage"]["prompt_tokens"] == 15
        assert chunks[-1]["usage"]["completion_tokens"] == 5


@pytest.mark.asyncio
async def test_agent_run_failure_emits_error_delta_and_syncs_error(monkeypatch):
    """Verify that when run_turn_stream raises an unhandled error, an error delta is emitted and synced."""
    sync_calls = []

    async def fake_sync(**kwargs):
        sync_calls.append(kwargs)
        return True

    async def fake_failing_run_turn_stream(self, **kwargs):
        yield StreamEvent(event_type=StreamEventType.TEXT_CHUNK, delta="Started task...")
        raise RuntimeError("400 INVALID_ARGUMENT: Function call is missing a thought_signature in functionCall parts")

    monkeypatch.setattr(server, "SessionStore", _SessionStoreStub)
    monkeypatch.setattr(server, "MemoryStore", lambda: object())
    monkeypatch.setattr(server, "_get_provider_instance", lambda *_: (object(), "gemini-3.7-flash", "gemini"))
    monkeypatch.setattr(server, "_build_server_tool_registry", lambda **_: object())
    monkeypatch.setattr(server, "_sync_turn_to_valstorm", fake_sync)
    monkeypatch.setattr(server.ReActEngine, "run_turn_stream", fake_failing_run_turn_stream)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        events = await _create_and_consume_run(ac)

    # 1. message.delta with error card must have been emitted
    deltas = [e["delta"] for e in events if e.get("event") == "message.delta"]
    assert any("Started task..." in d for d in deltas)
    assert any("❌ [Agent Run Terminated]" in d and "thought_signature" in d for d in deltas)

    # 2. run.failed event must have been emitted
    failed_events = [e for e in events if e.get("event") == "run.failed"]
    assert len(failed_events) == 1
    assert "thought_signature" in failed_events[0]["error"]

    # 3. _sync_turn_to_valstorm must have received status="error" and output_text containing the error notice
    assert len(sync_calls) == 1
    assert sync_calls[0]["status"] == "error"
    assert "❌ [Agent Run Terminated]" in sync_calls[0]["output_text"]
    assert "thought_signature" in sync_calls[0]["error"]




