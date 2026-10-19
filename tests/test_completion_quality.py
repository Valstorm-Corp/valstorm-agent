"""Regression tests for the completion-quality fixes (see 00_AGENT_COMPLETION_QUALITY_TESTING.md).

Covers: persistent working directory across tools, the continuation guard, finish-reason handling,
per-call vs. per-turn completion events, partial-turn preservation, parallel tool-result pairing,
compaction digest placement, and the Gemini provider's thinking config / valstorm pass-through fallback.
"""

import os
from typing import Any, Dict, List, Optional

import pytest

from core.models import Message, SessionState, StreamEventType, ToolCall, ToolResult, UsageMetadata
from core.react import ReActEngine


# --------------------------------------------------------------------------- helpers
class _Registry:
    def __init__(self, names=("terminal_exec", "read_file")):
        self.names = list(names)
        self.calls: List[str] = []

    def get_schemas(self) -> List[Dict[str, Any]]:
        return [{"name": n, "description": n, "parameters": {"type": "object", "properties": {}}} for n in self.names]

    def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        self.calls.append(name)
        return "ok"


class _Provider:
    def __init__(self, responses: List[Message]):
        self.responses = list(responses)
        self.seen: List[List[Message]] = []

    async def generate(self, messages, tools=None, model=None):
        self.seen.append(list(messages))
        if self.responses:
            return self.responses.pop(0)
        return Message(role="assistant", content="All done and verified.")


async def _run(engine: ReActEngine, prompt: str, session: Optional[SessionState] = None):
    session = session or SessionState(active_model="m", active_provider="p")
    events = []
    async for ev in engine.run_turn_stream(session=session, user_input=prompt, model="m"):
        events.append(ev)
    return session, events


# --------------------------------------------------------------------------- finish reasons
@pytest.mark.asyncio
async def test_truncated_output_asks_model_to_continue():
    provider = _Provider([
        Message(role="assistant", content="Here is the first half of", finish_reason="length"),
        Message(role="assistant", content="the answer, completed."),
    ])
    engine = ReActEngine(provider=provider, tools=_Registry())
    session, _ = await _run(engine, "Explain")
    assert len(provider.seen) == 2
    assert any("cut off" in (m.content or "") for m in session.messages if m.role == "user")


@pytest.mark.asyncio
async def test_malformed_tool_call_is_retried():
    provider = _Provider([
        Message(role="assistant", content=None, finish_reason="malformed_tool_call"),
        Message(role="assistant", content="Recovered."),
    ])
    engine = ReActEngine(provider=provider, tools=_Registry())
    session, _ = await _run(engine, "Do it")
    assert len(provider.seen) == 2
    assert any("malformed" in (m.content or "") for m in session.messages if m.role == "user")


# --------------------------------------------------------------------------- event labelling
@pytest.mark.asyncio
async def test_provider_turn_complete_is_relabelled_per_call():
    from core.models import StreamEvent

    class _StreamProvider:
        def __init__(self):
            self.n = 0

        async def generate_stream(self, messages, tools=None, model=None):
            self.n += 1
            if self.n == 1:
                msg = Message(role="assistant", tool_calls=[ToolCall(id="c1", name="read_file", arguments={})])
            else:
                msg = Message(role="assistant", content="Finished.")
            yield StreamEvent(event_type=StreamEventType.TURN_COMPLETE, message=msg)

    engine = ReActEngine(provider=_StreamProvider(), tools=_Registry())
    _, events = await _run(engine, "go")
    kinds = [e.event_type for e in events]
    assert kinds.count(StreamEventType.TURN_COMPLETE) == 1
    assert kinds.count(StreamEventType.LLM_CALL_COMPLETE) == 2
    assert kinds[-1] == StreamEventType.TURN_COMPLETE


# --------------------------------------------------------------------------- sandbox cwd
@pytest.mark.asyncio
async def test_cd_persists_and_file_tools_follow(tmp_path):
    from core.sandbox import HostSandbox, set_current_sandbox
    from tools.developer_tools import patch_file, read_file, terminal_exec

    sub = tmp_path / "bench" / "pkg"
    sub.mkdir(parents=True)
    (sub / "mod.py").write_text("x = 1\n")
    sb = HostSandbox(base_dir=str(tmp_path))
    token = set_current_sandbox(sb)
    try:
        out = await terminal_exec(command="cd bench/pkg && echo hi")
        assert "hi" in out
        assert "Working directory is now" in out
        assert "__VSAGENT_CWD__" not in out
        assert sb.cwd == sub.resolve()

        pwd_out = await terminal_exec(command="pwd")
        assert str(sub.resolve()) in pwd_out
        assert "Working directory is now" not in pwd_out

        res = await patch_file(path="mod.py", old_string="x = 1", new_string="x = 2")
        assert str((sub / "mod.py").resolve()) in res
        assert (sub / "mod.py").read_text() == "x = 2\n"

        content = await read_file(path="mod.py")
        assert str((sub / "mod.py").resolve()) in content

        fail = await terminal_exec(command="exit 3")
        assert fail.startswith("Command exited with code 3")
    finally:
        from core.sandbox import _current_sandbox
        _current_sandbox.reset(token)


# --------------------------------------------------------------------------- partial turns
def test_preserve_partial_turn_keeps_completed_work():
    from cli.helpers import preserve_partial_turn

    s = SessionState(active_model="m", active_provider="p")
    s.add_message(Message(role="system", content="sys"))
    initial = len(s.messages)
    s.add_message(Message(role="user", content="do it"))
    s.add_message(Message(role="assistant", tool_calls=[ToolCall(id="a", name="read_file")]))
    s.add_message(Message(role="tool", content="r", tool_result=ToolResult(call_id="a", name="read_file", output="r")))
    s.add_message(Message(role="assistant", tool_calls=[ToolCall(id="b", name="terminal_exec"), ToolCall(id="c", name="read_file")]))
    s.add_message(Message(role="tool", content="r", tool_result=ToolResult(call_id="b", name="terminal_exec", output="r")))

    preserve_partial_turn(s, initial, "API error: boom")
    roles = [m.role for m in s.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert "interrupted" in s.messages[-1].content


# --------------------------------------------------------------------------- openai pairing
def test_openai_parallel_tool_results_stay_tool_messages():
    from providers.openai import OpenAIProvider

    p = OpenAIProvider(api_key="k", base_url="http://x", provider_name="valstorm")
    msgs = [
        Message(role="user", content="go"),
        Message(role="assistant", tool_calls=[ToolCall(id=f"t{i}", name="read_file") for i in range(3)]),
    ] + [
        Message(role="tool", content=f"r{i}", tool_result=ToolResult(call_id=f"t{i}", name="read_file", output=f"r{i}"))
        for i in range(3)
    ]
    out = p._format_messages(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "tool", "tool", "tool"]


# --------------------------------------------------------------------------- compaction
def test_compaction_digest_is_not_a_second_system_message():
    from core.compaction import ContextCompactor

    s = SessionState(active_model="gemini-flash-latest", active_provider="valstorm")
    s.add_message(Message(role="system", content="REAL SYSTEM PROMPT"))
    for i in range(40):
        s.add_message(Message(role="user", content=f"question {i} " + "x" * 2000))
        s.add_message(Message(role="assistant", content=f"answer {i} " + "y" * 2000))
    res = ContextCompactor().compact(s, target_tokens=5000)
    assert res.compacted
    systems = [m for m in s.messages if m.role == "system"]
    assert len(systems) == 1 and systems[0].content == "REAL SYSTEM PROMPT"
    assert any("HISTORICAL CONTEXT DIGEST" in (m.content or "") for m in s.messages if m.role == "user")


# --------------------------------------------------------------------------- gemini provider
def test_gemini_thinking_config_default_and_off(monkeypatch):
    from providers.gemini import GeminiProvider

    monkeypatch.delenv("VALSTORM_THINKING_LEVEL", raising=False)
    p = GeminiProvider(api_key="k", backend="aistudio")
    cfg = p._build_config("sys", None, {})
    assert cfg.thinking_config is not None

    monkeypatch.setenv("VALSTORM_THINKING_LEVEL", "off")
    cfg2 = p._build_config("sys", None, {})
    assert cfg2.thinking_config is None


def test_gemini_backend_env_override(monkeypatch):
    from providers.gemini import GeminiProvider

    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    assert GeminiProvider.is_enterprise_mode() is True
    monkeypatch.setenv("VALSTORM_GEMINI_BACKEND", "aistudio")
    assert GeminiProvider.is_enterprise_mode() is False


@pytest.mark.asyncio
async def test_valstorm_passthrough_falls_back_to_openai_gateway_on_404():
    from providers.gemini import GeminiProvider

    class _NotFound(Exception):
        code = 404

    class _Models:
        async def generate_content_stream(self, **kw):
            raise _NotFound("404 Not Found")

    class _Aio:
        models = _Models()

    class _Client:
        aio = _Aio()

    class _Compat:
        called = 0

        def set_request_context(self, **kw):
            pass

        async def generate_stream(self, messages, tools=None, model=None, **kw):
            _Compat.called += 1
            from core.models import StreamEvent
            yield StreamEvent(event_type=StreamEventType.TURN_COMPLETE, message=Message(role="assistant", content="via compat"))

    p = GeminiProvider(api_key="tok", backend="valstorm", valstorm_base_url="http://x/v1/ai/gemini",
                       client=_Client(), compat_provider=_Compat())
    items = [i async for i in p.generate_stream(messages=[Message(role="user", content="hi")])]
    assert _Compat.called == 1
    assert p._passthrough_unavailable is True
    assert p.backend_label() == "valstorm-compat"
    assert any(getattr(i, "message", None) and i.message.content == "via compat" for i in items)


@pytest.mark.asyncio
async def test_action_intent_nudge_handles_adverbs_and_resets_after_tool_calls():
    from core.react import _has_unfulfilled_action_intent, evaluate_action_intent_s1

    assert _has_unfulfilled_action_intent("I will now write the Connecticut Market Research script at x.py.") is True
    assert _has_unfulfilled_action_intent("I will now query the Google Ads Keyword Planner API.") is True
    assert _has_unfulfilled_action_intent("Let me invoke the calculator tool to compute the total.") is True
    assert _has_unfulfilled_action_intent("I'll go ahead and apply the patch to server.py now.") is True
    # Should not false-trigger when asking a question, deferring until confirmation, or quoting in a markdown table
    assert _has_unfulfilled_action_intent("I will write a script. Would you like me to proceed?") is False
    assert _has_unfulfilled_action_intent("I will modify those files once you confirm.") is False
    assert _has_unfulfilled_action_intent("I will run the migration after you approve the plan.") is False
    assert _has_unfulfilled_action_intent("If you want, I will create the database schema.") is False
    assert _has_unfulfilled_action_intent("Summary:\n\n| Turn | Text |\n| --- | --- |\n| 1 | I will write a script |") is False

    s1_pos = evaluate_action_intent_s1("I will now write the Connecticut Market Research script at x.py.")
    assert s1_pos["unfulfilled_intent"] is True
    assert s1_pos["noul"] >= 0.55

    s1_defer = evaluate_action_intent_s1("I will modify those files once you confirm.")
    assert s1_defer["unfulfilled_intent"] is False
    assert s1_defer["has_deferral"] is True

    provider = _Provider([
        Message(role="assistant", content="I will now write the research script."),
        Message(role="assistant", tool_calls=[ToolCall(id="c1", name="terminal_exec", arguments={"command": "echo 1"})]),
        Message(role="assistant", content="I will now run the second step."),
        Message(role="assistant", content="All steps completed."),
    ])
    engine = ReActEngine(provider=provider, tools=_Registry())
    session, _ = await _run(engine, "Execute research")
    assert len(provider.seen) == 4
    assert session.messages[-1].content == "All steps completed."


def test_resolve_default_workdir_avoids_filesystem_root(monkeypatch, tmp_path):
    from pathlib import Path
    from core.sandbox import HostSandbox, resolve_default_workdir

    monkeypatch.setenv("VALSTORM_WORKSPACE_DIR", "")
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: Path("/")))
    resolved = resolve_default_workdir()
    assert resolved != Path("/")
    assert resolved != Path.home().resolve()
    sb = HostSandbox()
    assert sb.cwd != Path("/")
    assert sb.cwd != Path.home().resolve()


def test_resolve_default_workdir_avoids_raw_home_dir(monkeypatch, tmp_path):
    from pathlib import Path
    from core.sandbox import HostSandbox, resolve_default_workdir

    fake_home = tmp_path / "fake_user_home"
    fake_home.mkdir()
    monkeypatch.setenv("VALSTORM_WORKSPACE_DIR", "")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: fake_home))

    resolved = resolve_default_workdir()
    assert resolved == (fake_home / ".valstorm" / "workspace").resolve()
    assert resolved.is_dir()

    custom_vault = tmp_path / "my_project"
    custom_vault.mkdir()
    sb = HostSandbox(base_dir=str(custom_vault))
    assert sb.cwd == custom_vault.resolve()


def test_search_files_ignores_macos_library_and_app_containers(tmp_path):
    from tools.developer_tools import _search_files_sync, IGNORED_DIRECTORIES

    assert "Library" in IGNORED_DIRECTORIES
    assert "Applications" in IGNORED_DIRECTORIES
    assert ".Trash" in IGNORED_DIRECTORIES

    # Simulate a user directory containing Library/Containers and normal Documents
    lib_container = tmp_path / "Library" / "Containers" / "com.apple.mail"
    lib_container.mkdir(parents=True)
    (lib_container / "secret_target.py").write_text("SECRET_TOKEN = '123'\n", encoding="utf-8")

    app_bundle = tmp_path / "SomeApp.app" / "Contents"
    app_bundle.mkdir(parents=True)
    (app_bundle / "secret_target.py").write_text("SECRET_TOKEN = '456'\n", encoding="utf-8")

    project_dir = tmp_path / "Documents" / "Project"
    project_dir.mkdir(parents=True)
    (project_dir / "secret_target.py").write_text("SECRET_TOKEN = 'allowed'\n", encoding="utf-8")

    file_res = _search_files_sync("secret_target.py", target="files", path=str(tmp_path))
    assert "Documents/Project/secret_target.py" in file_res
    assert "Library" not in file_res
    assert "SomeApp.app" not in file_res

    content_res = _search_files_sync("SECRET_TOKEN", target="content", path=str(tmp_path))
    assert "Documents/Project/secret_target.py" in content_res
    assert "Library" not in content_res
    assert "SomeApp.app" not in content_res


@pytest.mark.asyncio
async def test_vfs_search_does_not_double_prefix_v1(monkeypatch):
    from tools.valstorm_client import ValstormApiClient

    called_endpoints = []

    async def fake_req(self, method, endpoint, **kwargs):
        called_endpoints.append(endpoint)

        class _Resp:
            def json(self):
                return {"results": []}

        return _Resp()

    monkeypatch.setattr(ValstormApiClient, "_request_with_retry", fake_req)
    client = ValstormApiClient(token="fake_tok", base_url="https://api.valstorm.com/v1")
    await client.vfs_search("marketing")
    assert called_endpoints == ["/search"]


@pytest.mark.asyncio
async def test_valstorm_truncated_sse_stream_recovers_via_generate_content():
    from providers.gemini import GeminiProvider

    class _TruncatedChunk:
        text = "I will now write the research script."
        candidates = []
        usage_metadata = None
        model_version = None

    async def _truncated_stream():
        yield _TruncatedChunk()
        # Stream closes prematurely without finish_reason, tool_calls, or usage_metadata

    class _RecoveredFC:
        id = "call_rec_1"
        name = "write_file"
        args = {"path": "run_ct_market_research.py", "content": "print('ok')"}

    class _RecoveredPart:
        thought = False
        text = "I will now write the research script."
        function_call = _RecoveredFC()
        thought_signature = b"sig_bytes"

    class _RecoveredCandidate:
        finish_reason = "STOP"
        content = type("Content", (), {"parts": [_RecoveredPart()]})()

    class _RecoveredUsage:
        prompt_token_count = 45000
        candidates_token_count = 600
        total_token_count = 45600
        cached_content_token_count = 0
        thoughts_token_count = 100

    class _RecoveredResp:
        candidates = [_RecoveredCandidate()]
        usage_metadata = _RecoveredUsage()
        model_version = "gemini-4-argon"

    class _Models:
        async def generate_content_stream(self, **kw):
            return _truncated_stream()

        async def generate_content(self, **kw):
            return _RecoveredResp()

    class _Client:
        aio = type("Aio", (), {"models": _Models()})()

    p = GeminiProvider(api_key="tok", backend="valstorm", client=_Client())
    items = [i async for i in p.generate_stream(messages=[Message(role="user", content="go")])]
    tool_events = [i for i in items if getattr(i, "event_type", None) == StreamEventType.TOOL_CALL_DETECTED]
    assert len(tool_events) == 1
    assert tool_events[0].tool_call.name == "write_file"
    final_tuple = [i for i in items if isinstance(i, tuple)]
    assert len(final_tuple) == 1
    msg, usage = final_tuple[0]
    assert msg.tool_calls and msg.tool_calls[0].name == "write_file"
    assert usage.prompt_tokens == 45000


def test_reasoning_effort_across_gemini_openai_and_anthropic_providers():
    from providers.gemini import GeminiProvider
    from providers.openai import OpenAIProvider
    from providers.anthropic import AnthropicProvider

    # 1. GeminiProvider honors reasoning_effort from request_context or kwargs and sets X-Valstorm-Thinking-Level
    gp = GeminiProvider(api_key="tok", backend="valstorm")
    gp.set_request_context(chat_id="aich_123", reasoning_effort="off")
    cfg_off = gp._build_config("sys", None, {"model": "gemini-3.8-flash"})
    assert cfg_off.thinking_config is None
    assert cfg_off.http_options.headers["X-Valstorm-Thinking-Level"] == "off"

    gp.set_request_context(chat_id="aich_123", reasoning_effort="med")
    cfg_med = gp._build_config("sys", None, {"model": "gemini-3.8-flash"})
    assert cfg_med.thinking_config is not None
    assert cfg_med.http_options.headers["X-Valstorm-Thinking-Level"] == "medium"

    # 2. OpenAIProvider sets X-Valstorm-Thinking-Level on valstorm gateway and reasoning_effort on gpt-5.6-sol
    op_valstorm = OpenAIProvider(api_key="tok", base_url="http://x/v1/ai", provider_name="valstorm")
    op_valstorm.set_request_context(chat_id="aich_123", reasoning_effort="high")
    req_kw = {"model": "claude-sonnet-5-5", "messages": []}
    op_valstorm._apply_valstorm_headers(req_kw)
    assert req_kw["extra_headers"]["X-Valstorm-Chat-Id"] == "aich_123"
    assert req_kw["extra_headers"]["X-Valstorm-Thinking-Level"] == "high"

    op_direct = OpenAIProvider(api_key="tok", provider_name="openai")
    op_direct.set_request_context(reasoning_effort="low")
    req_kw2 = {"model": "gpt-5.6-sol", "messages": []}
    op_direct._apply_valstorm_headers(req_kw2)
    assert req_kw2["reasoning_effort"] == "low"

    # 3. AnthropicProvider defaults to claude-haiku-5-5
    ap = AnthropicProvider(api_key="tok")
    assert ap.default_model == "claude-haiku-5-5"



