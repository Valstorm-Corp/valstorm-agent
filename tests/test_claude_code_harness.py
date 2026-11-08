"""Tests for the Claude Code head-to-head harness: CLI flags, env isolation, stream-json telemetry,
grader-integrity checks, and the vsagent ToolRegistry MCP bridge."""

import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from benchmarks.adapters import (
    BenchmarkConfig,
    ExecutionTrace,
    ExternalCliBenchmarkAdapter,
    build_claude_env,
    build_cli_command,
    build_mcp_bridge_config,
    detect_grader_tampering,
    parse_claude_stream,
)
from benchmarks.adapters.external_cli_adapter import configure_claude_auth
from benchmarks.runner import get_ablation_configs, load_tasks
from benchmarks.scorer import EnvironmentStateScorer

RUNTIME_ROOT = Path(__file__).resolve().parent.parent
BRIDGE = RUNTIME_ROOT / "benchmarks" / "mcp_bridge.py"


def _stream_fixture(workspace: str = "/ws") -> list:
    events = [
        {"type": "system", "subtype": "init", "model": "claude-opus-5-5",
         "mcp_servers": [{"name": "vsagent", "status": "connected"}]},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Working on it"},
            {"type": "tool_use", "id": "t1", "name": "mcp__vsagent__valstorm_sql_query", "input": {"query": "SELECT id FROM contact"}},
            {"type": "tool_use", "id": "t2", "name": "Write", "input": {"file_path": f"{workspace}/out.txt", "content": "hi"}},
        ]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "[{\"id\": \"con_master_1\"}]"}]},
            {"type": "tool_result", "tool_use_id": "t2", "content": "File written"},
        ]}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "t3", "name": "Bash", "input": {"command": "cat valstorm_sandbox_db.json"}},
        ]}},
        {"type": "result", "subtype": "success", "is_error": False, "result": "Done.", "num_turns": 4,
         "total_cost_usd": 0.0421, "duration_api_ms": 5300,
         "usage": {"input_tokens": 1, "output_tokens": 1},
         "modelUsage": {
             "claude-opus-5-5": {"inputTokens": 120, "outputTokens": 800, "cacheReadInputTokens": 30000, "cacheCreationInputTokens": 9000},
             "claude-haiku-4-5": {"inputTokens": 50, "outputTokens": 40, "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0},
         }},
    ]
    return [json.dumps(e) + "\n" for e in events]


def test_claude_command_isolation_flags():
    cfg = BenchmarkConfig(config_id="c", label="c", harness="claude", model="opus", thinking_level="high")
    cmd = build_cli_command("claude", "Do it", "opus", config=cfg)
    assert cmd[:3] == ["claude", "-p", "Do it"]
    assert cmd[cmd.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in cmd and "--strict-mcp-config" in cmd and "--no-session-persistence" in cmd
    assert cmd[cmd.index("--setting-sources") + 1] == "project,local"
    assert "--disable-slash-commands" in cmd and "--bare" not in cmd
    assert cmd[cmd.index("--model") + 1] == "opus"
    assert cmd[cmd.index("--effort") + 1] == "high"
    assert "AskUserQuestion" in cmd[cmd.index("--disallowedTools") + 1]

    # Bare mode replaces setting-source scoping; non-Claude models are never forwarded.
    bare_cmd = build_cli_command("claude", "Do it", "gemini-flash-latest", config=cfg, bare=True)
    assert "--bare" in bare_cmd and "--setting-sources" not in bare_cmd and "--model" not in bare_cmd

    no_tools = BenchmarkConfig(config_id="n", label="n", harness="claude", tools_mode="none", thinking_level="adaptive")
    nt_cmd = build_cli_command("claude", "Do it", "", config=no_tools)
    assert nt_cmd[nt_cmd.index("--tools") + 1] == "" and "--effort" not in nt_cmd


def test_claude_env_scrubs_parent_session_vars():
    env = build_claude_env({
        "PATH": "/bin",
        "CLAUDECODE": "1",
        "CLAUDE_CODE_SIMPLE": "1",
        "CLAUDE_CODE_SESSION_ID": "x",
        "ANTHROPIC_BASE_URL": "http://localhost:9999",
        "CLAUDE_CODE_OAUTH_TOKEN": "tok",
        "ANTHROPIC_API_KEY": "key",
    })
    assert env == {"PATH": "/bin", "CLAUDE_CODE_OAUTH_TOKEN": "tok", "ANTHROPIC_API_KEY": "key"}


def test_claude_auth_mode_selection():
    assert configure_claude_auth({"ANTHROPIC_API_KEY": "k"}) == ("api_key", None)
    assert configure_claude_auth({}) == ("login", None)

    creds = json.dumps({"type": "service_account", "project_id": "p"})
    env = {
        "ANTHROPIC_USE_VERTEX_AI": "true",
        "ANTHROPIC_GEAP_PROJECT_ID": "proj",
        "ANTHROPIC_GEAP_REGION": "global",
        "GOOGLE_APPLICATION_CREDENTIALS_JSON": creds,
    }
    mode, creds_file = configure_claude_auth(env)
    try:
        assert mode == "vertex"
        assert env["CLAUDE_CODE_USE_VERTEX"] == "1"
        assert env["ANTHROPIC_VERTEX_PROJECT_ID"] == "proj" and env["CLOUD_ML_REGION"] == "global"
        assert creds_file is not None and json.loads(creds_file.read_text()) == json.loads(creds)
        assert env["GOOGLE_APPLICATION_CREDENTIALS"] == str(creds_file)
        assert oct(creds_file.stat().st_mode & 0o777) == "0o600"
    finally:
        creds_file.unlink(missing_ok=True)

    # An explicit login request ignores both API keys and Vertex config.
    assert configure_claude_auth({"VSBENCH_CLAUDE_AUTH": "login", "ANTHROPIC_API_KEY": "k", **env})[0] == "login"


def test_mcp_bridge_config_scopes_task_tools(tmp_path: Path):
    task = {"id": "OF-02", "tool_chain": ["valstorm_schema_inspect", "valstorm_record_cud"]}
    cfg = BenchmarkConfig(config_id="d", label="d", harness="claude", tools_mode="scoped", mcp_bridge=True)
    server = build_mcp_bridge_config(task, cfg, tmp_path)["mcpServers"]["vsagent"]
    assert server["command"] == sys.executable
    assert server["args"][0] == str(BRIDGE)
    assert server["args"][server["args"].index("--workspace") + 1] == str(tmp_path)
    assert server["args"][server["args"].index("--tools") + 1] == "valstorm_schema_inspect,valstorm_record_cud"


def test_parse_claude_stream_telemetry():
    parsed = parse_claude_stream(_stream_fixture())
    assert parsed["got_result"] and not parsed["is_error"]
    assert parsed["raw_tool_calls"] == ["mcp__vsagent__valstorm_sql_query", "Write", "Bash"]
    assert parsed["tool_calls"] == ["valstorm_sql_query", "write_file", "terminal_exec"]
    assert parsed["tool_outputs"] == ['[{"id": "con_master_1"}]', "File written"]
    assert parsed["num_turns"] == 4 and parsed["final_output"] == "Done."
    assert parsed["cost_usd"] == pytest.approx(0.0421)
    # modelUsage (all models incl. subagents) wins over the main-thread `usage` block.
    assert parsed["prompt_tokens"] == 120 + 30000 + 9000 + 50
    assert parsed["completion_tokens"] == 840
    assert parsed["cached_tokens"] == 30000
    assert parsed["model"] == "claude-opus-5-5"
    assert parsed["integrity_violations"] == ["grader_tampering:Bash->valstorm_sandbox_db.json"]


def test_grader_tampering_detection_and_scoring(tmp_path: Path):
    assert detect_grader_tampering("valstorm_record_cud", {"note": "valstorm_sandbox_db.json"}) is None
    assert detect_grader_tampering("mcp__vsagent__valstorm_record_merge", "valstorm_sandbox_db.json") is None
    assert detect_grader_tampering("write_file", {"path": "valstorm_sandbox_db.json"}).startswith("grader_tampering:")

    (tmp_path / "out.txt").write_text("hello", encoding="utf-8")
    task = {"id": "T", "success_criteria": {"file": "out.txt", "content_regex": "hello"}}
    cfg = BenchmarkConfig(config_id="c", label="c", harness="claude")
    clean = ExecutionTrace(task_id="T", config_id="c", harness="claude", status="completed")
    dirty = ExecutionTrace(task_id="T", config_id="c", harness="claude", status="completed",
                           integrity_violations=["grader_tampering:Write->valstorm_sandbox_db.json"])
    scorer = EnvironmentStateScorer()
    assert scorer.score(task, cfg, clean, tmp_path).passed is True
    dirty_score = scorer.score(task, cfg, dirty, tmp_path)
    assert dirty_score.passed is False and "grader_tampering" in dirty_score.failure_reason


def test_head_to_head_preset_arms():
    arms = {c.config_id: c for c in get_ablation_configs("head_to_head", claude_model="claude-opus-5-5")}
    assert list(arms) == ["h2h_a_vsagent_gemini", "h2h_b_vsagent_claude", "h2h_c_claude_code", "h2h_d_claude_code_mcp"]
    assert arms["h2h_a_vsagent_gemini"].auto_tune is True
    assert arms["h2h_b_vsagent_claude"].provider == "anthropic" and arms["h2h_b_vsagent_claude"].isolate_provider
    assert arms["h2h_c_claude_code"].harness == "claude" and not arms["h2h_c_claude_code"].mcp_bridge
    assert arms["h2h_d_claude_code_mcp"].mcp_bridge is True
    assert {c.model for c in arms.values() if c.harness == "claude"} == {"claude-opus-5-5"}


def _rpc(workspace: Path, messages: list, extra_args: list = ()) -> dict:
    payload = "".join(json.dumps(m) + "\n" for m in messages)
    proc = subprocess.run(
        [sys.executable, str(BRIDGE), "--workspace", str(workspace), *extra_args],
        input=payload, capture_output=True, text=True, timeout=60, cwd=str(RUNTIME_ROOT),
    )
    return {r["id"]: r for r in (json.loads(l) for l in proc.stdout.splitlines() if l.strip())}


def test_mcp_bridge_serves_vsagent_tools(tmp_path: Path):
    trace_file = tmp_path / "trace.jsonl"
    responses = _rpc(tmp_path, [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "valstorm_sql_query", "arguments": {"query": "SELECT id FROM contact"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "nope"},
    ], ["--trace-file", str(trace_file)])

    assert responses[1]["result"]["serverInfo"]["name"] == "vsagent"
    names = {t["name"] for t in responses[2]["result"]["tools"]}
    assert {"valstorm_sql_query", "valstorm_vfs_discover_knowledge", "valstorm_record_merge"} <= names
    # Built-in duplicates and host-state mutators are never exposed.
    assert not names & {"write_file", "terminal_exec", "memory_manage", "skill_manage", "delegate_task"}
    for t in responses[2]["result"]["tools"]:
        assert t["inputSchema"]["type"] == "object"

    call = responses[3]["result"]
    assert call["isError"] is False and "con_master_1" in call["content"][0]["text"]
    assert responses[4]["error"]["code"] == -32601
    assert (tmp_path / "valstorm_sandbox_db.json").is_file()
    assert json.loads(trace_file.read_text().splitlines()[0])["tool"] == "valstorm_sql_query"


def test_mcp_bridge_scoped_whitelist(tmp_path: Path):
    responses = _rpc(tmp_path, [{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}],
                     ["--tools-mode", "scoped", "--tools", "valstorm_schema_inspect,valstorm_record_cud,write_file"])
    assert {t["name"] for t in responses[1]["result"]["tools"]} == {"valstorm_schema_inspect", "valstorm_record_cud"}


@pytest.mark.asyncio
async def test_external_adapter_end_to_end_with_fake_claude(tmp_path: Path, monkeypatch):
    """Runs the full adapter + scorer pipeline against a stand-in `claude` binary emitting stream-json."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "claude"
    stream = "".join(_stream_fixture())
    fake.write_text(
        "#!" + sys.executable + "\n"
        "import pathlib, sys\n"
        "pathlib.Path('calculation_result.txt').write_text('18.8495')\n"
        f"sys.stdout.write({stream!r})\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("CLAUDE_CODE_SIMPLE", "1")

    task = load_tasks(suite="core", task_id="G-S01")[0]
    cfg = BenchmarkConfig(config_id="c", label="c", harness="claude", model="claude-opus-5-5", mcp_bridge=True, timeout_sec=30)
    ws = tmp_path / "ws"
    ws.mkdir()
    trace = await ExternalCliBenchmarkAdapter().execute_task(task, cfg, ws, mode="live")

    assert trace.status == "completed", trace.error
    assert trace.turns_taken == 4 and trace.cached_tokens == 30000
    assert trace.estimated_cost_usd == pytest.approx(0.0421) and trace.model_used == "claude-opus-5-5"
    assert trace.tool_calls_made == ["valstorm_sql_query", "write_file", "terminal_exec"]

    score = EnvironmentStateScorer().score(task, cfg, trace, ws)
    # The fixture's Bash call reads the grader DB, so the run is failed despite a correct file.
    assert "content_regex_match:calculation_result.txt" in score.checks_passed
    assert score.passed is False


@pytest.mark.asyncio
async def test_external_adapter_timeout_keeps_partial_trace(tmp_path: Path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "claude"
    init = json.dumps({"type": "system", "subtype": "init", "model": "claude-opus-5-5", "mcp_servers": []})
    fake.write_text(
        "#!" + sys.executable + "\n"
        "import sys, time\n"
        f"sys.stdout.write({init!r} + '\\n'); sys.stdout.flush()\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    task = {"id": "T", "goal": "wait", "success_criteria": {}}
    cfg = BenchmarkConfig(config_id="c", label="c", harness="claude", timeout_sec=1.5)
    trace = await ExternalCliBenchmarkAdapter().execute_task(task, cfg, tmp_path, mode="live")
    assert trace.status == "timed_out"
    assert trace.model_used == "claude-opus-5-5"
    assert trace.duration_sec < 10
