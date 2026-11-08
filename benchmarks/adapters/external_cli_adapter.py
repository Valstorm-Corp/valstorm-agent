"""External CLI Harness Adapter (Terminal-Bench / Harbor Pattern).

Enables head-to-head benchmarking of external agent CLIs against `vsagent` using
the exact same isolated workspace and `EnvironmentStateScorer`:
  - `claude`: Anthropic Claude Code (`claude -p "<goal>" --output-format stream-json ...`)
  - `codex`: OpenAI Codex CLI (`codex exec --full-auto "<goal>"`)
  - `opencode`: OpenCode CLI (`opencode run "<goal>"`)
  - `hermes`: Hermes Agent (`hermes run "<goal>"`)
  - `oh-my-pi`: Oh My Pi Agent (`omp run "<goal>"`)
  - `vsagent`: Valstorm Agent CLI (`vsagent chat -p "<goal>"`)

Claude Code runs are fully instrumented: the `stream-json` event stream yields every tool call,
tool result, turn count, token usage (incl. prompt-cache reads) and Claude Code's own cost figure.
Runs are isolated from the host's `~/.claude` config (user settings, hooks, plugins, skills, MCP
servers) and from any parent Claude Code session's environment.
"""

import asyncio
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import time
from typing import Any, Dict, List, Literal, Optional

from ..sandbox_valstorm import provision_craft_suite_sandbox
from .base import BenchmarkAdapter, BenchmarkConfig, ExecutionTrace, detect_grader_tampering
from .vsagent_adapter import VsAgentBenchmarkAdapter


HARNESS_BINARY_MAP: Dict[str, str] = {
    "claude": "claude",
    "codex": "codex",
    "opencode": "opencode",
    "hermes": "hermes",
    "oh-my-pi": "omp",
    "vsagent": "vsagent",
}

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent
MCP_BRIDGE_SCRIPT = BENCHMARKS_DIR / "mcp_bridge.py"
MCP_BRIDGE_SERVER_NAME = "vsagent"

# Claude Code built-in tool -> equivalent vsagent tool, so trajectory overhead ratios compare like for like.
CLAUDE_TOOL_ALIASES: Dict[str, str] = {
    "Write": "write_file",
    "Edit": "patch_file",
    "MultiEdit": "patch_file",
    "NotebookEdit": "patch_file",
    "Read": "read_file",
    "Bash": "terminal_exec",
    "Glob": "search_files",
    "Grep": "search_files",
    "WebSearch": "web_search",
    "WebFetch": "web_scrape",
    "Task": "delegate_task",
    "Agent": "delegate_task",
}

# Interactive / scheduling tools that have no meaning in a one-shot headless benchmark run.
CLAUDE_DISALLOWED_TOOLS: List[str] = [
    "AskUserQuestion",
    "CronCreate",
    "CronDelete",
    "CronList",
    "ScheduleWakeup",
    "PushNotification",
    "RemoteTrigger",
    "EnterWorktree",
    "ExitWorktree",
]

_CLAUDE_MODEL_ALIASES = ("opus", "sonnet", "haiku", "fable")

# Env vars a parent Claude Code session exports that would hijack a nested `claude` process
# (e.g. CLAUDE_CODE_SIMPLE forces bare mode, ANTHROPIC_BASE_URL points at the host's proxy).
_CLAUDE_ENV_PASSTHROUGH = frozenset({"CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CONFIG_DIR"})


def _is_claude_model(model: str) -> bool:
    m = (model or "").strip().lower()
    return "claude" in m or m in _CLAUDE_MODEL_ALIASES


def _claude_effort(thinking_level: str) -> Optional[str]:
    """Maps the vsagent thinking ladder onto Claude Code's `--effort` levels."""
    return {"none": "low", "low": "low", "medium": "medium", "high": "high"}.get(thinking_level)


def build_claude_env(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Returns a clean environment for a nested `claude` process."""
    env = dict(os.environ if base_env is None else base_env)
    for key in list(env):
        if key == "CLAUDECODE" or key == "ANTHROPIC_BASE_URL":
            env.pop(key)
        elif key.startswith("CLAUDE") and key not in _CLAUDE_ENV_PASSTHROUGH:
            env.pop(key)
    return env


def configure_claude_auth(env: Dict[str, str]) -> tuple[str, Optional[Path]]:
    """Picks how the nested `claude` authenticates and mutates `env` to match.

    `VSBENCH_CLAUDE_AUTH` = auto (default) | api_key | vertex | login. In auto mode an
    ANTHROPIC_API_KEY wins, then the same Vertex AI (GEAP) project vsagent's anthropic provider
    uses (ANTHROPIC_USE_VERTEX_AI / ANTHROPIC_GEAP_PROJECT_ID / ANTHROPIC_GEAP_REGION), then the
    host's `claude /login` session. Returns (mode, temp credentials file to delete afterwards).
    """
    requested = env.get("VSBENCH_CLAUDE_AUTH", "auto").strip().lower()
    if requested in ("auto", "api_key") and env.get("ANTHROPIC_API_KEY"):
        return "api_key", None

    vertex_on = env.get("ANTHROPIC_USE_VERTEX_AI", "").lower() in ("true", "1", "yes")
    project, region = env.get("ANTHROPIC_GEAP_PROJECT_ID"), env.get("ANTHROPIC_GEAP_REGION")
    if requested == "vertex" or (requested == "auto" and vertex_on and project and region):
        env["CLAUDE_CODE_USE_VERTEX"] = "1"
        if project:
            env["ANTHROPIC_VERTEX_PROJECT_ID"] = project
        if region:
            env["CLOUD_ML_REGION"] = region
        creds_file: Optional[Path] = None
        inline_creds = env.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if not env.get("GOOGLE_APPLICATION_CREDENTIALS") and inline_creds:
            try:
                json.loads(inline_creds)
                fd, tmp = tempfile.mkstemp(prefix="vsbench_gcp_", suffix=".json")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(inline_creds)
                creds_file = Path(tmp)
                env["GOOGLE_APPLICATION_CREDENTIALS"] = tmp
            except json.JSONDecodeError:
                pass
        return "vertex", creds_file

    return "login", None


def build_mcp_bridge_config(task: Dict[str, Any], config: BenchmarkConfig, workspace_dir: Path) -> Dict[str, Any]:
    """Builds the `--mcp-config` payload that serves the vsagent ToolRegistry to Claude Code."""
    args = [str(MCP_BRIDGE_SCRIPT), "--workspace", str(workspace_dir), "--tools-mode", config.tools_mode]
    if config.tools_mode == "scoped" and task.get("tool_chain"):
        args.extend(["--tools", ",".join(task["tool_chain"])])
    return {
        "mcpServers": {
            MCP_BRIDGE_SERVER_NAME: {
                "type": "stdio",
                "command": sys.executable,
                "args": args,
            }
        }
    }


def build_cli_command(
    harness: str,
    goal: str,
    model: str = "",
    config: Optional[BenchmarkConfig] = None,
    mcp_config: Optional[Dict[str, Any]] = None,
    bare: bool = False,
) -> List[str]:
    """Constructs the non-interactive headless CLI invocation for a given agent harness."""
    h = harness.strip().lower()
    if h == "claude":
        cmd = [
            "claude", "-p", goal,
            "--output-format", "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
            "--no-session-persistence",
            "--strict-mcp-config",
        ]
        if bare:
            cmd.append("--bare")
        else:
            # Exclude ~/.claude user settings (hooks, plugins, permissions) and all skills.
            cmd.extend(["--setting-sources", "project,local", "--disable-slash-commands"])
        if model and _is_claude_model(model):
            cmd.extend(["--model", model])
        if config is not None:
            effort = _claude_effort(config.thinking_level)
            if effort:
                cmd.extend(["--effort", effort])
            if config.tools_mode == "none":
                cmd.extend(["--tools", ""])
        cmd.extend(["--disallowedTools", ",".join(CLAUDE_DISALLOWED_TOOLS)])
        if mcp_config:
            cmd.extend(["--mcp-config", json.dumps(mcp_config)])
        return cmd
    if h == "codex":
        cmd = ["codex", "exec", "--full-auto", goal]
        if model and ("gpt" in model or "o" in model):
            cmd.extend(["-m", model])
        return cmd
    if h == "opencode":
        cmd = ["opencode", "run", goal]
        if model:
            cmd.extend(["--model", model])
        return cmd
    if h == "hermes":
        return ["hermes", "run", goal]
    if h == "oh-my-pi":
        return ["omp", "run", goal]
    # Default vsagent headless CLI
    cmd = ["vsagent", "chat", "-p", goal]
    if model:
        cmd.extend(["--model", model])
    return cmd


def _normalize_claude_tool(name: str) -> str:
    prefix = f"mcp__{MCP_BRIDGE_SERVER_NAME}__"
    if name.startswith(prefix):
        return name[len(prefix):]
    return CLAUDE_TOOL_ALIASES.get(name, name)


def _tool_result_text(block: Dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(c.get("text", "")) for c in content if isinstance(c, dict) and c.get("type") == "text")
    return ""


def parse_claude_stream(lines: List[str]) -> Dict[str, Any]:
    """Parses Claude Code `--output-format stream-json` lines into benchmark telemetry."""
    parsed: Dict[str, Any] = {
        "model": "",
        "tool_calls": [],
        "raw_tool_calls": [],
        "tool_outputs": [],
        "final_output": "",
        "num_turns": 0,
        "cost_usd": 0.0,
        "duration_api_ms": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "is_error": False,
        "got_result": False,
        "integrity_violations": [],
        "mcp_servers": [],
    }
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = event.get("type")

        if etype == "system" and event.get("subtype") == "init":
            parsed["model"] = event.get("model") or parsed["model"]
            parsed["mcp_servers"] = event.get("mcp_servers") or []

        elif etype == "assistant":
            for block in (event.get("message") or {}).get("content") or []:
                if block.get("type") == "tool_use":
                    raw_name = str(block.get("name", ""))
                    parsed["raw_tool_calls"].append(raw_name)
                    parsed["tool_calls"].append(_normalize_claude_tool(raw_name))
                    violation = detect_grader_tampering(raw_name, block.get("input"))
                    if violation:
                        parsed["integrity_violations"].append(violation)

        elif etype == "user":
            for block in (event.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    text = _tool_result_text(block)
                    if text:
                        parsed["tool_outputs"].append(text)

        elif etype == "result":
            parsed["got_result"] = True
            parsed["is_error"] = bool(event.get("is_error"))
            parsed["final_output"] = str(event.get("result") or "")
            parsed["num_turns"] = int(event.get("num_turns") or 0)
            parsed["cost_usd"] = float(event.get("total_cost_usd") or 0.0)
            parsed["duration_api_ms"] = int(event.get("duration_api_ms") or 0)

            # modelUsage aggregates every model the run touched (incl. subagents); fall back to `usage`.
            model_usage = event.get("modelUsage") or {}
            if model_usage:
                prompt = completion = cached = 0
                top_model, top_out = "", -1
                for m_name, mu in model_usage.items():
                    cache_read = int(mu.get("cacheReadInputTokens") or 0)
                    prompt += int(mu.get("inputTokens") or 0) + cache_read + int(mu.get("cacheCreationInputTokens") or 0)
                    out = int(mu.get("outputTokens") or 0)
                    completion += out
                    cached += cache_read
                    if out > top_out:
                        top_model, top_out = m_name, out
                parsed.update(prompt_tokens=prompt, completion_tokens=completion, cached_tokens=cached)
                parsed["model"] = top_model or parsed["model"]
            else:
                usage = event.get("usage") or {}
                cache_read = int(usage.get("cache_read_input_tokens") or 0)
                parsed["prompt_tokens"] = (
                    int(usage.get("input_tokens") or 0) + cache_read + int(usage.get("cache_creation_input_tokens") or 0)
                )
                parsed["completion_tokens"] = int(usage.get("output_tokens") or 0)
                parsed["cached_tokens"] = cache_read
    return parsed


def is_harness_installed(harness: str) -> bool:
    """Checks whether the target external CLI binary is available on system PATH."""
    binary = HARNESS_BINARY_MAP.get(harness.lower(), harness.lower())
    return shutil.which(binary) is not None


async def _run_subprocess(
    cmd: List[str], cwd: Path, env: Optional[Dict[str, str]], timeout_sec: float
) -> tuple[List[str], str, Optional[int], bool]:
    """Runs `cmd`, streaming stdout line-by-line so a timed-out run still yields a partial trace."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        env=env,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=64 * 1024 * 1024,
        start_new_session=True,
    )
    stdout_lines: List[str] = []
    stderr_chunks: List[bytes] = []

    async def _drain_stdout():
        assert proc.stdout is not None
        async for raw in proc.stdout:
            stdout_lines.append(raw.decode("utf-8", errors="replace"))

    async def _drain_stderr():
        assert proc.stderr is not None
        stderr_chunks.append(await proc.stderr.read())

    timed_out = False
    try:
        await asyncio.wait_for(asyncio.gather(_drain_stdout(), _drain_stderr(), proc.wait()), timeout=timeout_sec)
    except asyncio.TimeoutError:
        timed_out = True
        try:
            # Kill the whole process group so MCP servers / subagent shells die with the CLI.
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await proc.wait()
    stderr_str = b"".join(stderr_chunks).decode("utf-8", errors="replace").strip()
    return stdout_lines, stderr_str, proc.returncode, timed_out


class ExternalCliBenchmarkAdapter(BenchmarkAdapter):
    """Runs external CLI coding agents in an isolated temporary directory and captures telemetry."""

    async def execute_task(
        self,
        task: Dict[str, Any],
        config: BenchmarkConfig,
        workspace_dir: Path,
        mode: Literal["auto", "mock", "live"] = "auto",
    ) -> ExecutionTrace:
        if mode == "mock":
            # In deterministic mock mode, delegate execution through VsAgentBenchmarkAdapter
            # to verify the end-to-end scoring & reporting pipeline across harness labels.
            delegate = VsAgentBenchmarkAdapter()
            trace = await delegate.execute_task(task=task, config=config, workspace_dir=workspace_dir, mode="mock")
            trace.harness = config.harness
            return trace

        binary = HARNESS_BINARY_MAP.get(config.harness.lower(), config.harness.lower())
        if config.harness.lower() == "claude":
            # VSBENCH_CLAUDE_BIN pins a specific Claude Code build (e.g. the desktop app's bundled one).
            binary = os.environ.get("VSBENCH_CLAUDE_BIN") or binary
        if not shutil.which(binary):
            return ExecutionTrace(
                task_id=task["id"],
                config_id=config.config_id,
                harness=config.harness,
                status="skipped",
                error=f"CLI binary '{binary}' not found on PATH",
            )

        # Same workspace fixtures the in-process vsagent adapter provisions.
        if str(task.get("id", "")).startswith("CR-") or str(task.get("category", "")).startswith("creative_"):
            provision_craft_suite_sandbox(workspace_dir)
        if task.get("ui_state"):
            (workspace_dir / "ui_state.json").write_text(json.dumps(task["ui_state"], indent=2), encoding="utf-8")

        if config.harness.lower() == "claude":
            return await self._execute_claude(task, config, workspace_dir, binary)

        cmd = build_cli_command(config.harness, task["goal"], config.model)
        t0 = time.perf_counter()
        try:
            stdout_lines, stderr_str, returncode, timed_out = await _run_subprocess(
                cmd, workspace_dir, None, config.timeout_sec
            )
        except Exception as e:
            return ExecutionTrace(
                task_id=task["id"],
                config_id=config.config_id,
                harness=config.harness,
                status="failed",
                duration_sec=round(time.perf_counter() - t0, 3),
                error=str(e),
            )

        duration_sec = round(time.perf_counter() - t0, 3)
        if timed_out:
            return ExecutionTrace(
                task_id=task["id"],
                config_id=config.config_id,
                harness=config.harness,
                status="timed_out",
                duration_sec=duration_sec,
                error=f"External CLI timed out after {config.timeout_sec}s",
            )
        return ExecutionTrace(
            task_id=task["id"],
            config_id=config.config_id,
            harness=config.harness,
            status="completed" if returncode == 0 else "failed",
            final_output="".join(stdout_lines).strip(),
            turns_taken=1,
            duration_sec=duration_sec,
            llm_time_sec=duration_sec,
            error=stderr_str if returncode != 0 else None,
        )

    async def _execute_claude(
        self, task: Dict[str, Any], config: BenchmarkConfig, workspace_dir: Path, binary: str = "claude"
    ) -> ExecutionTrace:
        env = build_claude_env()
        auth_mode, creds_file = configure_claude_auth(env)
        # `--bare` gives the strictest isolation (and Claude Code's minimal system prompt) but never reads
        # the OAuth login. VSBENCH_CLAUDE_BARE=0 benchmarks the default prompt with setting-source scoping.
        bare = auth_mode in ("api_key", "vertex") and env.get("VSBENCH_CLAUDE_BARE", "1") != "0"
        mcp_config = build_mcp_bridge_config(task, config, workspace_dir) if config.mcp_bridge else None
        cmd = build_cli_command("claude", task["goal"], config.model, config=config, mcp_config=mcp_config, bare=bare)
        cmd[0] = binary

        t0 = time.perf_counter()
        try:
            stdout_lines, stderr_str, returncode, timed_out = await _run_subprocess(
                cmd, workspace_dir, env, config.timeout_sec
            )
        except Exception as e:
            return ExecutionTrace(
                task_id=task["id"],
                config_id=config.config_id,
                harness=config.harness,
                status="failed",
                duration_sec=round(time.perf_counter() - t0, 3),
                error=str(e),
            )
        finally:
            if creds_file is not None:
                creds_file.unlink(missing_ok=True)
        duration_sec = round(time.perf_counter() - t0, 3)
        parsed = parse_claude_stream(stdout_lines)

        error: Optional[str] = None
        if timed_out:
            status: Literal["completed", "failed", "timed_out", "skipped"] = "timed_out"
            error = f"Claude Code timed out after {config.timeout_sec}s"
        elif parsed["is_error"] or returncode != 0 or not parsed["got_result"]:
            status = "failed"
            error = parsed["final_output"] or stderr_str or f"claude exited with code {returncode}"
        else:
            status = "completed"

        if config.mcp_bridge and not timed_out:
            bridge = next((s for s in parsed["mcp_servers"] if s.get("name") == MCP_BRIDGE_SERVER_NAME), None)
            if not bridge or bridge.get("status") != "connected":
                status = "failed"
                error = f"vsagent MCP bridge not connected: {bridge}"

        llm_time_sec = round(parsed["duration_api_ms"] / 1000.0, 3)
        return ExecutionTrace(
            task_id=task["id"],
            config_id=config.config_id,
            harness=config.harness,
            status=status,
            final_output=parsed["final_output"],
            tool_calls_made=parsed["tool_calls"],
            tool_outputs=parsed["tool_outputs"],
            turns_taken=parsed["num_turns"],
            duration_sec=duration_sec,
            llm_time_sec=llm_time_sec,
            tool_time_sec=round(max(0.0, duration_sec - llm_time_sec), 3),
            prompt_tokens=parsed["prompt_tokens"],
            completion_tokens=parsed["completion_tokens"],
            total_tokens=parsed["prompt_tokens"] + parsed["completion_tokens"],
            cached_tokens=parsed["cached_tokens"],
            estimated_cost_usd=round(parsed["cost_usd"], 6),
            model_used=parsed["model"],
            subagents_spawned=parsed["tool_calls"].count("delegate_task"),
            integrity_violations=parsed["integrity_violations"],
            error=error,
        )
