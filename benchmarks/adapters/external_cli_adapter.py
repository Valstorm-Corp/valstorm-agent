"""External CLI Harness Adapter (Terminal-Bench / Harbor Pattern).

Enables head-to-head benchmarking of external agent CLIs against `vsagent` using
the exact same isolated workspace and `EnvironmentStateScorer`:
  - `claude`: Anthropic Claude Code (`claude -p "<goal>" --dangerously-skip-permissions`)
  - `codex`: OpenAI Codex CLI (`codex exec --full-auto "<goal>"`)
  - `opencode`: OpenCode CLI (`opencode run "<goal>"`)
  - `hermes`: Hermes Agent (`hermes run "<goal>"`)
  - `oh-my-pi`: Oh My Pi Agent (`omp run "<goal>"`)
  - `vsagent`: Valstorm Agent CLI (`vsagent chat -p "<goal>"`)
"""

import asyncio
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Literal

from .base import BenchmarkAdapter, BenchmarkConfig, ExecutionTrace
from .vsagent_adapter import VsAgentBenchmarkAdapter


HARNESS_BINARY_MAP: Dict[str, str] = {
    "claude": "claude",
    "codex": "codex",
    "opencode": "opencode",
    "hermes": "hermes",
    "oh-my-pi": "omp",
    "vsagent": "vsagent",
}


def build_cli_command(harness: str, goal: str, model: str = "") -> List[str]:
    """Constructs the non-interactive headless CLI invocation for a given agent harness."""
    h = harness.strip().lower()
    if h == "claude":
        cmd = ["claude", "-p", goal, "--dangerously-skip-permissions"]
        if model and "claude" in model:
            cmd.extend(["--model", model])
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


def is_harness_installed(harness: str) -> bool:
    """Checks whether the target external CLI binary is available on system PATH."""
    binary = HARNESS_BINARY_MAP.get(harness.lower(), harness.lower())
    return shutil.which(binary) is not None


class ExternalCliBenchmarkAdapter(BenchmarkAdapter):
    """Runs external CLI coding agents in an isolated temporary directory and captures stdout/timing."""

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
        if not shutil.which(binary):
            return ExecutionTrace(
                task_id=task["id"],
                config_id=config.config_id,
                harness=config.harness,
                status="skipped",
                error=f"CLI binary '{binary}' not found on PATH",
            )

        cmd = build_cli_command(config.harness, task["goal"], config.model)
        t0 = time.perf_counter()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(workspace_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=config.timeout_sec)
            except asyncio.TimeoutError:
                proc.kill()
                return ExecutionTrace(
                    task_id=task["id"],
                    config_id=config.config_id,
                    harness=config.harness,
                    status="timed_out",
                    duration_sec=round(time.perf_counter() - t0, 3),
                    error=f"External CLI timed out after {config.timeout_sec}s",
                )

            duration_sec = round(time.perf_counter() - t0, 3)
            stdout_str = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr_str = stderr_bytes.decode("utf-8", errors="replace").strip()

            status: Literal["completed", "failed", "timed_out", "skipped"] = (
                "completed" if proc.returncode == 0 else "failed"
            )
            return ExecutionTrace(
                task_id=task["id"],
                config_id=config.config_id,
                harness=config.harness,
                status=status,
                final_output=stdout_str,
                turns_taken=1,
                duration_sec=duration_sec,
                llm_time_sec=duration_sec,
                error=stderr_str if proc.returncode != 0 else None,
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
