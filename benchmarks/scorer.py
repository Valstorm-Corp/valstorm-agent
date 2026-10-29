"""Deterministic Environment-State and Trajectory Scorer for Agent Benchmarks.

Implements the UK AISI Inspect AI / Terminal-Bench scoring pattern:
1. Environment-State Verification (files created, regex content match, executable bits, shell verification commands)
2. Trajectory Efficiency Analysis (expected tool chain vs. actual tool calls, overhead ratio)
3. Multi-Trial Reliability Aggregation (Pass@k and tau-bench pass^k consistency metrics)
"""

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, Optional

from .adapters.base import BenchmarkConfig, ExecutionTrace


@dataclass
class ScoreResult:
    """Complete evaluation scorecard for a single (Task, Config, Trial) run."""

    task_id: str
    complexity: str
    category: str
    config_id: str
    config_label: str
    harness: str
    passed: bool
    score: float  # 0.0 to 1.0
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    expected_tools: List[str] = field(default_factory=list)
    actual_tools: List[str] = field(default_factory=list)
    tool_overhead_ratio: float = 1.0
    turns_taken: int = 0
    duration_sec: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    subagents_spawned: int = 0
    parallel_speedup: float = 1.0
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EnvironmentStateScorer:
    """Grades post-execution workspace state and agent trajectory against task success_criteria."""

    def score(
        self,
        task: Dict[str, Any],
        config: BenchmarkConfig,
        trace: ExecutionTrace,
        workspace_dir: Path,
    ) -> ScoreResult:
        criteria = task.get("success_criteria") or {}
        expected_tools = task.get("tool_chain") or []
        actual_tools = list(trace.tool_calls_made)

        checks_passed: List[str] = []
        checks_failed: List[str] = []

        # If adapter skipped (e.g. external CLI binary not installed on host)
        if trace.status == "skipped":
            return ScoreResult(
                task_id=task["id"],
                complexity=task.get("complexity", "Unknown"),
                category=task.get("category", "general"),
                config_id=config.config_id,
                config_label=config.label,
                harness=config.harness,
                passed=False,
                score=0.0,
                expected_tools=expected_tools,
                actual_tools=actual_tools,
                turns_taken=trace.turns_taken,
                duration_sec=trace.duration_sec,
                failure_reason=trace.error or "Skipped",
            )

        # 1. Check `files_exist` list if specified
        files_exist = criteria.get("files_exist") or []
        for rel_path in files_exist:
            fpath = workspace_dir / rel_path
            if fpath.exists():
                checks_passed.append(f"file_exists:{rel_path}")
            else:
                checks_failed.append(f"missing_file:{rel_path}")

        # 2. Check primary `file` + `content_regex` + `require_executable`
        target_file = criteria.get("file")
        if target_file:
            fpath = workspace_dir / target_file
            if not fpath.is_file():
                checks_failed.append(f"missing_target_file:{target_file}")
            else:
                checks_passed.append(f"target_file_exists:{target_file}")
                content = fpath.read_text(encoding="utf-8", errors="replace").strip()
                content_regex = criteria.get("content_regex")
                if content_regex:
                    if re.search(content_regex, content, re.MULTILINE | re.DOTALL):
                        checks_passed.append(f"content_regex_match:{target_file}")
                    else:
                        checks_failed.append(
                            f"content_regex_mismatch:{target_file} (expected /{content_regex}/, got '{content[:80]}')"
                        )

                if criteria.get("require_executable"):
                    if os.access(fpath, os.X_OK):
                        checks_passed.append(f"executable_permission:{target_file}")
                    else:
                        checks_failed.append(f"missing_executable_chmod:{target_file}")

        # 3. Check `verify_command` + optional `verify_stdout_regex`
        verify_cmd = criteria.get("verify_command")
        if verify_cmd:
            try:
                proc = subprocess.run(
                    verify_cmd,
                    shell=True,
                    cwd=str(workspace_dir),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                stdout_str = (proc.stdout or "").strip()
                if proc.returncode == 0:
                    checks_passed.append("verify_command_exit_0")
                    stdout_regex = criteria.get("verify_stdout_regex")
                    if stdout_regex:
                        if re.search(stdout_regex, stdout_str, re.MULTILINE):
                            checks_passed.append("verify_stdout_regex_match")
                        else:
                            checks_failed.append(
                                f"verify_stdout_mismatch (expected /{stdout_regex}/, got '{stdout_str[:80]}')"
                            )
                else:
                    stderr_str = (proc.stderr or "").strip()
                    checks_failed.append(
                        f"verify_command_failed(exit={proc.returncode}): {stderr_str[:120] or stdout_str[:120]}"
                    )
            except Exception as e:
                checks_failed.append(f"verify_command_exception:{e}")

        # 4. Check `output_regex` against final output or tool outputs
        output_regex = criteria.get("output_regex")
        if output_regex:
            combined_output = "\n".join([trace.final_output] + trace.tool_outputs)
            if re.search(output_regex, combined_output, re.MULTILINE | re.DOTALL):
                checks_passed.append("output_regex_match")
            else:
                checks_failed.append(f"output_regex_mismatch (expected /{output_regex}/)")

        total_checks = len(checks_passed) + len(checks_failed)
        score_val = round(len(checks_passed) / total_checks, 3) if total_checks > 0 else 0.0
        passed = (len(checks_failed) == 0 and total_checks > 0 and trace.status in ("completed", "timed_out"))

        # Trajectory overhead ratio: actual calls / expected calls
        expected_len = max(1, len(expected_tools))
        overhead_ratio = round(len(actual_tools) / expected_len, 2) if actual_tools else 0.0

        failure_reason = "; ".join(checks_failed) if checks_failed else trace.error

        return ScoreResult(
            task_id=task["id"],
            complexity=task.get("complexity", "Unknown"),
            category=task.get("category", "general"),
            config_id=config.config_id,
            config_label=config.label,
            harness=config.harness,
            passed=passed,
            score=score_val,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            expected_tools=expected_tools,
            actual_tools=actual_tools,
            tool_overhead_ratio=overhead_ratio,
            turns_taken=trace.turns_taken,
            duration_sec=round(trace.duration_sec, 3),
            prompt_tokens=trace.prompt_tokens,
            completion_tokens=trace.completion_tokens,
            total_tokens=trace.total_tokens,
            estimated_cost_usd=trace.estimated_cost_usd,
            subagents_spawned=trace.subagents_spawned,
            parallel_speedup=round(trace.parallel_speedup, 2),
            failure_reason=failure_reason,
        )


def compute_multi_trial_metrics(trial_results: List[ScoreResult]) -> Dict[str, Any]:
    """Computes Pass@k (capability) and pass^k (tau-bench reliability) over k trials."""
    k = len(trial_results)
    if k == 0:
        return {"trials": 0, "pass_rate": 0.0, "pass_at_k": 0.0, "pass_hat_k": 0.0}

    successes = sum(1 for r in trial_results if r.passed)
    pass_rate = successes / k
    # Pass@k: probability of at least 1 success in k trials
    pass_at_k = 1.0 if successes > 0 else 0.0
    # pass^k (tau-bench): probability that all k trials succeed consistently (p^k)
    pass_hat_k = round(pass_rate ** k, 4)

    avg_duration = sum(r.duration_sec for r in trial_results) / k
    avg_tokens = sum(r.total_tokens for r in trial_results) / k
    avg_cost = sum(r.estimated_cost_usd for r in trial_results) / k

    return {
        "trials": k,
        "successes": successes,
        "pass_rate": round(pass_rate, 4),
        "pass_at_k": pass_at_k,
        "pass_hat_k": pass_hat_k,
        "avg_duration_sec": round(avg_duration, 3),
        "avg_total_tokens": round(avg_tokens, 1),
        "avg_cost_usd": round(avg_cost, 6),
    }
