#!/usr/bin/env python3
"""Entrypoint wrapper for the Valstorm Agent Runtime Benchmark Harness.

Delegates to `apps/agent-runtime/benchmarks/runner.py` while preserving backwards
compatibility for `run_g_s01_benchmark()`.
"""

import asyncio
from pathlib import Path
import sys
from typing import Any, Dict

RUNTIME_ROOT = Path(__file__).resolve().parent.parent
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from benchmarks.runner import get_ablation_configs, load_tasks, main, run_benchmark_matrix


def run_g_s01_benchmark(mode: str = "mock") -> Dict[str, Any]:
    """Executes the G-S01 task through the structured Benchmark Harness."""
    tasks = load_tasks(suite="core", task_id="G-S01")
    configs = get_ablation_configs("baseline")
    report = asyncio.run(run_benchmark_matrix(tasks=tasks, configs=configs, mode=mode, trials=1))  # type: ignore[arg-type]
    run_info = report["runs"][0]
    return {
        "task_id": run_info["task_id"],
        "status": "PASS" if run_info["passed"] else "FAIL",
        "score": run_info["score"],
        "duration_sec": run_info["duration_sec"],
        "turns_taken": run_info["turns_taken"],
        "actual_tools": run_info["actual_tools"],
        "total_tokens": run_info["total_tokens"],
        "message": "Calculation and file write verified." if run_info["passed"] else run_info["failure_reason"],
    }


if __name__ == "__main__":
    main()
