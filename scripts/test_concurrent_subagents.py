#!/usr/bin/env python3
"""Benchmark and Verification Script for Concurrent Subagent Delegation in Valstorm Agent Runtime.

This script tests and mathematically proves that multiple subagents can be spawned
simultaneously using `delegate_task(background=True)` and executed concurrently
in parallel rather than sequentially.

Usage:
    # Run automatic test (uses mock or live provider based on environment):
    uv run --directory apps/agent-runtime python scripts/test_concurrent_subagents.py

    # Run deterministic mock test (simulates multi-agent workload with precise timing):
    uv run --directory apps/agent-runtime python scripts/test_concurrent_subagents.py --mode mock

    # Run live LLM test against active provider fallback chain:
    uv run --directory apps/agent-runtime python scripts/test_concurrent_subagents.py --mode live
"""

import argparse
import asyncio
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure apps/agent-runtime is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
RUNTIME_ROOT = SCRIPT_DIR.parent
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

# Load local environment if present
from dotenv import load_dotenv
load_dotenv(RUNTIME_ROOT / ".env", override=False)
load_dotenv(RUNTIME_ROOT.parent.parent / ".env", override=False)

from core.models import Message, UsageMetadata
from tools.delegation import (
    SubagentRegistry,
    delegate_task,
    set_subagent_provider_override,
    subagent_manage,
)


class MockConcurrentProvider:
    """Simulates realistic asynchronous processing latency for testing concurrency."""

    def __init__(self, simulated_delay_sec: float = 2.0):
        self.simulated_delay_sec = simulated_delay_sec
        self.call_count = 0

    async def generate(self, messages: List[Any], tools: Optional[List[Any]] = None, model: Optional[str] = None, **kwargs):
        self.call_count += 1
        call_idx = self.call_count
        # Simulate LLM inference time
        await asyncio.sleep(self.simulated_delay_sec)
        last_msg = messages[-1].content if messages else "Task"
        return (
            Message(
                role="assistant",
                content=f"[Mock Worker {call_idx} Output] Successfully processed: {last_msg[:60]}...",
            ),
            UsageMetadata(prompt_tokens=45, completion_tokens=25, total_tokens=70),
        )


def _format_box(title: str, lines: List[str]) -> str:
    width = max(len(line) for line in [title] + lines) + 4
    width = max(width, 70)
    border = "═" * width
    res = [f"╔{border}╗", f"║  {title.ljust(width - 2)}║", f"╠{border}╣"]
    for l in lines:
        res.append(f"║  {l.ljust(width - 2)}║")
    res.append(f"╚{border}╝")
    return "\n".join(res)


async def run_concurrency_test(mode: str = "auto", num_subagents: int = 3, mock_delay: float = 2.0) -> bool:
    print("\n" + "=" * 76)
    print("🚀 VALSTORM AGENT RUNTIME — CONCURRENT SUBAGENT BENCHMARK")
    print("=" * 76)

    # Determine execution mode
    use_mock = False
    if mode == "mock":
        use_mock = True
    elif mode == "live":
        use_mock = False
    else:  # auto
        from tools.delegation import _get_subagent_provider
        prov = _get_subagent_provider("gemini")
        use_mock = prov is None

    if use_mock:
        print(f"🔧 Execution Mode: DETERMINISTIC MOCK (simulated latency = {mock_delay:.1f}s per worker)")
        mock_provider = MockConcurrentProvider(simulated_delay_sec=mock_delay)
        set_subagent_provider_override(mock_provider)
    else:
        print("⚡ Execution Mode: LIVE LLM (running against active provider fallback chain)")
        set_subagent_provider_override(None)

    registry = SubagentRegistry.get_instance()
    registry.clear()

    # Define test tasks
    test_specs = [
        {
            "profile": "developer",
            "goal": "Calculate prime numbers up to 50 and return as JSON list [2, 3, 5, ...].",
            "context": "Keep output brief and precise.",
        },
        {
            "profile": "researcher",
            "goal": "List the 3 main layers of the Valstorm monorepo: apps, packages, mobile-apps.",
            "context": "Include a single bullet summary for each.",
        },
        {
            "profile": "marketer",
            "goal": "Write a punchy one-sentence value proposition for Valstorm parallel agent swarms.",
            "context": "Tone: Enterprise, direct, impactful.",
        },
    ]

    selected_tasks = test_specs[:num_subagents]

    # -------------------------------------------------------------
    # PHASE 1: Asynchronous Non-Blocking Spawning
    # -------------------------------------------------------------
    print(f"\n[PHASE 1] Spawning {len(selected_tasks)} subagents concurrently with background=True...")
    t_spawn_start = time.time()
    spawn_results = []

    for idx, spec in enumerate(selected_tasks, 1):
        res = await delegate_task(
            profile=spec["profile"],
            goal=spec["goal"],
            context=spec.get("context"),
            background=True,
        )
        spawn_results.append(res)
        print(f"  ➜ Worker #{idx} spawned: profile='{spec['profile']}' | goal='{spec['goal'][:45]}...'")

    t_spawn_elapsed = time.time() - t_spawn_start
    print(f"  ✔ All {len(selected_tasks)} subagents launched non-blocking in {t_spawn_elapsed * 1000:.1f}ms")

    # -------------------------------------------------------------
    # PHASE 2: Live Concurrency State Inspection (Polling)
    # -------------------------------------------------------------
    print("\n[PHASE 2] Inspecting active subagent states in SubagentRegistry...")
    active_tasks = registry.list_tasks()
    running_count = sum(1 for t in active_tasks if t.status == "running")
    print(f"  Active tasks in registry: {len(active_tasks)}")
    print(f"  Currently running in parallel: {running_count} / {len(active_tasks)}")

    for t in active_tasks:
        print(f"    - [{t.status.upper()}] Task ID: {t.task_id} | Profile: {t.profile} | Elapsed: {t.duration_sec}s")

    # -------------------------------------------------------------
    # PHASE 3: Batch Parallel Await (task_id='all')
    # -------------------------------------------------------------
    print("\n[PHASE 3] Awaiting all running subagents via delegate_task(action='wait', task_id='all')...")
    t_wall_start = time.time()
    wait_summary = await delegate_task(
        action="wait",
        task_id="all",
        profile="developer",  # satisfies tool schema
        goal="await_all",
        timeout_sec=120.0,
    )
    t_wall_elapsed = time.time() - t_wall_start
    print(f"  ✔ All parallel subagent tasks finished. Batch await duration: {t_wall_elapsed:.2f}s")

    # -------------------------------------------------------------
    # PHASE 4: Metrics Synthesis & Concurrency Proof
    # -------------------------------------------------------------
    print("\n[PHASE 4] Analyzing Concurrency Metrics & Speedup Ratio...")
    completed_tasks = registry.list_tasks()
    all_success = all(t.status == "completed" for t in completed_tasks)

    individual_durations = [t.duration_sec for t in completed_tasks]
    sum_individual_durations = sum(individual_durations)
    max_individual_duration = max(individual_durations) if individual_durations else 0.0

    # Sequential estimate is the sum of times each would have taken if run one after another
    sequential_estimate = sum_individual_durations
    actual_parallel_wall_time = t_wall_elapsed

    # Concurrency speedup factor
    speedup = (sequential_estimate / actual_parallel_wall_time) if actual_parallel_wall_time > 0 else 1.0
    time_saved = max(0.0, sequential_estimate - actual_parallel_wall_time)
    efficiency_pct = (time_saved / sequential_estimate * 100) if sequential_estimate > 0 else 0.0

    print("\n" + "-" * 76)
    print(f"{'Task ID':<18} | {'Profile':<12} | {'Status':<10} | {'Duration':<10} | {'Outcome Preview'}")
    print("-" * 76)
    for t in completed_tasks:
        preview = (t.outcome or "").replace("\n", " ")[:36]
        print(f"{t.task_id:<18} | {t.profile:<12} | {t.status:<10} | {t.duration_sec:.2f}s     | {preview}...")
    print("-" * 76)

    metric_lines = [
        f"Total Workers Executed:        {len(completed_tasks)}",
        f"All Workers Completed:         {'✅ YES' if all_success else '❌ NO'}",
        f"Individual Durations:          {', '.join(f'{d:.2f}s' for d in individual_durations)}",
        f"Sequential Time (Sum):         {sequential_estimate:.2f}s",
        f"Parallel Wall-Clock Time:      {actual_parallel_wall_time:.2f}s",
        f"Theoretical Floor (Max Task):  {max_individual_duration:.2f}s",
        f"Speedup Factor:                {speedup:.2f}x faster",
        f"Time Saved:                    {time_saved:.2f}s ({efficiency_pct:.1f}% reduction)",
    ]

    print("\n" + _format_box("CONCURRENCY VERIFICATION SUMMARY", metric_lines) + "\n")

    # Clean up any provider override
    set_subagent_provider_override(None)

    # Concurrency verification assertion:
    # In a parallel execution, actual wall time must be significantly lower than sequential sum
    # (unless workers finished in 0s). For num_subagents >= 2, speedup should be > 1.2x.
    is_parallel = (speedup > 1.15) if len(completed_tasks) > 1 else True

    if all_success and is_parallel:
        print("🎉 VERIFICATION PASSED: Parallel subagent execution succeeded!")
        print(f"   Demonstrated {speedup:.2f}x speedup over sequential execution.\n")
        return True
    else:
        print("❌ VERIFICATION FAILED: Did not demonstrate expected concurrency or tasks failed.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test concurrent subagent delegation in Valstorm Agent Runtime.")
    parser.add_argument("--mode", choices=["auto", "mock", "live"], default="auto", help="Execution mode (default: auto)")
    parser.add_argument("--workers", type=int, default=3, help="Number of concurrent workers (default: 3)")
    parser.add_argument("--delay", type=float, default=2.0, help="Simulated worker delay for mock mode (default: 2.0s)")
    args = parser.parse_args()

    success = asyncio.run(run_concurrency_test(mode=args.mode, num_subagents=args.workers, mock_delay=args.delay))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
