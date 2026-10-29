#!/usr/bin/env python3
"""Multi-Axis Ablation & Benchmark Matrix Runner for Valstorm Agent Runtime.

Executes structured benchmark suites (`core_tasks.json`, `swarm_tasks.json`) across
isolated ablation configurations:
  - Axis 1 (Skills):      `with_skills=True` vs. `with_skills=False`
  - Axis 2 (ReAct Loop):  `with_react_loop=True` vs. `with_react_loop=False` (Single-Shot)
  - Axis 3 (Tools):       `tools_mode='scoped'` vs. `'full'` vs. `'none'`
  - Axis 4 (Reasoning):   `thinking_level='none' | 'low' | 'medium' | 'high'`
  - Axis 5 (Swarm):       `swarm_mode=False` (Sequential) vs. `swarm_mode=True` (Parallel Swarm)
  - Axis 6 (Harnesses):   `vsagent` vs. `claude` vs. `codex` vs. `opencode` vs. `hermes` vs. `oh-my-pi`

Usage:
    # Run baseline suite in deterministic mock mode:
    uv run --directory apps/agent-runtime python -m benchmarks.runner --ablation baseline --mode mock

    # Run all 6 scientific ablations and generate Markdown report:
    uv run --directory apps/agent-runtime python -m benchmarks.runner --ablation all --mode mock

    # Run a specific task with live LLM inference across reasoning levels:
    uv run --directory apps/agent-runtime python -m benchmarks.runner --ablation reasoning --task G-S01 --mode live
"""

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List, Literal, Optional

BENCHMARKS_DIR = Path(__file__).resolve().parent
RUNTIME_ROOT = BENCHMARKS_DIR.parent
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from dotenv import load_dotenv
load_dotenv(RUNTIME_ROOT / ".env", override=False)
load_dotenv(RUNTIME_ROOT.parent.parent / ".env", override=False)

from benchmarks.adapters import (
    BenchmarkConfig,
    ExternalCliBenchmarkAdapter,
    VsAgentBenchmarkAdapter,
)
from benchmarks.scorer import EnvironmentStateScorer, ScoreResult, compute_multi_trial_metrics


def get_ablation_configs(
    ablation: str,
    provider: str = "gemini",
    model: str = "gemini-flash-latest",
) -> List[BenchmarkConfig]:
    """Returns the scientific ablation configurations for a given axis."""
    ab = ablation.strip().lower()

    if ab == "baseline":
        return [
            BenchmarkConfig(
                config_id="baseline",
                label="Baseline (Skills + ReAct + Scoped Tools + Medium Thinking)",
                provider=provider,
                model=model,
                thinking_level="medium",
                with_skills=True,
                with_react_loop=True,
                tools_mode="scoped",
            )
        ]

    if ab == "skills":
        return [
            BenchmarkConfig(
                config_id="skills_on",
                label="1. With Procedural Skills",
                provider=provider,
                model=model,
                thinking_level="low",
                with_skills=True,
                with_react_loop=True,
                tools_mode="scoped",
            ),
            BenchmarkConfig(
                config_id="skills_off",
                label="2. Without Procedural Skills",
                provider=provider,
                model=model,
                thinking_level="low",
                with_skills=False,
                with_react_loop=True,
                tools_mode="scoped",
            ),
        ]

    if ab == "react":
        return [
            BenchmarkConfig(
                config_id="react_on",
                label="3. With ReAct Loop (Iterative Multi-Step, max_turns=15)",
                provider=provider,
                model=model,
                with_skills=True,
                with_react_loop=True,
                max_turns=15,
                tools_mode="scoped",
            ),
            BenchmarkConfig(
                config_id="react_off",
                label="4. Without ReAct Loop (Single-Shot, max_turns=1)",
                provider=provider,
                model=model,
                with_skills=True,
                with_react_loop=False,
                max_turns=1,
                tools_mode="scoped",
            ),
        ]

    if ab == "tools":
        return [
            BenchmarkConfig(
                config_id="tools_scoped",
                label="5a. With Scoped Tools (Task Whitelist)",
                provider=provider,
                model=model,
                with_skills=True,
                with_react_loop=True,
                tools_mode="scoped",
            ),
            BenchmarkConfig(
                config_id="tools_full",
                label="5b. With Full Tool Registry (All Runtime Tools)",
                provider=provider,
                model=model,
                with_skills=True,
                with_react_loop=True,
                tools_mode="full",
            ),
            BenchmarkConfig(
                config_id="tools_none",
                label="6. Without Tools (Pure Text Generation)",
                provider=provider,
                model=model,
                with_skills=True,
                with_react_loop=True,
                tools_mode="none",
            ),
        ]

    if ab == "reasoning":
        return [
            BenchmarkConfig(
                config_id=f"thinking_{lvl}",
                label=f"7. {model} (Thinking: {lvl.upper()})",
                provider=provider,
                model=model,
                thinking_level=lvl,  # type: ignore[arg-type]
                with_skills=True,
                with_react_loop=True,
                tools_mode="scoped",
            )
            for lvl in ("none", "low", "medium", "high")
        ]

    if ab == "swarm":
        return [
            BenchmarkConfig(
                config_id="seq_single",
                label="10a. Single Agent (Sequential Execution)",
                provider=provider,
                model=model,
                with_skills=True,
                with_react_loop=True,
                tools_mode="scoped",
                swarm_mode=False,
                swarm_workers=1,
            ),
            BenchmarkConfig(
                config_id="swarm_parallel",
                label="10b. Parallel Subagent Swarm (background=True, Static Model)",
                provider=provider,
                model=model,
                with_skills=True,
                with_react_loop=True,
                tools_mode="scoped",
                swarm_mode=True,
                swarm_workers=3,
                auto_tune=False,
            ),
            BenchmarkConfig(
                config_id="swarm_autotuned",
                label="10c. Parallel Subagent Swarm + System-1 Subagent Auto-Tune (background=True)",
                provider=provider,
                model=model,
                thinking_level="adaptive",
                with_skills=True,
                with_react_loop=True,
                tools_mode="scoped",
                swarm_mode=True,
                swarm_workers=3,
                auto_tune=True,
            ),
        ]

    if ab == "harnesses":
        harnesses: List[Literal["vsagent", "claude", "codex", "opencode", "hermes", "oh-my-pi"]] = [
            "vsagent",
            "claude",
            "codex",
            "opencode",
            "hermes",
            "oh-my-pi",
        ]
        return [
            BenchmarkConfig(
                config_id=f"harness_{h}",
                label=f"9. Harness: {h}",
                harness=h,
                provider=provider,
                model=model,
            )
            for h in harnesses
        ]

    if ab == "adaptive_thinking":
        return [
            BenchmarkConfig(
                config_id="thinking_adaptive_s1",
                label="8a. System-1 Adaptive Thinking Router (Dynamic none/low/medium/high)",
                provider=provider,
                model=model,
                thinking_level="adaptive",
                with_skills=False,
                with_react_loop=True,
                tools_mode="scoped",
            ),
            BenchmarkConfig(
                config_id="thinking_high_static",
                label="8b. Static HIGH Thinking (Always High without Skills)",
                provider=provider,
                model=model,
                thinking_level="high",
                with_skills=False,
                with_react_loop=True,
                tools_mode="scoped",
            ),
            BenchmarkConfig(
                config_id="thinking_none_static",
                label="8c. Static NONE Thinking (Always None without Skills)",
                provider=provider,
                model=model,
                thinking_level="none",
                with_skills=False,
                with_react_loop=True,
                tools_mode="scoped",
            ),
        ]

    if ab == "message_autotune":
        return [
            BenchmarkConfig(
                config_id="autotune_s1_full",
                label="9a. System-1 ai_chat_message Auto-Tuner (Dynamic Tier + Intra-Turn Step-Down)",
                provider=provider,
                model="gemini-flash-latest",
                thinking_level="adaptive",
                with_skills=False,
                with_react_loop=True,
                tools_mode="scoped",
                auto_tune=True,
            ),
            BenchmarkConfig(
                config_id="static_thinker_high",
                label="9b. Static Tier-1 THINKER (`gemini-pro-latest` @ HIGH, No Skills)",
                provider=provider,
                model="gemini-pro-latest",
                thinking_level="high",
                with_skills=False,
                with_react_loop=True,
                tools_mode="scoped",
                auto_tune=False,
            ),
            BenchmarkConfig(
                config_id="static_reasoner_high",
                label="9c. Static Tier-2 REASONER (`gemini-flash-latest` @ HIGH, No Skills)",
                provider=provider,
                model="gemini-flash-latest",
                thinking_level="high",
                with_skills=False,
                with_react_loop=True,
                tools_mode="scoped",
                auto_tune=False,
            ),
            BenchmarkConfig(
                config_id="static_worker_none",
                label="9d. Static Tier-3 WORKER (`gemini-flash-lite-latest` @ NONE, No Skills)",
                provider=provider,
                model="gemini-flash-lite-latest",
                thinking_level="none",
                with_skills=False,
                with_react_loop=True,
                tools_mode="scoped",
                auto_tune=False,
            ),
        ]

    if ab in ("ui_nav", "decision_model"):
        return [
            BenchmarkConfig(
                config_id="system_one_ui_nav",
                label="11a. System-1 Decision Model UI/Canvas Navigator (Parallel Closed-Set)",
                provider=provider,
                model=model,
                thinking_level="low",
                with_skills=True,
                with_react_loop=True,
                tools_mode="scoped",
                use_decision_model=True,
            ),
            BenchmarkConfig(
                config_id="llm_react_ui_nav",
                label="11b. System-2 Full LLM ReAct UI/Canvas Navigation (Step-by-Step)",
                provider=provider,
                model=model,
                thinking_level="medium",
                with_skills=True,
                with_react_loop=True,
                tools_mode="scoped",
                use_decision_model=False,
            ),
        ]

    # GEAP Explicit Baselines + DigitalOcean Open-Weight Model Catalog (No Proprietary 3rd-Party Frontier Models)
    do_thinker_configs = [
        BenchmarkConfig(
            config_id="geap_gemini_4_argon",
            label="GEAP Baseline Thinker (`gemini-4-argon`, 1M/1M)",
            provider="gemini",
            model="gemini-4-argon",
            thinking_level="high",
            with_skills=True,
            with_react_loop=True,
            tools_mode="scoped",
            isolate_provider=True,
            dedicated_eligible=False,
        ),
        BenchmarkConfig(
            config_id="do_qwen3_8_max",
            label="DO Open Thinker (`qwen3.8-max`, 2.4T MoE, 1M/262K)",
            provider="do",
            model="qwen3.8-max",
            thinking_level="high",
            with_skills=True,
            with_react_loop=True,
            tools_mode="scoped",
            isolate_provider=True,
            dedicated_eligible=False,
        ),
        BenchmarkConfig(
            config_id="do_deepseek_v4_pro_0813",
            label="DO Open Thinker (`deepseek-v4-pro-0813`, 1M/1M, $0.044 Cache)",
            provider="do",
            model="deepseek-v4-pro-0813",
            thinking_level="high",
            with_skills=True,
            with_react_loop=True,
            tools_mode="scoped",
            isolate_provider=True,
            dedicated_eligible=False,
        ),
        BenchmarkConfig(
            config_id="do_glm_5_3",
            label="DO Open Thinker (`glm-5.3`, 753B, Always-On Reasoning)",
            provider="do",
            model="glm-5.3",
            thinking_level="high",
            with_skills=True,
            with_react_loop=True,
            tools_mode="scoped",
            isolate_provider=True,
            dedicated_eligible=False,
        ),
        BenchmarkConfig(
            config_id="do_kimi_k2_6",
            label="DO Open Thinker [Dedicated+Serverless] (`kimi-k2.6`, 1T, 262K/262K)",
            provider="do",
            model="kimi-k2.6",
            thinking_level="high",
            with_skills=True,
            with_react_loop=True,
            tools_mode="scoped",
            isolate_provider=True,
            dedicated_eligible=True,
        ),
        BenchmarkConfig(
            config_id="do_mimo_v2_5_pro",
            label="DO Open Thinker [Dedicated+Serverless] (`mimo-v2.5-pro`, 1T MIT, 262K/262K)",
            provider="do",
            model="mimo-v2.5-pro",
            thinking_level="high",
            with_skills=True,
            with_react_loop=True,
            tools_mode="scoped",
            isolate_provider=True,
            dedicated_eligible=True,
        ),
    ]

    do_reasoner_configs = [
        BenchmarkConfig(
            config_id="geap_gemini_3_8_flash",
            label="GEAP Baseline Reasoner (`gemini-3.8-flash`, 1M/64K)",
            provider="gemini",
            model="gemini-3.8-flash",
            thinking_level="medium",
            with_skills=True,
            with_react_loop=True,
            tools_mode="scoped",
            isolate_provider=True,
            dedicated_eligible=False,
        ),
        BenchmarkConfig(
            config_id="do_qwen3_5_397b_a17b",
            label="DO Open Reasoner [Dedicated+Serverless] (`qwen3.5-397b-a17b`, 17B Active MoE)",
            provider="do",
            model="qwen3.5-397b-a17b",
            thinking_level="medium",
            with_skills=True,
            with_react_loop=True,
            tools_mode="scoped",
            isolate_provider=True,
            dedicated_eligible=True,
        ),
        BenchmarkConfig(
            config_id="do_deepseek_v4_1_flash",
            label="DO Open Reasoner (`deepseek-v4.1-flash`, 552B Vision, $0.006 Cache)",
            provider="do",
            model="deepseek-v4.1-flash",
            thinking_level="medium",
            with_skills=True,
            with_react_loop=True,
            tools_mode="scoped",
            isolate_provider=True,
            dedicated_eligible=False,
        ),
        BenchmarkConfig(
            config_id="do_deepseek_3_2",
            label="DO Open Reasoner [Dedicated+Serverless] (`deepseek-3.2`, 680B, 163K)",
            provider="do",
            model="deepseek-3.2",
            thinking_level="medium",
            with_skills=True,
            with_react_loop=True,
            tools_mode="scoped",
            isolate_provider=True,
            dedicated_eligible=True,
        ),
        BenchmarkConfig(
            config_id="do_glm_5_2",
            label="DO Open Reasoner [Dedicated+Serverless] (`glm-5.2`, MIT, 262K/131K)",
            provider="do",
            model="glm-5.2",
            thinking_level="medium",
            with_skills=True,
            with_react_loop=True,
            tools_mode="scoped",
            isolate_provider=True,
            dedicated_eligible=True,
        ),
        BenchmarkConfig(
            config_id="do_nemotron_3_ultra",
            label="DO Open Reasoner [Dedicated+Serverless] (`nemotron-3-ultra-550b`, 550B)",
            provider="do",
            model="nemotron-3-ultra-550b",
            thinking_level="medium",
            with_skills=True,
            with_react_loop=True,
            tools_mode="scoped",
            isolate_provider=True,
            dedicated_eligible=True,
        ),
        BenchmarkConfig(
            config_id="do_minimax_m2_5",
            label="DO Open Reasoner [Dedicated+Serverless] (`minimax-m2.5`, 230B, 65K)",
            provider="do",
            model="minimax-m2.5",
            thinking_level="medium",
            with_skills=True,
            with_react_loop=True,
            tools_mode="scoped",
            isolate_provider=True,
            dedicated_eligible=True,
        ),
    ]

    do_worker_configs = [
        BenchmarkConfig(
            config_id="geap_gemini_3_5_flash_lite",
            label="GEAP Baseline Worker (`gemini-3.5-flash-lite`, 1M/64K)",
            provider="gemini",
            model="gemini-3.5-flash-lite",
            thinking_level="low",
            with_skills=True,
            with_react_loop=True,
            tools_mode="scoped",
            isolate_provider=True,
            dedicated_eligible=False,
        ),
        BenchmarkConfig(
            config_id="do_deepseek_v4_flash_0731",
            label="DO Open Worker (`deepseek-v4-flash-0731`, 284B, 1M/1M, $0.14/$0.28)",
            provider="do",
            model="deepseek-v4-flash-0731",
            thinking_level="low",
            with_skills=True,
            with_react_loop=True,
            tools_mode="scoped",
            isolate_provider=True,
            dedicated_eligible=False,
        ),
        BenchmarkConfig(
            config_id="do_glm_5_3_flash",
            label="DO Open Worker (`glm-5.3-flash`, 320B Multimodal, $0.15/$0.50)",
            provider="do",
            model="glm-5.3-flash",
            thinking_level="low",
            with_skills=True,
            with_react_loop=True,
            tools_mode="scoped",
            isolate_provider=True,
            dedicated_eligible=False,
        ),
        BenchmarkConfig(
            config_id="do_llama_4_maverick",
            label="DO Open Worker [Dedicated+Serverless] (`llama-4-maverick`, 17B-128E, 16K Out)",
            provider="do",
            model="llama-4-maverick",
            thinking_level="low",
            with_skills=True,
            with_react_loop=True,
            tools_mode="scoped",
            isolate_provider=True,
            dedicated_eligible=True,
        ),
        BenchmarkConfig(
            config_id="do_gpt_oss_120b",
            label="DO Open Worker [Dedicated+Serverless] (`openai-gpt-oss-120b`, 4K Max Out)",
            provider="do",
            model="openai-gpt-oss-120b",
            thinking_level="low",
            with_skills=True,
            with_react_loop=True,
            tools_mode="scoped",
            isolate_provider=True,
            dedicated_eligible=True,
        ),
        BenchmarkConfig(
            config_id="do_gemma_4_31b",
            label="DO Open Worker [1x GPU Dedicated] (`gemma-4-31B-it`, 31B, 256K/8K)",
            provider="do",
            model="gemma-4-31B-it",
            thinking_level="low",
            with_skills=True,
            with_react_loop=True,
            tools_mode="scoped",
            isolate_provider=True,
            dedicated_eligible=True,
        ),
        BenchmarkConfig(
            config_id="do_mistral_3_14b",
            label="DO Open Worker [1x GPU Dedicated] (`mistral-3-14B`, 14B, 262K)",
            provider="do",
            model="mistral-3-14B",
            thinking_level="low",
            with_skills=True,
            with_react_loop=True,
            tools_mode="scoped",
            isolate_provider=True,
            dedicated_eligible=True,
        ),
    ]

    if ab == "do_thinkers":
        return do_thinker_configs
    if ab == "do_reasoners":
        return do_reasoner_configs
    if ab == "do_workers":
        return do_worker_configs
    if ab == "do_open_models":
        return do_thinker_configs + do_reasoner_configs + do_worker_configs
    if ab == "do_dedicated_models":
        all_do = do_thinker_configs + do_reasoner_configs + do_worker_configs
        return [c for c in all_do if c.provider == "gemini" or c.dedicated_eligible]

    raise ValueError(f"Unknown ablation preset: '{ablation}'")


def load_tasks(
    suite: Literal["core", "swarm", "ui", "office", "creative", "all", "full"] = "core",
    task_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Loads benchmark task definitions from `benchmarks/tasks/*.json`."""
    tasks_dir = BENCHMARKS_DIR / "tasks"
    loaded: List[Dict[str, Any]] = []
    tid_upper = task_id.strip().upper() if task_id else ""

    if suite in ("core", "all", "full") or tid_upper.startswith("G-"):
        core_path = tasks_dir / "core_tasks.json"
        if core_path.is_file():
            loaded.extend(json.loads(core_path.read_text(encoding="utf-8")))

    if suite in ("swarm", "all", "full") or tid_upper.startswith("SW-"):
        swarm_path = tasks_dir / "swarm_tasks.json"
        if swarm_path.is_file():
            loaded.extend(json.loads(swarm_path.read_text(encoding="utf-8")))

    if suite in ("ui", "full") or tid_upper.startswith("UI-"):
        ui_path = tasks_dir / "ui_navigation_tasks.json"
        if ui_path.is_file():
            loaded.extend(json.loads(ui_path.read_text(encoding="utf-8")))

    if suite in ("office", "full") or tid_upper.startswith("OF-"):
        office_path = tasks_dir / "office_tasks.json"
        if office_path.is_file():
            loaded.extend(json.loads(office_path.read_text(encoding="utf-8")))

    if suite in ("creative", "full") or tid_upper.startswith("CR-"):
        creative_path = tasks_dir / "creative_tasks.json"
        if creative_path.is_file():
            loaded.extend(json.loads(creative_path.read_text(encoding="utf-8")))

    if task_id:
        loaded = [t for t in loaded if t.get("id", "").upper() == tid_upper]

    # Deduplicate by task id while preserving order
    seen_ids = set()
    unique_loaded = []
    for t in loaded:
        tid = t.get("id")
        if tid not in seen_ids:
            seen_ids.add(tid)
            unique_loaded.append(t)

    return unique_loaded


async def run_benchmark_matrix(
    tasks: List[Dict[str, Any]],
    configs: List[BenchmarkConfig],
    mode: Literal["auto", "mock", "live"] = "auto",
    trials: int = 1,
) -> Dict[str, Any]:
    """Executes the cartesian product of `configs x tasks x trials` in isolated workspaces."""
    vs_adapter = VsAgentBenchmarkAdapter()
    ext_adapter = ExternalCliBenchmarkAdapter()
    scorer = EnvironmentStateScorer()

    all_scores: List[ScoreResult] = []
    config_summaries: List[Dict[str, Any]] = []

    for cfg in configs:
        adapter = vs_adapter if cfg.harness == "vsagent" else ext_adapter
        cfg_scores: List[ScoreResult] = []
        task_trial_groups: Dict[str, List[ScoreResult]] = {}

        for task in tasks:
            t_id = task["id"]
            task_trial_groups[t_id] = []
            for _trial_idx in range(trials):
                with tempfile.TemporaryDirectory(prefix=f"vsbench_{t_id}_{cfg.config_id}_") as tmp_dir:
                    ws_path = Path(tmp_dir).resolve()
                    trace = await adapter.execute_task(
                        task=task,
                        config=cfg,
                        workspace_dir=ws_path,
                        mode=mode,
                    )
                    score_res = scorer.score(
                        task=task,
                        config=cfg,
                        trace=trace,
                        workspace_dir=ws_path,
                    )
                    all_scores.append(score_res)
                    cfg_scores.append(score_res)
                    task_trial_groups[t_id].append(score_res)

        # Aggregate per-config metrics
        total_runs = len(cfg_scores)
        passed_runs = sum(1 for r in cfg_scores if r.passed)
        pass_rate = round(passed_runs / total_runs, 4) if total_runs > 0 else 0.0
        avg_dur = round(sum(r.duration_sec for r in cfg_scores) / total_runs, 3) if total_runs > 0 else 0.0
        avg_turns = round(sum(r.turns_taken for r in cfg_scores) / total_runs, 2) if total_runs > 0 else 0.0
        avg_tokens = round(sum(r.total_tokens for r in cfg_scores) / total_runs, 1) if total_runs > 0 else 0.0
        total_cost = round(sum(r.estimated_cost_usd for r in cfg_scores), 6)
        avg_speedup = round(sum(r.parallel_speedup for r in cfg_scores) / total_runs, 2) if total_runs > 0 else 1.0

        per_task_reliability = {
            tid: compute_multi_trial_metrics(t_list)
            for tid, t_list in task_trial_groups.items()
        }
        avg_pass_hat_k = (
            round(sum(m["pass_hat_k"] for m in per_task_reliability.values()) / len(per_task_reliability), 4)
            if per_task_reliability
            else 0.0
        )

        config_summaries.append({
            "config_id": cfg.config_id,
            "label": cfg.label,
            "harness": cfg.harness,
            "model": cfg.model,
            "thinking_level": cfg.thinking_level,
            "with_skills": cfg.with_skills,
            "with_react_loop": cfg.with_react_loop,
            "tools_mode": cfg.tools_mode,
            "swarm_mode": cfg.swarm_mode,
            "total_runs": total_runs,
            "passed_runs": passed_runs,
            "pass_rate": pass_rate,
            "pass_hat_k": avg_pass_hat_k,
            "avg_duration_sec": avg_dur,
            "avg_turns": avg_turns,
            "avg_tokens": avg_tokens,
            "total_cost_usd": total_cost,
            "avg_parallel_speedup": avg_speedup,
            "per_task": per_task_reliability,
        })

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "trials_per_task": trials,
        "task_count": len(tasks),
        "config_count": len(configs),
        "config_summaries": config_summaries,
        "runs": [r.to_dict() for r in all_scores],
    }


def format_markdown_report(report: Dict[str, Any], title: str = "Valstorm Agent Runtime — Benchmark Report") -> str:
    """Formats the benchmark matrix summary into a Markdown table report."""
    lines = [
        f"# {title}",
        "",
        f"- **Timestamp (UTC):** `{report['timestamp']}`",
        f"- **Execution Mode:** `{report['mode']}`",
        f"- **Tasks Evaluated:** `{report['task_count']}` | **Trials per Task (`k`):** `{report['trials_per_task']}`",
        "",
        "## 1. Configuration Ablation Summary",
        "",
        "| Config ID | Configuration Label | Pass Rate | `pass^k` | Avg Turns | Avg Latency | Avg Tokens | Est. Cost ($) | Parallel Speedup |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cs in report["config_summaries"]:
        pr_pct = f"{cs['pass_rate'] * 100:.1f}% ({cs['passed_runs']}/{cs['total_runs']})"
        ph_pct = f"{cs['pass_hat_k'] * 100:.1f}%"
        lines.append(
            f"| `{cs['config_id']}` | {cs['label']} | **{pr_pct}** | `{ph_pct}` | "
            f"{cs['avg_turns']} | {cs['avg_duration_sec']:.2f}s | {cs['avg_tokens']:.0f} | "
            f"${cs['total_cost_usd']:.5f} | {cs['avg_parallel_speedup']:.2f}x |"
        )

    lines.extend([
        "",
        "## 2. Granular Task-by-Task Scorecard",
        "",
        "| Task ID | Complexity | Config ID | Outcome | Score | Turns | Tools Called | Duration | Failure / Notes |",
        "| :--- | :---: | :--- | :---: | :---: | :---: | :--- | :---: | :--- |",
    ])

    for r in report["runs"]:
        outcome_badge = "✅ PASS" if r["passed"] else ("⏭️ SKIP" if r["failure_reason"] and "not found on PATH" in r["failure_reason"] else "❌ FAIL")
        tools_str = ", ".join(r["actual_tools"]) if r["actual_tools"] else "*(none)*"
        notes = (r["failure_reason"] or "All checks verified")[:65]
        lines.append(
            f"| `{r['task_id']}` | {r['complexity']} | `{r['config_id']}` | {outcome_badge} | "
            f"{r['score']:.2f} | {r['turns_taken']} | `{tools_str}` | {r['duration_sec']:.2f}s | {notes} |"
        )

    lines.append("")
    return "\n".join(lines)


def save_benchmark_artifacts(report: Dict[str, Any], markdown_text: str) -> Dict[str, Path]:
    """Saves JSONL and Markdown benchmark reports to `benchmarks/results/`."""
    results_dir = BENCHMARKS_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    md_path = results_dir / "latest_report.md"
    json_path = results_dir / "latest_run.json"

    md_path.write_text(markdown_text, encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"markdown": md_path, "json": json_path}


async def run_all_ablations_demo(mode: Literal["auto", "mock", "live"] = "mock", trials: int = 1) -> Dict[str, Any]:
    """Runs a comprehensive scientific sweep across all ablation axes (Core + Swarm + System-1 UI/Canvas Nav)."""
    core_tasks = load_tasks(suite="core")
    swarm_tasks = load_tasks(suite="swarm")
    ui_tasks = load_tasks(suite="ui")

    # 1. Run Core Tasks across Skills, ReAct, Tools, Reasoning, Adaptive Thinking, and Message Auto-Tune ablations
    core_configs: List[BenchmarkConfig] = []
    for axis in ("skills", "react", "tools", "reasoning", "adaptive_thinking", "message_autotune"):
        core_configs.extend(get_ablation_configs(axis))

    core_report = await run_benchmark_matrix(tasks=core_tasks, configs=core_configs, mode=mode, trials=trials)

    # 2. Run Swarm Tasks across Sequential vs Parallel Swarm ablation
    swarm_configs = get_ablation_configs("swarm")
    swarm_report = await run_benchmark_matrix(tasks=swarm_tasks, configs=swarm_configs, mode=mode, trials=trials)

    # 3. Run UI & Crafting Canvas Tasks across System-1 Decision Model vs System-2 LLM ReAct
    ui_configs = get_ablation_configs("decision_model")
    ui_report = await run_benchmark_matrix(tasks=ui_tasks, configs=ui_configs, mode=mode, trials=trials)

    # Combine reports
    combined = {
        "timestamp": core_report["timestamp"],
        "mode": mode,
        "trials_per_task": trials,
        "task_count": len(core_tasks) + len(swarm_tasks) + len(ui_tasks),
        "config_count": len(core_configs) + len(swarm_configs) + len(ui_configs),
        "config_summaries": (
            core_report["config_summaries"]
            + swarm_report["config_summaries"]
            + ui_report["config_summaries"]
        ),
        "runs": core_report["runs"] + swarm_report["runs"] + ui_report["runs"],
    }
    return combined


def main():
    parser = argparse.ArgumentParser(description="Valstorm Agent Runtime — Multi-Axis Benchmark & Ablation Runner")
    parser.add_argument(
        "--ablation",
        choices=[
            "baseline", "skills", "react", "tools", "reasoning",
            "adaptive_thinking", "message_autotune", "swarm", "harnesses",
            "ui_nav", "decision_model", "do_open_models", "do_dedicated_models",
            "do_thinkers", "do_reasoners", "do_workers", "all",
        ],
        default="baseline",
        help="Ablation preset to execute (default: baseline)",
    )
    parser.add_argument(
        "--suite",
        choices=["core", "swarm", "ui", "office", "creative", "all", "full"],
        default="core",
        help="Task suite to load (default: core)",
    )
    parser.add_argument("--task", type=str, default=None, help="Optional single task ID filter (e.g. G-S01, G-H03, SW-01)")
    parser.add_argument("--mode", choices=["auto", "mock", "live"], default="auto", help="Execution mode (default: auto)")
    parser.add_argument("--provider", type=str, default="gemini", help="Target provider (default: gemini)")
    parser.add_argument("--model", type=str, default="gemini-flash-latest", help="Target model (default: gemini-flash-latest)")
    parser.add_argument("--trials", type=int, default=1, help="Number of trials (k) per task for pass^k calculation")
    args = parser.parse_args()

    if args.ablation == "all" and not args.task:
        report = asyncio.run(run_all_ablations_demo(mode=args.mode, trials=args.trials))
    else:
        if args.ablation == "swarm" and args.suite == "core" and not args.task:
            effective_suite = "swarm"
        elif args.ablation in ("ui_nav", "decision_model") and args.suite == "core" and not args.task:
            effective_suite = "ui"
        else:
            effective_suite = args.suite
        tasks = load_tasks(suite=effective_suite, task_id=args.task)
        if not tasks:
            print(f"No tasks matched suite='{effective_suite}' task='{args.task}'.")
            sys.exit(1)
        configs = get_ablation_configs(args.ablation, provider=args.provider, model=args.model)
        report = asyncio.run(run_benchmark_matrix(tasks=tasks, configs=configs, mode=args.mode, trials=args.trials))

    md_text = format_markdown_report(report, title=f"Valstorm Agent Benchmark Report (`--ablation {args.ablation}`)")
    paths = save_benchmark_artifacts(report, md_text)

    print("\n" + md_text)
    print(f"📊 Saved Markdown report to: {paths['markdown']}")
    print(f"📁 Saved JSON telemetry to:  {paths['json']}\n")


if __name__ == "__main__":
    main()
