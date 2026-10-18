"""Unit and integration tests for the Valstorm Agent Runtime Multi-Axis Benchmark Harness."""

from pathlib import Path
import pytest

from benchmarks.adapters import (
    BenchmarkConfig,
    ExecutionTrace,
    build_benchmark_tool_registry,
    build_cli_command,
    estimate_cost_usd,
)
from benchmarks.runner import (
    format_markdown_report,
    get_ablation_configs,
    load_tasks,
    run_benchmark_matrix,
)
from benchmarks.scorer import (
    EnvironmentStateScorer,
    ScoreResult,
    compute_multi_trial_metrics,
)


def test_load_core_and_swarm_tasks():
    """Verify core_tasks.json (9 tasks) and swarm_tasks.json (2 tasks) load cleanly."""
    core = load_tasks(suite="core")
    swarm = load_tasks(suite="swarm")
    all_tasks = load_tasks(suite="all")

    assert len(core) == 9
    assert len(swarm) == 2
    assert len(all_tasks) == 11

    single = load_tasks(suite="core", task_id="G-H03")
    assert len(single) == 1
    assert single[0]["id"] == "G-H03"
    assert single[0]["success_criteria"]["require_executable"] is True


def test_ablation_presets_cover_all_axes():
    """Verify all 6 scientific ablation axes generate valid BenchmarkConfig objects."""
    skills_cfgs = get_ablation_configs("skills")
    assert len(skills_cfgs) == 2
    assert skills_cfgs[0].with_skills is True
    assert skills_cfgs[1].with_skills is False

    react_cfgs = get_ablation_configs("react")
    assert len(react_cfgs) == 2
    assert react_cfgs[0].effective_max_turns == 15
    assert react_cfgs[1].effective_max_turns == 1

    tools_cfgs = get_ablation_configs("tools")
    assert [c.tools_mode for c in tools_cfgs] == ["scoped", "full", "none"]

    reasoning_cfgs = get_ablation_configs("reasoning")
    assert [c.thinking_level for c in reasoning_cfgs] == ["none", "low", "medium", "high"]

    swarm_cfgs = get_ablation_configs("swarm")
    assert swarm_cfgs[0].swarm_mode is False
    assert swarm_cfgs[1].swarm_mode is True

    harness_cfgs = get_ablation_configs("harnesses")
    assert [c.harness for c in harness_cfgs] == [
        "vsagent",
        "claude",
        "codex",
        "opencode",
        "hermes",
        "oh-my-pi",
    ]


def test_external_cli_command_builder():
    """Verify headless CLI command generation for Claude Code, Codex, OpenCode, Hermes, Oh My Pi, and vsagent."""
    assert build_cli_command("claude", "Fix bug", "claude-sonnet-4-5") == [
        "claude",
        "-p",
        "Fix bug",
        "--dangerously-skip-permissions",
        "--model",
        "claude-sonnet-4-5",
    ]
    assert build_cli_command("codex", "Fix bug", "gpt-5") == [
        "codex",
        "exec",
        "--full-auto",
        "Fix bug",
        "-m",
        "gpt-5",
    ]
    assert build_cli_command("opencode", "Fix bug", "gemini-flash") == [
        "opencode",
        "run",
        "Fix bug",
        "--model",
        "gemini-flash",
    ]
    assert build_cli_command("hermes", "Fix bug") == ["hermes", "run", "Fix bug"]
    assert build_cli_command("oh-my-pi", "Fix bug") == ["omp", "run", "Fix bug"]
    assert build_cli_command("vsagent", "Fix bug", "gemini-flash-latest") == [
        "vsagent",
        "chat",
        "-p",
        "Fix bug",
        "--model",
        "gemini-flash-latest",
    ]


def test_tool_registry_scoping_ablation():
    """Verify tools_mode='none', 'scoped', and 'full' construct appropriately sized registries."""
    task = {"id": "G-S01", "tool_chain": ["calculator", "write_file"]}

    reg_none = build_benchmark_tool_registry("none", task)
    assert len(reg_none.list_tools()) == 0

    reg_scoped = build_benchmark_tool_registry("scoped", task)
    assert set(reg_scoped.list_tools()) == {"calculator", "write_file"}

    reg_swarm = build_benchmark_tool_registry("scoped", task, swarm_mode=True)
    assert "delegate_task" in reg_swarm.list_tools()

    reg_full = build_benchmark_tool_registry("full", task)
    assert len(reg_full.list_tools()) > 15


def test_environment_state_scorer_and_chmod_guardrail(tmp_path: Path):
    """Verify EnvironmentStateScorer checks file contents, executable permissions, and commands."""
    scorer = EnvironmentStateScorer()
    cfg = BenchmarkConfig(config_id="test", label="Test")
    task = {
        "id": "G-H03",
        "complexity": "High",
        "category": "terminal_ops",
        "tool_chain": ["write_file", "terminal_exec"],
        "success_criteria": {
            "file": "setup_env.sh",
            "require_executable": True,
        },
    }
    trace = ExecutionTrace(
        task_id="G-H03",
        config_id="test",
        harness="vsagent",
        status="completed",
        tool_calls_made=["write_file", "terminal_exec"],
        turns_taken=2,
        duration_sec=0.5,
    )

    script_file = tmp_path / "setup_env.sh"
    script_file.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")

    # 1. Without chmod +x -> must fail require_executable
    res_no_chmod = scorer.score(task, cfg, trace, tmp_path)
    assert res_no_chmod.passed is False
    assert "missing_executable_chmod:setup_env.sh" in res_no_chmod.checks_failed

    # 2. With chmod +x -> must pass
    script_file.chmod(0o755)
    res_with_chmod = scorer.score(task, cfg, trace, tmp_path)
    assert res_with_chmod.passed is True
    assert res_with_chmod.score == 1.0


def test_tau_bench_pass_hat_k_calculation():
    """Verify Pass@k and tau-bench pass^k calculation across multiple trials."""
    r_pass = ScoreResult(
        task_id="G-S01",
        complexity="Simple",
        category="math",
        config_id="c1",
        config_label="C1",
        harness="vsagent",
        passed=True,
        score=1.0,
        duration_sec=1.0,
        total_tokens=100,
        estimated_cost_usd=0.001,
    )
    r_fail = ScoreResult(
        task_id="G-S01",
        complexity="Simple",
        category="math",
        config_id="c1",
        config_label="C1",
        harness="vsagent",
        passed=False,
        score=0.0,
        duration_sec=1.0,
        total_tokens=100,
        estimated_cost_usd=0.001,
    )

    # 2 out of 3 pass -> pass_rate = 0.6667, Pass@3 = 1.0, pass^3 = (2/3)^3 = 0.2963
    metrics = compute_multi_trial_metrics([r_pass, r_pass, r_fail])
    assert metrics["trials"] == 3
    assert metrics["pass_at_k"] == 1.0
    assert abs(metrics["pass_hat_k"] - round((2 / 3) ** 3, 4)) < 1e-4


@pytest.mark.asyncio
async def test_ablation_matrix_execution_skills_react_and_swarm():
    """Verify end-to-end execution of Skills, ReAct, and Swarm ablations in mock mode."""
    # 1. Skills ablation on G-H03
    gh03 = load_tasks(suite="core", task_id="G-H03")
    skills_report = await run_benchmark_matrix(
        tasks=gh03,
        configs=get_ablation_configs("skills"),
        mode="mock",
        trials=1,
    )
    summaries = {s["config_id"]: s for s in skills_report["config_summaries"]}
    assert summaries["skills_on"]["pass_rate"] == 1.0
    assert summaries["skills_off"]["pass_rate"] == 0.0

    # 2. ReAct ablation on G-H01 (TDD requires multiple iterations)
    gh01 = load_tasks(suite="core", task_id="G-H01")
    react_report = await run_benchmark_matrix(
        tasks=gh01,
        configs=get_ablation_configs("react"),
        mode="mock",
        trials=1,
    )
    r_summaries = {s["config_id"]: s for s in react_report["config_summaries"]}
    assert r_summaries["react_on"]["pass_rate"] == 1.0
    assert r_summaries["react_off"]["pass_rate"] == 0.0

    # 3. Swarm ablation on SW-01 (Parallel Swarm must be faster than Sequential Single Agent)
    sw01 = load_tasks(suite="swarm", task_id="SW-01")
    swarm_report = await run_benchmark_matrix(
        tasks=sw01,
        configs=get_ablation_configs("swarm"),
        mode="mock",
        trials=1,
    )
    sw_summaries = {s["config_id"]: s for s in swarm_report["config_summaries"]}
    assert sw_summaries["seq_single"]["pass_rate"] == 1.0
    assert sw_summaries["swarm_parallel"]["pass_rate"] == 1.0
    assert sw_summaries["swarm_parallel"]["avg_duration_sec"] < sw_summaries["seq_single"]["avg_duration_sec"]
    assert sw_summaries["swarm_parallel"]["avg_parallel_speedup"] > 1.3

    md = format_markdown_report(swarm_report)
    assert "Configuration Ablation Summary" in md
    assert "SW-01" in md
