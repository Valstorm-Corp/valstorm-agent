# Valstorm Agent Runtime — Multi-Axis Benchmark & Ablation Harness

This directory houses the scientific evaluation and ablation framework for the Valstorm Agent Runtime (`apps/agent-runtime`), commercially open-sourced at **[`https://github.com/Valstorm-Corp/valstorm-agent`](https://github.com/Valstorm-Corp/valstorm-agent)** under the **[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0)** (with commercial and enterprise deployments governed by the Valstorm Commercial License).

Rather than inventing a custom evaluation methodology from scratch, this harness combines four established industry standards:
1. **UK AISI `Inspect AI` Architecture**: Strictly decouples **Tasks** (`tasks/*.json`), **Solvers/Configs** (`BenchmarkConfig` + `adapters/`), and **Scorers** (`EnvironmentStateScorer` in `scorer.py`).
2. **Terminal-Bench 2.0 / Harbor Pattern**: Grades **post-execution environment state** (files created, regex matches, executable permission bits, pytest/shell exit codes) inside an isolated temporary directory so internal `vsagent` runs and external CLI agents (`claude`, `codex`, `opencode`, `hermes`, `oh-my-pi`) are evaluated identically.
3. **$\tau$-bench (`pass^k`) Consistency Metric**: Tracks both optimistic capability (`Pass@k`: at least 1 success in $k$ trials) and enterprise reliability (`pass^k`: probability that all $k$ trials succeed consistently, $p^k$).
4. **GAIA & SWE-bench Micro-Task Taxonomy**: Covers math/IO (`G-S01`), code generation (`G-S02`, `G-M02`), terminal/git scripting & permission guardrails (`G-S03`, `G-H03`), web intelligence (`G-M01`, `G-H02`), `patch_file` precision (`G-M03`), Red-Green-Refactor TDD loops (`G-H01`), and multi-agent parallel swarms (`SW-01`, `SW-02`).

---

## 1. Directory Structure

```text
apps/agent-runtime/benchmarks/
├── README.md                        # Architecture, ablation axes, & CLI reference
├── __init__.py
├── runner.py                        # CLI & programmatic matrix runner
├── scorer.py                        # EnvironmentStateScorer + pass^k / trajectory metrics
├── adapters/
│   ├── __init__.py
│   ├── base.py                      # BenchmarkConfig (10 factors) & ExecutionTrace models
│   ├── vsagent_adapter.py           # In-process vsagent runner (Mock & Live LLM modes)
│   └── external_cli_adapter.py      # Headless subprocess adapter for external CLIs
├── tasks/
│   ├── core_tasks.json              # 9 core tasks (G-S01..G-H03: Simple, Medium, High)
│   └── swarm_tasks.json             # Multi-deliverable parallel swarm tasks (SW-01, SW-02)
└── results/
    ├── latest_report.md             # Auto-generated Markdown scorecard
    └── latest_run.json              # Full JSON telemetry & per-run traces
```

---

## 2. The 6 Scientific Ablation Axes (Covering All 10 Factors)

To avoid confounding variables, each preset isolates **one independent variable** while holding the baseline configuration constant:

| Ablation Preset (`--ablation`) | Factors Covered | Configurations Compared | What It Measures |
| :--- | :--- | :--- | :--- |
| **`skills`** | **#1** With Skills vs.<br>**#2** Without Skills | `skills_on` vs. `skills_off` | Quantifies how procedural skills (`bash-scripting`, `test-driven-development`) prevent guardrail failures (e.g., `chmod +x` on `G-H03`) and reduce wasted turns. |
| **`react`** | **#3** With ReAct Loop vs.<br>**#4** Without ReAct Loop | `react_on` (`max_turns=15`) vs.<br>`react_off` (`max_turns=1` single-shot) | Quantifies **Self-Correction & Multi-Step Value**. Single-shot fails multi-step tool chains (`G-S01`, `G-M03`, `G-H01`, `G-H03`) where observation feedback is required. |
| **`tools`** | **#5** With Tools (`scoped` / `full`) vs.<br>**#6** Without Tools (`none`) | `tools_scoped` vs. `tools_full` vs. `tools_none` | Quantifies **Tool Schema Context Bloat** vs. capability. Live runs show `tools_scoped` cuts prompt token usage and cost by **~65%** (`10.9k` vs `31.2k` tokens on `G-S01`) compared to `tools_full`. |
| **`reasoning`** | **#7** Model $\times$ Thinking Level | `thinking_none`, `thinking_low`, `thinking_medium`, `thinking_high` | Measures the latency and token trade-off across reasoning effort levels (`none` $\rightarrow$ `high`) and model tiers (`gemini-flash-latest`, `gemini-3.1-pro-preview`, `deepseek-chat`, `claude-sonnet`). |
| **`swarm`** | **#10** Swarm / Parallel vs. Sequential | `seq_single` (`swarm_mode=False`) vs.<br>`swarm_parallel` (`background=True`, 3 workers) | Measures **Wall-Clock Compression** ($\max(D_i)$ vs. $\sum D_i$) and multi-agent token overhead on composite tasks (`SW-01`, `SW-02`). |
| **`harnesses`** | **#9** Cross-Agent CLI Comparison | `vsagent`, `claude`, `codex`, `opencode`, `hermes`, `oh-my-pi` | Runs external agent CLIs headlessly in an isolated temp folder and grades post-run filesystem state with the exact same `EnvironmentStateScorer`. |

---

## 3. Scorecard Metrics Captured

Every cell in the benchmark matrix records 5 orthogonal dimensions:
1. **Accuracy & Reliability**: `Pass Rate`, `Pass@k` (optimistic capability), and $\tau$-bench `pass^k` ($p^k$ consistency).
2. **Trajectory Efficiency**: Expected `tool_chain` vs. `actual_tools` called, `turns_taken`, and `tool_overhead_ratio`.
3. **Latency Breakdown**: Total wall-clock time (`duration_sec`), `llm_time_sec`, `tool_time_sec`, and `parallel_speedup` ($\frac{\sum D_{\text{subagent}}}{T_{\text{wall}}}$).
4. **Token & Cost Economics**: `prompt_tokens`, `completion_tokens`, `total_tokens`, and `estimated_cost_usd`.
5. **Guardrail Compliance**: Post-run verification of executable permissions (`require_executable: true` for shell scripts) and test exit codes.

---

## 4. Quick Start & CLI Commands

### Run Full 6-Axis Ablation Suite (Deterministic Mock Mode)
Executes all 11 tasks across all ablation configurations in ~3 seconds without consuming API quota:
```bash
uv run --directory apps/agent-runtime python -m benchmarks.runner --ablation all --mode mock
```

### Run Live LLM Ablations
Compare **Scoped Tools vs. Full Tool Registry vs. No Tools** on a live model:
```bash
uv run --directory apps/agent-runtime python -m benchmarks.runner --ablation tools --task G-S01 --mode live
```

Compare **With ReAct Loop vs. Without ReAct Loop (Single-Shot)** on a live model:
```bash
uv run --directory apps/agent-runtime python -m benchmarks.runner --ablation react --task G-S01 --mode live
```

Compare **Thinking Levels (`none`, `low`, `medium`, `high`)** on a specific model:
```bash
uv run --directory apps/agent-runtime python -m benchmarks.runner \
  --ablation reasoning \
  --model gemini-flash-latest \
  --task G-M03 \
  --mode live
```

Compare **Sequential Single-Agent vs. Parallel Subagent Swarm (`background=True`)**:
```bash
uv run --directory apps/agent-runtime python -m benchmarks.runner --ablation swarm --suite swarm --mode mock
# Or standalone concurrency timing benchmark:
uv run --directory apps/agent-runtime python scripts/test_concurrent_subagents.py --mode live --workers 2
```

Run **Multi-Trial Consistency (`pass^k` with $k=3$)**:
```bash
uv run --directory apps/agent-runtime python -m benchmarks.runner --ablation skills --trials 3 --mode mock
```

---

## 5. Sample Empirical Findings (Live `gemini-flash-latest` on `G-S01`)

| Ablation Axis | Configuration | Pass Rate | Turns | Tools Called | Latency | Total Tokens | Est. Cost ($) |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| **ReAct Loop** | `react_on` (`max_turns=15`) | **100%** | 3 | `calculator, write_file` | 7.16s | 10,844 | $0.00169 |
| **ReAct Loop** | `react_off` (`max_turns=1`) | **0%** | 1 | `calculator` *(halted before write)* | 2.06s | 3,413 | $0.00052 |
| **Tool Surface** | `tools_scoped` (2 tools) | **100%** | 3 | `calculator, write_file` | 6.58s | **10,976** | **$0.00170** |
| **Tool Surface** | `tools_full` (30+ tools) | **100%** | 4 | `calculator, write_file, read_file` | 7.55s | **31,200** | **$0.00477** |
| **Tool Surface** | `tools_none` (0 tools) | **0%** | 1 | *(none)* | 10.96s | 3,457 | $0.00061 |
