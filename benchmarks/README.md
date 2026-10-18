# Valstorm Agent Runtime — Scientific Benchmark & Ablation Harness (`v4.1.0`)

This package implements a reproducible, **9-axis scientific evaluation and ablation suite** for `apps/agent-runtime` (`vsagent`), combining four industry evaluation methodologies (**UK AISI Inspect AI**, **Terminal-Bench 2.0 / Harbor**, **$\tau$-bench (`pass^k`)**, and **GAIA / SWE-bench Verified**) with **System-1 Calibrated Decision Model** benchmarks (`core/decision_model.py`, `core/message_autotuner.py`, `core/ui_navigator.py`, `core/thinking_router.py`).

---

## 1. The 9 Isolated Ablation Axes (`--ablation`)

| Ablation Preset (`--ablation`) | Factors Tested | Configurations Compared | What It Measures |
| :--- | :--- | :--- | :--- |
| **`skills`** | **#1** With Skills vs. **#2** Without Skills | `skills_on` vs. `skills_off` | Impact of injecting domain `SKILL.md` SOPs (`bash-scripting`, `test-driven-development`) on operational guardrails (`chmod +x` on `G-H03`). |
| **`react`** | **#3** With ReAct Loop vs. **#4** Without ReAct Loop | `react_on` (`max_turns=15`) vs. `react_off` (`max_turns=1`) | Pass-rate lift (+44.4%) of iterative tool observation & TDD Red-Green-Refactor loops over single-shot generation. |
| **`tools`** | **#5** Scoped / Full Tools vs. **#6** No Tools | `tools_scoped` vs. `tools_full` vs. `tools_none` | Token bloat (-64.8% when scoped) and cost reduction (2.8x cheaper) of profile-scoped tool whitelists. |
| **`reasoning`** | **#7 & #8** Model Tier $\times$ Static Thinking Effort | `thinking_none`, `thinking_low`, `thinking_medium`, `thinking_high` | Latency and token cost scaling across static reasoning budgets (`none` $\rightarrow$ `high`). |
| **`adaptive_thinking`** *(v4.0)* | **#11** System-1 Dynamic Thinking Router vs. Static Extremes | `thinking_adaptive_s1` vs. `thinking_high_static` vs. `thinking_none_static` | Uses `SystemOneThinkingRouter` (`<3ms`, zero KV prefix cache invalidation) to dynamically pick `none`/`low`/`medium`/`high` per turn—achieving **100% pass rate (9/9)** while running **2.21x faster** on simple tasks (`5.64s` vs `12.45s` live). |
| **`message_autotune`** *(New in v4.1)* | **#13** Two-Layer `ai_chat_message` Auto-Tuner (Tier + Effort) vs. Static Tiers | `autotune_s1_full` vs. `static_thinker_high` vs. `static_reasoner_high` vs. `static_worker_none` | Selects `WORKER` (`flash-lite`), `REASONER` (`flash`), or `THINKER` (`pro`) at `iteration == 0` per `ai_chat_message` and locks the model mid-turn while stepping `reasoning_effort` down/up (**5.01x faster & 16.6x cheaper than Thinker on `G-S01` live; 7.94x cheaper on `G-M03` live; 100% pass rate**). |
| **`swarm`** | **#10** Parallel Swarm vs. Sequential Single Agent | `seq_single` vs. `swarm_parallel` (`background=True`, 3 workers) | Wall-clock speedup (**1.53x live**, **2.04x–3.00x deterministic**) from non-blocking `asyncio.create_task` subagent delegation. |
| **`decision_model` (`ui_nav`)** *(v4.0)* | **#12** System-1 Closed-Set UI/Canvas Navigator vs. System-2 LLM ReAct | `system_one_ui_nav` (`use_decision_model=True`) vs. `llm_react_ui_nav` (`use_decision_model=False`) | Compares single-pass System-1 UI & 2D/3D Crafting Canvas navigation (`UI-01..UI-04`, including `storytold/artcraft` workflows) against step-by-step LLM ReAct turns (**3.34x faster live on `UI-01`**, **1.65x faster live on `UI-02`**, **7.93x faster mock**, **0% selector hallucination**). |
| **`harnesses`** | **#9** Cross-Agent Headless CLI Comparison | `vsagent`, `claude`, `codex`, `opencode`, `hermes`, `oh-my-pi` | Head-to-head execution inside identical isolated `tmp_path` workspaces graded by `EnvironmentStateScorer`. |

---

## 2. Quick-Start Commands

```bash
# 1. Run the complete 9-axis ablation matrix (15 tasks, 22 configs, 171 runs)
uv run --directory apps/agent-runtime python -m benchmarks.runner --ablation all --mode mock

# 2. Run the System-1 ai_chat_message Auto-Tuner ablation (Tier + Intra-Turn Step-Down)
uv run --directory apps/agent-runtime python -m benchmarks.runner --ablation message_autotune --mode mock
uv run --directory apps/agent-runtime python -m benchmarks.runner --ablation message_autotune --task G-S01 --mode live
uv run --directory apps/agent-runtime python -m benchmarks.runner --ablation message_autotune --task G-M03 --mode live

# 3. Run the System-1 UI & ArtCraft Canvas Navigator ablation (UI-01 .. UI-04)
uv run --directory apps/agent-runtime python -m benchmarks.runner --ablation decision_model --mode mock
uv run --directory apps/agent-runtime python -m benchmarks.runner --ablation decision_model --task UI-01 --mode live

# 4. Run the automated 40-test pytest verification suite
uv run --directory apps/agent-runtime pytest tests/test_message_autotuner.py tests/test_system_one_ui_navigator.py tests/test_benchmark_harness.py tests/test_profile_and_delegation.py -v
```
