# Valstorm Agent Benchmark Report (`--ablation all`)

- **Timestamp (UTC):** `2026-10-18T19:48:11.038195+00:00`
- **Execution Mode:** `mock`
- **Tasks Evaluated:** `15` | **Trials per Task (`k`):** `1`

## 1. Configuration Ablation Summary

| Config ID | Configuration Label | Pass Rate | `pass^k` | Avg Turns | Avg Latency | Avg Tokens | Est. Cost ($) | Parallel Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `skills_on` | 1. With Procedural Skills | **100.0% (9/9)** | `100.0%` | 3.0 | 0.27s | 945 | $0.00171 | 1.00x |
| `skills_off` | 2. Without Procedural Skills | **88.9% (8/9)** | `88.9%` | 3.0 | 0.23s | 705 | $0.00139 | 1.00x |
| `react_on` | 3. With ReAct Loop (Iterative Multi-Step, max_turns=15) | **100.0% (9/9)** | `100.0%` | 3.0 | 0.30s | 972 | $0.00186 | 1.00x |
| `react_off` | 4. Without ReAct Loop (Single-Shot, max_turns=1) | **55.6% (5/9)** | `55.6%` | 1.0 | 0.08s | 323 | $0.00062 | 1.00x |
| `tools_scoped` | 5a. With Scoped Tools (Task Whitelist) | **100.0% (9/9)** | `100.0%` | 3.0 | 0.30s | 972 | $0.00186 | 1.00x |
| `tools_full` | 5b. With Full Tool Registry (All Runtime Tools) | **100.0% (9/9)** | `100.0%` | 3.0 | 0.30s | 3885 | $0.00579 | 1.00x |
| `tools_none` | 6. Without Tools (Pure Text Generation) | **0.0% (0/9)** | `0.0%` | 1.0 | 0.08s | 245 | $0.00051 | 1.00x |
| `thinking_none` | 7. gemini-flash-latest (Thinking: NONE) | **100.0% (9/9)** | `100.0%` | 3.0 | 0.18s | 903 | $0.00149 | 1.00x |
| `thinking_low` | 7. gemini-flash-latest (Thinking: LOW) | **100.0% (9/9)** | `100.0%` | 3.0 | 0.25s | 945 | $0.00171 | 1.00x |
| `thinking_medium` | 7. gemini-flash-latest (Thinking: MEDIUM) | **100.0% (9/9)** | `100.0%` | 3.0 | 0.30s | 972 | $0.00186 | 1.00x |
| `thinking_high` | 7. gemini-flash-latest (Thinking: HIGH) | **100.0% (9/9)** | `100.0%` | 3.0 | 0.42s | 1038 | $0.00221 | 1.00x |
| `thinking_adaptive_s1` | 8a. System-1 Adaptive Thinking Router (Dynamic none/low/medium/high) | **100.0% (9/9)** | `100.0%` | 3.0 | 0.30s | 727 | $0.00151 | 1.00x |
| `thinking_high_static` | 8b. Static HIGH Thinking (Always High without Skills) | **100.0% (9/9)** | `100.0%` | 3.0 | 0.42s | 798 | $0.00189 | 1.00x |
| `thinking_none_static` | 8c. Static NONE Thinking (Always None without Skills) | **88.9% (8/9)** | `88.9%` | 3.0 | 0.16s | 663 | $0.00116 | 1.00x |
| `seq_single` | 10a. Single Agent (Sequential Execution) | **100.0% (2/2)** | `100.0%` | 4.0 | 1.08s | 1260 | $0.00054 | 1.00x |
| `swarm_parallel` | 10b. Parallel Subagent Swarm (background=True, 3 Workers) | **100.0% (2/2)** | `100.0%` | 3.0 | 0.67s | 7840 | $0.00272 | 2.09x |
| `system_one_ui_nav` | 11a. System-1 Decision Model UI/Canvas Navigator (Parallel Closed-Set) | **100.0% (4/4)** | `100.0%` | 2.0 | 0.15s | 542 | $0.00046 | 1.00x |
| `llm_react_ui_nav` | 11b. System-2 Full LLM ReAct UI/Canvas Navigation (Step-by-Step) | **100.0% (4/4)** | `100.0%` | 4.5 | 1.18s | 5108 | $0.00456 | 1.00x |

## 2. Granular Task-by-Task Scorecard

| Task ID | Complexity | Config ID | Outcome | Score | Turns | Tools Called | Duration | Failure / Notes |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- | :---: | :--- |
| `G-S01` | Simple | `skills_on` | ✅ PASS | 1.00 | 3 | `calculator, write_file` | 0.20s | All checks verified |
| `G-S02` | Simple | `skills_on` | ✅ PASS | 1.00 | 3 | `write_file, read_file` | 0.20s | All checks verified |
| `G-S03` | Simple | `skills_on` | ✅ PASS | 1.00 | 3 | `write_file, write_file, search_files` | 0.22s | All checks verified |
| `G-M01` | Medium | `skills_on` | ✅ PASS | 1.00 | 2 | `write_file` | 0.13s | All checks verified |
| `G-M02` | Medium | `skills_on` | ✅ PASS | 1.00 | 3 | `write_file, terminal_exec` | 0.22s | All checks verified |
| `G-M03` | Medium | `skills_on` | ✅ PASS | 1.00 | 4 | `write_file, patch_file, read_file` | 0.26s | All checks verified |
| `G-H01` | High | `skills_on` | ✅ PASS | 1.00 | 5 | `write_file, write_file, terminal_exec, write_file, terminal_exec` | 0.66s | All checks verified |
| `G-H02` | High | `skills_on` | ✅ PASS | 1.00 | 1 | `*(none)*` | 0.07s | All checks verified |
| `G-H03` | High | `skills_on` | ✅ PASS | 1.00 | 3 | `write_file, terminal_exec` | 0.47s | All checks verified |
| `G-S01` | Simple | `skills_off` | ✅ PASS | 1.00 | 3 | `calculator, write_file` | 0.20s | All checks verified |
| `G-S02` | Simple | `skills_off` | ✅ PASS | 1.00 | 3 | `write_file, read_file` | 0.20s | All checks verified |
| `G-S03` | Simple | `skills_off` | ✅ PASS | 1.00 | 3 | `write_file, write_file, search_files` | 0.21s | All checks verified |
| `G-M01` | Medium | `skills_off` | ✅ PASS | 1.00 | 2 | `write_file` | 0.13s | All checks verified |
| `G-M02` | Medium | `skills_off` | ✅ PASS | 1.00 | 3 | `write_file, terminal_exec` | 0.22s | All checks verified |
| `G-M03` | Medium | `skills_off` | ✅ PASS | 1.00 | 4 | `write_file, patch_file, read_file` | 0.26s | All checks verified |
| `G-H01` | High | `skills_off` | ✅ PASS | 1.00 | 5 | `write_file, write_file, terminal_exec, write_file, terminal_exec` | 0.61s | All checks verified |
| `G-H02` | High | `skills_off` | ✅ PASS | 1.00 | 1 | `*(none)*` | 0.07s | All checks verified |
| `G-H03` | High | `skills_off` | ❌ FAIL | 0.60 | 3 | `write_file, terminal_exec` | 0.22s | missing_executable_chmod:setup_env.sh; verify_command_failed(exit |
| `G-S01` | Simple | `react_on` | ✅ PASS | 1.00 | 3 | `calculator, write_file` | 0.24s | All checks verified |
| `G-S02` | Simple | `react_on` | ✅ PASS | 1.00 | 3 | `write_file, read_file` | 0.24s | All checks verified |
| `G-S03` | Simple | `react_on` | ✅ PASS | 1.00 | 3 | `write_file, write_file, search_files` | 0.26s | All checks verified |
| `G-M01` | Medium | `react_on` | ✅ PASS | 1.00 | 2 | `write_file` | 0.16s | All checks verified |
| `G-M02` | Medium | `react_on` | ✅ PASS | 1.00 | 3 | `write_file, terminal_exec` | 0.27s | All checks verified |
| `G-M03` | Medium | `react_on` | ✅ PASS | 1.00 | 4 | `write_file, patch_file, read_file` | 0.33s | All checks verified |
| `G-H01` | High | `react_on` | ✅ PASS | 1.00 | 5 | `write_file, write_file, terminal_exec, write_file, terminal_exec` | 0.71s | All checks verified |
| `G-H02` | High | `react_on` | ✅ PASS | 1.00 | 1 | `*(none)*` | 0.08s | All checks verified |
| `G-H03` | High | `react_on` | ✅ PASS | 1.00 | 3 | `write_file, terminal_exec` | 0.43s | All checks verified |
| `G-S01` | Simple | `react_off` | ❌ FAIL | 0.00 | 1 | `calculator` | 0.08s | missing_target_file:calculation_result.txt |
| `G-S02` | Simple | `react_off` | ✅ PASS | 1.00 | 1 | `write_file` | 0.08s | All checks verified |
| `G-S03` | Simple | `react_off` | ✅ PASS | 1.00 | 1 | `write_file, write_file` | 0.08s | All checks verified |
| `G-M01` | Medium | `react_off` | ✅ PASS | 1.00 | 1 | `write_file` | 0.08s | All checks verified |
| `G-M02` | Medium | `react_off` | ✅ PASS | 1.00 | 1 | `write_file` | 0.08s | All checks verified |
| `G-M03` | Medium | `react_off` | ❌ FAIL | 0.50 | 1 | `write_file` | 0.08s | content_regex_mismatch:config.json (expected /http://api\.new\.co |
| `G-H01` | High | `react_off` | ❌ FAIL | 0.67 | 1 | `write_file, write_file` | 0.08s | verify_command_failed(exit=1): ============================= test |
| `G-H02` | High | `react_off` | ✅ PASS | 1.00 | 1 | `*(none)*` | 0.08s | All checks verified |
| `G-H03` | High | `react_off` | ❌ FAIL | 0.40 | 1 | `write_file` | 0.08s | missing_file:temp_proj/.gitignore; missing_executable_chmod:setup |
| `G-S01` | Simple | `tools_scoped` | ✅ PASS | 1.00 | 3 | `calculator, write_file` | 0.24s | All checks verified |
| `G-S02` | Simple | `tools_scoped` | ✅ PASS | 1.00 | 3 | `write_file, read_file` | 0.24s | All checks verified |
| `G-S03` | Simple | `tools_scoped` | ✅ PASS | 1.00 | 3 | `write_file, write_file, search_files` | 0.26s | All checks verified |
| `G-M01` | Medium | `tools_scoped` | ✅ PASS | 1.00 | 2 | `write_file` | 0.16s | All checks verified |
| `G-M02` | Medium | `tools_scoped` | ✅ PASS | 1.00 | 3 | `write_file, terminal_exec` | 0.27s | All checks verified |
| `G-M03` | Medium | `tools_scoped` | ✅ PASS | 1.00 | 4 | `write_file, patch_file, read_file` | 0.33s | All checks verified |
| `G-H01` | High | `tools_scoped` | ✅ PASS | 1.00 | 5 | `write_file, write_file, terminal_exec, write_file, terminal_exec` | 0.70s | All checks verified |
| `G-H02` | High | `tools_scoped` | ✅ PASS | 1.00 | 1 | `*(none)*` | 0.08s | All checks verified |
| `G-H03` | High | `tools_scoped` | ✅ PASS | 1.00 | 3 | `write_file, terminal_exec` | 0.40s | All checks verified |
| `G-S01` | Simple | `tools_full` | ✅ PASS | 1.00 | 3 | `calculator, write_file` | 0.24s | All checks verified |
| `G-S02` | Simple | `tools_full` | ✅ PASS | 1.00 | 3 | `write_file, read_file` | 0.24s | All checks verified |
| `G-S03` | Simple | `tools_full` | ✅ PASS | 1.00 | 3 | `write_file, write_file, search_files` | 0.26s | All checks verified |
| `G-M01` | Medium | `tools_full` | ✅ PASS | 1.00 | 2 | `write_file` | 0.16s | All checks verified |
| `G-M02` | Medium | `tools_full` | ✅ PASS | 1.00 | 3 | `write_file, terminal_exec` | 0.27s | All checks verified |
| `G-M03` | Medium | `tools_full` | ✅ PASS | 1.00 | 4 | `write_file, patch_file, read_file` | 0.33s | All checks verified |
| `G-H01` | High | `tools_full` | ✅ PASS | 1.00 | 5 | `write_file, write_file, terminal_exec, write_file, terminal_exec` | 0.69s | All checks verified |
| `G-H02` | High | `tools_full` | ✅ PASS | 1.00 | 1 | `*(none)*` | 0.08s | All checks verified |
| `G-H03` | High | `tools_full` | ✅ PASS | 1.00 | 3 | `write_file, terminal_exec` | 0.42s | All checks verified |
| `G-S01` | Simple | `tools_none` | ❌ FAIL | 0.00 | 1 | `*(none)*` | 0.08s | missing_target_file:calculation_result.txt |
| `G-S02` | Simple | `tools_none` | ❌ FAIL | 0.00 | 1 | `*(none)*` | 0.08s | missing_target_file:index.js |
| `G-S03` | Simple | `tools_none` | ❌ FAIL | 0.00 | 1 | `*(none)*` | 0.08s | missing_file:src/main.py; missing_file:test/test_main.py; verify_ |
| `G-M01` | Medium | `tools_none` | ❌ FAIL | 0.00 | 1 | `*(none)*` | 0.08s | missing_target_file:documentation_source.md |
| `G-M02` | Medium | `tools_none` | ❌ FAIL | 0.00 | 1 | `*(none)*` | 0.08s | missing_target_file:sum.py; verify_command_failed(exit=2): python |
| `G-M03` | Medium | `tools_none` | ❌ FAIL | 0.00 | 1 | `*(none)*` | 0.08s | missing_target_file:config.json |
| `G-H01` | High | `tools_none` | ❌ FAIL | 0.00 | 1 | `*(none)*` | 0.08s | missing_file:src/utils.py; missing_file:test/test_reverse.py; ver |
| `G-H02` | High | `tools_none` | ❌ FAIL | 0.00 | 1 | `*(none)*` | 0.08s | output_regex_mismatch (expected /(?i)(unchanged|changed|diff|iden |
| `G-H03` | High | `tools_none` | ❌ FAIL | 0.00 | 1 | `*(none)*` | 0.08s | missing_file:setup_env.sh; missing_file:temp_proj/.gitignore; mis |
| `G-S01` | Simple | `thinking_none` | ✅ PASS | 1.00 | 3 | `calculator, write_file` | 0.12s | All checks verified |
| `G-S02` | Simple | `thinking_none` | ✅ PASS | 1.00 | 3 | `write_file, read_file` | 0.12s | All checks verified |
| `G-S03` | Simple | `thinking_none` | ✅ PASS | 1.00 | 3 | `write_file, write_file, search_files` | 0.13s | All checks verified |
| `G-M01` | Medium | `thinking_none` | ✅ PASS | 1.00 | 2 | `write_file` | 0.08s | All checks verified |
| `G-M02` | Medium | `thinking_none` | ✅ PASS | 1.00 | 3 | `write_file, terminal_exec` | 0.15s | All checks verified |
| `G-M03` | Medium | `thinking_none` | ✅ PASS | 1.00 | 4 | `write_file, patch_file, read_file` | 0.17s | All checks verified |
| `G-H01` | High | `thinking_none` | ✅ PASS | 1.00 | 5 | `write_file, write_file, terminal_exec, write_file, terminal_exec` | 0.50s | All checks verified |
| `G-H02` | High | `thinking_none` | ✅ PASS | 1.00 | 1 | `*(none)*` | 0.04s | All checks verified |
| `G-H03` | High | `thinking_none` | ✅ PASS | 1.00 | 3 | `write_file, terminal_exec` | 0.29s | All checks verified |
| `G-S01` | Simple | `thinking_low` | ✅ PASS | 1.00 | 3 | `calculator, write_file` | 0.20s | All checks verified |
| `G-S02` | Simple | `thinking_low` | ✅ PASS | 1.00 | 3 | `write_file, read_file` | 0.20s | All checks verified |
| `G-S03` | Simple | `thinking_low` | ✅ PASS | 1.00 | 3 | `write_file, write_file, search_files` | 0.21s | All checks verified |
| `G-M01` | Medium | `thinking_low` | ✅ PASS | 1.00 | 2 | `write_file` | 0.13s | All checks verified |
| `G-M02` | Medium | `thinking_low` | ✅ PASS | 1.00 | 3 | `write_file, terminal_exec` | 0.22s | All checks verified |
| `G-M03` | Medium | `thinking_low` | ✅ PASS | 1.00 | 4 | `write_file, patch_file, read_file` | 0.26s | All checks verified |
| `G-H01` | High | `thinking_low` | ✅ PASS | 1.00 | 5 | `write_file, write_file, terminal_exec, write_file, terminal_exec` | 0.61s | All checks verified |
| `G-H02` | High | `thinking_low` | ✅ PASS | 1.00 | 1 | `*(none)*` | 0.07s | All checks verified |
| `G-H03` | High | `thinking_low` | ✅ PASS | 1.00 | 3 | `write_file, terminal_exec` | 0.38s | All checks verified |
| `G-S01` | Simple | `thinking_medium` | ✅ PASS | 1.00 | 3 | `calculator, write_file` | 0.24s | All checks verified |
| `G-S02` | Simple | `thinking_medium` | ✅ PASS | 1.00 | 3 | `write_file, read_file` | 0.24s | All checks verified |
| `G-S03` | Simple | `thinking_medium` | ✅ PASS | 1.00 | 3 | `write_file, write_file, search_files` | 0.25s | All checks verified |
| `G-M01` | Medium | `thinking_medium` | ✅ PASS | 1.00 | 2 | `write_file` | 0.16s | All checks verified |
| `G-M02` | Medium | `thinking_medium` | ✅ PASS | 1.00 | 3 | `write_file, terminal_exec` | 0.27s | All checks verified |
| `G-M03` | Medium | `thinking_medium` | ✅ PASS | 1.00 | 4 | `write_file, patch_file, read_file` | 0.33s | All checks verified |
| `G-H01` | High | `thinking_medium` | ✅ PASS | 1.00 | 5 | `write_file, write_file, terminal_exec, write_file, terminal_exec` | 0.72s | All checks verified |
| `G-H02` | High | `thinking_medium` | ✅ PASS | 1.00 | 1 | `*(none)*` | 0.08s | All checks verified |
| `G-H03` | High | `thinking_medium` | ✅ PASS | 1.00 | 3 | `write_file, terminal_exec` | 0.41s | All checks verified |
| `G-S01` | Simple | `thinking_high` | ✅ PASS | 1.00 | 3 | `calculator, write_file` | 0.36s | All checks verified |
| `G-S02` | Simple | `thinking_high` | ✅ PASS | 1.00 | 3 | `write_file, read_file` | 0.36s | All checks verified |
| `G-S03` | Simple | `thinking_high` | ✅ PASS | 1.00 | 3 | `write_file, write_file, search_files` | 0.37s | All checks verified |
| `G-M01` | Medium | `thinking_high` | ✅ PASS | 1.00 | 2 | `write_file` | 0.24s | All checks verified |
| `G-M02` | Medium | `thinking_high` | ✅ PASS | 1.00 | 3 | `write_file, terminal_exec` | 0.39s | All checks verified |
| `G-M03` | Medium | `thinking_high` | ✅ PASS | 1.00 | 4 | `write_file, patch_file, read_file` | 0.49s | All checks verified |
| `G-H01` | High | `thinking_high` | ✅ PASS | 1.00 | 5 | `write_file, write_file, terminal_exec, write_file, terminal_exec` | 0.92s | All checks verified |
| `G-H02` | High | `thinking_high` | ✅ PASS | 1.00 | 1 | `*(none)*` | 0.12s | All checks verified |
| `G-H03` | High | `thinking_high` | ✅ PASS | 1.00 | 3 | `write_file, terminal_exec` | 0.53s | All checks verified |
| `G-S01` | Simple | `thinking_adaptive_s1` | ✅ PASS | 1.00 | 3 | `calculator, write_file` | 0.13s | All checks verified |
| `G-S02` | Simple | `thinking_adaptive_s1` | ✅ PASS | 1.00 | 3 | `write_file, read_file` | 0.22s | All checks verified |
| `G-S03` | Simple | `thinking_adaptive_s1` | ✅ PASS | 1.00 | 3 | `write_file, write_file, search_files` | 0.23s | All checks verified |
| `G-M01` | Medium | `thinking_adaptive_s1` | ✅ PASS | 1.00 | 2 | `write_file` | 0.15s | All checks verified |
| `G-M02` | Medium | `thinking_adaptive_s1` | ✅ PASS | 1.00 | 3 | `write_file, terminal_exec` | 0.24s | All checks verified |
| `G-M03` | Medium | `thinking_adaptive_s1` | ✅ PASS | 1.00 | 4 | `write_file, patch_file, read_file` | 0.28s | All checks verified |
| `G-H01` | High | `thinking_adaptive_s1` | ✅ PASS | 1.00 | 5 | `write_file, write_file, terminal_exec, write_file, terminal_exec` | 0.79s | All checks verified |
| `G-H02` | High | `thinking_adaptive_s1` | ✅ PASS | 1.00 | 1 | `*(none)*` | 0.08s | All checks verified |
| `G-H03` | High | `thinking_adaptive_s1` | ✅ PASS | 1.00 | 3 | `write_file, terminal_exec` | 0.55s | All checks verified |
| `G-S01` | Simple | `thinking_high_static` | ✅ PASS | 1.00 | 3 | `calculator, write_file` | 0.36s | All checks verified |
| `G-S02` | Simple | `thinking_high_static` | ✅ PASS | 1.00 | 3 | `write_file, read_file` | 0.36s | All checks verified |
| `G-S03` | Simple | `thinking_high_static` | ✅ PASS | 1.00 | 3 | `write_file, write_file, search_files` | 0.37s | All checks verified |
| `G-M01` | Medium | `thinking_high_static` | ✅ PASS | 1.00 | 2 | `write_file` | 0.24s | All checks verified |
| `G-M02` | Medium | `thinking_high_static` | ✅ PASS | 1.00 | 3 | `write_file, terminal_exec` | 0.39s | All checks verified |
| `G-M03` | Medium | `thinking_high_static` | ✅ PASS | 1.00 | 4 | `write_file, patch_file, read_file` | 0.49s | All checks verified |
| `G-H01` | High | `thinking_high_static` | ✅ PASS | 1.00 | 5 | `write_file, write_file, terminal_exec, write_file, terminal_exec` | 0.89s | All checks verified |
| `G-H02` | High | `thinking_high_static` | ✅ PASS | 1.00 | 1 | `*(none)*` | 0.12s | All checks verified |
| `G-H03` | High | `thinking_high_static` | ✅ PASS | 1.00 | 3 | `write_file, terminal_exec` | 0.54s | All checks verified |
| `G-S01` | Simple | `thinking_none_static` | ✅ PASS | 1.00 | 3 | `calculator, write_file` | 0.12s | All checks verified |
| `G-S02` | Simple | `thinking_none_static` | ✅ PASS | 1.00 | 3 | `write_file, read_file` | 0.12s | All checks verified |
| `G-S03` | Simple | `thinking_none_static` | ✅ PASS | 1.00 | 3 | `write_file, write_file, search_files` | 0.14s | All checks verified |
| `G-M01` | Medium | `thinking_none_static` | ✅ PASS | 1.00 | 2 | `write_file` | 0.08s | All checks verified |
| `G-M02` | Medium | `thinking_none_static` | ✅ PASS | 1.00 | 3 | `write_file, terminal_exec` | 0.15s | All checks verified |
| `G-M03` | Medium | `thinking_none_static` | ✅ PASS | 1.00 | 4 | `write_file, patch_file, read_file` | 0.17s | All checks verified |
| `G-H01` | High | `thinking_none_static` | ✅ PASS | 1.00 | 5 | `write_file, write_file, terminal_exec, write_file, terminal_exec` | 0.48s | All checks verified |
| `G-H02` | High | `thinking_none_static` | ✅ PASS | 1.00 | 1 | `*(none)*` | 0.04s | All checks verified |
| `G-H03` | High | `thinking_none_static` | ❌ FAIL | 0.60 | 3 | `write_file, terminal_exec` | 0.15s | missing_executable_chmod:setup_env.sh; verify_command_failed(exit |
| `SW-01` | High | `seq_single` | ✅ PASS | 1.00 | 4 | `write_file, write_file, write_file` | 1.08s | All checks verified |
| `SW-02` | High | `seq_single` | ✅ PASS | 1.00 | 4 | `write_file, write_file, write_file` | 1.08s | All checks verified |
| `SW-01` | High | `swarm_parallel` | ✅ PASS | 1.00 | 3 | `delegate_task, delegate_task, delegate_task, delegate_task` | 0.75s | All checks verified |
| `SW-02` | High | `swarm_parallel` | ✅ PASS | 1.00 | 3 | `delegate_task, delegate_task, delegate_task, delegate_task` | 0.60s | All checks verified |
| `UI-01` | Medium | `system_one_ui_nav` | ✅ PASS | 1.00 | 2 | `ui_navigate` | 0.15s | All checks verified |
| `UI-02` | High | `system_one_ui_nav` | ✅ PASS | 1.00 | 2 | `ui_navigate` | 0.15s | All checks verified |
| `UI-03` | Medium | `system_one_ui_nav` | ✅ PASS | 1.00 | 2 | `ui_navigate` | 0.14s | All checks verified |
| `UI-04` | High | `system_one_ui_nav` | ✅ PASS | 1.00 | 2 | `ui_navigate` | 0.15s | All checks verified |
| `UI-01` | Medium | `llm_react_ui_nav` | ✅ PASS | 1.00 | 4 | `write_file, write_file, write_file` | 1.05s | All checks verified |
| `UI-02` | High | `llm_react_ui_nav` | ✅ PASS | 1.00 | 5 | `write_file, write_file, write_file, write_file` | 1.31s | All checks verified |
| `UI-03` | Medium | `llm_react_ui_nav` | ✅ PASS | 1.00 | 4 | `write_file, write_file, write_file` | 1.05s | All checks verified |
| `UI-04` | High | `llm_react_ui_nav` | ✅ PASS | 1.00 | 5 | `write_file, write_file, write_file, write_file` | 1.31s | All checks verified |
