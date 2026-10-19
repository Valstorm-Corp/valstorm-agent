# Valstorm Agent Benchmark Report (`--ablation swarm`)

- **Timestamp (UTC):** `2026-10-19T01:07:34.438056+00:00`
- **Execution Mode:** `mock`
- **Tasks Evaluated:** `2` | **Trials per Task (`k`):** `1`

## 1. Configuration Ablation Summary

| Config ID | Configuration Label | Pass Rate | `pass^k` | Avg Turns | Avg Latency | Avg Tokens | Est. Cost ($) | Parallel Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `seq_single` | 10a. Single Agent (Sequential Execution) | **100.0% (2/2)** | `100.0%` | 4.0 | 1.09s | 1260 | $0.00054 | 1.00x |
| `swarm_parallel` | 10b. Parallel Subagent Swarm (background=True, Static Model) | **100.0% (2/2)** | `100.0%` | 3.0 | 0.71s | 8085 | $0.00279 | 1.93x |
| `swarm_autotuned` | 10c. Parallel Subagent Swarm + System-1 Subagent Auto-Tune (background=True) | **100.0% (2/2)** | `100.0%` | 3.0 | 0.47s | 7906 | $0.00188 | 1.94x |

## 2. Granular Task-by-Task Scorecard

| Task ID | Complexity | Config ID | Outcome | Score | Turns | Tools Called | Duration | Failure / Notes |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- | :---: | :--- |
| `SW-01` | High | `seq_single` | ✅ PASS | 1.00 | 4 | `write_file, write_file, write_file` | 1.09s | All checks verified |
| `SW-02` | High | `seq_single` | ✅ PASS | 1.00 | 4 | `write_file, write_file, write_file` | 1.08s | All checks verified |
| `SW-01` | High | `swarm_parallel` | ✅ PASS | 1.00 | 3 | `delegate_task, delegate_task, delegate_task, delegate_task` | 0.80s | All checks verified |
| `SW-02` | High | `swarm_parallel` | ✅ PASS | 1.00 | 3 | `delegate_task, delegate_task, delegate_task, delegate_task` | 0.62s | All checks verified |
| `SW-01` | High | `swarm_autotuned` | ✅ PASS | 1.00 | 3 | `delegate_task, delegate_task, delegate_task, delegate_task` | 0.53s | All checks verified |
| `SW-02` | High | `swarm_autotuned` | ✅ PASS | 1.00 | 3 | `delegate_task, delegate_task, delegate_task, delegate_task` | 0.40s | All checks verified |
