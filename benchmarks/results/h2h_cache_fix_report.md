# Valstorm Agent Benchmark Report (`--ablation head_to_head`)

- **Timestamp (UTC):** `2026-11-02T19:05:17.249270+00:00`
- **Execution Mode:** `live`
- **Tasks Evaluated:** `5` | **Trials per Task (`k`):** `3`

## 1. Configuration Ablation Summary

| Config ID | Configuration Label | Pass Rate | `pass^k` | Avg Turns | Avg Latency | Avg Tokens | Cached % | Est. Cost ($) | $ / Pass | Parallel Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `h2h_a_vsagent_gemini` | A. vsagent + Gemini auto-tune (`gemini-flash-latest` base, skills, scoped tools) | **100.0% (15/15)** | `100.0%` | 4.4 | 12.66s | 19204 | 31% | $0.15799 | $0.01053 | 1.00x |
| `h2h_b_vsagent_claude` | B. vsagent + Claude (`claude-opus-5-5`, skills, scoped tools) | **100.0% (15/15)** | `100.0%` | 4.07 | 14.31s | 32135 | 31% | $1.97062 | $0.13137 | 1.00x |

## 2. Granular Task-by-Task Scorecard

| Task ID | Complexity | Config ID | Outcome | Score | Turns | Tools Called | Duration | Failure / Notes |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- | :---: | :--- |
| `G-M03` | Medium | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 4 | `write_file, patch_file, read_file` | 10.95s | All checks verified |
| `G-M03` | Medium | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 4 | `write_file, patch_file, read_file` | 8.92s | All checks verified |
| `G-M03` | Medium | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 4 | `write_file, patch_file, read_file` | 31.97s | All checks verified |
| `G-H01` | High | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 7 | `write_file, write_file, terminal_exec, terminal_exec, write_file, terminal_exec` | 19.99s | All checks verified |
| `G-H01` | High | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 7 | `write_file, write_file, terminal_exec, terminal_exec, write_file, terminal_exec` | 19.50s | All checks verified |
| `G-H01` | High | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 8 | `terminal_exec, write_file, write_file, terminal_exec, terminal_exec, write_file, terminal_exec` | 19.63s | All checks verified |
| `G-H03` | High | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 5 | `write_file, terminal_exec, terminal_exec, terminal_exec` | 27.06s | All checks verified |
| `G-H03` | High | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 5 | `write_file, terminal_exec, terminal_exec, terminal_exec` | 19.27s | All checks verified |
| `G-H03` | High | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 5 | `write_file, terminal_exec, terminal_exec, terminal_exec` | 18.83s | All checks verified |
| `OF-02` | Medium | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 5 | `valstorm_schema_inspect, valstorm_record_cud, valstorm_record_cud` | 3.78s | All checks verified |
| `OF-02` | Medium | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 2.16s | All checks verified |
| `OF-02` | Medium | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 2.60s | All checks verified |
| `OF-04` | High | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 2 | `valstorm_sql_query, valstorm_record_merge` | 1.72s | All checks verified |
| `OF-04` | High | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 2 | `valstorm_sql_query, valstorm_record_merge` | 2.00s | All checks verified |
| `OF-04` | High | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 2 | `valstorm_sql_query, valstorm_record_merge` | 1.53s | All checks verified |
| `G-M03` | Medium | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 3 | `write_file, patch_file, read_file` | 10.34s | All checks verified |
| `G-M03` | Medium | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 4 | `write_file, patch_file, read_file` | 9.78s | All checks verified |
| `G-M03` | Medium | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 4 | `write_file, patch_file, read_file` | 9.81s | All checks verified |
| `G-H01` | High | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 5 | `terminal_exec, terminal_exec, write_file, write_file, terminal_exec, write_file, terminal_exec` | 19.25s | All checks verified |
| `G-H01` | High | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 6 | `terminal_exec, terminal_exec, write_file, write_file, terminal_exec, write_file, terminal_exec` | 19.26s | All checks verified |
| `G-H01` | High | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 5 | `terminal_exec, terminal_exec, write_file, write_file, terminal_exec, write_file, terminal_exec` | 17.63s | All checks verified |
| `G-H03` | High | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 4 | `write_file, terminal_exec, terminal_exec` | 16.43s | All checks verified |
| `G-H03` | High | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 4 | `write_file, terminal_exec, terminal_exec` | 17.17s | All checks verified |
| `G-H03` | High | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 4 | `write_file, terminal_exec, terminal_exec` | 17.20s | All checks verified |
| `OF-02` | Medium | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 7.71s | All checks verified |
| `OF-02` | Medium | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 9.53s | All checks verified |
| `OF-02` | Medium | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 9.08s | All checks verified |
| `OF-04` | High | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 4 | `valstorm_sql_query, valstorm_sql_query, valstorm_record_merge, valstorm_sql_query, valstorm_sql_query` | 17.84s | All checks verified |
| `OF-04` | High | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 5 | `valstorm_sql_query, valstorm_sql_query, valstorm_record_merge, valstorm_sql_query, valstorm_sql_query` | 17.03s | All checks verified |
| `OF-04` | High | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 4 | `valstorm_sql_query, valstorm_sql_query, valstorm_record_merge, valstorm_sql_query, valstorm_sql_query` | 16.54s | All checks verified |
