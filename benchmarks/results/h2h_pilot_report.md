# Valstorm Agent Benchmark Report (`--ablation head_to_head`)

- **Timestamp (UTC):** `2026-11-02T18:44:45.452634+00:00`
- **Execution Mode:** `live`
- **Tasks Evaluated:** `5` | **Trials per Task (`k`):** `3`

## 1. Configuration Ablation Summary

| Config ID | Configuration Label | Pass Rate | `pass^k` | Avg Turns | Avg Latency | Avg Tokens | Cached % | Est. Cost ($) | $ / Pass | Parallel Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `h2h_a_vsagent_gemini` | A. vsagent + Gemini auto-tune (`gemini-flash-latest` base, skills, scoped tools) | **93.3% (14/15)** | `85.9%` | 4.87 | 15.45s | 21596 | 23% | $0.26810 | $0.01915 | 1.00x |
| `h2h_b_vsagent_claude` | B. vsagent + Claude (`claude-opus-5-5`, skills, scoped tools) | **93.3% (14/15)** | `85.9%` | 4.4 | 15.81s | 34615 | 0% | $2.35230 | $0.16802 | 1.00x |
| `h2h_c_claude_code` | C. Claude Code (`claude-opus-5-5`, built-in tools only) | **60.0% (9/15)** | `60.0%` | 6.2 | 24.72s | 32668 | 69% | $1.27792 | $0.14199 | 1.00x |
| `h2h_d_claude_code_mcp` | D. Claude Code (`claude-opus-5-5`) + vsagent tools via MCP bridge | **100.0% (15/15)** | `100.0%` | 4.73 | 15.95s | 12235 | 58% | $0.71916 | $0.04794 | 1.00x |

## 2. Granular Task-by-Task Scorecard

| Task ID | Complexity | Config ID | Outcome | Score | Turns | Tools Called | Duration | Failure / Notes |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- | :---: | :--- |
| `G-M03` | Medium | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 4 | `write_file, patch_file, read_file` | 37.07s | All checks verified |
| `G-M03` | Medium | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 4 | `write_file, patch_file, read_file` | 8.71s | All checks verified |
| `G-M03` | Medium | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 4 | `write_file, patch_file, read_file` | 28.27s | All checks verified |
| `G-H01` | High | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 9 | `terminal_exec, write_file, terminal_exec, write_file, terminal_exec, terminal_exec, write_file, terminal_exec` | 21.02s | All checks verified |
| `G-H01` | High | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 14 | `terminal_exec, write_file, write_file, terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec, write_file, terminal_exec` | 45.75s | All checks verified |
| `G-H01` | High | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 8 | `terminal_exec, write_file, write_file, terminal_exec, terminal_exec, write_file, terminal_exec` | 18.02s | All checks verified |
| `G-H03` | High | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 4 | `write_file, terminal_exec, terminal_exec` | 14.36s | All checks verified |
| `G-H03` | High | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 5 | `write_file, terminal_exec, terminal_exec, terminal_exec` | 20.95s | All checks verified |
| `G-H03` | High | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 6 | `write_file, terminal_exec, terminal_exec, terminal_exec, terminal_exec` | 19.65s | All checks verified |
| `OF-02` | Medium | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 2.20s | All checks verified |
| `OF-02` | Medium | `h2h_a_vsagent_gemini` | ❌ FAIL | 0.33 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 2.36s | content_regex_mismatch:valstorm_sandbox_db.json (expected /(?=[\s |
| `OF-02` | Medium | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 2.12s | All checks verified |
| `OF-04` | High | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 2 | `valstorm_sql_query, valstorm_record_merge` | 1.79s | All checks verified |
| `OF-04` | High | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 2 | `valstorm_sql_query, valstorm_record_merge, valstorm_sql_query, valstorm_sql_query` | 7.92s | All checks verified |
| `OF-04` | High | `h2h_a_vsagent_gemini` | ✅ PASS | 1.00 | 2 | `valstorm_sql_query, valstorm_record_merge` | 1.58s | All checks verified |
| `G-M03` | Medium | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 4 | `write_file, patch_file, read_file` | 10.29s | All checks verified |
| `G-M03` | Medium | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 4 | `write_file, patch_file, read_file` | 9.87s | All checks verified |
| `G-M03` | Medium | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 4 | `write_file, patch_file, read_file` | 10.54s | All checks verified |
| `G-H01` | High | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 6 | `terminal_exec, terminal_exec, terminal_exec, write_file, write_file, terminal_exec, write_file, terminal_exec` | 24.13s | All checks verified |
| `G-H01` | High | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 8 | `terminal_exec, terminal_exec, terminal_exec, write_file, write_file, write_file, terminal_exec, write_file, terminal_exec` | 30.43s | All checks verified |
| `G-H01` | High | `h2h_b_vsagent_claude` | ❌ FAIL | 0.67 | 7 | `terminal_exec, terminal_exec, terminal_exec, write_file, write_file, terminal_exec, write_file, terminal_exec` | 28.43s | verify_command_failed(exit=1): /Users/jared/.pyenv/versions/3.10. |
| `G-H03` | High | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 4 | `write_file, terminal_exec, terminal_exec` | 16.05s | All checks verified |
| `G-H03` | High | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 4 | `write_file, terminal_exec, terminal_exec` | 16.68s | All checks verified |
| `G-H03` | High | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 4 | `write_file, terminal_exec, terminal_exec` | 15.90s | All checks verified |
| `OF-02` | Medium | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 7.79s | All checks verified |
| `OF-02` | Medium | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 9.51s | All checks verified |
| `OF-02` | Medium | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 9.59s | All checks verified |
| `OF-04` | High | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 4 | `valstorm_sql_query, valstorm_sql_query, valstorm_record_merge, valstorm_sql_query, valstorm_sql_query` | 14.88s | All checks verified |
| `OF-04` | High | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 4 | `valstorm_sql_query, valstorm_sql_query, valstorm_record_merge, valstorm_sql_query, valstorm_sql_query` | 16.71s | All checks verified |
| `OF-04` | High | `h2h_b_vsagent_claude` | ✅ PASS | 1.00 | 4 | `valstorm_sql_query, valstorm_sql_query, valstorm_record_merge, valstorm_sql_query, valstorm_sql_query` | 16.36s | All checks verified |
| `G-M03` | Medium | `h2h_c_claude_code` | ✅ PASS | 1.00 | 4 | `terminal_exec, patch_file, terminal_exec` | 12.61s | All checks verified |
| `G-M03` | Medium | `h2h_c_claude_code` | ✅ PASS | 1.00 | 5 | `terminal_exec, read_file, patch_file, terminal_exec` | 12.88s | All checks verified |
| `G-M03` | Medium | `h2h_c_claude_code` | ✅ PASS | 1.00 | 5 | `terminal_exec, read_file, patch_file, terminal_exec` | 14.21s | All checks verified |
| `G-H01` | High | `h2h_c_claude_code` | ✅ PASS | 1.00 | 6 | `terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec` | 25.41s | All checks verified |
| `G-H01` | High | `h2h_c_claude_code` | ✅ PASS | 1.00 | 7 | `terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec` | 27.29s | All checks verified |
| `G-H01` | High | `h2h_c_claude_code` | ✅ PASS | 1.00 | 6 | `terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec` | 25.29s | All checks verified |
| `G-H03` | High | `h2h_c_claude_code` | ✅ PASS | 1.00 | 3 | `terminal_exec, terminal_exec` | 18.15s | All checks verified |
| `G-H03` | High | `h2h_c_claude_code` | ✅ PASS | 1.00 | 3 | `terminal_exec, terminal_exec` | 15.58s | All checks verified |
| `G-H03` | High | `h2h_c_claude_code` | ✅ PASS | 1.00 | 6 | `terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec` | 24.23s | All checks verified |
| `OF-02` | Medium | `h2h_c_claude_code` | ❌ FAIL | 0.00 | 11 | `terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec` | 36.94s | missing_target_file:valstorm_sandbox_db.json; verify_command_fail |
| `OF-02` | Medium | `h2h_c_claude_code` | ❌ FAIL | 0.00 | 9 | `terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec` | 31.71s | missing_target_file:valstorm_sandbox_db.json; verify_command_fail |
| `OF-02` | Medium | `h2h_c_claude_code` | ❌ FAIL | 0.00 | 10 | `terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec` | 33.81s | missing_target_file:valstorm_sandbox_db.json; verify_command_fail |
| `OF-04` | High | `h2h_c_claude_code` | ❌ FAIL | 0.00 | 6 | `terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec` | 28.96s | missing_target_file:valstorm_sandbox_db.json; verify_command_fail |
| `OF-04` | High | `h2h_c_claude_code` | ❌ FAIL | 0.00 | 6 | `terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec` | 32.34s | missing_target_file:valstorm_sandbox_db.json; verify_command_fail |
| `OF-04` | High | `h2h_c_claude_code` | ❌ FAIL | 0.00 | 6 | `terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec` | 31.41s | missing_target_file:valstorm_sandbox_db.json; verify_command_fail |
| `G-M03` | Medium | `h2h_d_claude_code_mcp` | ✅ PASS | 1.00 | 4 | `terminal_exec, patch_file, terminal_exec` | 11.85s | All checks verified |
| `G-M03` | Medium | `h2h_d_claude_code_mcp` | ✅ PASS | 1.00 | 5 | `terminal_exec, read_file, patch_file, terminal_exec` | 14.37s | All checks verified |
| `G-M03` | Medium | `h2h_d_claude_code_mcp` | ✅ PASS | 1.00 | 5 | `terminal_exec, read_file, patch_file, terminal_exec` | 14.01s | All checks verified |
| `G-H01` | High | `h2h_d_claude_code_mcp` | ✅ PASS | 1.00 | 7 | `terminal_exec, terminal_exec, terminal_exec, terminal_exec, patch_file, terminal_exec` | 26.71s | All checks verified |
| `G-H01` | High | `h2h_d_claude_code_mcp` | ✅ PASS | 1.00 | 5 | `terminal_exec, terminal_exec, terminal_exec, terminal_exec` | 18.76s | All checks verified |
| `G-H01` | High | `h2h_d_claude_code_mcp` | ✅ PASS | 1.00 | 7 | `terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec, terminal_exec` | 27.24s | All checks verified |
| `G-H03` | High | `h2h_d_claude_code_mcp` | ✅ PASS | 1.00 | 5 | `terminal_exec, terminal_exec, terminal_exec, terminal_exec` | 16.52s | All checks verified |
| `G-H03` | High | `h2h_d_claude_code_mcp` | ✅ PASS | 1.00 | 5 | `terminal_exec, terminal_exec, terminal_exec, terminal_exec` | 20.88s | All checks verified |
| `G-H03` | High | `h2h_d_claude_code_mcp` | ✅ PASS | 1.00 | 4 | `terminal_exec, terminal_exec, terminal_exec` | 17.64s | All checks verified |
| `OF-02` | Medium | `h2h_d_claude_code_mcp` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 8.04s | All checks verified |
| `OF-02` | Medium | `h2h_d_claude_code_mcp` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 8.18s | All checks verified |
| `OF-02` | Medium | `h2h_d_claude_code_mcp` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 7.64s | All checks verified |
| `OF-04` | High | `h2h_d_claude_code_mcp` | ✅ PASS | 1.00 | 5 | `valstorm_sql_query, valstorm_record_merge, valstorm_sql_query, valstorm_sql_query` | 14.19s | All checks verified |
| `OF-04` | High | `h2h_d_claude_code_mcp` | ✅ PASS | 1.00 | 5 | `valstorm_sql_query, valstorm_record_merge, valstorm_sql_query, valstorm_sql_query` | 15.36s | All checks verified |
| `OF-04` | High | `h2h_d_claude_code_mcp` | ✅ PASS | 1.00 | 5 | `valstorm_sql_query, valstorm_record_merge, valstorm_sql_query, valstorm_sql_query` | 17.85s | All checks verified |
