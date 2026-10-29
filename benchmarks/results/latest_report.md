# Valstorm Live Benchmark: GEAP Gemini Baselines vs. DigitalOcean Open-Weight Models

- **Timestamp (UTC):** `2026-10-21T15:26:44.042280+00:00`
- **Execution Mode:** `live`
- **Tasks Evaluated:** `3` | **Trials per Task (`k`):** `1`

## 1. Configuration Ablation Summary

| Config ID | Configuration Label | Pass Rate | `pass^k` | Avg Turns | Avg Latency | Avg Tokens | Est. Cost ($) | Parallel Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `geap_gemini_4_argon` | GEAP Baseline Thinker (`gemini-4-argon`, 1M/1M) | **100.0% (3/3)** | `100.0%` | 2.33 | 29.62s | 10440 | $0.07935 | 1.00x |
| `do_qwen3_8_max` | DO Open Thinker (`qwen3.8-max`, 2.4T MoE, 1M/262K) | **100.0% (3/3)** | `100.0%` | 3.33 | 40.26s | 15522 | $0.10489 | 1.00x |
| `do_deepseek_v4_pro_0813` | DO Open Thinker (`deepseek-v4-pro-0813`, 1M/1M, $0.044 Cache) | **100.0% (3/3)** | `100.0%` | 3.67 | 17.96s | 16387 | $0.07208 | 1.00x |
| `do_glm_5_3` | DO Open Thinker (`glm-5.3`, 753B, Always-On Reasoning) | **66.7% (2/3)** | `66.7%` | 2.33 | 43.25s | 10711 | $0.05685 | 1.00x |
| `do_kimi_k2_6` | DO Open Thinker [Dedicated+Serverless] (`kimi-k2.6`, 1T, 262K/262K) | **100.0% (3/3)** | `100.0%` | 3.0 | 33.55s | 12061 | $0.04444 | 1.00x |
| `do_mimo_v2_5_pro` | DO Open Thinker [Dedicated+Serverless] (`mimo-v2.5-pro`, 1T MIT, 262K/262K) | **66.7% (2/3)** | `66.7%` | 2.33 | 27.12s | 10165 | $0.02761 | 1.00x |
| `geap_gemini_3_8_flash` | GEAP Baseline Reasoner (`gemini-3.8-flash`, 1M/64K) | **100.0% (3/3)** | `100.0%` | 3.33 | 33.35s | 16343 | $0.08628 | 1.00x |
| `do_qwen3_5_397b_a17b` | DO Open Reasoner [Dedicated+Serverless] (`qwen3.5-397b-a17b`, 17B Active MoE) | **66.7% (2/3)** | `66.7%` | 2.33 | 24.82s | 10301 | $0.02003 | 1.00x |
| `do_deepseek_v4_1_flash` | DO Open Reasoner (`deepseek-v4.1-flash`, 552B Vision, $0.006 Cache) | **100.0% (3/3)** | `100.0%` | 2.67 | 23.70s | 12277 | $0.01432 | 1.00x |
| `do_deepseek_3_2` | DO Open Reasoner [Dedicated+Serverless] (`deepseek-3.2`, 680B, 163K) | **100.0% (3/3)** | `100.0%` | 3.33 | 23.07s | 15407 | $0.02695 | 1.00x |
| `do_glm_5_2` | DO Open Reasoner [Dedicated+Serverless] (`glm-5.2`, MIT, 262K/131K) | **66.7% (2/3)** | `66.7%` | 2.33 | 24.62s | 10187 | $0.05033 | 1.00x |
| `do_nemotron_3_ultra` | DO Open Reasoner [Dedicated+Serverless] (`nemotron-3-ultra-550b`, 550B) | **100.0% (3/3)** | `100.0%` | 3.67 | 11.15s | 16322 | $0.04616 | 1.00x |
| `do_minimax_m2_5` | DO Open Reasoner [Dedicated+Serverless] (`minimax-m2.5`, 230B, 65K) | **66.7% (2/3)** | `66.7%` | 3.33 | 16.74s | 14664 | $0.01640 | 1.00x |
| `geap_gemini_3_5_flash_lite` | GEAP Baseline Worker (`gemini-3.5-flash-lite`, 1M/64K) | **100.0% (3/3)** | `100.0%` | 3.33 | 4.47s | 13644 | $0.01615 | 1.00x |
| `do_deepseek_v4_flash_0731` | DO Open Worker (`deepseek-v4-flash-0731`, 284B, 1M/1M, $0.14/$0.28) | **66.7% (2/3)** | `66.7%` | 2.33 | 33.28s | 11143 | $0.00510 | 1.00x |
| `do_glm_5_3_flash` | DO Open Worker (`glm-5.3-flash`, 320B Multimodal, $0.15/$0.50) | **66.7% (2/3)** | `66.7%` | 2.67 | 19.59s | 11457 | $0.00641 | 1.00x |
| `do_llama_4_maverick` | DO Open Worker [Dedicated+Serverless] (`llama-4-maverick`, 17B-128E, 16K Out) | **0.0% (0/3)** | `0.0%` | 1.0 | 23.95s | 3856 | $0.00340 | 1.00x |
| `do_gpt_oss_120b` | DO Open Worker [Dedicated+Serverless] (`openai-gpt-oss-120b`, 4K Max Out) | **66.7% (2/3)** | `66.7%` | 3.33 | 14.02s | 13150 | $0.00546 | 1.00x |
| `do_gemma_4_31b` | DO Open Worker [1x GPU Dedicated] (`gemma-4-31B-it`, 31B, 256K/8K) | **100.0% (3/3)** | `100.0%` | 3.0 | 12.88s | 12447 | $0.00719 | 1.00x |
| `do_mistral_3_14b` | DO Open Worker [1x GPU Dedicated] (`mistral-3-14B`, 14B, 262K) | **66.7% (2/3)** | `66.7%` | 3.67 | 14.16s | 15592 | $0.00935 | 1.00x |

## 2. Granular Task-by-Task Scorecard

| Task ID | Complexity | Config ID | Outcome | Score | Turns | Tools Called | Duration | Failure / Notes |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- | :---: | :--- |
| `G-H03` | High | `geap_gemini_4_argon` | ✅ PASS | 1.00 | 3 | `write_file, terminal_exec` | 23.76s | All checks verified |
| `G-H03` | High | `do_qwen3_8_max` | ✅ PASS | 1.00 | 5 | `write_file, terminal_exec, terminal_exec, terminal_exec` | 53.29s | All checks verified |
| `G-H03` | High | `do_deepseek_v4_pro_0813` | ✅ PASS | 1.00 | 5 | `write_file, terminal_exec, terminal_exec, terminal_exec` | 21.77s | All checks verified |
| `G-H03` | High | `do_glm_5_3` | ✅ PASS | 1.00 | 4 | `write_file, terminal_exec, terminal_exec` | 66.99s | All checks verified |
| `G-H03` | High | `do_kimi_k2_6` | ✅ PASS | 1.00 | 5 | `write_file, terminal_exec, terminal_exec, terminal_exec` | 35.52s | All checks verified |
| `G-H03` | High | `do_mimo_v2_5_pro` | ✅ PASS | 1.00 | 4 | `write_file, terminal_exec, terminal_exec` | 19.39s | All checks verified |
| `OF-02` | Medium | `geap_gemini_4_argon` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 20.11s | All checks verified |
| `OF-02` | Medium | `do_qwen3_8_max` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 22.49s | All checks verified |
| `OF-02` | Medium | `do_deepseek_v4_pro_0813` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 8.96s | All checks verified |
| `OF-02` | Medium | `do_glm_5_3` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 17.77s | All checks verified |
| `OF-02` | Medium | `do_kimi_k2_6` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 20.08s | All checks verified |
| `OF-02` | Medium | `do_mimo_v2_5_pro` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 23.98s | All checks verified |
| `CR-01` | Medium | `geap_gemini_4_argon` | ✅ PASS | 1.00 | 1 | `write_file` | 45.00s | All checks verified |
| `CR-01` | Medium | `do_qwen3_8_max` | ✅ PASS | 1.00 | 2 | `write_file, analyze_image` | 45.00s | All checks verified |
| `CR-01` | Medium | `do_deepseek_v4_pro_0813` | ✅ PASS | 1.00 | 3 | `write_file, analyze_image` | 23.14s | All checks verified |
| `CR-01` | Medium | `do_glm_5_3` | ❌ FAIL | 0.00 | 0 | `*(none)*` | 45.00s | missing_target_file:ad_plate.psd.json; verify_command_failed(exit |
| `CR-01` | Medium | `do_kimi_k2_6` | ✅ PASS | 1.00 | 1 | `write_file` | 45.04s | All checks verified |
| `CR-01` | Medium | `do_mimo_v2_5_pro` | ❌ FAIL | 0.00 | 0 | `*(none)*` | 38.00s | missing_target_file:ad_plate.psd.json; verify_command_failed(exit |
| `G-H03` | High | `geap_gemini_3_8_flash` | ✅ PASS | 1.00 | 4 | `write_file, terminal_exec, terminal_exec, terminal_exec` | 45.00s | All checks verified |
| `G-H03` | High | `do_qwen3_5_397b_a17b` | ✅ PASS | 1.00 | 4 | `write_file, terminal_exec, terminal_exec` | 19.75s | All checks verified |
| `G-H03` | High | `do_deepseek_v4_1_flash` | ✅ PASS | 1.00 | 4 | `write_file, terminal_exec, terminal_exec` | 15.67s | All checks verified |
| `G-H03` | High | `do_deepseek_3_2` | ✅ PASS | 1.00 | 5 | `write_file, terminal_exec, terminal_exec, terminal_exec, terminal_exec` | 8.93s | All checks verified |
| `G-H03` | High | `do_glm_5_2` | ✅ PASS | 1.00 | 4 | `write_file, terminal_exec, terminal_exec` | 13.90s | All checks verified |
| `G-H03` | High | `do_nemotron_3_ultra` | ✅ PASS | 1.00 | 5 | `write_file, terminal_exec, terminal_exec, terminal_exec, terminal_exec` | 9.73s | All checks verified |
| `G-H03` | High | `do_minimax_m2_5` | ✅ PASS | 1.00 | 4 | `write_file, terminal_exec, terminal_exec` | 14.58s | All checks verified |
| `OF-02` | Medium | `geap_gemini_3_8_flash` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 10.05s | All checks verified |
| `OF-02` | Medium | `do_qwen3_5_397b_a17b` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 16.70s | All checks verified |
| `OF-02` | Medium | `do_deepseek_v4_1_flash` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 10.43s | All checks verified |
| `OF-02` | Medium | `do_deepseek_3_2` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 22.28s | All checks verified |
| `OF-02` | Medium | `do_glm_5_2` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 21.96s | All checks verified |
| `OF-02` | Medium | `do_nemotron_3_ultra` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 6.85s | All checks verified |
| `OF-02` | Medium | `do_minimax_m2_5` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 7.89s | All checks verified |
| `CR-01` | Medium | `geap_gemini_3_8_flash` | ✅ PASS | 1.00 | 3 | `analyze_image, write_file, analyze_image` | 45.00s | All checks verified |
| `CR-01` | Medium | `do_qwen3_5_397b_a17b` | ❌ FAIL | 0.00 | 0 | `*(none)*` | 38.00s | missing_target_file:ad_plate.psd.json; verify_command_failed(exit |
| `CR-01` | Medium | `do_deepseek_v4_1_flash` | ✅ PASS | 1.00 | 1 | `write_file, analyze_image` | 45.00s | All checks verified |
| `CR-01` | Medium | `do_deepseek_3_2` | ✅ PASS | 1.00 | 2 | `write_file, analyze_image` | 38.00s | All checks verified |
| `CR-01` | Medium | `do_glm_5_2` | ❌ FAIL | 0.00 | 0 | `*(none)*` | 38.00s | missing_target_file:ad_plate.psd.json; verify_command_failed(exit |
| `CR-01` | Medium | `do_nemotron_3_ultra` | ✅ PASS | 1.00 | 3 | `write_file, analyze_image` | 16.87s | All checks verified |
| `CR-01` | Medium | `do_minimax_m2_5` | ❌ FAIL | 0.33 | 3 | `write_file, analyze_image` | 27.75s | content_regex_mismatch:ad_plate.psd.json (expected /(?=[\s\S]*sha |
| `G-H03` | High | `geap_gemini_3_5_flash_lite` | ✅ PASS | 1.00 | 4 | `write_file, terminal_exec, terminal_exec` | 5.76s | All checks verified |
| `G-H03` | High | `do_deepseek_v4_flash_0731` | ✅ PASS | 1.00 | 4 | `terminal_exec, terminal_exec, write_file, terminal_exec, terminal_exec` | 32.00s | All checks verified |
| `G-H03` | High | `do_glm_5_3_flash` | ✅ PASS | 1.00 | 5 | `write_file, write_file, terminal_exec, terminal_exec, terminal_exec` | 15.54s | All checks verified |
| `G-H03` | High | `do_llama_4_maverick` | ❌ FAIL | 0.00 | 1 | `*(none)*` | 29.43s | missing_file:setup_env.sh; missing_file:temp_proj/.gitignore; mis |
| `G-H03` | High | `do_gpt_oss_120b` | ✅ PASS | 1.00 | 4 | `write_file, terminal_exec, terminal_exec` | 12.31s | All checks verified |
| `G-H03` | High | `do_gemma_4_31b` | ✅ PASS | 1.00 | 4 | `write_file, terminal_exec, terminal_exec` | 9.30s | All checks verified |
| `G-H03` | High | `do_mistral_3_14b` | ✅ PASS | 1.00 | 5 | `write_file, terminal_exec, terminal_exec, terminal_exec, terminal_exec` | 8.04s | All checks verified |
| `OF-02` | Medium | `geap_gemini_3_5_flash_lite` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 2.88s | All checks verified |
| `OF-02` | Medium | `do_deepseek_v4_flash_0731` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 29.82s | All checks verified |
| `OF-02` | Medium | `do_glm_5_3_flash` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 5.21s | All checks verified |
| `OF-02` | Medium | `do_llama_4_maverick` | ❌ FAIL | 0.33 | 1 | `*(none)*` | 10.37s | content_regex_mismatch:valstorm_sandbox_db.json (expected /(?=[\s |
| `OF-02` | Medium | `do_gpt_oss_120b` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 7.54s | All checks verified |
| `OF-02` | Medium | `do_gemma_4_31b` | ✅ PASS | 1.00 | 3 | `valstorm_schema_inspect, valstorm_record_cud` | 7.99s | All checks verified |
| `OF-02` | Medium | `do_mistral_3_14b` | ✅ PASS | 1.00 | 2 | `valstorm_schema_inspect, valstorm_record_cud` | 5.71s | All checks verified |
| `CR-01` | Medium | `geap_gemini_3_5_flash_lite` | ✅ PASS | 1.00 | 3 | `write_file, analyze_image` | 4.78s | All checks verified |
| `CR-01` | Medium | `do_deepseek_v4_flash_0731` | ❌ FAIL | 0.00 | 0 | `*(none)*` | 38.00s | missing_target_file:ad_plate.psd.json; verify_command_failed(exit |
| `CR-01` | Medium | `do_glm_5_3_flash` | ❌ FAIL | 0.00 | 0 | `*(none)*` | 38.00s | missing_target_file:ad_plate.psd.json; verify_command_failed(exit |
| `CR-01` | Medium | `do_llama_4_maverick` | ❌ FAIL | 0.00 | 1 | `write_file, analyze_image` | 32.05s | missing_target_file:ad_plate.psd.json; verify_command_failed(exit |
| `CR-01` | Medium | `do_gpt_oss_120b` | ❌ FAIL | 0.33 | 3 | `write_file, analyze_image` | 22.22s | content_regex_mismatch:ad_plate.psd.json (expected /(?=[\s\S]*sha |
| `CR-01` | Medium | `do_gemma_4_31b` | ✅ PASS | 1.00 | 2 | `write_file, analyze_image` | 21.35s | All checks verified |
| `CR-01` | Medium | `do_mistral_3_14b` | ❌ FAIL | 0.33 | 4 | `write_file, write_file, analyze_image` | 28.71s | content_regex_mismatch:ad_plate.psd.json (expected /(?=[\s\S]*sha |
