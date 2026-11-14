# FI-558 (P1): Iteration-0 "Blind Discovery" Thinking Step-Down & Post-Discovery Effort Step-Up — Benchmark Report

- **Ticket ID:** `tick_z3feIX6BdiRLOI9G` (`FI-558`)
- **Worktree:** `<worktrees>/feat-FI-558-blind-discovery-thinking-stepdown`
- **Branch:** `feat/FI-558-blind-discovery-thinking-stepdown`
- **Date (UTC):** `2026-11-14`

---

## 1. Architectural Changes Implemented

1. **Iteration-0 Blind Discovery Step-Down (`core/message_autotuner.py`, `core/thinking_router.py`):**
   - Added `DISCOVERY_READ_TOOLS` (`read_file`, `search_files`, `code_outline`, `find_symbols`, `package_blast_radius`, `valstorm_schema_inspect`, `valstorm_sql_query`, `valstorm_vfs_search`, `web_search`, `web_scrape`, etc.) and `MessageAutoTuner.is_blind_discovery_prompt()`.
   - When `iteration == 0` must first emit a discovery tool call before synthesis, the model's `tier` and `locked_model` remain 100% locked for KV prefix cache safety, while `reasoning_effort` steps down from `'high'`/`'medium'` $\rightarrow$ `'low'` and records `deferred_synthesis_effort`.
2. **Post-Discovery Effort Step-Up (`core/message_autotuner.py`, `core/thinking_router.py`, `core/react.py`):**
   - On `iteration >= 1`, after a `DISCOVERY_READ_TOOLS` tool call hydrates file/schema/search content into `session.messages` (and before any `MUTATION_EXECUTION_TOOLS` has run), `tune_iteration` steps `reasoning_effort` **UP** to `deferred_synthesis_effort` (`'medium'` or `'high'`) for code/patch synthesis, then steps **DOWN** to `'none'` on subsequent verification/wrap-up turns.
   - Fixed substring false-positive where `"JavaScript"` matched `"script"` in `_REASONER_SIGNALS` / `_MEDIUM_COMPLEXITY_SIGNALS` by enforcing word-boundary regex `\bscript\b`.
3. **Provider-Level Thinking Budget Step-Down (`providers/gemini.py`):**
   - Previously, `reasoning_effort="none"` returned `thinking_config=None`, which caused Gemini 3 (`gemini-3.8-flash`) to fall back to default HIGH thinking (`100+` `thoughts_tokens`) and caused `gemini-2.5-pro` to run unlimited dynamic thinking on every post-tool iteration.
   - Mapped `reasoning_effort="none"` to `ThinkingConfig(thinking_level="LOW")` on Gemini 3 (`0` `thoughts_tokens`) and mapped `reasoning_effort` (`none`/`low`/`medium`/`high`) to calibrated `ThinkingConfig(thinking_budget=128|256|512)` on `gemini-2.5-pro` while keeping explicit `"off"` returning `None`.

---

## 2. Live Gemini API Head-to-Head Benchmark (`PRE_P1` vs. `POST_P1`)

| Task ID | Task Category | Config | Model Selected | Outcome | Turns | Wall-Clock Latency | Prompt Tokens | Completion Tokens | `thoughts_tokens` | Est. Cost ($) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`G-S02`** | JS File Write + Verify | `PRE_P1` | `gemini-flash-latest` | ✅ PASS | 3 | 7.39s | 10,584 | 126 | 0 | $0.00166 |
| **`G-S02`** | JS File Write + Verify | **`POST_P1`** | `gemini-flash-lite-latest` | **✅ PASS** | **3** | **2.31s (-68.7%)** | **10,433** | **75** | **2** | **$0.00081 (-51.2%)** |
| **`G-M01`** | Blind Web Search $\rightarrow$ Write | `PRE_P1` | `gemini-flash-latest` | ✅ PASS | 10 | 30.47s | 41,153 | 266 | 2,200 | $0.00765 |
| **`G-M01`** | Blind Web Search $\rightarrow$ Write | **`POST_P1`** | `gemini-flash-latest` | **✅ PASS** | **7** | **11.94s (-60.8%)** | **25,484** | **279** | **0 (-100%)** | **$0.00399 (-47.8%)** |
| **`G-M03`** | Write + Patch + Verify | `PRE_P1` | `gemini-flash-latest` | ✅ PASS | 4 | 8.31s | 15,675 | 324 | 200 | $0.00266 |
| **`G-M03`** | Write + Patch + Verify | **`POST_P1`** | `gemini-flash-latest` | **✅ PASS** | **4** | **7.20s (-13.4%)** | **15,691** | **306** | **188 (-6.0%)** | **$0.00265** |
| **`G-H01`** | TDD Red-Green-Refactor | `PRE_P1` | `gemini-pro-latest` | ✅ PASS | 7 | 15.98s | 27,997 | 319 | 413 | $0.02708 |
| **`G-H01`** | TDD Red-Green-Refactor | **`POST_P1`** | `gemini-pro-latest` | **✅ PASS** | **7** | **14.40s (-9.9%)** | **28,143** | **448** | **215 (-47.9%)** | **$0.02642** |
| **`G-H02`** | Blind Web Scrape $\rightarrow$ Diff | `PRE_P1` | `gemini-flash-latest` | ✅ PASS | 3 | 4.97s | 15,513 | 114 | 134 | $0.00248 |
| **`G-H02`** | Blind Web Scrape $\rightarrow$ Diff | **`POST_P1`** | `gemini-flash-latest` | **✅ PASS** | **3** | **5.39s** | **15,343** | **115** | **0 (-100%)** | **$0.00237 (-4.4%)** |
| **`G-H03`** | Shell Script + `chmod +x` | `PRE_P1` | `gemini-pro-latest` | ✅ PASS | 4 | 15.30s | 14,765 | 280 | 800 | $0.01554 |
| **`G-H03`** | Shell Script + `chmod +x` | **`POST_P1`** | `gemini-pro-latest` | **✅ PASS** | **3** | **12.41s (-18.9%)** | **11,179** | **318** | **574 (-28.3%)** | **$0.01908** |
| **`OF-02`** | Schema Inspect $\rightarrow$ CUD | `PRE_P1` | `gemini-flash-lite-latest` | ✅ PASS | 4 | 3.23s | 20,635 | 203 | 2 | $0.00135 |
| **`OF-02`** | Schema Inspect $\rightarrow$ CUD | **`POST_P1`** | `gemini-flash-lite-latest` | **✅ PASS** | **3** | **2.37s (-26.6%)** | **15,246** | **309** | **0 (-100%)** | **$0.00082 (-39.3%)** |
| **`OF-04`** | SQL Query $\rightarrow$ Merge | `PRE_P1` | `gemini-flash-lite-latest` | ✅ PASS | 2 | 1.66s | 10,444 | 214 | 0 | $0.00085 |
| **`OF-04`** | SQL Query $\rightarrow$ Merge | **`POST_P1`** | `gemini-flash-lite-latest` | **✅ PASS** | **2** | **1.86s** | **10,442** | **247** | **2** | **$0.00086** |
| **`P1-DISC-PATCH`** | Blind `read_file` $\rightarrow$ `patch_file` $\rightarrow$ `pytest` | `PRE_P1` | `gemini-pro-latest` | ✅ PASS | 5 | 18.61s | 21,802 | 442 | 891 | $0.02727 |
| **`P1-DISC-PATCH`** | Blind `read_file` $\rightarrow$ `patch_file` $\rightarrow$ `pytest` | **`POST_P1`** | `gemini-pro-latest` | **✅ PASS** | **6** | **16.69s (-10.3%)** | **25,957** | **506** | **733 (-17.7%)** | **$0.02868** |

### Aggregate Live Summary (Across All 9 Tasks)
- **Pass Rate:** `100.0% (9/9)` on both `PRE_P1` and `POST_P1` (zero quality regression).
- **Total Wall-Clock Duration:** `105.92s` (`PRE_P1`) $\rightarrow$ **`74.57s` (`POST_P1`) — `29.6%` faster overall (`-31.35s` saved)**.
- **Total `thoughts_tokens`:** `4,640` (`PRE_P1`) $\rightarrow$ **`1,714` (`POST_P1`) — `63.1%` reduction in thinking tokens (`-2,926` `thoughts_tokens` eliminated)**.

---

## 3. Full 14-Task Mock Ablation Suite (`Core` + `Office`)

| Config ID | Pass Rate | Avg Latency | Avg Tokens | Avg `thoughts_tokens` | Total Cost ($) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`autotune_s1_full` (P1)** | **100.0% (14/14)** | **0.21s** | **668.9** | **58.9 (-91.8%)** | **$0.00448 (-81.6%)** |
| `static_thinker_high` | 100.0% (14/14) | 0.60s | 788.5 | 720.0 | $0.02436 |
| `static_reasoner_high` | 100.0% (14/14) | 0.41s | 788.5 | 720.0 | $0.00292 |
| `static_worker_none` | 92.9% (13/14) | 0.12s | 653.5 | 0.0 | $0.00090 |
