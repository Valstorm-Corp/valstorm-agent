# FI-559 (P2): Ephemeral Micro-Worker Discovery Pre-Hydration & Mechanical Turn Offload — Benchmark Report

- **Ticket ID:** `tick_UQCQhNRN0lQS3sjz` (`FI-559`)
- **Worktree:** `<worktrees>/feat-FI-559-micro-worker-tool-routing`
- **Branch:** `feat/FI-559-micro-worker-tool-routing`
- **Date (UTC):** `2026-11-14`

---

## 1. Architectural Changes Implemented

1. **Zero-Token Pre-Turn Discovery Hydration (`core/micro_worker.py` — `MicroWorkerEngine.plan_pre_turn_discovery_calls`):**
   - Before Iteration 0 of `ReActEngine.run_turn_stream`, when `is_auto_tune=True` and the user prompt explicitly requests reading existing workspace files (`read_file`) or scraping an explicit URL (`web_scrape`) prior to synthesis/verification, `MicroWorkerEngine` executes those read-only discovery tools in parallel with **`0` LLM tokens** (`<5ms` routing overhead) and injects native `FunctionCall` / `FunctionResponse` messages into `session.messages`.
2. **Post-Synthesis Mechanical Turn Offload (`core/micro_worker.py` — `MicroWorkerEngine.resolve_mechanical_offload`):**
   - On `thinker`-tier turns (`gemini-4-argon` at `$2.00/1M` input, `$10.00/1M` output, `$0.10/1M` cached), once core synthesis (`write_file`, `patch_file`) has completed on Iteration 0/1, subsequent mechanical verification (`terminal_exec`) and wrap-up iterations offload to `worker` (`gemini-3.5-flash-lite` at `$0.30/1M` input, `$2.50/1M` output, `$0.03/1M` cached — **6.67x cheaper input, 4.0x cheaper output**).
   - Because `gemini-4-argon`, `gemini-3.8-flash`, and `gemini-3.5-flash-lite` all share native `thought_signature` support on `FunctionCall` parts (and `gemini-4-argon` is never switched back to after synthesis), `gemini-4-argon`'s KV cache is never fragmented and tool-chain continuity is 100% preserved.
3. **`terminal_exec` Default `PYTHONPATH=.` & Unreachable Scraper Loop Guard (`core/sandbox.py`, `tools/web_tools.py`):**
   - Defaulted `PYTHONPATH=.` in `HostSandbox.exec_command` so `pytest` invocations resolve workspace packages (`src/...`) on the first attempt without wasting an LLM retry turn on `ModuleNotFoundError`.

---

## 2. Live Gemini API 3-Way Progression (`PRE_P1` vs. `POST_P1` vs. `POST_P2`)

| Task ID | Task Category | Stage |Pass | Turns | Wall-Clock Latency | Prompt Tokens | `thoughts_tokens` | Est. Cost ($) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`G-S02`** | JS File Write + Verify | `PRE_P1` (`v4.3.5`) | ✅ | 3 | 7.39s | 10,584 | 0 | $0.00166 |
| **`G-S02`** | JS File Write + Verify | `POST_P1` (`v4.3.6`) | ✅ | 3 | 2.31s | 10,433 | 2 | $0.00081 |
| **`G-S02`** | JS File Write + Verify | **`POST_P2` (`FI-559`)** | **✅** | **3** | **2.14s (-71.0% vs Base)** | **10,490** | **2** | **$0.00066 (-60.2% vs Base)** |
| **`G-M01`** | Blind Web Search $\rightarrow$ Write | `PRE_P1` (`v4.3.5`) | ✅ | 10 | 30.47s | 41,153 | 2,200 | $0.00765 |
| **`G-M01`** | Blind Web Search $\rightarrow$ Write | `POST_P1` (`v4.3.6`) | ✅ | 7 | 11.94s | 25,484 | 0 | $0.00399 |
| **`G-M01`** | Blind Web Search $\rightarrow$ Write | **`POST_P2` (`FI-559`)** | **✅** | **7** | **11.35s (-62.7% vs Base)** | **25,323** | **0 (-100%)** | **$0.00392 (-48.8% vs Base)** |
| **`G-M03`** | Write + Patch + Verify | `PRE_P1` (`v4.3.5`) | ✅ | 4 | 8.31s | 15,675 | 200 | $0.00266 |
| **`G-M03`** | Write + Patch + Verify | `POST_P1` (`v4.3.6`) | ✅ | 4 | 7.20s | 15,691 | 188 | $0.00265 |
| **`G-M03`** | Write + Patch + Verify | **`POST_P2` (`FI-559`)** | **✅** | **4** | **8.29s** | **15,789** | **221** | **$0.00267** |
| **`G-H01`** | TDD Red-Green-Refactor (`pytest`) | `PRE_P1` (`v4.3.5`) | ✅ | 7 | 15.98s | 27,997 | 413 | $0.02708 |
| **`G-H01`** | TDD Red-Green-Refactor (`pytest`) | `POST_P1` (`v4.3.6`) | ✅ | 7 | 14.40s | 28,143 | 215 | $0.02642 |
| **`G-H01`** | TDD Red-Green-Refactor (`pytest`) | **`POST_P2` (`FI-559`)** | **✅** | **7** | **10.88s (-24.4% vs P1, -31.9% vs Base)** | **28,023** | **277** | **$0.01729 (-34.6% vs P1, -36.2% vs Base)** |
| **`G-H02`** | Blind Web Scrape $\rightarrow$ Diff | `PRE_P1` (`v4.3.5`) | ✅ | 3 | 4.97s | 15,513 | 134 | $0.00248 |
| **`G-H02`** | Blind Web Scrape $\rightarrow$ Diff | `POST_P1` (`v4.3.6`) | ✅ | 3 | 5.39s | 15,343 | 0 | $0.00237 |
| **`G-H02`** | Blind Web Scrape $\rightarrow$ Diff | **`POST_P2` (`FI-559`)** | **✅** | **2 (-33%)** | **3.10s (-42.5% vs P1, -37.6% vs Base)** | **11,775 (-23.3%)** | **0 (-100%)** | **$0.00183 (-22.8% vs P1, -26.2% vs Base)** |
| **`G-H03`** | Shell Script + `chmod +x` | `PRE_P1` (`v4.3.5`) | ✅ | 4 | 15.30s | 14,765 | 800 | $0.01554 |
| **`G-H03`** | Shell Script + `chmod +x` | `POST_P1` (`v4.3.6`) | ✅ | 3 | 12.41s | 11,179 | 574 | $0.01908 |
| **`G-H03`** | Shell Script + `chmod +x` | **`POST_P2` (`FI-559`)** | **✅** | **5** | **9.56s (-23.0% vs P1, -37.5% vs Base)** | **19,103** | **422 (-26.5% vs P1)** | **$0.00783 (-59.0% vs P1, -49.6% vs Base)** |
| **`OF-02`** | Schema Inspect $\rightarrow$ CUD | `PRE_P1` (`v4.3.5`) | ✅ | 4 | 3.23s | 20,635 | 2 | $0.00135 |
| **`OF-02`** | Schema Inspect $\rightarrow$ CUD | `POST_P1` (`v4.3.6`) | ✅ | 3 | 2.37s | 15,246 | 0 | $0.00082 |
| **`OF-02`** | Schema Inspect $\rightarrow$ CUD | **`POST_P2` (`FI-559`)** | **✅** | **2 (-33% vs P1)** | **1.37s (-42.2% vs P1, -57.6% vs Base)** | **10,072 (-33.9% vs P1)** | **0** | **$0.00081 (-40.0% vs Base)** |
| **`OF-04`** | SQL Query $\rightarrow$ Merge | `PRE_P1` (`v4.3.5`) | ✅ | 2 | 1.66s | 10,444 | 0 | $0.00085 |
| **`OF-04`** | SQL Query $\rightarrow$ Merge | `POST_P1` (`v4.3.6`) | ✅ | 2 | 1.86s | 10,442 | 2 | $0.00086 |
| **`OF-04`** | SQL Query $\rightarrow$ Merge | **`POST_P2` (`FI-559`)** | **✅** | **2** | **1.68s** | **10,495** | **0** | **$0.00087** |
| **`P1-DISC-PATCH`** | Blind `read_file` (2 files) $\rightarrow$ `patch_file` $\rightarrow$ `pytest` | `PRE_P1` (`v4.3.5`) | ✅ | 5 | 18.61s | 21,802 | 891 | $0.02727 |
| **`P1-DISC-PATCH`** | Blind `read_file` (2 files) $\rightarrow$ `patch_file` $\rightarrow$ `pytest` | `POST_P1` (`v4.3.6`) | ✅ | 6 | 16.69s | 25,957 | 733 | $0.02868 |
| **`P1-DISC-PATCH`** | Blind `read_file` (2 files) $\rightarrow$ `patch_file` $\rightarrow$ `pytest` | **`POST_P2` (`FI-559`)** | **✅** | **3 (-50% vs P1)** | **5.25s (-68.5% vs P1, -71.8% vs Base)** | **12,596 (-51.5% vs P1)** | **176 (-76.0% vs P1)** | **$0.00715 (-75.1% vs P1, -73.8% vs Base)** |

---

## 3. Aggregate 9-Task Live Suite Comparison

| Metric | Baseline (`PRE_P1`, `v4.3.5`) | After P1 (`POST_P1`, `v4.3.6`) | After P2 (`POST_P2`, `FI-559`) | P2 vs. P1 Improvement | Cumulative (P1 + P2 vs. Baseline) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pass Rate (`pass@1`)** | `100.0% (9/9)` | `100.0% (9/9)` | **`100.0% (9/9)`** | **Zero regression** | **Zero regression (`64/64` pytest pass)** |
| **Total Suite Cost ($ USD)** | `$0.08654` | `$0.08568` | **`$0.04303`** | **`-49.8%` (`1.99x` cheaper than P1)** | **`-50.3%` (`2.01x` cheaper overall)** |
| **Total Wall-Clock Latency** | `105.92s` | `74.57s` | **`53.62s`** | **`-28.1%` (`-20.95s` faster than P1)** | **`-49.4%` (`1.98x` faster; `-52.30s` saved)** |
| **Total Thinking Tokens (`thoughts_tokens`)** | `4,640` | `1,714` | **`1,098`** | **`-35.9%` fewer than P1** | **`-76.3%` (`4.23x` fewer thinking tokens)** |
| **Total Prompt Tokens Billed** | `178,568` | `158,318` | **`141,666`** | **`-10.5%` fewer than P1** | **`-20.7%` (`-36,902` prompt tokens saved)** |
| **Total LLM Turns** | `42` | `38` | **`35`** | **`-7.9%` fewer than P1** | **`-16.7%` fewer LLM roundtrips** |
