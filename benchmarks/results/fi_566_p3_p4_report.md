# FI-566 (P3 + P4): System-1 Discovery Pre-Fetcher, Dynamic Tool Schema Pruning & Post-Turn Consumed Observation Folding — Benchmark Report

- **Ticket ID:** `tick_mJYP6lB6K4hYWYLA` (`FI-566`)
- **Worktree:** `<worktrees>/feat-FI-566-s1-prefetch-and-observation-pruning`
- **Branch:** `feat/FI-566-s1-prefetch-and-observation-pruning`
- **Date (UTC):** `2026-11-14`
- **Models Evaluated:** `gemini-4-argon` (Thinker), `gemini-3.8-flash` (Reasoner), `gemini-3.5-flash-lite` (Worker)

---

## 1. Architectural Changes Implemented

### Priority 3 (P3): System-1 Discovery Pre-Fetcher & Dynamic Tool Schema Pruning (`core/micro_worker.py`, `core/react.py`)
1. **System-1 `valstorm_schema_inspect` Pre-Hydration (`MicroWorkerEngine.plan_pre_turn_discovery_calls`):**
   - Uses `SystemOneClient` (`ChoiceQuestion`, `<2ms`) to evaluate whether a user prompt requesting Valstorm schema inspection (`valstorm_schema_inspect`) should be pre-executed before Iteration 0 (`confidence >= 0.50`).
   - Appends an explicit continuation note inside `tool_result.output` (`[System Note: Step 1 valstorm_schema_inspect above has already been executed. Do NOT call valstorm_schema_inspect again — immediately invoke the next required tool call...]`) so `gemini-3.5-flash-lite` immediately executes Step 2 (`valstorm_record_cud`) on Iteration 0 without pausing.
2. **Dynamic Tool Schema Pruning (`MicroWorkerEngine.filter_schemas_for_turn`):**
   - When a turn has a large tool registry (`>12` tools, e.g., 37–55 tools in `unscoped` or full `developer` / `valstorm-assistant` profiles), scopes tool declarations down to the core developer + domain-matched tools on `worker` / `reasoner` turns and down to `MECHANICAL_VERIFICATION_TOOLS` on offloaded post-synthesis iterations.
   - Deterministic per `(user_input, locked_tier, is_offloaded)` so `tools_hash` and KV prefix caches never drift across iterations of the same model.

### Priority 4 (P4): System-1 Intra-Tool Relevance Pruning & Post-Turn Consumed Observation Folding (`core/compaction.py`, `core/react.py`)
1. **Intra-Tool System-1 Relevance Pruning (`prune_tool_observation_s1`):**
   - When noisy discovery tools (`search_files`, `web_scrape`, `valstorm_schema_inspect`, `valstorm_sql_query`, `valstorm_vfs_search`) return outputs `>2,800` chars, scores middle lines/blocks against `user_query` tokens using System-1 lexical/structural scoring and prunes low-relevance filler lines before the `ToolResult` enters `session.messages`.
2. **Post-Turn Consumed Observation Folding (`ContextCompactor.fold_consumed_turn_observations`):**
   - At `StreamEventType.TURN_COMPLETE` (once all tool calls for the turn have finished and the assistant has synthesized its final response), folds bulky consumed tool outputs (`>360` chars) into compact `~90-token` structural receipts (`[Compacted Output | tool=...; file=...; symbols=[...]; records=...; ids=[...]]`) while keeping the assistant's final response 100% intact for subsequent turns.

---

## 2. Live Gemini API Proof-Check #1: Single-Turn `OF-02` (`valstorm_schema_inspect` $\rightarrow$ `valstorm_record_cud`)

| Stage | Model | Pass | LLM Turns | Wall-Clock Latency | Prompt Tokens | `thoughts_tokens` | Est. Cost ($) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PRE_P3_P4` (`v4.3.7` Post-P2)** | `gemini-3.5-flash-lite` | ✅ | 3 | 2.78s | 15,149 | 0 | $0.00312 |
| **`POST_P3_P4` (`FI-566`)** | `gemini-3.5-flash-lite` | **✅** | **2 (-33.3%)** | **2.04s (-26.6%)** | **10,460 (-31.0%)** | **0** | **$0.00268 (-14.1%)** |

---

## 3. Live Gemini API Proof-Check #2: 5-Turn Multi-Step Rolling Session (`PRE_P3_P4` vs. `POST_P3_P4`)

A realistic 5-turn multi-domain session executed against the live Gemini API (`gemini-4-argon`, `gemini-3.8-flash`, `gemini-3.5-flash-lite`) with the full 37-tool registry (`tools_mode="unscoped"`):
- **Turn 1 (`gemini-4-argon` $\rightarrow$ `gemini-3.5-flash-lite`):** Read `src/pool.py` (with 60-line module header) and `test/test_pool.py`, patch off-by-one capacity bug, and run `pytest`.
- **Turn 2 (`gemini-3.8-flash`):** Scrape `https://competitor.com/pricing` and run `web_content_diff`.
- **Turn 3 (`gemini-3.5-flash-lite`):** Inspect `campaign` schema via `valstorm_schema_inspect` and create `Q4 Enterprise Open Model Launch` via `valstorm_record_cud`.
- **Turn 4 (`gemini-3.5-flash-lite`):** Write `release_summary.json` via `write_file` and verify via `read_file`.
- **Turn 5 (`gemini-3.8-flash`):** Multi-turn recall test verifying Turn 1 bug fix details, Turn 3 campaign name, and Turn 4 filename without calling tools.

| Turn | Metric | `PRE_P3_P4` (`v4.3.7`) | `POST_P3_P4` (`FI-566`) | P3 + P4 Improvement |
| :--- | :--- | :---: | :---: | :--- |
| **Turn 1** (`Pool Bug Fix`) | Prompt Tokens / Cost | `40,721` / `$0.03653` (`3 calls`) | **`19,099` / `$0.00807` (`2 calls`)** | **`-53.1%` prompt tokens, `-77.9%` cost (`4.53x` cheaper)** |
| **Turn 2** (`Web Scrape + Diff`) | Prompt Tokens / Cost | `26,026` / `$0.02498` (`12.99s`) | **`5,262` / `$0.00878` (`10.36s`)** | **`-79.8%` prompt tokens (`4.95x` smaller), `-64.9%` cost, `-20.2%` latency** |
| **Turn 3** (`Schema + CUD`) | Prompt Tokens / Latency | `40,629` / `2.94s` (`3 calls`) | **`9,561` / `1.57s` (`2 calls`)** | **`-76.5%` prompt tokens (`4.25x` smaller), `-46.6%` faster, `-1` LLM call** |
| **Turn 4** (`Write + Read JSON`) | Prompt Tokens / Latency | `42,356` / `2.63s` (`3 calls`) | **`14,771` / `1.99s` (`3 calls`)** | **`-65.1%` prompt tokens (`2.87x` smaller), `-24.3%` faster** |
| **Turn 5** (`Multi-Turn Recall`) | Prompt Tokens / Latency | `14,331` / `2.84s` (`100% recall`) | **`5,151` / `2.03s` (`100% recall`)** | **`-64.1%` prompt tokens (`2.78x` smaller), `-28.5%` faster, `100%` recall** |
| **5-Turn Session Total** | **Cumulative Prompt Tokens** | `164,063` | **`53,844`** | **`-67.2%` (`3.05x` fewer prompt tokens; `-110,219` tokens saved)** |
| **5-Turn Session Total** | **Cumulative Session Cost ($)** | `$0.07359` | **`$0.03169`** | **`-56.9%` (`2.32x` cheaper across the 5-turn session)** |
| **5-Turn Session Total** | **Total LLM Calls & Recall** | `12 calls` / `100% recall` | **`10 calls` / `100% recall`** | **`-16.7%` fewer LLM calls, zero loss in multi-turn recall (`75/75` pytest pass)** |

---

## 4. Live Gemini API Proof-Check #3: Priority 5 (P5) Background Curators (`MemoryExtractor` & `SkillCurator`)

| Curation Scenario | `PRE_P5` (`gemini-4-argon` inherited from session) | **`POST_P5` (`System-1` + `gemini-3.5-flash-lite`)** | P5 Improvement |
| :--- | :---: | :---: | :--- |
| **1. Explicit Rule Memory Extraction** (`"Always run chmod +x..."`) | `6.82s` / `349` in / `163` thoughts / `$0.00275` | **`0.0007s` (`0.7ms`) / `0` tok / `$0.00000`** | **`-100.0%` cost (`0` LLM calls), `9,742x` faster** |
| **2. Complex Supersession Memory Extraction** (`"Switched from Twilio to AWS SNS"`) | `4.68s` / `348` in / `74` thoughts / `$0.00193` | **`0.72s` / `313` in / `0` thoughts / `$0.00023`** | **`-88.0%` cost (`8.39x` cheaper), `6.50x` faster** |
| **3. Post-Trajectory Skill Curation** (`error_recovery` trace) | `5.73s` / `626` in / `203` thoughts / `$0.00386` | **`0.77s` / `596` in / `0` thoughts / `$0.00032`** | **`-91.7%` cost (`12.1x` cheaper), `7.44x` faster** |
| **Total P5 Background Curation Suite** | `17.23s` / `440` thoughts / `$0.00854` | **`1.49s` / `0` thoughts / `$0.00055`** | **`-93.6%` cost (`15.5x` cheaper), `-91.4%` latency (`11.6x` faster)** |

---

## 5. Live Gemini API Proof-Check #4: Priority 6 (P6) System-1 Tool Argument Self-Healing (`core/tool_healer.py`)

| Mismatch Scenario | `PRE_P6` (No Self-Healing) | **`POST_P6` (`System-1` Self-Healing)** | P6 Improvement |
| :--- | :---: | :---: | :--- |
| **1. `P6-1` File Path Mismatch** (`'pool_manager.py'` $\rightarrow$ `'src/services/pool_manager.py'`) | `7 calls` / `12.94s` / `9,351` tok / `$0.01570` | **`3 calls` / `7.92s` / `4,643` tok / `$0.00806`** | **`-57.1%` calls, `-38.8%` latency, `-50.3%` tokens, `-48.7%` cost** |
| **2. `P6-2` SQL Plural Table & Column Mismatch** (`FROM leads` $\rightarrow$ `FROM lead`) | `5 calls` / `3.86s` / `15,484` tok / `$0.00529` | **`2 calls` / `1.67s` / `5,847` tok / `$0.00189`** | **`-60.0%` calls, `-56.7%` latency, `-62.2%` tokens, `-64.3%` cost** |
| **3. `P6-3` Record CUD Plural Object + Field Synonym + Lookup Dict** | `3 calls` / `5.80s` / `9,275` tok / `$0.01678` | **`2 calls` / `6.72s` / `6,046` tok / `$0.01037`** | **`-33.3%` calls, `-34.8%` tokens, `-38.2%` cost** |
| **Total P6 Self-Healing Suite** | `15 calls` / `22.60s` / `34,110` tok / `$0.03777` | **`7 calls` / `16.31s` / `16,536` tok / `$0.02032`** | **`-53.3%` fewer LLM calls, `-51.5%` prompt tokens, `-27.8%` latency, `-46.2%` cost** |

---

## 6. OpenAI GPT-6 Model Tier Mapping (`PROVIDER_TIER_MODELS["openai"]`)

| Capability Tier | Model ID | Role |
| :--- | :--- | :--- |
| **`worker`** | `gpt-6-luna` | High-speed mechanical tool routing, background curation (`MemoryExtractor`/`SkillCurator`), and lightweight tasks |
| **`reasoner`** | `gpt-6-1-sol` | Mid-tier workhorse for standard code generation, surgical patching, web search, and UI navigation |
| **`thinker`** | `gpt-6-astra` | Deep architectural reasoning, Red-Green-Refactor TDD loops, and complex debugging |


