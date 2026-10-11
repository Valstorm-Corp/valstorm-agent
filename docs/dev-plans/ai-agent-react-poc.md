# Bare-Bones ReAct AI Runtime & Token Tracking POC

## 1. Project Location in Monorepo
**Path:** `/Users/jared/Documents/Code/monorepo/apps/agent-runtime`

Isolated Python project managed via `uv` or standard venv with zero legacy coupling:
```text
apps/agent-runtime/
├── pyproject.toml
├── README.md
├── cli.py                    # Interactive REPL & single-turn runner
├── core/
│   ├── __init__.py
│   ├── models.py             # Message, ToolCall, ToolResult, UsageMetadata, SessionState
│   ├── keystore.py           # Multi-provider API key resolution
│   ├── tools.py              # @tool decorator & ToolRegistry (Schema introspection)
│   └── react.py              # ReAct loop engine (Thought -> Action -> Observation)
├── providers/
│   ├── __init__.py
│   ├── base.py               # BaseProvider interface
│   ├── gemini.py             # Gemini provider (tracks model & usage_metadata)
│   └── openai.py             # OpenAI provider (tracks model & usage tokens)
└── tests/
    ├── test_keystore.py
    ├── test_tools.py
    ├── test_providers.py
    └── test_react_loop.py
```

---

## 2. Parallel Delegation Strategy (3 Concurrent Subagents)

To achieve maximum build velocity, the implementation is partitioned into 3 isolated workstreams:

### Agent 1: Data Contracts & Keystore
- **Scope:** `core/models.py`, `core/keystore.py`, `tests/test_keystore.py`
- **Responsibilities:**
  - Standardized Pydantic schemas for `UsageMetadata` (prompt_tokens, completion_tokens, total_tokens), `Message` (role, model, provider, usage, content, tool_calls), `ToolCall`, `ToolResult`.
  - Secure credential loader (CLI flag $\rightarrow$ Environment variables $\rightarrow$ `~/.config/valstorm/keys.json` / `.env`).

### Agent 2: Provider Adapters & Tool Registry
- **Scope:** `core/tools.py`, `providers/base.py`, `providers/gemini.py`, `providers/openai.py`, `tests/test_tools.py`, `tests/test_providers.py`
- **Responsibilities:**
  - `@tool` decorator converting Python functions + docstrings to standard JSON schemas.
  - Built-in test tools (`calculator`, `read_file`, `mock_db_lookup`).
  - Provider adapters mapping responses and raw token counts into unified `UsageMetadata`.

### Agent 3: ReAct Loop Engine & CLI
- **Scope:** `core/react.py`, `cli.py`, `pyproject.toml`, `tests/test_react_loop.py`
- **Responsibilities:**
  - ReAct execution loop handling multi-step function call dispatch, iteration limits, error recovery, and mid-session model tracking.
  - Clean CLI terminal interface supporting single-shot queries and multi-turn REPL chat with visible per-turn token telemetry.

---

## 3. Verification & Acceptance Criteria
1. Unit tests pass cleanly with `pytest apps/agent-runtime/tests/`.
2. Running `python cli.py "Calculate (45 * 12) + 180 and explain"` runs the loop, calls the `calculator` tool, returns the result, and outputs the exact model name and token breakdown.
3. Mid-session model switching in REPL preserves message history and tags each message with its specific generating model.
