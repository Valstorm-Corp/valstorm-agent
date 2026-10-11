# Architecture Overview (Valstorm Internal Developers)

## Overview
`apps/agent-runtime` is an isolated, zero-legacy execution engine that powers autonomous ReAct agent loops in the Valstorm ecosystem. It decouples LLM generation, tool schema introspection, and telemetry tracking from specific web or database frameworks.

```
┌─────────────────────────────────────────────────────────────┐
│                            CLI                              │
│   (Interactive REPL / single-run task invocation)          │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                         ReActEngine                         │
│   Thought -> Tool Execution -> Observation -> Final Answer  │
│   • Tracks per-step: Model, Duration, In/Out Tokens         │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
┌──────────────▼──────────────┐┌──────────────▼───────────────┐
│     KeyStore Keystore       ││        ToolRegistry          │
│  (Env vars + local config/  ││  (@tool decorator ->         │
│   keyring for API keys)     ││   JSON Schema tool defs)     │
└──────────────┬──────────────┘└──────────────────────────────┘
               │
┌──────────────▼──────────────┐
│     BaseProvider Adapters   │
│  • GeminiProvider           │
│  • OpenAIProvider           │
└─────────────────────────────┘
```

## Core Modules

### 1. `core/models.py`
Defines the strict Pydantic schemas for the runtime:
- **`UsageMetadata`**: `prompt_tokens`, `completion_tokens`, `total_tokens`, `cached_tokens`.
- **`ToolCall`**: Unique `id`, tool `name`, parsed `arguments` dictionary.
- **`ToolResult`**: Associated `call_id`, `name`, `output`, `is_error` flag.
- **`Message`**: Unified message representation storing `id`, `role`, `content`, `model`, `provider`, `tool_calls`, `tool_result`, `usage`, and `timestamp`.
- **`SessionState`**: Holds conversation history, `active_model`, `active_provider`, and computes running totals for `total_prompt_tokens`, `total_completion_tokens`, and `total_tokens`.

### 2. `core/react.py`
The ReAct execution loop orchestrator:
- Manages conversational turns.
- Introspects tool schemas from the `ToolRegistry`.
- Dispatches tool invocations and creates `ToolResult` messages.
- Prevents infinite cycles with `max_iterations`.
- Invokes real-time telemetry callbacks (`step_callback`).

### 3. `core/keystore.py`
Credential management engine enforcing hierarchical resolution order.

### 4. `core/tools.py`
Introspects Python signatures and docstrings into JSON Schema definitions.
