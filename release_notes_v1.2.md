# Architecture Release Notes v1.2

## Overview
`apps/agent-runtime` is an isolated, zero-legacy execution engine that powers autonomous ReAct agent loops in the Valstorm ecosystem. It decouples LLM generation, tool schema introspection, and telemetry tracking from specific web or database frameworks.

This version (v1.2) introduces foundational architectural changes paving the way for the v2.0 "Supercharger" specification, with a primary focus on universal model compatibility.

## New Features in v1.2

### Universal Model Provider via OpenAI API Specification
- **BYOK (Bring Your Own Key):** `vsagent` can now connect to any OpenAI-compatible API endpoint. This enables the use of a wide range of models and providers without requiring dedicated SDKs.
- **Supported Providers:** Out-of-the-box support for custom endpoints and pre-configured aliases for major providers like DeepSeek, Groq, and local inference servers (Ollama, vLLM).
- **Configuration:** API keys and base URLs are managed via the `vsagent keys` CLI and the hierarchical `core/keystore.py` engine.

## Core Module Updates

### 1. `providers/openai.py` & `providers/__init__.py`
- The `OpenAIProvider` has been enhanced to support dynamic `base_url` and custom headers.
- The provider factory now dynamically instantiates `OpenAIProvider` instances for different provider strings ("deepseek", "ollama", "groq", "custom").

### 2. `core/keystore.py`
- Expanded to manage provider-specific base URLs in addition to API keys, enabling seamless switching between model endpoints.

### 3. `core/models.py`
Defines the strict Pydantic schemas for the runtime:
- **`UsageMetadata`**: `prompt_tokens`, `completion_tokens`, `total_tokens`, `cached_tokens`.
- **`ToolCall`**: Unique `id`, tool `name`, parsed `arguments` dictionary.
- **`ToolResult`**: Associated `call_id`, `name`, `output`, `is_error` flag.
- **`Message`**: Unified message representation storing `id`, `role`, `content`, `model`, `provider`, `tool_calls`, `tool_result`, `usage`, and `timestamp`.
- **`SessionState`**: Holds conversation history, `active_model`, `active_provider`, and computes running totals for `total_prompt_tokens`, `total_completion_tokens`, and `total_tokens`.

### 4. `core/react.py`
The ReAct execution loop orchestrator:
- Manages conversational turns.
- Introspects tool schemas from the `ToolRegistry`.
- Dispatches tool invocations and creates `ToolResult` messages.
- Prevents infinite cycles with `max_iterations`.
- Invokes real-time telemetry callbacks (`step_callback`).

### 5. `core/tools.py`
Introspects Python signatures and docstrings into JSON Schema definitions.

### 6. Telemetry and Observability (`core/telemetry.py`)
- **Real-time Callbacks:** The ReAct loop now supports a `step_callback` function, allowing for real-time monitoring of each step in the agent's execution cycle.
- **Detailed Token Tracking:** Granular token usage data (prompt, completion, total) is now captured for every model interaction and aggregated at the session level.
- **Error Logging:** All tool execution errors are explicitly captured in `ToolResult` messages, providing clear visibility into failures.
- **Performance Metrics:** Timestamps on each message allow for precise performance analysis of both model response times and tool execution durations.
