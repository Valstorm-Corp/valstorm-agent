# Agent Runtime Documentation

Welcome to the **Valstorm Agent Runtime** documentation. This runtime is a modular, high-performance ReAct AI execution engine featuring automatic tool schema generation, multi-provider support, real-time token tracking per message turn, sub-millisecond micro-telemetry, developer execution tools, SQLite WAL session persistence, FTS5 historical search, declarative long-term memory, and authenticated Valstorm platform tool integration.

---

## Documentation Categories

The documentation is organized into three tailored tracks:

### 1. [Valstorm Internal Developers](./internal-developers/)
*For engineers maintaining the runtime core, monorepo integrations, provider adapters, and billing pipelines.*
- **[Architecture Overview](./internal-developers/architecture-overview.md)**: Engine design, data contracts, and monorepo structure.
- **[Session Storage & Declarative Memory](./internal-developers/session-storage-and-memory.md)**: SQLite WAL database, FTS5 virtual tables, and memory stores.
- **[Valstorm REST Tools Integration](./internal-developers/valstorm-tools-integration.md)**: Direct REST API tool adapters, security gates, and endpoints.
- **[Provider Implementation](./internal-developers/provider-implementation.md)**: How LLM providers (Gemini, OpenAI, Anthropic) are implemented and normalized.
- **[Token Telemetry & Billing](./internal-developers/token-telemetry-and-billing.md)**: How token usage metadata is extracted and aggregated.
- **[ReAct Loop Internals](./internal-developers/react-loop-internals.md)**: Concurrent parallel tool dispatch, micro-telemetry, and smart windowing.

---

### 2. [Developers](./developers/)
*For engineers building tools, integrating the runtime into applications, or extending agents.*
- **[Getting Started](./developers/getting-started.md)**: Environment setup, dependencies, and basic execution.
- **[Session Persistence & State](./developers/session-and-state.md)**: Working with `SessionStore`, SQLite persistence, and FTS5 search in Python.
- **[Valstorm Platform Tools](./developers/valstorm-tools.md)**: Using SQL, VFS search, CUD, and schema inspection tools in Python.
- **[Creating Custom Tools](./developers/creating-tools.md)**: Using the `@tool` decorator, type hints, and schemas.
- **[Testing & Mocks](./developers/testing-and-mocks.md)**: Writing unit and integration tests with mock providers.

---

### 3. [Users & Operators](./users/)
*For end users, operators, and developers testing prompts via the CLI.*
- **[Agent Profiles & Procedural Skills Guide](./users/profiles-and-skills-guide.md)**: Using named profiles (`developer`, `researcher`, `orchestrator`), 170+ procedural skills, and subagent delegation.
- **[CLI User Guide](./users/cli-guide.md)**: Interactive chat REPL (`chat`), profile switching (`/profile`), and single-shot task runner (`run`).
- **[Managing Sessions & Memory](./users/managing-sessions-and-memory.md)**: Resuming sessions (`--resume`), FTS5 search (`/search`), and viewing facts (`/memory`).
- **[Using Valstorm Tools Guide](./users/valstorm-tools-guide.md)**: Running live queries, VFS file search, and mutations in the CLI.
- **[Managing API Keys](./users/managing-api-keys.md)**: Configuring keys via CLI, interactive prompts, and config files.
- **[Switching Models Mid-Session](./users/switching-models.md)**: Changing models during active conversations without losing context.
- **[Inspecting Token Usage](./users/inspecting-tokens.md)**: Reading prompt, completion, and cumulative token metrics.
