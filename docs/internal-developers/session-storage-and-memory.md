# Session Persistence & Declarative Memory (Internal Developers)

## Overview
Phase 4 adds local, zero-dependency persistence to the Agent Runtime using **SQLite WAL Mode** and **Declarative Fact JSON Storage**.

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Runtime (CLI)                      │
│  • Auto-saves session & messages after every turn           │
│  • Restores full message history on --resume <id>           │
│  • Injects memory facts into dynamic system prompt          │
└──────────────────────────────┬──────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
  ~/.valstorm/agent_state.db       ~/.valstorm/memories.json
     (SQLite WAL + FTS5)             (Declarative Facts)
```

---

## 1. SQLite Database Architecture (`core/storage.py`)

- **Location**: `~/.valstorm/agent_state.db`
- **Concurrency**: `PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;`
- **Tables**:
  - `sessions`: Stores session UUID, title, created/updated timestamps, active model/provider, cumulative token telemetry, and JSON metadata.
  - `messages`: Stores individual turns (`role`, `content`, `model`, `provider`, `tool_calls_json`, `tool_result_json`, `usage_json`).
  - `messages_fts`: FTS5 virtual table indexing message content with Porter stemming for instant full-text search.

---

## 2. Declarative Memory Architecture (`core/memory.py`)

- **Location**: `~/.valstorm/memories.json`
- **Stores**:
  - `user`: User profile facts, role, tone, and communication style.
  - `memory`: Environment facts, local port mappings, monorepo conventions, tool quirks.
- **System Prompt Injection**: Formatted as compact markdown bullet points automatically injected by `WorkspaceContextManager`.

---

## 3. Tool Bindings (`tools/memory_tool.py`)

- `memory_manage(action='add'|'remove'|'list', target='user'|'memory', content=..., old_text=...)`: Allows agents to proactively save and delete facts.
- `session_search(query=..., limit=5)`: Allows agents to search across past historical sessions using SQLite FTS5.
