# Phase 4 Architectural Implementation Spec: State, Session Persistence & Declarative Memory

## 1. Overview & Core Philosophy

Phase 4 establishes the persistent **State & Memory Layer** for the Valstorm Agent Runtime (`apps/agent-runtime`).

### Core Requirements:
1. **Zero State Loss Across Restarts**: Conversations automatically serialize to a local SQLite database in WAL mode after every turn.
2. **Session Lifecycle Management**: Users and subagents can list (`/sessions`), resume (`--resume <id>`), and delete sessions.
3. **High-Speed FTS5 Search**: Full-text search across all previous conversation turns, code artifacts, and tool outputs using SQLite FTS5.
4. **Declarative Fact Memory**: Long-term persistent facts (user preferences, environment quirks, lessons learned) that survive across restarts and inject into system prompts without token bloat.
5. **1:1 Valstorm Cloud Schema Mirroring**: SQLite tables directly mirror `ai_chat` and `ai_chat_message` schemas for instant, zero-transformation cloud synchronization.

---

## 2. Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Agent Runtime Client (CLI / REPL)                  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                              ReAct Engine                             │  │
│  │  • Runs Turn -> Emits StreamEvents -> Completes Turn                  │  │
│  │  • Calls memory_manage & session_search tools                         │  │
│  └──────────────────────────────────┬────────────────────────────────────┘  │
│                                     │ Auto-Saves Session Turn                │
│                                     ▼                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                           Storage & Memory Layer                      │  │
│  │                                                                       │  │
│  │   ┌──────────────────────────────┐   ┌────────────────────────────┐   │  │
│  │   │        SessionStore          │   │        MemoryStore         │   │  │
│  │   │  - SQLite (WAL Mode)         │   │  - User Profile Facts      │   │  │
│  │   │  - ai_chat & message Tables  │   │  - Environment Notes       │   │  │
│  │   │  - FTS5 Full-Text Index      │   │  - System Prompt Injector  │   │  │
│  │   └──────────────┬───────────────┘   └─────────────┬──────────────┘   │  │
│  └──────────────────┼─────────────────────────────────┼──────────────────┘  │
└─────────────────────┼─────────────────────────────────┼─────────────────────┘
                      │                                 │
                      ▼                                 ▼
         ~/.valstorm/agent_state.db           ~/.valstorm/memories.json
             (SQLite WAL + FTS5)                 (Declarative Facts)
```

---

## 3. Database Schema (SQLite WAL Mode — Valstorm 1:1 Mirror)

```sql
-- 1. ai_chat Table (Mirrors Valstorm ai_chat.json schema)
CREATE TABLE IF NOT EXISTS ai_chat (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT DEFAULT 'Active',
    created_date TEXT NOT NULL,
    modified_date TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    total_input_tokens INTEGER DEFAULT 0,
    total_output_tokens INTEGER DEFAULT 0,
    metadata_json TEXT DEFAULT '{}'
);

-- 2. ai_chat_message Table (Mirrors Valstorm ai_chat_message.json schema)
CREATE TABLE IF NOT EXISTS ai_chat_message (
    id TEXT PRIMARY KEY,
    ai_chat TEXT NOT NULL,
    name TEXT NOT NULL,
    role TEXT NOT NULL,
    body TEXT,
    tool_calls TEXT,
    tool_result TEXT,
    system_prompt TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    model TEXT,
    provider TEXT,
    created_date TEXT NOT NULL,
    modified_date TEXT NOT NULL,
    FOREIGN KEY(ai_chat) REFERENCES ai_chat(id) ON DELETE CASCADE
);

-- 3. FTS5 Virtual Index for Full-Text Search
CREATE VIRTUAL TABLE IF NOT EXISTS ai_chat_message_fts USING fts5(
    ai_chat UNINDEXED,
    message_id UNINDEXED,
    role UNINDEXED,
    body,
    tokenize='porter unicode61'
);
```

---

## 4. Component Breakdown & Deliverables

### A. `apps/agent-runtime/core/storage.py` (`SessionStore`)
- **WAL Mode Connection**: Opens SQLite with `PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;`.
- **`save_session(session: SessionState, title: Optional[str] = None) -> None`**: Upserts `ai_chat` row and appends uncommitted `ai_chat_message` records with base64-encoded `thought_signature` bytes.
- **`load_session(session_id: str) -> Optional[SessionState]`**: Reconstructs complete `SessionState` including message history, model tracking, thought signatures, and token aggregates.
- **`list_sessions(limit: int = 20) -> List[Dict[str, Any]]`**: Returns recent sessions with title, updated timestamp, total tokens, and message count.
- **`search_sessions(query: str, limit: int = 5) -> List[Dict[str, Any]]`**: Executes FTS5 query with snippet highlighting.

### B. `apps/agent-runtime/core/memory.py` (`MemoryStore`)
- **Declarative Memory Structure**:
  - `user`: User details, preferences, communication style.
  - `memory`: Environment facts, monorepo conventions, lessons learned.
- **`add_fact(target: str, content: str) -> str`**: Deduplicates and stores fact.
- **`remove_fact(target: str, old_text: str) -> bool`**: Fuzzy/substring match removal.
- **`format_for_system_prompt() -> str`**: Generates compact, high-signal system prompt block.

### C. `apps/agent-runtime/tools/memory_tool.py`
- `@tool memory_manage(action: str, target: str, content: Optional[str], old_text: Optional[str]) -> str`
- `@tool session_search(query: str, limit: int = 5) -> str`

### D. CLI Commands & Workflow Updates (`cli.py`)
- `--resume <session_id>`: Resume prior session.
- REPL `/sessions`: Display interactive list of recent sessions.
- REPL `/resume <id>`: Switch session mid-REPL.
- REPL `/memory`: View and inspect memory stores.
- REPL `/search <query>`: FTS5 historical search.
