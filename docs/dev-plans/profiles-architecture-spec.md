# Architectural Specification: Valstorm Agent Profile System

## 1. Executive Summary & Core Decisions

The **Valstorm Agent Profile System** enables users and teams to configure, switch between, and orchestrate specialized AI personas (e.g. `orchestrator`, `developer`, `researcher`, `valstorm-assistant`).

### Core Architectural Decisions:
1. **Hybrid Source of Truth**: Profiles are canonically authored in the Valstorm platform as `ai_agent` records and cached locally at `~/.valstorm/profiles/<name>.json` for zero-latency startup and offline reliability.
2. **Comprehensive Profile Scope**: A profile encapsulates:
   - Default Model & Provider (e.g. `gemini-flash-latest`, `gpt-4o-mini`, `claude-sonnet-5`).
   - System Instructions & Persona.
   - Toolset Whitelist (restricts which tools the profile can see and execute).
   - Profile-Scoped Memory (facts specific to that persona).
3. **Unified Single Database with Profile Tagging**: Sessions and messages live in the shared SQLite database (`~/.valstorm/agent_state.db`), tagged with `profile` for cross-profile search and analytics without database fragmentation.
4. **Named Profile Delegation (`delegate_task`)**: Orchestrators can spawn specialist subagents by profile name (e.g. `delegate_task(profile="developer", goal="...")`), automatically provisioning the child agent with the specialist's system prompt, model, and toolset.

---

## 2. Architecture & Data Flow

```
                                  ┌─────────────────────────────────────────────────────────┐
                                  │            Valstorm Cloud (ai_agent Schema)             │
                                  │  • Canonical Persona Registry                           │
                                  │  • System Prompts, Default Models, Whitelisted Tools    │
                                  └───────────────────────────┬─────────────────────────────┘
                                                              │
                                            Sync on Login / On-Demand Pull
                                                              │
┌─────────────────────────────────────────────────────────────▼─────────────────────────────────────────────────────────┐
│                                             Local Valstorm Agent Runtime                                              │
│                                                                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                                          ProfileManager (core/profile.py)                                       │  │
│  │  • Resolves active profile from: CLI flag (--profile) | REST payload | Local Cache (~/.valstorm/profiles/)      │  │
│  │  • Filters ToolRegistry to only include profile's whitelisted tools                                             │  │
│  │  • Injects profile-specific instructions and memory into WorkspaceContextManager                                │  │
│  └──────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────┘  │
│                                                             │                                                         │
│                                  ┌──────────────────────────┴──────────────────────────┐                              │
│                                  ▼                                                     ▼                              │
│  ┌─────────────────────────────────────────────────────────────┐   ┌───────────────────────────────────────────────┐  │
│  │                ReAct Execution Engine                       │   │             Shared SQLite State DB            │  │
│  │  • Active Model: Profile.model                              │   │          (~/.valstorm/agent_state.db)         │  │
│  │  • Allowed Tools: Profile.tools                             │   │                                               │  │
│  │  • Subagent Delegation: Spawns Named Profile Subagents      │   │  • ai_chat (tagged with profile column)       │  │
│  │    (e.g., delegate_task(profile="developer", goal="..."))   │   │  • ai_chat_message (tagged with profile)      │  │
│  └─────────────────────────────────────────────────────────────┘   └───────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Profile Schema & Data Contract

### Profile JSON Definition (`~/.valstorm/profiles/<name>.json`):
```json
{
  "id": "agent_developer_01",
  "name": "developer",
  "display_name": "Full-Stack Developer",
  "description": "Specialized in monorepo coding, refactoring, test execution, and debugging.",
  "provider": "gemini",
  "model": "gemini-flash-latest",
  "system_prompt": "You are an expert full-stack developer. Always write tests and verify with terminal_exec.",
  "allowed_tools": [
    "terminal_exec",
    "patch_file",
    "write_file",
    "read_file",
    "search_files",
    "valstorm_sql_query",
    "valstorm_schema_inspect",
    "calculator"
  ],
  "max_turns": 30,
  "created_at": "2026-08-18T12:00:00Z"
}
```

---

## 4. Component Deliverables

### A. `core/profile.py` (`ProfileManager`)
- **`get_profile(name: str) -> Profile`**:
  - Resolves profile from local cache `~/.valstorm/profiles/<name>.json`.
  - If missing locally, pulls `ai_agent` record from Valstorm Cloud API (`GET /schema/ai_agent` or `GET /object/ai_agent`) and caches it.
  - Built-in standard fallbacks: `orchestrator`, `developer`, `writer`, `valstorm-assistant`, `default`.
- **`list_profiles() -> List[Profile]`**:
  - Lists all local and cloud-synced profiles.
- **`sync_profiles(client: ValstormApiClient) -> int`**:
  - Syncs all cloud `ai_agent` records to `~/.valstorm/profiles/`.

### B. Dynamic Tool Filtering (`core/tools.py`)
- `ToolRegistry.filter_by_whitelist(allowed_tool_names: List[str]) -> ToolRegistry`:
  - Clones or scopes the registry so that the LLM only receives schemas for tools allowed by the active profile.

### C. Multi-Agent Delegation (`tools/delegation_tool.py`)
- `@tool delegate_task(profile: str, goal: str, context: Optional[str] = None) -> str`:
  - Spawns an isolated subagent turn running the named `profile`.
  - Subagent inherits parent's credentials but runs with the child profile's model, prompt, and tool whitelist.
  - Returns structured result summary to the parent orchestrator.

### D. Server & CLI Integration (`server.py` & `cli.py`)
- **`server.py`**: When `payload.model` or `payload.agent_id` specifies a profile name (e.g. `"model": "orchestrator"`), resolves the profile, applies its model, instructions, and tool whitelist.
- **`cli.py`**: Supports `--profile <name>` CLI option and `/profile <name>` REPL command to switch active profiles on the fly.

---

## 5. Implementation Roadmap

1. **Step 1:** Implement `core/profile.py` (`Profile` dataclass & `ProfileManager`) with default built-in profile templates.
2. **Step 2:** Add `ToolRegistry.filter_by_whitelist()` in `core/tools.py`.
3. **Step 3:** Implement `tools/delegation_tool.py` (`delegate_task`) for named profile subagent spawning.
4. **Step 4:** Integrate `ProfileManager` into `server.py` and `cli.py`.
5. **Step 5:** Write unit and live integration test suite (`tests/test_profiles.py`).
