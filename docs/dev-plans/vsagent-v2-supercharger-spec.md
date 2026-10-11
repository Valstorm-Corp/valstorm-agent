# Valstorm Agent Runtime (vsagent) v2.0 Architecture & Supercharger Specification

## 1. Executive Summary

This document specifies the architectural evolution of **`vsagent` (Valstorm Agent Runtime)** inspired by the strengths of modern coding agents (such as OpenCode) while preserving and extending Valstorm's native enterprise platform capabilities (CRM, multi-tenant DB, VFS, Slack, Cloud sandboxes, and 170+ procedural skills).

The v2.0 upgrade focuses on four core architectural pillars:
1. **Universal Model Freedom & BYOK** via standard OpenAI `/v1/chat/completions` API specification (DeepSeek, Groq, Kimi, Ollama, vLLM).
2. **Compiler-Driven Language Server Protocol (LSP) Loop** for real-time type/syntax self-correction before running tests.
3. **Transactional Git Snapshots & Instant `/undo` Rollback**.
4. **Aggressive Swarm & Cost Optimization** using cognitive tier routing to slash token consumption by 80–90%.
5. **Local-First Privacy Mode & Vendor-Neutral `AGENTS.md` Directives**.

---

## 2. High-Level Architecture Overview

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                           vsagent v2.0 Supercharger                               │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         │
     ┌───────────────────┬───────────────┴───────────────┬───────────────────┐
     ▼                   ▼                               ▼                   ▼
┌──────────────┐ ┌────────────────┐             ┌────────────────┐ ┌─────────────────┐
│   PHASE 1    │ │    PHASE 2     │             │    PHASE 3     │ │     PHASE 4     │
│ OpenAI-Spec  │ │ Language Server│             │ Transactional  │ │   Aggressive    │
│  BYOK Engine │ │ Protocol (LSP) │             │ Git Snapshots  │ │ Subagent Swarm  │
│ (DeepSeek,   │ │  Loop for Code │             │   & /undo      │ │   Cost Routing  │
│ Ollama, vLLM)│ │  Diagnostics   │             │   Rollbacks    │ │  (80%+ Savings) │
└──────────────┘ └────────────────┘             └────────────────┘ └─────────────────┘
```

---

## 3. Comparison & Architectural Matrix

| Feature | `vsagent` (Current) | OpenCode Reference | `vsagent v2.0` (Target) |
| :--- | :--- | :--- | :--- |
| **Model Connectivity** | Gemini, Claude, OpenAI native SDKs | 75+ Providers via OpenAI spec + Ollama/LM Studio | Native Multi-Provider + Universal OpenAI-compatible endpoint router (`base_url` + custom auth) |
| **Code Validation** | Runtime test execution (`terminal_exec`, `pytest`) | Real-time Language Server Protocol (LSP) diagnostics | In-memory LSP daemon attaching compiler diagnostics directly to `patch_file`/`write_file` results |
| **State & Rollbacks** | SQLite session persistence & memory | SQLite session history + Git tree snapshots (`/undo`) | Atomic pre-mutation Git tree snapshotting + interactive `/undo` rollback command |
| **Subagents & Routing** | Named profiles (`developer`, `architect`, etc.) | Generic subagents (`Explore`, `Scout`, `General`) | Tier-optimized swarms: Fast open-weights for exploration, heavy reasoners for synthesis |
| **Enterprise Tooling** | SQL, Mongo, VFS, Record CUD, Slack, Skills | None (pure coding/file tools) | Retains 100% full-stack Valstorm platform integration + cloud profile/skill sync |
| **Privacy & Directives** | `CLAUDE.md`, Local SQLite storage | `AGENTS.md` standard, offline mode | `AGENTS.md` + `CLAUDE.md` fallback, explicit `--offline` zero-telemetry enforcement |

---

## 4. Phase-by-Phase Implementation Specifications

### Phase 1: Universal Model Freedom via OpenAI API Spec (BYOK)

#### Objective
Enable `vsagent` to connect to **any OpenAI-compatible API endpoint** (DeepSeek, Groq, Kimi/Moonshot, Mistral, Together AI, or 100% offline local inference via Ollama / vLLM / LM Studio) without extra SDK dependencies.

#### Key Updates:
1. **`providers/openai.py` Enhancement**:
   * Support dynamic `base_url`, custom default headers, and non-standard model parameters.
   * Auto-detect and configure endpoints for DeepSeek (`https://api.deepseek.com/v1`), Groq (`https://api.groq.com/openai/v1`), Ollama (`http://localhost:11434/v1`), and vLLM (`http://localhost:8000/v1`).
2. **`core/keystore.py` & CLI Config**:
   * Expand keystore resolution to manage provider base URLs and model aliases.
   * CLI commands:
     ```bash
     vsagent keys set deepseek <API_KEY>
     vsagent keys set ollama http://localhost:11434/v1 --type base_url
     ```
3. **Dynamic Provider Instantiation (`providers/__init__.py`)**:
   * Map provider strings (`"deepseek"`, `"ollama"`, `"groq"`, `"custom"`) directly to configured `OpenAIProvider` instances.

---

### Phase 2: Live Language Server Protocol (LSP) & Diagnostic Loop

#### Objective
Turn `vsagent` into a compiler-driven, self-correcting coder. When editing files (`patch_file`, `write_file`), the agent immediately receives compiler and linter diagnostics in the tool result, catching type mismatches, missing imports, and syntax errors *before* running tests.

#### Key Updates:
1. **New LSP Engine (`core/lsp/`)**:
   * **`lsp_manager.py`**: Lightweight background process manager for language servers:
     * **TypeScript/JavaScript**: `typescript-language-server` or `tsc --noEmit`
     * **Python**: `pyright`, `ruff`, or `flake8`
     * **Rust/Go/JSON**: `rust-analyzer` / `gopls`
   * Asynchronous, non-blocking diagnostic query with a strict timeout budget (<= 1.5s).
2. **Tool Mutation Interceptor (`tools/developer_tools.py`)**:
   * Hook into `patch_file` and `write_file`:
     * After successfully applying changes to disk, query the LSP daemon for errors in the modified file.
     * If errors/warnings exist, append them directly to the tool's return string:
       ```
       Successfully patched apps/api/routers/contacts.py.

       [LSP Compiler Diagnostics]:
       - Line 42: Type error: Argument 'tenant_id' cannot be None (expected str).
       - Line 48: NameError: 'datetime' is not imported.
       ```
3. **ReAct Prompt Directive Update (`core/context.py`)**:
   * Add a ReAct loop directive:
     > *"If a file modification returns [LSP Compiler Diagnostics], you MUST address and fix the reported compiler/type errors immediately on your next turn before proceeding to execution."*

---

### Phase 3: Transactional Git Snapshots & Instant `/undo`

#### Objective
Provide instant checkpointing and rollback capabilities for agent actions, giving developers complete safety during complex refactors.

#### Key Updates:
1. **Snapshot Manager (`core/snapshot.py`)**:
   * Hook into the ReAct turn lifecycle before executing mutating tools (`patch_file`, `write_file`, `terminal_exec` with write flags).
   * Create lightweight git tree references / stash hashes (`git stash create` or working-tree commit snapshots) mapped to `session_id` and `turn_number`.
2. **Rollback Mechanism & Tooling**:
   * **CLI Command**: `vsagent undo` or `/undo` in the interactive REPL.
   * **Core Engine**: Restores the working directory to the pre-turn snapshot and updates the SQLite session history, marking the undone turn as reverted so context remains pristine.
3. **Audit Log in SQLite (`core/storage.py`)**:
   * Add `git_snapshot_hash` and `changed_files` columns to the message/turn records in `storage.db`.

---

### Phase 4: Aggressive Subagent Swarms & Cost Routing

#### Objective
Maximize execution speed and achieve 80–90% token cost reduction by aggressively routing tasks to the optimal cognitive model tier.

#### Key Updates:
1. **Profile Tier Re-mapping (`core/context.py` & `~/.valstorm/profiles/`)**:
   * **Tier 3 / High-Speed Open-Weight Workers** (DeepSeek V3, Gemini Flash, Local Ollama Qwen 2.5 Coder):
     * `researcher`: Deep file searches, grep analysis, reading logs, doc indexing (~$0.14/1M tokens or $0.00 locally).
     * `scout`: External documentation scraping and VFS browsing.
     * `backend-tester`: Running test suites, collecting output, and summarizing failure traces.
   * **Tier 1 / Heavy Reasoning Engines** (Claude 3.7 Sonnet, Gemini 2.5 Pro, o3-mini):
     * `architect`: High-level system design, migration planning, complex cross-file refactoring.
     * `orchestrator`: Breaking user prompts into subagent swarm tasks and synthesizing final outputs.
2. **Context Budgeting for Subagents (`tools/delegation.py`)**:
   * When `delegate_task` is invoked, pass only relevant file paths and instructions rather than dumping orchestrator conversation history.
   * On completion, return a compact distilled report back to the parent agent.

---

### Phase 5: Workspace Standard (`AGENTS.md`) & Zero-Telemetry Mode

#### Objective
Ensure 100% portability and local privacy compliance for enterprise MSP deployments.

#### Key Updates:
1. **Rule Discovery (`core/context.py`)**:
   * Support hierarchical rule loading:
     1. `AGENTS.md` (universal standard)
     2. `CLAUDE.md` (legacy fallback)
     3. `.cursorrules` / `.vscode/` rules
2. **Explicit Offline / Local-Only Flag**:
   * Add `vsagent chat --offline` or `VALSTORM_LOCAL_ONLY=1`:
     * Disables any external telemetry, cloud sync checks, or external tool callbacks.
     * Routes all generation through local Ollama / vLLM endpoints.

---

## 5. Execution Order & Milestone Roadmap

| Milestone | Phase | Primary Deliverables | Target Area |
| :--- | :--- | :--- | :--- |
| **M1** | Universal BYOK & OpenAI-Spec | Configurable `base_url`, DeepSeek/Ollama provider aliases, key manager updates. | `providers/`, `core/keystore.py` |
| **M2** | LSP Diagnostic Feedback Loop | `core/lsp/`, diagnostic interceptor in `patch_file`/`write_file`, prompt updates. | `core/lsp/`, `tools/developer_tools.py` |
| **M3** | Git Snapshots & `/undo` | Snapshot manager, REPL `/undo` command, SQLite turn state tracking. | `core/snapshot.py`, `cli/repl.py` |
| **M4** | Subagent Swarm Cost Optimization | Updated profile tiers, fast open-weight model defaults for `researcher`/`scout`. | `core/context.py`, `tools/delegation.py` |
| **M5** | `AGENTS.md` & Local Privacy Mode | `AGENTS.md` discovery, `--offline` enforcement, zero-leak verification. | `core/context.py`, `cli/main.py` |
