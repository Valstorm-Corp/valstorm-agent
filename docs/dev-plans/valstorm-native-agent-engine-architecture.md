# Valstorm-Native Agent Engine: Architectural Blueprint & Strategic Roadmap

## 1. Vision & Core Philosophy

The goal is a lean, enterprise-grade, high-performance **Valstorm Agent Engine** designed to operate seamlessly across both local developer machines (Desktop/CLI) and cloud cluster workers.

Unlike generalized consumer agent frameworks that suffer from feature creep (skins, pet mascots, sprawling directory clones, direct consumer chat bots), the Valstorm Agent Engine focuses on:
- **Deterministic, high-reliability execution** (file operations, PTY subprocesses, code patching, tool calling).
- **Native platform integration** (Valstorm CLI, VFS, custom objects/records, automation flows).
- **Unified Hybrid Vector RAG** (matching local workspace context with cloud Qdrant collections).
- **Hybrid Client-Cloud Connectivity** (Valstorm cloud acts as the permissioned OAuth/telephony/messaging hub, streaming tasks down to local or cluster agent workers).
- **Zero Session Bleed & Hermetic Isolation** (complete architectural separation between UI chats, CLI runs, and automated workers).
- **Universal Metering & Telemetry (BYOK + Platform Monetization)** (accurate token/tool accounting for per-action billing and platform markups).
- **Native Multimodal Media Engine** (seamless support for screenshot pasting, image analysis, audio transcripts, and video inspection).

---

## 2. Multimodal Media Pipeline (Images, Screenshots, Audio & Video)

When working on modern applications, dropping screenshots, error logs, UI mockups, audio recordings, or video walkthroughs into the chat is a first-class requirement.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Multimodal Ingestion                           │
│   - Desktop Paste (Clipboard Image)   - Drag-and-Drop Video / Audio File │
│   - VFS File Attachment               - CLI File Path Reference          │
└─────────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   Standardized Media Attachment Layer                    │
│   - Local Cache / Hashing: ~/.valstorm/media_cache/<sha256>.<ext>        │
│   - MIME Inspection & Automatic Downsampling / Pre-processing            │
│   - VFS Upload Hook (Presigned S3/Vault Sync)                            │
└─────────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   Provider-Specific Multimodal Adapter                   │
│                                                                          │
│  ┌───────────────────────┐  ┌───────────────────────┐  ┌──────────────┐  │
│  │    Google Gemini      │  │   Anthropic Claude    │  │  OpenAI GPT  │  │
│  │ - Inline Image Data   │  │ - Base64 Image Blocks │  │ - Image URLs │  │
│  │ - Native Audio / Video│  │ - PDF Document Blocks │  │ - Audio Chunks│ │
│  │   File API URI Upload │  │                       │  │              │  │
│  └───────────────────────┘  └───────────────────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

### Multimodal Format Capabilities by Asset Type:
1. **Screenshots & Images (PNG, JPEG, WebP):**
   - Direct clipboard paste in Desktop/Web UI (`⌘V`).
   - Automatically converted to base64 or inline data blocks for vision-capable models (Claude 3.5/3.7, Gemini 1.5/2.0, GPT-4o).
2. **Audio (MP3, WAV, M4A, OGG):**
   - **Native Multimodal Models (e.g. Gemini):** Raw audio bytes or File API URIs passed directly for native acoustic understanding, tone, and speech nuances.
   - **Text-Only Models (e.g. Claude):** Automatic upstream transcription via Whisper / FastWhisper before entering the ReAct turn loop.
3. **Video (MP4, WebM):**
   - **Native Video Models (Gemini):** Full video ingestion via Valstorm File API / Gemini File URI for multi-minute screen recordings and bug walkthroughs.
   - **Vision-Only Models (Claude/OpenAI):** Intelligent frame sampling (e.g., 1 frame every 2 seconds + extracted audio transcript).
4. **Documents (PDF, Office, CSV, JSON):**
   - Direct text layer extraction via PyMuPDF/Fast-parser + vision OCR rendering fallback for scanned/image-only pages.

---

## 3. Universal Metering, BYOK & Monetization Architecture

### How the Engine Enables BYOK + Platform Usage Fees
Whether a customer uses Valstorm-managed model endpoints, BYOK (Bring Your Own Key for Claude, Gemini, OpenAI, DeepSeek), or local self-hosted models (Ollama, vLLM, llama.cpp), the agent engine acts as the authoritative **Telemetry & Metering Layer**.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      Valstorm Agent Execution Core                       │
│                                                                          │
│  ┌───────────────────────┐   Raw Stream    ┌──────────────────────────┐  │
│  │ LLM Provider Adapter  │ ──────────────> │ Stream Parser & Metering │  │
│  │ (Anthropic/Gemini/    │                 │ - Exact Token Counter    │  │
│  │  OpenAI/BYOK/Ollama)  │                 │ - Cache Hit / Miss Ratio │  │
│  │  + Multimodal Payloads│                 │ - Tool Compute Latency   │  │
│  └───────────────────────┘                 └─────────────┬────────────┘  │
│                                                          │               │
│                                            Usage Receipt │ (Signed)      │
│                                                          ▼               │
│                                            ┌──────────────────────────┐  │
│                                            │  Local SQLite Ledger     │  │
│                                            │  (Append-Only Batch Sync)│  │
│                                            └─────────────┬────────────┘  │
└──────────────────────────────────────────────────────────┼───────────────┘
                                                           │ HTTPS / SSE Sync
                                                           ▼
                                             ┌──────────────────────────┐
                                             │ Valstorm Billing Service │
                                             │ - Stripe Metered Usage   │
                                             │ - Platform Fee / Markup  │
                                             │ - Org Usage Analytics    │
                                             └──────────────────────────┘
```

### Key Metering Capabilities
1. **Granular Usage Capture per Run:**
   - Prompt input tokens (distinguishing text tokens vs. vision/multimodal image tokens vs. cached tokens).
   - Output/completion tokens and reasoning/thinking tokens.
   - Tool execution metrics (number of tool calls, wall-clock compute duration, disk I/O).
   - Metadata payload: `org_id`, `user_id`, `session_id`, `model_id`, `provider`, `auth_type` (`platform` vs `byok`).
2. **Tamper-Resistant Local Ledger:**
   - Telemetry is recorded to an append-only SQLite usage ledger locally.
   - Synchronized periodically and upon run completion via `POST /v1/billing/usage-events` (or during existing `/desktop-sync` handshakes).
   - Offline grace policy: Offline runs are queued in the local ledger; subsequent online sync flushes pending usage batches before token quotas refresh.
3. **Flexible Monetization Models Supported:**
   - **BYOK Platform Markup:** Base API cost ($0 to Valstorm) + $X per 1M tokens processed or fixed platform fee per agent run.
   - **Hybrid Credits / Value Pricing:** Charge based on high-value actions (e.g., successful code patches, VFS RAG extractions, automated CRM updates).

---

## 4. Deep Dive: OpenCode Evaluation (Build vs. Wrap vs. Adopt)

### What Makes OpenCode Effective?
OpenCode's strengths in code creation stem from:
1. **LSP (Language Server Protocol) & Tree-Sitter AST Parsing:** Instead of treating code purely as raw text, it inspects semantic symbol definitions, references, and diagnostics.
2. **Specialized Agent Roles:** Clear separation between "Planning/Architect" agents and "Execution/Code" agents.
3. **Structured Tool Schemas:** Strict boundaries on multi-file edits and diff verification.

### Recommendation: "Inspect & Implement Natively" (Do Not Wrap as External Tool)
- **Why Not Wrap OpenCode:** Wrapping OpenCode as an external CLI binary introduces external Node/NPM dependencies, secondary auth mechanisms, separate process management, and disjointed session logs.
- **The Native Approach:** Inspect OpenCode's AST/LSP primitives and tool schemas, then implement them directly into our engine. This gives us:
  - Native Rust/Tree-sitter bindings for instant AST diffing.
  - Native integration with Valstorm's `valstorm-cli` and project context.
  - Zero external Node/NPM dependencies on the host machine.

---

## 5. Solving the Hermes `/runs` Flaws (Context Bleed & Dropped Tool Calls)

### Diagnosing the Root Causes in Hermes
1. **Context Bleed (Cross-Chat Contamination):**
   - *Flaw:* Hermes API daemon historically maintained global "active session" fallbacks or shared in-memory conversation singletons. When concurrent requests hit `/runs` from both the terminal and the Electron UI, state could leak or bind to the wrong session ID.
2. **Dropped Tool Calls & Oddities:**
   - *Flaw:* Hermes uses loose regex/JSON parsing over streaming chunks when translating between different LLM provider formats (OpenAI format vs. Anthropic content blocks vs. Gemini function calls). When parallel tool calls arrive or partial JSON chunks break across packet boundaries, calls silently fail or vanish.

### How Our Engine Eliminates This Completely:
1. **Actor-Based Session Isolation (Zero Shared State):**
   - Every session is a distinct, stateful Actor (or isolated async state machine) bound strictly to a `UUIDv7` session ID.
   - No fallback to "latest session". A run cannot start without an explicit, validated `session_id`.
   - SQLite FTS5 transactions use WAL (Write-Ahead Logging) mode with per-session row locking.
2. **Typed Streaming Tool Call State Machine:**
   - Provider responses are decoded into a strict AST streaming parser (Rust / Pydantic).
   - Partial tool call arguments are buffered and validated against strict schemas before execution.
   - If an LLM emits malformed JSON or drops a tool call, the state machine triggers an immediate automated correction turn rather than silently failing.

---

## 6. Strategic Technical Decisions

### Decision 1: Progressive Python-to-Rust Migration Path
- **Stage 1 (Rapid Proof of Concept - Python Async):**
  - Build the core ReAct loop, tool registry, memory manager, metering interceptor, and multimodal pipeline in clean, modern Python (managed via `uv`).
  - Rapidly prototype and prove out tool calling, prompt caching invariants, and Valstorm API/VFS bridges.
- **Stage 2 (Targeted Rust Offloading via PyO3):**
  - Replace CPU-bound and reliability-critical components with compiled Rust modules:
    1. *Fuzzy AST/Text Code Patcher* (Tree-sitter AST diffing + multi-strategy text matching).
    2. *Fast File & Codebase Search* (direct `ripgrep` / `ignore` crate bindings).
    3. *Session State & Metering Store* (SQLite + FTS5 high-concurrency locking and retrieval).
- **Stage 3 (Full Standalone Rust Core Binary):**
  - Once contracts, schemas, and runtime semantics are locked in, compile the entire agent runner into a single static binary for deployment on user machines with zero Python environment dependencies.

---

### Decision 2: Vector Store Architecture — Local Qdrant vs. Cloud Qdrant
- **The Cloud Architecture:** Valstorm cloud runs Qdrant as a StatefulSet on Kubernetes (`valstorm_vfs_chunks` partition by `org_id`, 384d FastEmbed embeddings).
- **The Local Strategy:**
  - Standardize on **Qdrant** across both local and cloud to ensure 100% schema, filter, and embedding parity.
  - **Local Deployment Options:**
    1. *Qdrant Embedded / In-Process (or lightweight background container/binary):* Low-footprint, persistent storage for local project codebases, git history, and local workspace indexing.
    2. *Hybrid Query Router:* The agent's RAG pipeline simultaneously queries the **Local Qdrant collection** (local repo files) and the **Cloud Valstorm VFS Qdrant cluster** (org-wide policies, tickets, remote documents), merging results via Reciprocal Rank Fusion (RRF).

---

### Decision 3: Cloud-Brokered Gateway & Platform Integrations
- Rather than baking platform-specific API clients (Slack, Teams, Telegram) directly into the local agent process, adopt a **Hub-and-Spoke Architecture**:
  - **Valstorm Cloud (The Hub):** Handles OAuth 2.0 PKCE, Slack/Teams webhooks, enterprise permissions, RBAC, and telephony (Twilio/Email).
  - **Agent Engine (The Spoke / Worker):** Connects to the Valstorm Platform over secure WebSockets / SSE / IPC.
  - **Execution Flow:**
    1. A message arrives on Slack/Teams or Valstorm Web Chat.
    2. Valstorm Cloud authenticates the user, verifies org permissions, and generates a structured Task payload.
    3. The payload is dispatched to the user's Local Agent (via Electron/CLI bridge) or a Cloud Cluster Worker (via Celery).
    4. The agent executes tools locally/remotely and streams events back to the cloud hub for rendering in Slack/Teams/Web.

---

## 7. High-Level System Architecture

```
                               ┌─────────────────────────────────────────┐
                               │         Valstorm Cloud Platform         │
                               │  - OAuth & Scoped RBAC                  │
                               │  - Slack / Teams / Twilio Integrations  │
                               │  - VFS Storage & Cloud Qdrant Cluster   │
                               └────────────────────┬────────────────────┘
                                                    │
                                  Encrypted WebSocket / SSE Stream
                                                    │
┌───────────────────────────────────────────────────▼───────────────────────────────────────────────────┐
│                                 Valstorm Agent Engine (Local / Server)                                │
│                                                                                                       │
│   ┌───────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                               Orchestration Engine (Python ➔ Rust)                            │   │
│   │   - ReAct Turn FSM              - Prompt Cache & Context Manager   - Dynamic Skill Loader     │   │
│   │   - Isolated Session Actors     - Streaming Tool State Machine     - Tree-sitter AST Engine   │   │
│   │   - Multimodal Ingestion Layer  - Token & Usage Metering Intercept - Local Ledger Storage     │   │
│   └───────────────────────────────┬───────────────────────────────┬───────────────────────────────┘   │
│                                   │                               │                                   │
│   ┌───────────────────────────────▼───────────────┐   ┌───────────▼───────────────────────────────┐   │
│   │            Execution & Tool Layer             │   │            Memory & State Layer           │   │
│   │   - PTY Process / Subprocess Runner           │   │   - SQLite + FTS5 Canonical Session Store │   │
│   │   - Fuzzy AST & Text Diff Engine (Rust)       │   │   - Declarative Fact Store (Memory/User)  │   │
│   │   - Ripgrep File Searcher                     │   │   - Local Embedded Qdrant (Code Index)    │   │
│   │   - Valstorm CLI & API Native Toolset         │   │   - Hybrid Cloud RAG Dispatcher (RRF)     │   │
│   └───────────────────────────────────────────────┘   └───────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Swarm R&D Blueprint & Phase Breakdown

Using the Hermes multi-agent swarm, the development workflow can be orchestrated cleanly:

### Phase 1: Specifications & Interface Contracts (No Code Execution)
- Architect subagents draft exact JSON schemas for:
  - Strict actor-based session lifecycle and SSE event streams.
  - Multimodal attachment data models (images, video chunks, audio URIs).
  - Metering telemetry payloads (tokens, image tokens, cache hits/misses, model, tool duration).
  - Typed Tool calling contracts (`read_file`, `patch`, `terminal`, `valstorm_query`, `valstorm_vfs`).
  - SQLite WAL schemas (isolated session turns, message metadata, FTS5 indexes, billing ledger).

### Phase 2: Python Reference Implementation & Test Suite
- Specialists build the reference Python async agent runner and conformance test harness (`pytest`).
- Validate strict ReAct loop turn alternation, zero session context bleed, prompt caching preservation, multimodal ingestion, token metering accuracy, and tool sandbox boundaries.

### Phase 3: Rust Core Component Offloading
- Specialists implement high-performance Rust crates for:
  1. `valstorm-patch` (Tree-sitter AST & fuzzy text patcher).
  2. `valstorm-search` (Ripgrep/fast file traversal).
  3. `valstorm-state` (SQLite FTS5 & Metering ledger wrapper with zero-copy query execution).
- Bind into Python via PyO3 and verify 100% test suite parity.

### Phase 4: Local Qdrant RAG & Valstorm Platform Bridge
- Connect local Qdrant indexer with Valstorm VFS hybrid search API.
- Wire into the Valstorm Desktop Electron IPC layer and CLI client.
