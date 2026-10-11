# Architectural Specification: Skill Linking, Token Telemetry Pipeline & Cloud Sandbox Blueprint

## 1. Executive Summary

This architecture addresses three foundational capabilities required to scale the Valstorm Agent Platform from local development to multi-tenant enterprise production:

1. **Two-Stage Skill & Agent Cloud Migration**: Uploading the 169 modular `ai_skill` records first, extracting their generated IDs, and linking them into `ai_agent.ai_skills` (`lookup_list`).
2. **Full-Pipeline Token & Model Telemetry**: Upgrading `DesktopSyncRequest`, `ai_service.py`, and `ai_chat_message` creation so every conversational turn permanently records `input_tokens`, `output_tokens`, `model`, `provider`, and updates parent `ai_chat` rollup totals.
3. **Cloud Ephemeral Sandbox Architecture Blueprint**: The collaborative architecture where Valstorm backend servers orchestrate disposable, zero-install cloud micro-sandboxes (Firecracker / Docker) for 99% of non-developer users.

---

## 2. Stage 1: Two-Stage Cloud Skill & Profile Migration

Because `ai_agent` maintains a `lookup_list` referencing `ai_skill` (`ai_skills: ["aisk_..."]`), migration must execute in two atomic phases:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STAGE 1: MIGRATION PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 1.1: Upload All Skills                                                │
│    • Ingests 169 normalized skills from ~/.valstorm/migrated_skills.json     │
│    • Executes POST /object/ai_skill in batch                                │
│    • Builds in-memory slug -> aisk_id lookup table                          │
│                                                                             │
│  Step 1.2: Map & Upload AI Agents                                           │
│    • Ingests 14 profiles from ~/.valstorm/migrated_profiles.json            │
│    • Resolves skill references (e.g. ['youtube-content', 'valstorm-sql'])   │
│      into explicit ID arrays (e.g. ['aisk_c5a5d1f711025b97', ...])          │
│    • Executes POST /object/ai_agent with populated ai_skills list           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Migration Script: `apps/agent-runtime/scripts/upload_skills_and_agents.py`
```python
# 1. Batch create ai_skill records
skills_res = await client.records_create("ai_skill", skills_payload)

# 2. Build map: slug -> generated/authoritative skill ID
skill_slug_to_id = {s["api_name"]: s["id"] for s in skills_res}

# 3. Attach matching skills to each agent profile
for agent in agents_payload:
    matching_skill_ids = [
        skill_slug_to_id[slug]
        for slug in agent.get("attached_skill_slugs", [])
        if slug in skill_slug_to_id
    ]
    agent["ai_skills"] = matching_skill_ids

# 4. Batch create ai_agent records
agents_res = await client.records_create("ai_agent", agents_payload)
```

---

## 3. Stage 2: Token Telemetry & Model Tracking Pipeline

Currently, `desktop-sync` receives live agent turns but does not populate token fields in MongoDB.

### Data Flow Upgrades:

```
┌────────────────────────────┐
│ Valstorm Agent Runtime /   │
│ Desktop IPC (8650 / 8642)  │
└─────────────┬──────────────┘
              │ POST /v1/ai/chat/{chat_id}/desktop-sync
              │ Payload includes: input_tokens, output_tokens, model, provider
              ▼
┌────────────────────────────┐
│ FastAPI Backend            │
│ (ai/ai_service.py)         │
├────────────────────────────┤
│ 1. Writes ai_chat_message: │
│    - input_tokens: 385     │
│    - output_tokens: 28     │
│    - model: "gemini-3.5"   │
│    - provider: "Gemini"    │
│                            │
│ 2. Atomically increments   │
│    parent ai_chat totals:  │
│    - total_input_tokens    │
│    - total_output_tokens   │
└────────────────────────────┘
```

### Updates Required in `valstorm_platform/models.py` & `ai_service.py`:
1. Add fields to `DesktopSyncRequest`:
   - `input_tokens: Optional[int] = 0`
   - `output_tokens: Optional[int] = 0`
   - `model: Optional[str] = None`
   - `provider: Optional[str] = None`
2. Update `AIService.handle_desktop_sync()`:
   - When creating `ai_chat_message`, inject `input_tokens`, `output_tokens`, `model`, and `provider`.
   - Update parent `ai_chat`:
     ```python
     await platform.records.update("ai_chat", [{
         "id": chat_id,
         "total_input_tokens": current_input + data.input_tokens,
         "total_output_tokens": current_output + data.output_tokens,
         "model": data.model,
         "provider": data.provider,
     }])
     ```

---

## 4. Stage 3: Cloud Ephemeral Sandboxes (Future VM Architecture)

For 99% of enterprise business users (Sales, HR, Support, Operations), the agent will run in **cloud ephemeral micro-sandboxes** rather than on their local computer.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Valstorm Cloud Platform (Hub)                      │
│                                                                             │
│   • User prompts in Web Browser / Mobile / Slack / Teams                    │
│   • Backend checks OAuth scope & tenant DB permissions                      │
│   • Spawns on-demand ephemeral Micro-Sandbox Worker (Firecracker / Docker)   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       │ 1. Provision & Pass Task Payload
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Disposable Ephemeral Micro-Sandbox (<150ms boot)          │
│                                                                             │
│   • Pre-installed: Python, Node, uv, git, valstorm-cli                      │
│   • Ephemeral Filesystem (Scratch disk wiped immediately upon completion)    │
│   • Ingests transient user files via Valstorm VFS API / presigned S3 URLs    │
│   • Runs ReAct execution loop & tools strictly over authenticated REST       │
│   • Egress Network Filtering: Outbound connections limited to allowlist      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       │ 2. Streams events & token stats back
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Canonical Ledger & Persistence                      │
│   • Saves ai_chat and ai_chat_message records with exact token telemetry    │
│   • Real-time typewriter updates pushed over WebSockets to client UI        │
│   • Micro-Sandbox is destroyed immediately (Zero data residue)               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Security & File Handling Principles for Cloud Sandboxes:
1. **File Access via VFS**: Documents are retrieved on-demand via authenticated `/vfs` REST calls using the user's scoped Bearer token.
2. **Transient Upload Bridge**: If a user attaches a local file from their browser, it is uploaded to a temporary encrypted S3 scratch vault with a 1-hour TTL, read by the sandbox, and auto-purged.
3. **No Persistent Credentials**: Sandboxes receive only short-lived scoped JWTs. No master database keys or root cloud credentials ever enter the container.
