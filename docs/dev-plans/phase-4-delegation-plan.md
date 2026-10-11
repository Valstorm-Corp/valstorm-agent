# Multi-Agent Delegation Plan: Workstreams 4C & 4D

**Phase:** Phase 4 — Production Enterprise Agent Capabilities  
**Subsystem:** `apps/agent-runtime`  
**Execution Strategy:** Multi-Agent Swarm Delegation via `delegate_task`  

---

## 1. Subagent Swarm Roster & Responsibilities

```text
               ┌──────────────────────────────┐
               │    Lead Orchestrator Agent   │
               │         (orchestrator)       │
               └──────────────┬───────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ Full-Stack Dev 1 │ │ Full-Stack Dev 2 │ │  Backend Tester  │
│   (developer)    │ │   (developer)    │ │ (backend-tester) │
│  Workstream 4C   │ │  Workstream 4D   │ │  Verification    │
│  (Cancellation)  │ │ (Vision Pipeline)│ │   & Test Suite   │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

| Subagent Profile | Assigned Workstream | Core Goal & Scope | Key Tools |
| :--- | :--- | :--- | :--- |
| **`developer` (Agent 1)** | **Workstream 4C: Mid-Run Cancellation** | Implement `DELETE /v1/runs/{run_id}` in `server.py`, cancel active asyncio tasks, emit `run.cancelled` SSE events, clean up queues. | `patch_file`, `write_file`, `read_file` |
| **`developer` (Agent 2)** | **Workstream 4D: Native Multimodal Vision** | Add attachment extractor and MIME resolver. Wire inline binary parts into `gemini.py`, `openai.py`, and `anthropic.py`. | `patch_file`, `write_file`, `read_file` |
| **`backend-tester` (Agent 3)** | **Verification & QA** | Author `tests/test_cancellation.py` and `tests/test_vision.py`. Run test suite and assert 100% pass rate. | `terminal_exec`, `write_file` |
| **`writer` (Agent 4)** | **Documentation & Log Sync** | Update `README.md`, user guides, context bundles, and session coordination logs. | `write_file`, `patch_file` |

---

## 2. Workstream Breakdown & Acceptance Criteria

### Subagent 1: Workstream 4C (Mid-Run Cancellation)
* **Target:** `apps/agent-runtime/server.py`
* **Tasks:**
  1. Add `DELETE /v1/runs/{run_id}` endpoint.
  2. Locate running task in `active_run_tasks[run_id]` and call `task.cancel()`.
  3. Emit `{"event": "run.cancelled", "run_id": run_id, "status": "cancelled"}` over the active SSE queue.
  4. Ensure `asyncio.CancelledError` is caught in `_execute_agent_run` to cleanly terminate ReAct turn loop.
  5. Cleanly close stream by putting `None` sentinel.

### Subagent 2: Workstream 4D (Multimodal Vision Engine)
* **Target:** `providers/gemini.py`, `providers/openai.py`, `providers/anthropic.py`, `core/models.py`
* **Tasks:**
  1. Implement `_extract_image_attachments(text: str, images: Optional[List[str]]) -> List[Tuple[bytes, str]]` resolving local image files (`.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.pdf`) and Base64 Data URIs.
  2. In `providers/gemini.py`: Encode image attachments as `types.Part.from_bytes(data, mime_type)` in `contents`.
  3. In `providers/openai.py`: Encode image attachments as `{"type": "image_url", "image_url": {"url": "data:..."}}`.
  4. In `providers/anthropic.py`: Encode image attachments as `{"type": "image", "source": {"type": "base64", ...}}`.

### Subagent 3: Test Verification Suite
* **Target:** `tests/test_cancellation.py`, `tests/test_vision.py`
* **Tasks:**
  1. Test canceling active run via `DELETE /v1/runs/{id}` and verify `run.cancelled` SSE event.
  2. Test canceling non-existent run returns 404.
  3. Test Gemini vision formatting with PNG/JPEG files.
  4. Test OpenAI vision formatting with Base64 data URIs.
  5. Test Anthropic vision formatting with source blocks.
  6. Execute full offline test suite.

### Subagent 4: Documentation & Coordination Log
* **Target:** `coordination_log.md`, `README.md`, `context.md`
* **Tasks:**
  1. Document cancellation API and multimodal vision capabilities.
  2. Synchronize context bundles.
