# Token Telemetry & Usage-Based Billing (Valstorm Internal Developers)

## Telemetry Principles

1. **Per-Message Granularity**: Because models can be switched mid-session (e.g. starting with `gemini-2.5-flash` and escalating to `gemini-2.5-pro`), token counts are captured **per message turn** alongside the exact model name.
2. **Session Aggregates**: The `SessionState` maintains running counters:
   - `total_prompt_tokens`
   - `total_completion_tokens`
   - `total_tokens`
3. **Usage Model Integration**:
   When integrating with Valstorm's backend (`apps/api`):
   - Message turns map directly to MongoDB `ai_chat_message` records.
   - Aggregate counters update `ai_chat` totals.
   - Metrics feed into Valstorm's Usage-Based Billing (UBB) pipeline (`AI Inference: tokens`).

---

## Token Data Flow

```
[LLM SDK Response]
       │
       ▼
[Provider.generate] ──> Extracts (Message, UsageMetadata)
       │
       ▼
[ReActEngine.run_turn] ──> Attaches msg.usage = UsageMetadata
       │
       ▼
[SessionState.add_message] ──> session.total_tokens += msg.usage.total_tokens
       │
       ▼
[UI / Telemetry / UBB Egress]
```
