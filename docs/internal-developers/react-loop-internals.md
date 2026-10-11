# ReAct Loop Internals (Valstorm Internal Developers)

## High-Performance Concurrent Architecture

The `ReActEngine` in `core/react.py` executes agent loops with sub-millisecond overhead, asynchronous streaming, and parallel tool dispatch.

```
┌─────────────────────────────────────────────────────────────┐
│                       run_turn_stream                       │
│    (Yields real-time TEXT_CHUNKs, TOOL events, & METRICS)   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                ┌──────────────▼──────────────┐
                │   Model Generation Stream   │
                └──────────────┬──────────────┘
                               │
               ┌───────────────┴───────────────┐
               │                               │
       [Tool Calls Emitted]             [No Tool Calls]
               │                               │
    ┌──────────▼──────────┐            ┌───────▼───────┐
    │ Parallel Execution  │            │  TURN_COMPLETE│
    │   asyncio.gather    │            └───────────────┘
    │  (Tool 1, 2, ... N) │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ Precision Telemetry │
    │ (ms, bytes, window) │
    └──────────┬──────────┘
               │
         (Next Iteration)
```

---

## 1. Concurrent Parallel Tool Execution
When models emit multiple tool invocations in a single turn (e.g. reading two files and computing a formula simultaneously):
- The runtime launches all tool calls in parallel using Python's `asyncio.gather(*[self._execute_tool(tc) for tc in tool_calls])`.
- **Latency impact**: Total tool latency equals $\max(t_1, t_2, \dots, t_n)$ rather than $\sum t_i$.
- Tool results are collected and appended in deterministic order into the conversation history.

---

## 2. Microsecond Precision & Telemetry
Every tool execution records high-resolution metrics via `time.perf_counter()`:
- `duration_ms`: Execution time in milliseconds (e.g., `0.35ms`, `4.12ms`).
- `payload_bytes`: Exact byte size of the returned UTF-8 payload.
- `item_count`: Automatic detection of list length or dictionary key count.
- `truncated`: Boolean indicating whether payload windowing took place.
- `raw_size_bytes`: Original byte count before windowing.

---

## 3. Smart Payload Windowing (Context Blowup Guardrail)
To prevent rogue tools from blowing up the context window (e.g. accidentally querying 50,000 DB records):
- Payloads exceeding `max_payload_bytes` (default: 16 KB) are automatically trimmed with a standard model notice:
  `[... Payload Windowed: Showing 16000 of 84200 bytes (50 items). Filter or query specifically for details ...]`
- Protects downstream Time-to-First-Token (TTFT) and token budget.
