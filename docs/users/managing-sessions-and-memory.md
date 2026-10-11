# Managing Sessions & Persistent Memory (Users & Operators)

The Valstorm Agent Runtime automatically saves your conversation history and token metrics to a local SQLite database after every turn.

---

## 1. Listing and Resuming Past Sessions

### From the Command Line:
```bash
# Resume an existing session by ID
uv run --project apps/agent-runtime python apps/agent-runtime/cli.py chat --resume 8f2a1b9c
```

### Inside the Interactive REPL:
```text
[gemini-flash-lite-latest] > /sessions
--- Recent Stored Sessions ---
  • 8f2a1b9c | Fix CORS headers in API             | 6 msgs | 2026-08-18 10:45 (Active)
  • 3d91f04e | Search Task Schema                  | 2 msgs | 2026-08-18 09:30

[gemini-flash-lite-latest] > /resume 3d91f04e
[Switched Session] Loaded 3d91f04e (2 messages, 1420 tokens)
```

---

## 2. Searching Past Sessions with FTS5

You can search across historical turns, code snippets, and tool outputs directly:

```text
[gemini-flash-lite-latest] > /search CORS
--- FTS5 Search Results for 'CORS' ---
  • [Fix CORS headers in API] (8f2a1b9c · 2026-08-18):
    Snippet: Added <b>CORS</b> middleware configuration in main.py...
```

---

## 3. Viewing Declarative Memory Facts

Inspect the agent's long-term facts:

```text
[gemini-flash-lite-latest] > /memory
--- Persistent Declarative Memory ---
{
  "user": [
    "User prefers concise terminal outputs."
  ],
  "memory": [
    "Valstorm local API runs on port 8010."
  ]
}
```
