# CLI User Guide (Users & Operators)

The Valstorm Agent Runtime CLI provides real-time streaming, high-precision tool execution badges, multi-agent profile switching, and conversational session management.

---

## 1. Interactive Chat REPL (`chat`)

Start an interactive conversation with real-time token streaming and profile instructions:

```bash
# Launch with default profile
uv run --project apps/agent-runtime python apps/agent-runtime/cli.py chat

# Launch with a specific profile
uv run --project apps/agent-runtime python apps/agent-runtime/cli.py chat --profile developer
uv run --project apps/agent-runtime python apps/agent-runtime/cli.py chat --profile researcher
uv run --project apps/agent-runtime python apps/agent-runtime/cli.py chat --profile orchestrator

# Launch with custom model or iteration limits
uv run --project apps/agent-runtime python apps/agent-runtime/cli.py chat --model gemini-flash-latest --max-iterations 75
```

### Visual Tool Lifecycle Badges

When the model invokes tools, the CLI renders live execution states:

```text
[gemini-flash-latest] > Use skill_list to find testing skills, then inspect test-driven-development
⚡ [Tool Requested] skill_list(args={'query': 'testing'})
  ⚙ [Tool Executing] Running skill_list...
  ✔ [Tool Result [0.85ms · 412 B · 3 records]] 📚 Available Agent Skills (3 matching):
  • software-development/test-driven-development [tags: testing, tdd, quality]
  • valstorm-internal/valstorm-pytest-patterns [tags: python, pytest, backend]
  • valstorm-internal/playwright-electron-testing [tags: electron, e2e, testing]

⚡ [Tool Requested] skill_view(args={'name': 'test-driven-development'})
  ⚙ [Tool Executing] Running skill_view...
  ✔ [Tool Result [1.12ms · 3.4 KB]] === SKILL: software-development/test-driven-development (v1.1.0) ===
  # Test-Driven Development (TDD)
  ## Overview: Write the test first. Watch it fail. Write minimal code to pass...

[Usage Stats] Turn Tokens: Prompt=842 | Completion=190 | Total=1,032
[Session Cumulative] Prompt=842 | Completion=190 | Total=1,032
[Session Info] ID: aich_6a04fc17ffe84cb4 | Model: gemini-flash-latest (gemini)
[Resume Command] vsagent chat --session aich_6a04fc17ffe84cb4
```

---

## 2. In-Chat REPL Commands

| Command | Description |
| :--- | :--- |
| `/profile [slug]` | Switch agent profile (e.g. `/profile researcher`) or list all available profiles. |
| `/model <model> [provider]` | Switch active model mid-session (e.g. `/model gemini-flash-latest` or `/model gpt-4o openai`). |
| `/key <provider> <api_key>` | Save or update an API key in `~/.config/valstorm/keys.json`. |
| `/sessions` | List recent saved SQLite sessions with message counts and timestamps. |
| `/resume <session_id>` | Resume a past session by ID or prefix. |
| `/memory` | View persistent declarative memory facts. |
| `/search <query>` | Full-text search across past message history using SQLite FTS5. |
| `/stats` | View session token telemetry (prompt, completion, total). |
| `/status` | View telemetry badges, token budget %, and estimated cost. |
| `/steer <instruction>` | Queue an out-of-band instruction injected on the next tool loop iteration. |
| `/yolo` | Toggle human-in-the-loop confirmation safety mode. |
| `/compress [keep_last]` | Summarize historical turns to compress context token usage. |
| `/history` | View list of messages with model tags and per-message token counts. |
| `/exit`, `/quit` | Save and exit the session. |

---

## 3. Single-Shot Task Runner (`run`)

Run a prompt directly from the terminal without entering interactive mode:

```bash
# Execute with developer profile
uv run --project apps/agent-runtime python apps/agent-runtime/cli.py run \
  --profile developer \
  "Inspect apps/agent-runtime/core/react.py and execute test suite"

# Execute with deep researcher profile
uv run --project apps/agent-runtime python apps/agent-runtime/cli.py run \
  --profile researcher \
  "Summarize Valstorm virtual file service architecture"
```

---

## 4. API Key Management (`keys`)

```bash
# Check configured API keys status
uv run --project apps/agent-runtime python apps/agent-runtime/cli.py keys status

# Set an API key persistently
uv run --project apps/agent-runtime python apps/agent-runtime/cli.py keys set gemini AIzaSy...
uv run --project apps/agent-runtime python apps/agent-runtime/cli.py keys set openai sk-...
uv run --project apps/agent-runtime python apps/agent-runtime/cli.py keys set anthropic sk-ant-...
```
