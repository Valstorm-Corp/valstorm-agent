# Using Valstorm Platform Tools in the CLI (Users & Operators)

The Agent Runtime connects directly to your active Valstorm workspace (contacts, tasks, deals, documents, custom objects) via authenticated REST APIs with zero manual configuration.

---

## Automatic Authentication & Workspace Detection

The agent runtime automatically detects your active Valstorm environment and profile:
1. It reads `valstorm.json` in your workspace directory (e.g. `{"env": "local", "profile": "vdk"}`).
2. It loads your active session from `~/.valstorm/auth_{env}_{profile}.json`.
3. If your session token expires, it automatically refreshes it in the background and saves the fresh token back to disk.

---

## CLI Examples

### 1. Interactive Chat REPL
```bash
# Uses your active workspace credentials automatically:
uv run --project apps/agent-runtime python apps/agent-runtime/cli.py chat --model gemini-flash-lite-latest

# Explicit environment targeting:
uv run --project apps/agent-runtime python apps/agent-runtime/cli.py chat --valstorm-env dev --model gemini-flash-lite-latest
```

### 2. Single-Shot Task Execution
```bash
uv run --project apps/agent-runtime python apps/agent-runtime/cli.py run \
  "How many not started tasks do I have?" \
  --model gemini-flash-lite-latest
```

---

## Troubleshooting & Debugging

If you encounter authentication issues, check the detailed step-by-step resolution log at:
```text
apps/agent-runtime/auth_debug.log
```
This log tracks every checked profile file, token resolution, auto-refresh attempt, and API response status.
