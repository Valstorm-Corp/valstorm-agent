# Managing API Keys (Users & Operators)

The runtime supports multiple convenient ways to store and configure your API credentials.

---

## 1. Store API Keys via CLI (Recommended)
You can save your API key persistently so you don't need to re-enter it:

```bash
# Save Gemini API Key
uv run --project apps/agent-runtime python apps/agent-runtime/cli.py keys set gemini YOUR_GEMINI_KEY

# Save OpenAI API Key
uv run --project apps/agent-runtime python apps/agent-runtime/cli.py keys set openai YOUR_OPENAI_KEY
```
Keys are saved to `~/.config/valstorm/keys.json`.

---

## 2. Check Key Status
```bash
uv run --project apps/agent-runtime python apps/agent-runtime/cli.py keys status
```
Output:
```text
--- Configured API Keys ---
  Gemini    : Configured (AIza...W9f0)
  Openai    : Missing
  Anthropic : Missing
```

---

## 3. Interactive Prompt
If no key is configured when launching `chat` or `run`, the CLI will prompt you in the terminal and ask whether you'd like to save it.

---

## 4. Environment Variables
Alternatively, you can export keys in your shell:
```bash
export GEMINI_API_KEY=YOUR_GEMINI_KEY
export OPENAI_API_KEY=YOUR_OPENAI_KEY
```
