# Switching Models Mid-Session (Users & Operators)

One of the key features of the Agent Runtime is the ability to switch models on the fly without wiping conversation history or resetting tool context.

---

## How to Switch Models in Interactive Chat

Inside the REPL, use the `/model` command:

```text
[gemini-2.5-flash] > Hello, let's start a planning session.
[Final Reply] Hello! How can I help you plan today?

[gemini-2.5-flash] > /model gemini-2.5-pro
[System] Switched active model to: gemini-2.5-pro (gemini)

[gemini-2.5-pro] > Write a complex algorithm for task distribution.
```

You can also switch provider and model simultaneously:
```text
[gemini-2.5-pro] > /model gpt-4o openai
[System] Switched active model to: gpt-4o (openai)
```

---

## Recommended Models

| Purpose | Gemini | OpenAI |
| :--- | :--- | :--- |
| **Fast / Testing / Low Cost** (Recommended) | `gemini-2.5-flash`, `gemini-2.0-flash` | `gpt-4o-mini` |
| **Complex Reasoning / Code Synthesis** | `gemini-2.5-pro` | `gpt-4o` |
