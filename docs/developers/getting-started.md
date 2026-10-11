# Getting Started (Developer Guide)

## Prerequisites
- Python >= 3.10
- `uv` (recommended) or standard Python `venv`

## Installation
From the monorepo root:
```bash
# Sync dependencies
uv sync --project apps/agent-runtime
```

## Running the Test Suite
```bash
uv run --project apps/agent-runtime pytest apps/agent-runtime/tests
```

## Embedding ReActEngine in Your Code

```python
import asyncio
from core.models import SessionState
from core.keystore import KeyStore
from core.tools import get_default_registry
from core.react import ReActEngine
from providers.gemini import GeminiProvider

async def main():
    api_key = KeyStore.resolve_key("gemini")
    provider = GeminiProvider(api_key=api_key)
    tools = get_default_registry()
    engine = ReActEngine(provider=provider, tools=tools)

    session = SessionState(active_model="gemini-2.5-flash", active_provider="gemini")
    response = await engine.run_turn(session, "Calculate (15 * 40) + 120")

    print(f"Reply: {response.content}")
    print(f"Total Tokens: {session.total_tokens}")

if __name__ == "__main__":
    asyncio.run(main())
```
