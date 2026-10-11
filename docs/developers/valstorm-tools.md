# Valstorm Tools Developer Guide (Developers)

## Registering Valstorm Tools in Python

```python
import asyncio
from core.models import SessionState
from core.react import ReActEngine
from core.tools import get_default_registry
from providers.gemini import GeminiProvider
from tools.valstorm_tools import register_valstorm_tools

async def main():
    provider = GeminiProvider()
    tools = get_default_registry()
    
    # Automatically connects using workspace valstorm.json / ~/.valstorm profile
    # or passes explicit env ("local", "dev", "prod") and token overrides
    register_valstorm_tools(tools, env="local")

    engine = ReActEngine(provider=provider, tools=tools)
    session = SessionState(active_model="gemini-flash-lite-latest", active_provider="gemini")

    response = await engine.run_turn(
        session=session,
        user_input="How many active tasks are assigned to me?",
    )
    print(response.content)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Direct Client Access (`ValstormApiClient`)

If you need to make programmatic Valstorm API calls directly with transparent auto-refresh:

```python
import asyncio
from tools.valstorm_client import ValstormApiClient

async def query_tasks():
    async with ValstormApiClient(env="local") as client:
        # Transparently auto-refreshes token if expired
        result = await client.sql_query("SELECT id, name, status FROM task LIMIT 10")
        print("Records:", result["records"])

asyncio.run(query_tasks())
```
