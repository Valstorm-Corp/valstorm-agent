# Session Persistence & State Management (Developers)

## Persisting and Loading Sessions in Python

```python
import asyncio
from core.models import SessionState
from core.react import ReActEngine
from core.storage import SessionStore
from core.tools import get_default_registry
from providers.gemini import GeminiProvider

async def main():
    store = SessionStore()
    provider = GeminiProvider()
    tools = get_default_registry()
    engine = ReActEngine(provider=provider, tools=tools)

    # 1. Start or resume a session
    session = store.load_session("YOUR_SESSION_ID")
    if not session:
        session = SessionState(active_model="gemini-flash-lite-latest", active_provider="gemini")

    # 2. Execute a turn
    response = await engine.run_turn(
        session=session,
        user_input="Remember that we are deploying the API on port 8010.",
        model="gemini-flash-lite-latest",
    )

    # 3. Save session & message turn to SQLite WAL database
    store.save_session(session)
    print(f"Session {session.session_id} saved with {len(session.messages)} messages.")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Searching Sessions with SQLite FTS5

```python
from core.storage import SessionStore

store = SessionStore()
results = store.search_sessions("port 8010", limit=5)
for r in results:
    print(f"[{r['title']}] ({r['session_id']}): {r['snippet']}")
```
