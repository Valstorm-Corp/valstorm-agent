# Testing & Mocks (Developer Guide)

## Mocking Providers in Unit Tests
To test your tools and ReAct loops without hitting external LLM APIs, implement a simple mock provider:

```python
import pytest
from core.models import Message, ToolCall, UsageMetadata
from core.react import ReActEngine
from core.tools import ToolRegistry

class MockEchoProvider:
    name = "mock"

    async def generate(self, messages, tools=None, model=None):
        last_user = messages[-1].content
        return Message(
            role="assistant",
            content=f"Echo: {last_user}",
            usage=UsageMetadata(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        )

@pytest.mark.asyncio
async def test_mock_loop():
    engine = ReActEngine(provider=MockEchoProvider(), tools=ToolRegistry())
    session = SessionState(active_model="mock-model", active_provider="mock")
    res = await engine.run_turn(session, "Hello World")
    assert res.content == "Echo: Hello World"
    assert session.total_tokens == 15
```

---

## Testing Tool Execution
```python
from core.tools import get_default_registry

def test_calculator_tool():
    tools = get_default_registry()
    res = tools.execute("calculator", {"expression": "25 * 4"})
    assert res.output == "100"
    assert not res.is_error
```
