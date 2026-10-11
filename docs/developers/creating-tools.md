# Creating Custom Tools (Developer Guide)

## The `@tool` Decorator
You can convert any Python function into an LLM-callable tool using `@tool`. Type annotations and docstrings are automatically parsed into standard JSON Schema definitions.

```python
from core.tools import tool, ToolRegistry

@tool
def fetch_weather(city: str, unit: str = "celsius") -> str:
    """Fetch current weather for a specified city.

    Args:
        city: Name of the city (e.g. San Francisco).
        unit: Temperature unit (celsius or fahrenheit).
    """
    return f"Weather in {city}: 22 degrees {unit}."
```

---

## Tool Registry & Execution Configuration

```python
# Configure default timeouts and maximum payload byte windowing
registry = ToolRegistry(
    default_timeout_sec=10.0,
    max_payload_bytes=16_000 # Prevents context blowup on massive outputs
)

# Register tool directly
registry.register(fetch_weather)
```

---

## High-Precision Telemetry on Tool Execution
Every invocation returns a `ToolResult` with micro-telemetry:

```python
res = await registry.execute_async("fetch_weather", {"city": "Austin"})

print(f"Output: {res.output}")
print(f"Latency: {res.duration_ms} ms")
print(f"Payload Size: {res.payload_bytes} bytes")
print(f"Record Count: {res.item_count}")
print(f"Truncated: {res.truncated}")
```
