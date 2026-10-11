# Provider Implementation Guide (Valstorm Internal Developers)

## BaseProvider Contract
All model adapters in `providers/` must inherit from `BaseProvider` in `providers/base.py`:

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from core.models import Message, UsageMetadata

class BaseProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **kwargs: Any
    ) -> Tuple[Message, UsageMetadata]:
        """Returns tuple of (response_message, usage_metadata)."""
        pass
```

---

## 1. Gemini Implementation (`providers/gemini.py`)
- **SDK**: `google-genai` (official modern SDK) with fallback error handling.
- **Tool Mapping**: Transforms JSON Schema into `types.Tool(function_declarations=[...])`.
- **Token Extraction**:
  ```python
  usage_meta = UsageMetadata(
      prompt_tokens=getattr(response.usage_metadata, "prompt_token_count", 0),
      completion_tokens=getattr(response.usage_metadata, "candidates_token_count", 0),
      total_tokens=getattr(response.usage_metadata, "total_token_count", 0),
      cached_tokens=getattr(response.usage_metadata, "cached_content_token_count", None),
  )
  ```

---

## 2. OpenAI Implementation (`providers/openai.py`)
- **SDK**: `openai` client.
- **Tool Mapping**: Standard OpenAI tool format `{"type": "function", "function": schema}`.
- **Token Extraction**:
  ```python
  usage_meta = UsageMetadata(
      prompt_tokens=getattr(response.usage, "prompt_tokens", 0),
      completion_tokens=getattr(response.usage, "completion_tokens", 0),
      total_tokens=getattr(response.usage, "total_tokens", 0),
  )
  ```

---

## Adding a New Provider
To add Anthropic, Mistral, or Groq:
1. Create `providers/<name>.py` inheriting from `BaseProvider`.
2. Map incoming `List[Message]` to the provider's message structure.
3. Map tools from standard JSON Schema to the provider's tool schema format.
4. Extract tool calls and token counts into `Message(role="assistant", tool_calls=...)` and `UsageMetadata(...)`.
5. Register the provider in `providers/__init__.py` and `cli.py:get_provider()`.
