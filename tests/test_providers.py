"""Unit tests for BaseProvider, GeminiProvider, and OpenAIProvider adapters."""

import json
from unittest.mock import AsyncMock, MagicMock
import pytest

from core.models import Message, ToolCall, ToolResult, UsageMetadata
from providers.base import BaseProvider
from providers.gemini import GeminiProvider
from providers.openai import OpenAIProvider
from providers.anthropic import AnthropicProvider


class DummyProvider(BaseProvider):
    """Concrete implementation of BaseProvider for testing."""

    @property
    def provider_name(self) -> str:
        return "dummy"

    async def generate(self, messages, tools=None, model=None, **kwargs):
        usage = UsageMetadata(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        msg = Message(
            role="assistant",
            content="Dummy response",
            model=model or "dummy-model",
            provider="dummy",
            usage=usage,
        )
        return msg, usage


class TestBaseProvider:
    """Tests for BaseProvider abstract base class."""

    @pytest.mark.asyncio
    async def test_base_provider_contract(self):
        provider = DummyProvider(api_key="test-key", default_model="dummy-default")
        assert provider.provider_name == "dummy"
        assert provider.api_key == "test-key"
        assert provider.default_model == "dummy-default"

        messages = [Message(role="user", content="Hello")]
        resp_msg, usage = await provider.generate(messages)

        assert resp_msg.role == "assistant"
        assert resp_msg.content == "Dummy response"
        assert resp_msg.provider == "dummy"
        assert usage.total_tokens == 15


class TestGeminiProvider:
    """Tests for GeminiProvider adapter."""

    def test_tool_formatting(self):
        provider = GeminiProvider(api_key="fake-key")

        tools = [
            {
                "name": "calculator",
                "description": "Perform math",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            }
        ]

        formatted = provider._format_tools(tools)
        assert formatted is not None
        assert len(formatted) == 1
        decl = formatted[0].function_declarations[0]
        assert decl.name == "calculator"
        assert decl.description == "Perform math"

    def test_tool_formatting_openai_wrapped(self):
        provider = GeminiProvider(api_key="fake-key")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "fetch_user",
                    "description": "Fetch a user by ID",
                    "parameters": {
                        "type": "object",
                        "properties": {"user_id": {"type": "string"}},
                        "required": ["user_id"],
                    },
                },
            }
        ]

        formatted = provider._format_tools(tools)
        assert formatted is not None
        decl = formatted[0].function_declarations[0]
        assert decl.name == "fetch_user"

    def test_message_formatting(self):
        provider = GeminiProvider(api_key="fake-key")

        messages = [
            Message(role="system", content="System instruction text"),
            Message(role="user", content="What is 5 + 5?"),
            Message(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(name="calculator", arguments={"expression": "5+5"})],
            ),
            Message(
                role="tool",
                content="10",
                tool_result=ToolResult(call_id="call_1", name="calculator", output="10"),
            ),
        ]

        system_instruction, contents = provider._format_messages(messages)
        assert system_instruction == "System instruction text"
        assert len(contents) == 3

        # User message
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "What is 5 + 5?"

        # Assistant tool call message
        assert contents[1].role == "model"
        assert contents[1].parts[0].function_call.name == "calculator"
        assert contents[1].parts[0].function_call.args == {"expression": "5+5"}

        # Tool result message
        assert contents[2].role == "user"
        assert contents[2].parts[0].function_response.name == "calculator"
        assert contents[2].parts[0].function_response.response == {"result": "10"}

    def test_parse_text_response(self):
        provider = GeminiProvider(api_key="fake-key")

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 25
        mock_usage.candidates_token_count = 10
        mock_usage.total_token_count = 35
        mock_usage.cached_content_token_count = 5

        mock_part = MagicMock()
        mock_part.text = "The result is 42."
        mock_part.function_call = None

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        mock_response = MagicMock()
        mock_response.usage_metadata = mock_usage
        mock_response.function_calls = None
        mock_response.candidates = [mock_candidate]
        mock_response.text = "The result is 42."

        msg, usage = provider._parse_response(mock_response, "gemini-2.5-flash")

        assert msg.role == "assistant"
        assert msg.content == "The result is 42."
        assert msg.model == "gemini-2.5-flash"
        assert msg.provider == "gemini"
        assert msg.tool_calls is None

        assert usage.prompt_tokens == 25
        assert usage.completion_tokens == 10
        assert usage.total_tokens == 35
        assert usage.cached_tokens == 5

    def test_parse_function_call_response(self):
        provider = GeminiProvider(api_key="fake-key")

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 30
        mock_usage.candidates_token_count = 8
        mock_usage.total_token_count = 38
        mock_usage.cached_content_token_count = None

        mock_fc = MagicMock()
        mock_fc.id = "call_gemini_1"
        mock_fc.name = "calculator"
        mock_fc.args = {"expression": "(45 * 12) + 180"}

        mock_response = MagicMock()
        mock_response.usage_metadata = mock_usage
        mock_response.function_calls = [mock_fc]
        mock_response.candidates = []
        mock_response.text = None

        msg, usage = provider._parse_response(mock_response, "gemini-2.5-pro")

        assert msg.role == "assistant"
        assert msg.content is None
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == "call_gemini_1"
        assert msg.tool_calls[0].name == "calculator"
        assert msg.tool_calls[0].arguments == {"expression": "(45 * 12) + 180"}
        assert usage.prompt_tokens == 30

    @pytest.mark.asyncio
    async def test_generate_with_mock_client(self):
        mock_client = MagicMock()
        mock_generate = AsyncMock()

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 15
        mock_usage.candidates_token_count = 7
        mock_usage.total_token_count = 22
        mock_usage.cached_content_token_count = 0

        mock_part = MagicMock()
        mock_part.text = "Calculated successfully."
        mock_part.function_call = None

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        mock_response = MagicMock()
        mock_response.usage_metadata = mock_usage
        mock_response.function_calls = None
        mock_response.candidates = [mock_candidate]
        mock_response.text = "Calculated successfully."

        mock_generate.return_value = mock_response
        mock_client.aio.models.generate_content = mock_generate

        provider = GeminiProvider(api_key="fake-key", client=mock_client)
        messages = [Message(role="user", content="Calculate something")]

        msg, usage = await provider.generate(messages, model="gemini-2.5-flash")

        assert msg.content == "Calculated successfully."
        assert msg.model == "gemini-2.5-flash"
        assert usage.total_tokens == 22
        mock_generate.assert_awaited_once()


class TestOpenAIProvider:
    """Tests for OpenAIProvider adapter."""

    def test_tool_formatting(self):
        provider = OpenAIProvider(api_key="fake-key")

        tools = [
            {
                "name": "calculator",
                "description": "Evaluate math",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            }
        ]

        formatted = provider._format_tools(tools)
        assert formatted is not None
        assert len(formatted) == 1
        assert formatted[0]["type"] == "function"
        assert formatted[0]["function"]["name"] == "calculator"
        assert formatted[0]["function"]["description"] == "Evaluate math"

    def test_tool_formatting_already_wrapped(self):
        provider = OpenAIProvider(api_key="fake-key")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup data",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        formatted = provider._format_tools(tools)
        assert formatted is not None
        assert formatted[0]["function"]["name"] == "lookup"

    def test_message_formatting(self):
        provider = OpenAIProvider(api_key="fake-key")

        messages = [
            Message(role="system", content="System prompt"),
            Message(role="user", content="Run calculation"),
            Message(
                role="assistant",
                content="Thinking...",
                tool_calls=[ToolCall(id="call_99", name="calculator", arguments={"expression": "1+1"})],
            ),
            Message(
                role="tool",
                content="2",
                tool_result=ToolResult(call_id="call_99", name="calculator", output="2"),
            ),
        ]

        formatted = provider._format_messages(messages)
        assert len(formatted) == 4

        # System
        assert formatted[0] == {"role": "system", "content": "System prompt"}

        # User
        assert formatted[1] == {"role": "user", "content": "Run calculation"}

        # Assistant
        assert formatted[2]["role"] == "assistant"
        assert formatted[2]["content"] == "Thinking..."
        assert len(formatted[2]["tool_calls"]) == 1
        assert formatted[2]["tool_calls"][0]["id"] == "call_99"
        assert json.loads(formatted[2]["tool_calls"][0]["function"]["arguments"]) == {"expression": "1+1"}

        # Tool
        assert formatted[3] == {"role": "tool", "tool_call_id": "call_99", "content": "2"}

    def test_parse_text_response(self):
        provider = OpenAIProvider(api_key="fake-key")

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 40
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 60
        mock_tokens_details = MagicMock()
        mock_tokens_details.cached_tokens = 10
        mock_usage.prompt_tokens_details = mock_tokens_details

        mock_msg = MagicMock()
        mock_msg.content = "Here is the response."
        mock_msg.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_msg

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        msg, usage = provider._parse_response(mock_response, "gpt-4o")

        assert msg.role == "assistant"
        assert msg.content == "Here is the response."
        assert msg.model == "gpt-4o"
        assert msg.provider == "openai"
        assert msg.tool_calls is None

        assert usage.prompt_tokens == 40
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 60
        assert usage.cached_tokens == 10

    def test_parse_tool_call_response(self):
        provider = OpenAIProvider(api_key="fake-key")

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 50
        mock_usage.completion_tokens = 15
        mock_usage.total_tokens = 65
        mock_usage.prompt_tokens_details = None

        mock_func = MagicMock()
        mock_func.name = "mock_db_lookup"
        mock_func.arguments = '{"query": "users"}'

        mock_tc = MagicMock()
        mock_tc.id = "call_abc123"
        mock_tc.function = mock_func

        mock_msg = MagicMock()
        mock_msg.content = None
        mock_msg.tool_calls = [mock_tc]

        mock_choice = MagicMock()
        mock_choice.message = mock_msg

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        msg, usage = provider._parse_response(mock_response, "gpt-4o-mini")

        assert msg.role == "assistant"
        assert msg.content is None
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == "call_abc123"
        assert msg.tool_calls[0].name == "mock_db_lookup"
        assert msg.tool_calls[0].arguments == {"query": "users"}

        assert usage.prompt_tokens == 50
        assert usage.completion_tokens == 15
        assert usage.total_tokens == 65

    @pytest.mark.asyncio
    async def test_generate_with_mock_client(self):
        mock_client = MagicMock()
        mock_create = AsyncMock()

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 18
        mock_usage.completion_tokens = 9
        mock_usage.total_tokens = 27
        mock_usage.prompt_tokens_details = None

        mock_msg = MagicMock()
        mock_msg.content = "OpenAI response text"
        mock_msg.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_msg

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        mock_create.return_value = mock_response
        mock_client.chat.completions.create = mock_create

        provider = OpenAIProvider(api_key="fake-key", client=mock_client)
        messages = [Message(role="user", content="Hello GPT")]

        msg, usage = await provider.generate(messages, model="gpt-4o")

        assert msg.content == "OpenAI response text"
        assert msg.model == "gpt-4o"
        assert usage.total_tokens == 27
        mock_create.assert_awaited_once()

    def test_gemini_sanitizer_starting_with_model_or_orphan_tools(self):
        """Verify Gemini format_messages handles starting with model turn, orphan tool results, and multi-system messages."""
        provider = GeminiProvider(api_key="fake-key")

        messages = [
            Message(role="system", content="System instruction 1"),
            Message(role="system", content="[HISTORICAL CONTEXT DIGEST - 5 turns]"),
            # Starts with assistant model turn with tool call
            Message(
                role="assistant",
                content="Calling tool",
                tool_calls=[ToolCall(id="c1", name="search_files", arguments={"pattern": "*"})],
            ),
            # Orphan tool result (e.g. from mismatched id or prior call pruned)
            Message(
                role="tool",
                content="some_file.py",
                tool_result=ToolResult(call_id="c_diff", name="other_tool", output="file data"),
            ),
            Message(role="user", content="Next user query"),
        ]

        sys_inst, contents = provider._format_messages(messages)

        # 1. Multi-system messages combined
        assert "System instruction 1" in sys_inst
        assert "[HISTORICAL CONTEXT DIGEST - 5 turns]" in sys_inst

        # 2. First turn in contents MUST be role="user"
        assert contents[0].role == "user"

        # 3. Model turn follows
        assert contents[1].role == "model"
        assert contents[1].parts[0].text == "Calling tool"

        # 4. Orphan tool result converted to safe text part
        assert contents[2].role == "user"
        # Since 'other_tool' was not called in contents[1] (which called 'search_files'), it is safely sanitized to text
        assert any("[Historical tool result other_tool]" in (getattr(p, "text", "") or "") for p in contents[2].parts)

    def test_anthropic_sanitizer_multi_system_and_start_user(self):
        """Verify Anthropic format_messages merges system prompts and ensures starting user turn."""
        provider = AnthropicProvider(api_key="fake-key")

        messages = [
            Message(role="system", content="System A"),
            Message(role="system", content="System B"),
            Message(role="assistant", content="Direct assistant opening"),
            Message(role="user", content="User follow up"),
        ]

        sys_prompt, formatted = provider._format_messages(messages)
        assert "System A" in sys_prompt
        assert "System B" in sys_prompt
        assert formatted[0]["role"] == "user"
        assert formatted[1]["role"] == "assistant"
