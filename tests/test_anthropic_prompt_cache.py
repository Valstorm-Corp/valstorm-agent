"""Anthropic prompt-cache breakpoints and cache-aware usage accounting."""

from types import SimpleNamespace

import pytest

from benchmarks.adapters.base import estimate_cost_usd
from core.models import Message, ToolCall, ToolResult
from providers.anthropic import AnthropicProvider, apply_prompt_cache_breakpoints

EPHEMERAL = {"type": "ephemeral"}


class _CapturingClient:
    def __init__(self, usage):
        self.calls = []
        self.messages = self
        self._usage = usage

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="done")], usage=self._usage)


def _conversation():
    return [
        Message(role="system", content="You are Valstorm Agent."),
        Message(role="user", content="Write the file."),
        Message(role="assistant", content="", tool_calls=[ToolCall(id="t1", name="write_file", arguments={"path": "a"})]),
        Message(role="tool", content="ok", tool_result=ToolResult(call_id="t1", name="write_file", output="ok")),
    ]


def test_breakpoints_cover_tools_system_and_conversation_tail():
    req = {
        "tools": [{"name": "a", "input_schema": {}}, {"name": "b", "input_schema": {}}],
        "system": "persona",
        "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}, {"role": "user", "content": "go"}],
    }
    original_tool = req["tools"][-1]
    apply_prompt_cache_breakpoints(req)

    assert "cache_control" not in req["tools"][0] and req["tools"][-1]["cache_control"] == EPHEMERAL
    assert "cache_control" not in original_tool  # caller's tool dict is not mutated
    assert req["system"] == [{"type": "text", "text": "persona", "cache_control": EPHEMERAL}]
    assert req["messages"][0]["content"] == "hi"
    assert req["messages"][-1]["content"] == [{"type": "text", "text": "go", "cache_control": EPHEMERAL}]
    breakpoints = 1 + 1 + 1
    assert breakpoints <= 4


def test_breakpoint_lands_on_tool_result_block_and_skips_empty_text():
    req = {"messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "x"}]}]}
    apply_prompt_cache_breakpoints(req)
    assert req["messages"][-1]["content"][-1]["cache_control"] == EPHEMERAL

    empty = {"messages": [{"role": "user", "content": "   "}]}
    apply_prompt_cache_breakpoints(empty)
    assert empty["messages"][-1]["content"] == "   "


@pytest.mark.asyncio
async def test_generate_sends_breakpoints_and_folds_cache_usage(monkeypatch):
    monkeypatch.delenv("VALSTORM_ANTHROPIC_PROMPT_CACHE", raising=False)
    usage = SimpleNamespace(input_tokens=120, output_tokens=40, cache_read_input_tokens=9000, cache_creation_input_tokens=800)
    client = _CapturingClient(usage)
    provider = AnthropicProvider(api_key="k", default_model="claude-opus-5-5", client=client)

    tools = [{"name": "write_file", "description": "w", "parameters": {"type": "object", "properties": {}}}]
    _, u = await provider.generate(_conversation(), tools=tools)

    sent = client.calls[0]
    assert sent["tools"][-1]["cache_control"] == EPHEMERAL
    assert sent["system"][0]["cache_control"] == EPHEMERAL
    assert sent["messages"][-1]["content"][-1]["type"] == "tool_result"
    assert sent["messages"][-1]["content"][-1]["cache_control"] == EPHEMERAL

    assert u.prompt_tokens == 120 + 9000 + 800
    assert u.cached_tokens == 9000 and u.cache_write_tokens == 800
    assert u.total_tokens == u.prompt_tokens + 40


@pytest.mark.asyncio
async def test_prompt_cache_can_be_disabled(monkeypatch):
    monkeypatch.setenv("VALSTORM_ANTHROPIC_PROMPT_CACHE", "0")
    client = _CapturingClient(SimpleNamespace(input_tokens=10, output_tokens=2))
    provider = AnthropicProvider(api_key="k", client=client)
    _, u = await provider.generate(_conversation())
    assert isinstance(client.calls[0]["system"], str)
    assert u.prompt_tokens == 10 and u.cached_tokens == 0


def test_cost_estimate_prices_cache_reads_and_writes():
    # claude-opus-5-5: $4 in / $20 out / $0.20 cache read per 1M; writes at 1.25x input.
    uncached = estimate_cost_usd("claude-opus-5-5", 1_000_000, 0)
    assert uncached == pytest.approx(4.00)
    mixed = estimate_cost_usd("claude-opus-5-5", 1_000_000, 0, cached_tokens=800_000, cache_write_tokens=100_000)
    assert mixed == pytest.approx(0.1 * 4.00 + 0.8 * 0.20 + 0.1 * 5.00)
