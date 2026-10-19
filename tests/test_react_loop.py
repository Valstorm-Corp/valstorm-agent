"""Tests for the ReAct loop execution engine, tool invocation, and token tracking."""

import pytest
from typing import Any, Dict, List, Optional

from core.models import Message, SessionState, ToolCall, ToolResult, UsageMetadata
from core.react import ReActEngine
from core.tools import ToolRegistry


class MockToolRegistry:
    """Mock tool registry for testing ReAct loop execution."""

    def __init__(self):
        self.tools = {}
        self.call_history = []

    def register(self, name: str, fn):
        self.tools[name] = fn

    def get_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": name,
                "description": f"Mock schema for {name}",
                "parameters": {"type": "object", "properties": {}},
            }
            for name in self.tools
        ]

    def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        self.call_history.append((name, arguments))
        if name not in self.tools:
            raise ValueError(f"Tool {name} not found")
        return self.tools[name](**arguments)


class MockProvider:
    """Mock provider with canned sequential responses for ReAct turns."""

    def __init__(self, responses: List[Message], name: str = "mock-provider"):
        self.responses = list(responses)
        self.name = name
        self.call_count = 0
        self.received_messages: List[List[Message]] = []

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
    ) -> Message:
        self.received_messages.append(list(messages))
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            if model and not resp.model:
                resp.model = model
            return resp
        # Fallback if called more times than responses provided
        return Message(
            role="assistant",
            content="Fallback end of mock stream.",
            model=model or "default-mock",
            usage=UsageMetadata(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


@pytest.mark.asyncio
async def test_react_loop_tool_call_and_final_response():
    """Verify tool execution on step 1, observation feeding on step 2, and token accumulation."""
    registry = MockToolRegistry()
    registry.register("calculator", lambda expression: eval(expression))

    # Step 1: Assistant requests tool call
    step1_response = Message(
        role="assistant",
        content="Let me compute that for you.",
        model="mock-gpt-4o",
        provider="mock-provider",
        tool_calls=[
            ToolCall(
                id="call-calc-1",
                name="calculator",
                arguments={"expression": "45 * 12 + 180"},
            )
        ],
        usage=UsageMetadata(prompt_tokens=50, completion_tokens=20, total_tokens=70),
    )

    # Step 2: Assistant receives tool observation and provides final answer
    step2_response = Message(
        role="assistant",
        content="The calculated result is 720.",
        model="mock-gpt-4o",
        provider="mock-provider",
        tool_calls=None,
        usage=UsageMetadata(prompt_tokens=100, completion_tokens=15, total_tokens=115),
    )

    provider = MockProvider([step1_response, step2_response])
    engine = ReActEngine(provider=provider, tools=registry)

    session = SessionState(active_model="mock-gpt-4o", active_provider="mock-provider")

    final_msg = await engine.run_turn(session, "Calculate 45 * 12 + 180")

    # Assertions on final response
    assert final_msg.role == "assistant"
    assert final_msg.content == "The calculated result is 720."
    assert final_msg.model == "mock-gpt-4o"

    # Assertions on session messages structure
    assert len(session.messages) == 4
    assert [m.role for m in session.messages] == ["user", "assistant", "tool", "assistant"]

    # User message
    assert session.messages[0].content == "Calculate 45 * 12 + 180"

    # Assistant tool call step
    assert session.messages[1].tool_calls is not None
    assert len(session.messages[1].tool_calls) == 1
    assert session.messages[1].tool_calls[0].name == "calculator"

    # Tool execution result step
    assert session.messages[2].role == "tool"
    assert session.messages[2].tool_result is not None
    assert session.messages[2].tool_result.name == "calculator"
    assert str(session.messages[2].tool_result.output) == "720"
    assert session.messages[2].tool_result.is_error is False

    # Tool call history in registry
    assert len(registry.call_history) == 1
    assert registry.call_history[0] == ("calculator", {"expression": "45 * 12 + 180"})

    # Session token accumulation
    assert session.total_prompt_tokens == 150  # 50 + 100
    assert session.total_completion_tokens == 35  # 20 + 15
    assert session.total_tokens == 185  # 70 + 115


@pytest.mark.asyncio
async def test_react_loop_mid_session_model_switch():
    """Verify mid-session model switching preserves history and accurately tags models."""
    registry = MockToolRegistry()
    registry.register("calculator", lambda expression: eval(expression))

    # Turn 1 responses (model: model-alpha)
    t1_step1 = Message(
        role="assistant",
        content="Computing...",
        model="model-alpha",
        tool_calls=[
            ToolCall(
                id="call-1",
                name="calculator",
                arguments={"expression": "10 + 10"},
            )
        ],
        usage=UsageMetadata(prompt_tokens=20, completion_tokens=10, total_tokens=30),
    )
    t1_step2 = Message(
        role="assistant",
        content="10 + 10 = 20",
        model="model-alpha",
        usage=UsageMetadata(prompt_tokens=40, completion_tokens=8, total_tokens=48),
    )

    # Turn 2 response (model: model-beta)
    t2_step1 = Message(
        role="assistant",
        content="Hello from model-beta!",
        model="model-beta",
        usage=UsageMetadata(prompt_tokens=60, completion_tokens=12, total_tokens=72),
    )

    provider = MockProvider([t1_step1, t1_step2, t2_step1])
    engine = ReActEngine(provider=provider, tools=registry)

    session = SessionState(active_model="model-alpha", active_provider="mock")

    # Turn 1
    res1 = await engine.run_turn(session, "What is 10 + 10?", model="model-alpha")
    assert res1.content == "10 + 10 = 20"
    assert res1.model == "model-alpha"

    # Turn 2 with different model
    session.active_model = "model-beta"
    res2 = await engine.run_turn(session, "Hello!", model="model-beta")
    assert res2.content == "Hello from model-beta!"
    assert res2.model == "model-beta"

    # Verify message history contains correct tags for each message
    models_used = [m.model for m in session.messages if m.role == "assistant"]
    assert models_used == ["model-alpha", "model-alpha", "model-beta"]

    # Verify cumulative token counts
    assert session.total_prompt_tokens == 20 + 40 + 60
    assert session.total_completion_tokens == 10 + 8 + 12
    assert session.total_tokens == 30 + 48 + 72


@pytest.mark.asyncio
async def test_react_loop_tool_error_handling():
    """Verify tool errors are gracefully captured as ToolResult(is_error=True) and forwarded."""
    registry = MockToolRegistry()

    def failing_tool():
        raise RuntimeError("Database connection failed")

    registry.register("failing_tool", failing_tool)

    step1 = Message(
        role="assistant",
        content="Attempting db lookup...",
        model="mock-gpt",
        tool_calls=[ToolCall(id="fail-1", name="failing_tool", arguments={})],
        usage=UsageMetadata(prompt_tokens=30, completion_tokens=10, total_tokens=40),
    )
    step2 = Message(
        role="assistant",
        content="The database lookup failed, so I cannot complete the request.",
        model="mock-gpt",
        usage=UsageMetadata(prompt_tokens=50, completion_tokens=15, total_tokens=65),
    )

    provider = MockProvider([step1, step2])
    engine = ReActEngine(provider=provider, tools=registry)

    session = SessionState(active_model="mock-gpt", active_provider="mock")
    final_msg = await engine.run_turn(session, "Fetch DB data")

    assert final_msg.content == "The database lookup failed, so I cannot complete the request."
    assert len(session.messages) == 4
    tool_msg = session.messages[2]
    assert tool_msg.role == "tool"
    assert tool_msg.tool_result is not None
    assert tool_msg.tool_result.is_error is True
    assert "Database connection failed" in str(tool_msg.tool_result.output)


@pytest.mark.asyncio
async def test_react_loop_max_iterations_bound():
    """Verify that an infinite tool-calling loop terminates at max_iterations."""
    registry = MockToolRegistry()
    registry.register("noop", lambda: "ok")

    # Provider that always calls 'noop'
    infinite_tool_call = Message(
        role="assistant",
        content="Looping...",
        model="mock-gpt",
        tool_calls=[ToolCall(id="loop", name="noop", arguments={})],
        usage=UsageMetadata(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    provider = MockProvider([infinite_tool_call] * 10)
    engine = ReActEngine(provider=provider, tools=registry)

    session = SessionState(active_model="mock-gpt", active_provider="mock")
    max_iter = 3
    final_msg = await engine.run_turn(session, "Start loop", max_iterations=max_iter)

    # Number of assistant messages generated must be exactly max_iter
    assistant_msgs = [m for m in session.messages if m.role == "assistant"]
    assert len(assistant_msgs) == max_iter


@pytest.mark.asyncio
async def test_react_loop_parallel_tool_calls():
    """Verify that multiple tool calls emitted in a single turn execute concurrently."""
    execution_order = []

    async def async_tool_1(x: int):
        import asyncio
        await asyncio.sleep(0.02)
        execution_order.append("tool_1")
        return f"result_1_{x}"

    async def async_tool_2(y: int):
        import asyncio
        await asyncio.sleep(0.01)
        execution_order.append("tool_2")
        return f"result_2_{y}"

    registry = ToolRegistry()
    registry.register(async_tool_1, name="tool_1")
    registry.register(async_tool_2, name="tool_2")

    # Step 1: Model requests both tools simultaneously
    step1 = Message(
        role="assistant",
        content=None,
        model="mock-gpt",
        tool_calls=[
            ToolCall(id="c1", name="tool_1", arguments={"x": 10}),
            ToolCall(id="c2", name="tool_2", arguments={"y": 20}),
        ],
        usage=UsageMetadata(prompt_tokens=30, completion_tokens=10, total_tokens=40),
    )

    # Step 2: Model finishes with combined result
    step2 = Message(
        role="assistant",
        content="Both tools executed successfully.",
        model="mock-gpt",
        usage=UsageMetadata(prompt_tokens=60, completion_tokens=10, total_tokens=70),
    )

    provider = MockProvider([step1, step2])
    engine = ReActEngine(provider=provider, tools=registry)

    session = SessionState(active_model="mock-gpt", active_provider="mock")
    final_msg = await engine.run_turn(session, "Run both tools")

    assert final_msg.content == "Both tools executed successfully."
    # tool_2 sleeps less so it finishes first in parallel execution
    assert execution_order == ["tool_2", "tool_1"]

    tool_messages = [m for m in session.messages if m.role == "tool"]
    assert len(tool_messages) == 2
    assert tool_messages[0].tool_result.output == "result_1_10"
    assert tool_messages[1].tool_result.output == "result_2_20"
    assert tool_messages[0].tool_result.duration_ms > 0.0
    assert tool_messages[1].tool_result.duration_ms > 0.0


@pytest.mark.asyncio
async def test_react_loop_empty_response_nudge_and_recovery():
    """Verify that when a model produces an empty response after tool execution, the engine nudges and recovers."""
    registry = ToolRegistry()
    registry.register(lambda x: f"data_{x}", name="fetch_data")

    # Step 1: Model calls tool
    step1 = Message(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="c1", name="fetch_data", arguments={"x": "abc"})],
        usage=UsageMetadata(prompt_tokens=20, completion_tokens=10, total_tokens=30),
    )
    # Step 2: Model returns empty response (content=None, tool_calls=None)
    step2_empty = Message(
        role="assistant",
        content="",
        tool_calls=None,
        usage=UsageMetadata(prompt_tokens=40, completion_tokens=1, total_tokens=41),
    )
    # Step 3: Model recovers after nudge
    step3_recovered = Message(
        role="assistant",
        content="Here is the recovered data: data_abc",
        tool_calls=None,
        usage=UsageMetadata(prompt_tokens=55, completion_tokens=15, total_tokens=70),
    )

    provider = MockProvider([step1, step2_empty, step3_recovered])
    engine = ReActEngine(provider=provider, tools=registry)

    session = SessionState(active_model="mock-gpt", active_provider="mock")
    final_msg = await engine.run_turn(session, "Fetch my data")

    assert final_msg.content == "Here is the recovered data: data_abc"
    # Ensure system notice was added between step 2 and step 3
    system_nudges = [m for m in session.messages if "[System Notice:" in (m.content or "")]
    assert len(system_nudges) == 1


@pytest.mark.asyncio
async def test_react_loop_empty_response_fallback_exhausted():
    """Verify that when a model continuously returns empty responses, fallback text is provided."""
    registry = ToolRegistry()
    step_empty = Message(
        role="assistant",
        content="",
        tool_calls=None,
        usage=UsageMetadata(prompt_tokens=10, completion_tokens=0, total_tokens=10),
    )

    provider = MockProvider([step_empty, step_empty, step_empty])
    engine = ReActEngine(provider=provider, tools=registry)

    session = SessionState(active_model="mock-gpt", active_provider="mock")
    final_msg = await engine.run_turn(session, "Hello")

    assert "No response generated by model" in final_msg.content or "concluded turn without text response" in final_msg.content


@pytest.mark.asyncio
async def test_react_loop_action_intent_nudge_recovery():
    """Verify that when a model states an action intent without tool calls, it is nudged to invoke the tool."""
    registry = MockToolRegistry()
    registry.register("write_file", lambda path, content: f"Wrote {len(content)} bytes to {path}")

    # Step 1: Model states it will write a file, but omits tool_calls
    step1_intent = Message(
        role="assistant",
        content="I will write `apps/worker-go/internal/inbound/handler.go`.",
        tool_calls=None,
        usage=UsageMetadata(prompt_tokens=20, completion_tokens=10, total_tokens=30),
    )
    # Step 2: In response to ACTION_INTENT_NUDGE, model invokes the tool
    step2_tool_call = Message(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="tc_write_1", name="write_file", arguments={"path": "handler.go", "content": "package main"})],
        usage=UsageMetadata(prompt_tokens=40, completion_tokens=20, total_tokens=60),
    )
    # Step 3: Model concludes
    step3_final = Message(
        role="assistant",
        content="File written successfully.",
        tool_calls=None,
        usage=UsageMetadata(prompt_tokens=70, completion_tokens=10, total_tokens=80),
    )

    provider = MockProvider([step1_intent, step2_tool_call, step3_final])
    engine = ReActEngine(provider=provider, tools=registry)

    session = SessionState(active_model="mock-gpt", active_provider="mock")
    final_msg = await engine.run_turn(session, "Create the Go handler")

    assert final_msg.content == "File written successfully."
    # Verify nudge was injected
    nudges = [m for m in session.messages if "[System Directive:" in (m.content or "")]
    assert len(nudges) == 1
    assert "You stated your intent to perform an action, but did not execute a tool call" in nudges[0].content


@pytest.mark.asyncio
async def test_speculative_readonly_tool_execution_during_stream_and_raw_safety():
    """Verify speculative read-only tool execution overlaps I/O with LLM streaming and respects read-after-write safety."""
    import asyncio
    import time
    from core.models import StreamEvent, StreamEventType

    class StaggeredStreamProvider:
        provider_name = "mock-stream"

        def __init__(self):
            self.call_idx = 0

        async def generate_stream(self, messages, tools=None, model=None):
            self.call_idx += 1
            if self.call_idx in (1, 3):
                tc1 = ToolCall(id="tc_r1", name="read_file", arguments={"path": "a.py"})
                tc2 = ToolCall(id="tc_r2", name="search_files", arguments={"pattern": "foo"})
                tc3 = ToolCall(id="tc_r3", name="code_outline", arguments={"path": "b.py"})
                # Emit tool calls staggered during the LLM stream before TURN_COMPLETE
                yield StreamEvent(event_type=StreamEventType.TOOL_CALL_DETECTED, tool_call=tc1)
                await asyncio.sleep(0.10)
                yield StreamEvent(event_type=StreamEventType.TOOL_CALL_DETECTED, tool_call=tc2)
                await asyncio.sleep(0.10)
                yield StreamEvent(event_type=StreamEventType.TOOL_CALL_DETECTED, tool_call=tc3)
                await asyncio.sleep(0.12)
                final_m = Message(
                    role="assistant",
                    content="",
                    tool_calls=[tc1, tc2, tc3],
                    usage=UsageMetadata(prompt_tokens=60, completion_tokens=30, total_tokens=90),
                )
                yield StreamEvent(event_type=StreamEventType.TURN_COMPLETE, message=final_m, usage=final_m.usage)
            else:
                done_m = Message(
                    role="assistant",
                    content="All 3 files inspected.",
                    usage=UsageMetadata(prompt_tokens=90, completion_tokens=15, total_tokens=105),
                )
                yield StreamEvent(event_type=StreamEventType.TURN_COMPLETE, message=done_m, usage=done_m.usage)

    class AsyncTimedRegistry:
        def get_schemas(self):
            return []

        async def execute_async(self, name: str, args: dict, call_id: str = None):
            await asyncio.sleep(0.18)
            return f"Result for {name}({args})"

    prov = StaggeredStreamProvider()
    reg = AsyncTimedRegistry()
    engine = ReActEngine(provider=prov, tools=reg)

    # 1. With speculative_tools=True (default): tool I/O overlaps with stream generation
    s_spec = SessionState(active_model="mock-gpt", active_provider="mock-stream", metadata={"speculative_tools": True})
    t0 = time.perf_counter()
    msg_spec = await engine.run_turn(s_spec, "Inspect a.py, foo, and b.py")
    dur_spec = time.perf_counter() - t0
    assert msg_spec.content == "All 3 files inspected."
    assert s_spec.metadata.get("speculative_tools_launched") == 3

    # 2. With speculative_tools=False (baseline): tools wait until stream completes
    s_nospec = SessionState(active_model="mock-gpt", active_provider="mock-stream", metadata={"speculative_tools": False})
    t1 = time.perf_counter()
    msg_nospec = await engine.run_turn(s_nospec, "Inspect a.py, foo, and b.py")
    dur_nospec = time.perf_counter() - t1
    assert msg_nospec.content == "All 3 files inspected."
    assert s_nospec.metadata.get("speculative_tools_launched", 0) == 0
    # Speculative overlap saves ~120ms+ compared to waiting for TURN_COMPLETE
    assert dur_spec < dur_nospec - 0.08

    # 3. Verify Read-After-Write batch safety: if write_file precedes read_file in the same batch,
    # read_file is NOT launched speculatively before write_file runs.
    execution_order = []

    class RawSafetyStreamProvider:
        provider_name = "mock-raw"
        called = 0

        async def generate_stream(self, messages, tools=None, model=None):
            self.called += 1
            if self.called == 1:
                tc_w = ToolCall(id="tc_w1", name="write_file", arguments={"path": "x.txt", "content": "new"})
                tc_r = ToolCall(id="tc_r_after_w", name="read_file", arguments={"path": "x.txt"})
                yield StreamEvent(event_type=StreamEventType.TOOL_CALL_DETECTED, tool_call=tc_w)
                yield StreamEvent(event_type=StreamEventType.TOOL_CALL_DETECTED, tool_call=tc_r)
                await asyncio.sleep(0.08)
                m = Message(role="assistant", content="", tool_calls=[tc_w, tc_r])
                yield StreamEvent(event_type=StreamEventType.TURN_COMPLETE, message=m)
            else:
                m = Message(role="assistant", content="Write then read done.")
                yield StreamEvent(event_type=StreamEventType.TURN_COMPLETE, message=m)

    class OrderTrackingRegistry:
        def get_schemas(self):
            return []

        async def execute_async(self, name: str, args: dict, call_id: str = None):
            execution_order.append(name)
            return f"ok:{name}"

    raw_session = SessionState(active_model="mock-gpt", active_provider="mock-raw")
    raw_engine = ReActEngine(provider=RawSafetyStreamProvider(), tools=OrderTrackingRegistry())
    await raw_engine.run_turn(raw_session, "Write and read x.txt")
    # Neither write_file nor the subsequent read_file was launched speculatively
    assert raw_session.metadata.get("speculative_tools_launched", 0) == 0
    assert execution_order == ["write_file", "read_file"]

