"""Live Self-Hosting Acceptance Test for Valstorm Agent Runtime.

Tests that the agent can autonomously write code, run pytest via terminal_exec,
diagnose test failures, patch bugs, and achieve 100% test passes.
"""

import os
import shutil
import time
import pytest
from pathlib import Path

from core.context import WorkspaceContextManager
from core.keystore import KeyStore
from core.models import SessionState
from core.react import ReActEngine
from core.tools import ToolRegistry
from providers.gemini import GeminiProvider
from tools.developer_tools import register_developer_tools


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_agent_self_development_loop(tmp_path):
    """End-to-end self-hosting test: agent authors code, runs tests, fixes a bug, and passes."""
    api_key = KeyStore.resolve_key("gemini")
    if not api_key:
        pytest.skip("No GEMINI_API_KEY configured for live self-development test.")

    workdir = tmp_path / "sandbox_workspace"
    workdir.mkdir()

    # Setup tools & engine
    registry = ToolRegistry()
    register_developer_tools(registry)
    provider = GeminiProvider(api_key=api_key)
    engine = ReActEngine(provider=provider, tools=registry)

    ctx_mgr = WorkspaceContextManager(workdir=str(workdir))
    system_prompt = ctx_mgr.build_system_prompt()

    session = SessionState(active_model="gemini-flash-lite-latest", active_provider="gemini")

    task_prompt = (
        f"You are working in the directory `{workdir}`.\n"
        "Perform these exact steps using your developer tools:\n"
        f"1. Use `write_file` to create `{workdir}/math_ops.py` with a function `add_numbers(a, b)` that returns `a - b` (intentionally buggy).\n"
        f"2. Use `write_file` to create `{workdir}/test_math_ops.py` with `def test_add(): assert add_numbers(2, 3) == 5`.\n"
        f"3. Run `terminal_exec` with `pytest {workdir}/test_math_ops.py` and observe the test failure.\n"
        f"4. Use `patch_file` on `{workdir}/math_ops.py` to fix the bug so it correctly returns `a + b`.\n"
        f"5. Run `terminal_exec` with `pytest {workdir}/test_math_ops.py` to verify all tests pass.\n"
        "6. Return a concise final message confirming the test passed."
    )

    tools_called = []
    def on_step(event_type: str, payload):
        if event_type == "tool_call":
            t_name = payload.get("name") if isinstance(payload, dict) else getattr(payload, "name", str(payload))
            tools_called.append(t_name)

    start_t = time.time()
    response = await engine.run_turn(
        session=session,
        user_input=task_prompt,
        model="gemini-flash-lite-latest",
        step_callback=on_step,
    )
    duration = time.time() - start_t

    # Verify artifacts on disk
    math_file = workdir / "math_ops.py"
    test_file = workdir / "test_math_ops.py"

    assert math_file.is_file(), "Agent should have created math_ops.py"
    assert test_file.is_file(), "Agent should have created test_math_ops.py"
    assert "a + b" in math_file.read_text(), "Agent should have patched math_ops.py to return a + b"

    # Verify tool usage sequence
    assert "write_file" in tools_called
    assert "terminal_exec" in tools_called
    assert "patch_file" in tools_called

    print(f"\n[Self-Development Live Test Passed] Duration: {duration:.2f}s | Tools called: {tools_called}")
