"""Unit tests for Tier 1 Orchestration & Safety Tools.

Tests:
1. confirmation_required: Human-in-the-loop approval gate and risk level tracking.
2. clarify: Structured interactive user dialogs with single/multi-choice and open-ended modes.
3. execute_code: In-process safe Python script execution with tool helper bindings and timeouts.
4. delegate_task: Subagent delegation with named profile resolution, context isolation, and summaries.
5. process_manage: Background process lifecycle (start, list, poll, log, wait, kill, submit).
6. ToolRegistry integration and whitelist filtering.
"""

import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List
import pytest

from core.models import Message, ToolResult, UsageMetadata
from core.tools import ToolRegistry, get_default_registry
from tools import (
    clarify,
    confirmation_required,
    create_clarify_tools,
    create_confirmation_tools,
    create_delegation_tools,
    create_developer_tools,
    create_execute_code_tools,
    create_process_manager_tools,
    create_tier1_tools,
    delegate_task,
    execute_code,
    process_manage,
    register_clarify_tools,
    register_confirmation_tools,
    register_delegation_tools,
    register_developer_tools,
    register_execute_code_tools,
    register_process_manager_tools,
    register_tier1_tools,
)
from tools.delegation import BUILTIN_PROFILES, load_profile
from tools.process_manager import ProcessRegistry, ProcessSession


# =====================================================================
# 1. confirmation_required Tests
# =====================================================================


class TestConfirmationRequiredTool:
    """Tests for the confirmation_required human-in-the-loop safety gate."""

    def test_schema(self):
        assert getattr(confirmation_required, "_is_tool") is True
        schema = getattr(confirmation_required, "_tool_schema")
        assert schema["name"] == "confirmation_required"
        assert "action" in schema["parameters"]["properties"]
        assert "details" in schema["parameters"]["properties"]
        assert "risk_level" in schema["parameters"]["properties"]
        assert schema["parameters"]["required"] == ["action", "details"]

    def test_basic_confirmation_execution(self):
        res = confirmation_required(
            action="drop_table_customers",
            details={"database": "production", "table": "customers", "record_count": 50000},
            risk_level="critical",
        )
        assert "CONFIRMATION REQUIRED - RISK LEVEL: CRITICAL" in res
        assert "Confirmation ID: conf_" in res
        assert "PENDING_APPROVAL" in res
        assert "drop_table_customers" in res
        assert "production" in res
        assert "50000" in res

    def test_risk_level_defaults_and_normalization(self):
        # Default risk level
        res_default = confirmation_required(
            action="restart_service",
            details={"service": "redis"},
        )
        assert "RISK LEVEL: HIGH" in res_default

        # Custom risk level (case insensitive)
        res_medium = confirmation_required(
            action="update_flag",
            details={"flag": "beta_features"},
            risk_level="medium",
        )
        assert "RISK LEVEL: MEDIUM" in res_medium

        # Invalid risk level falls back to HIGH
        res_invalid = confirmation_required(
            action="unknown_action",
            details={},
            risk_level="super_ultra_high",
        )
        assert "RISK LEVEL: HIGH" in res_invalid

    def test_empty_action_validation(self):
        res = confirmation_required(action="", details={})
        assert "Error: An action description must be provided" in res

    def test_execution_in_tool_registry(self):
        registry = ToolRegistry()
        register_confirmation_tools(registry)

        assert "confirmation_required" in registry.list_tools()
        result = registry.execute(
            "confirmation_required",
            {
                "action": "delete_all_keys",
                "details": {"keystore": "prod_keys"},
                "risk_level": "critical",
            },
        )
        assert isinstance(result, ToolResult)
        assert result.is_error is False
        assert "CONFIRMATION_REQUIRED" in result.output
        assert "conf_" in result.output


# =====================================================================
# 2. clarify Tests
# =====================================================================


class TestClarifyTool:
    """Tests for the clarify interactive user dialog tool."""

    def test_schema(self):
        assert getattr(clarify, "_is_tool") is True
        schema = getattr(clarify, "_tool_schema")
        assert schema["name"] == "clarify"
        assert "question" in schema["parameters"]["properties"]
        assert "choices" in schema["parameters"]["properties"]
        assert "multi_select" in schema["parameters"]["properties"]
        assert schema["parameters"]["required"] == ["question"]

    def test_single_select_choices(self):
        res = clarify(
            question="Which database dialect should we use?",
            choices=["PostgreSQL", "SQLite", "MySQL"],
            multi_select=False,
        )
        assert "CLARIFICATION REQUESTED - SINGLE SELECT" in res
        assert "Clarification ID: clarify_" in res
        assert "1. PostgreSQL" in res
        assert "2. SQLite" in res
        assert "3. MySQL" in res
        assert "Awaiting user response" in res

    def test_multi_select_choices(self):
        res = clarify(
            question="Select the export formats needed:",
            choices=["CSV", "JSON", "Parquet", "PDF"],
            multi_select=True,
        )
        assert "CLARIFICATION REQUESTED - MULTI SELECT" in res
        assert "1. CSV" in res
        assert "4. PDF" in res
        assert '"multi_select": true' in res.lower()

    def test_open_ended_question(self):
        res = clarify(
            question="What is the target customer segment for this campaign?",
            choices=None,
        )
        assert "CLARIFICATION REQUESTED - OPEN ENDED" in res
        assert "What is the target customer segment" in res
        assert '"mode": "open_ended"' in res

    def test_empty_question_validation(self):
        res = clarify(question="")
        assert "Error: A question must be provided" in res

    def test_execution_in_tool_registry(self):
        registry = ToolRegistry()
        register_clarify_tools(registry)

        assert "clarify" in registry.list_tools()
        result = registry.execute(
            "clarify",
            {
                "question": "Proceed with deployment?",
                "choices": ["Yes", "No"],
            },
        )
        assert isinstance(result, ToolResult)
        assert result.is_error is False
        assert "CLARIFICATION REQUESTED" in result.output
        assert "1. Yes" in result.output


# =====================================================================
# 3. execute_code Tests
# =====================================================================


class TestExecuteCodeTool:
    """Tests for the execute_code in-process safe Python script execution."""

    def test_schema(self):
        assert getattr(execute_code, "_is_tool") is True
        schema = getattr(execute_code, "_tool_schema")
        assert schema["name"] == "execute_code"
        assert "code" in schema["parameters"]["properties"]
        assert "timeout_sec" in schema["parameters"]["properties"]
        assert schema["parameters"]["required"] == ["code"]

    @pytest.mark.asyncio
    async def test_basic_python_execution(self):
        code = """
nums = [1, 2, 3, 4, 5]
squared = [x**2 for x in nums]
print(f"Total: {sum(squared)}")
"""
        res = await execute_code(code)
        assert "Total: 55" in res

    @pytest.mark.asyncio
    async def test_tool_helpers_bindings_file_io(self, tmp_path):
        test_file = tmp_path / "sample.txt"
        code = f"""
content = "Line 1: Alpha\\nLine 2: Beta\\nLine 3: Gamma"
w_res = write_file(r"{test_file}", content)
r_res = read_file(r"{test_file}")
print("READ_OUTPUT:")
print(r_res)
"""
        res = await execute_code(code)
        assert "READ_OUTPUT:" in res
        assert "Line 1: Alpha" in res
        assert "Line 3: Gamma" in res

    @pytest.mark.asyncio
    async def test_tool_helpers_bindings_calculator_and_math(self):
        code = """
calc_res = calculator("sqrt(144) * 5")
print(f"Calc result: {calc_res}")
print(f"Math pi: {round(math.pi, 2)}")
"""
        res = await execute_code(code)
        assert "Calc result: 60" in res
        assert "Math pi: 3.14" in res

    @pytest.mark.asyncio
    async def test_tool_helpers_mock_db_lookup(self):
        code = """
db_res = mock_db_lookup("users")
parsed = json.loads(db_res)
print(f"User count: {len(parsed)}")
print(f"First user: {parsed[0]['name']}")
"""
        res = await execute_code(code)
        assert "User count: 3" in res
        assert "First user: Alice Johnson" in res

    @pytest.mark.asyncio
    async def test_syntax_error_handling(self):
        code = "def broken_fn(:\n    pass"
        res = await execute_code(code)
        assert "Execution Error:" in res
        assert "SyntaxError" in res

    @pytest.mark.asyncio
    async def test_runtime_exception_handling(self):
        code = """
def divide(a, b):
    return a / b

divide(10, 0)
"""
        res = await execute_code(code)
        assert "Execution Error:" in res
        assert "ZeroDivisionError" in res

    @pytest.mark.asyncio
    async def test_timeout_enforcement(self):
        code = """
import time
time.sleep(3)
print("Finished sleep")
"""
        res = await execute_code(code, timeout_sec=1)
        assert "timed out after 1 seconds" in res

    @pytest.mark.asyncio
    async def test_empty_code_validation(self):
        res = await execute_code("")
        assert "Error: No Python code provided" in res

    @pytest.mark.asyncio
    async def test_registry_execution(self):
        registry = ToolRegistry()
        register_execute_code_tools(registry)

        assert "execute_code" in registry.list_tools()
        result = await registry.execute_async(
            "execute_code",
            {"code": "print('HELLO FROM REGISTRY')"},
        )
        assert isinstance(result, ToolResult)
        assert result.is_error is False
        assert "HELLO FROM REGISTRY" in result.output


# =====================================================================
# 4. delegate_task Tests
# =====================================================================


class TestDelegateTaskTool:
    """Tests for the delegate_task subagent delegation tool."""

    def test_schema(self):
        assert getattr(delegate_task, "_is_tool") is True
        schema = getattr(delegate_task, "_tool_schema")
        assert schema["name"] == "delegate_task"
        assert "profile" in schema["parameters"]["properties"]
        assert "goal" in schema["parameters"]["properties"]
        assert "context" in schema["parameters"]["properties"]
        assert schema["parameters"]["required"] == ["profile", "goal"]

    def test_load_builtin_profiles(self, tmp_path, monkeypatch):
        empty_dir = tmp_path / "empty_profiles"
        empty_dir.mkdir()
        monkeypatch.setenv("VALSTORM_PROFILES_DIR", str(empty_dir))

        dev_profile = load_profile("developer")
        assert dev_profile.get("api_name") == "developer" or "Developer" in dev_profile["name"]
        assert "terminal_exec" in dev_profile["allowed_tools"]
        assert dev_profile["provider"].lower() in ("gemini", "valstorm")

        researcher_profile = load_profile("researcher")
        assert researcher_profile.get("api_name") == "researcher" or "Researcher" in researcher_profile["name"]
        assert "mock_db_lookup" in researcher_profile["allowed_tools"]

        tester_profile = load_profile("backend-tester")
        assert tester_profile.get("api_name") == "backend-tester" or "Tester" in tester_profile["name"]

    def test_load_custom_profile_from_disk(self, tmp_path, monkeypatch):
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir(parents=True)
        monkeypatch.setenv("VALSTORM_PROFILES_DIR", str(profiles_dir))

        custom_data = {
            "name": "custom-security-auditor",
            "display_name": "Security Auditor",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "system_prompt": "Audit code for SQL injection and XSS.",
            "allowed_tools": ["read_file", "search_files"],
            "max_turns": 15,
        }
        (profiles_dir / "custom-security-auditor.json").write_text(json.dumps(custom_data), encoding="utf-8")

        loaded = load_profile("custom-security-auditor")
        assert loaded["name"] == "custom-security-auditor"
        assert loaded["model"] == "gpt-4o-mini"
        assert loaded["allowed_tools"] == ["read_file", "search_files"]

    @pytest.mark.asyncio
    async def test_delegate_developer_task(self):
        from tools.delegation import set_subagent_provider_override
        class MockProv:
            async def generate(self, *args, **kwargs):
                return Message(role="assistant", content="Refactored auth token resolution successfully."), UsageMetadata()
        set_subagent_provider_override(MockProv())
        try:
            res = await delegate_task(
                profile="developer",
                goal="Refactor auth token resolution to support Bearer prefix",
                context="See apps/api/auth.py for details.",
            )
            assert "SUBAGENT DELEGATION COMPLETE - PROFILE: DEVELOPER" in res
            assert "Child Session ID: aich_sub_" in res
            assert "Status: COMPLETED" in res
            assert "terminal_exec" in res
            assert "Refactor auth token resolution" in res
            assert "apps/api/auth.py" in res
        finally:
            set_subagent_provider_override(None)

    @pytest.mark.asyncio
    async def test_delegate_researcher_task(self):
        from tools.delegation import set_subagent_provider_override
        class MockProv:
            async def generate(self, *args, **kwargs):
                return Message(role="assistant", content="Analyzed multi-tenant MongoDB architecture."), UsageMetadata()
        set_subagent_provider_override(MockProv())
        try:
            res = await delegate_task(
                profile="researcher",
                goal="Analyze multi-tenant MongoDB routing architecture",
            )
            assert "SUBAGENT DELEGATION COMPLETE - PROFILE: RESEARCHER" in res
            assert "Child Session ID: aich_sub_" in res
            assert "read_file" in res or "mock_db_lookup" in res
            assert "Analyze multi-tenant MongoDB routing architecture" in res
        finally:
            set_subagent_provider_override(None)

    @pytest.mark.asyncio
    async def test_delegate_validation_errors(self):
        res_no_profile = await delegate_task(profile="", goal="Do something")
        assert "Error: A target profile name must be provided" in res_no_profile

        res_no_goal = await delegate_task(profile="developer", goal="")
        assert "Error: A goal must be provided" in res_no_goal

    @pytest.mark.asyncio
    async def test_registry_execution(self):
        registry = ToolRegistry()
        register_delegation_tools(registry)

        assert "delegate_task" in registry.list_tools()
        result = await registry.execute_async(
            "delegate_task",
            {
                "profile": "writer",
                "goal": "Draft architecture release notes for v1.2",
            },
        )
        assert isinstance(result, ToolResult)
        assert result.is_error is False
        assert "SUBAGENT DELEGATION COMPLETE" in result.output
        assert "aich_sub_" in result.output


# =====================================================================
# 5. process_manage Tests
# =====================================================================


class TestProcessManageTool:
    """Tests for the process_manage persistent background process manager."""

    def test_schema(self):
        assert getattr(process_manage, "_is_tool") is True
        schema = getattr(process_manage, "_tool_schema")
        assert schema["name"] == "process_manage"
        assert "action" in schema["parameters"]["properties"]
        assert "session_id" in schema["parameters"]["properties"]
        assert "command" in schema["parameters"]["properties"]
        assert "data" in schema["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_start_list_poll_and_wait_process(self):
        # 1. Start a short-lived process
        cmd = f'{sys.executable} -c "import time; print(\'STEP_1_START\'); time.sleep(0.2); print(\'STEP_2_DONE\')"'
        start_res = await process_manage(action="submit", command=cmd)
        assert "Started background process" in start_res
        assert "Session ID: proc_" in start_res
        assert "PID:" in start_res

        # Extract session_id
        session_id = None
        for line in start_res.splitlines():
            if "Session ID:" in line:
                session_id = line.split(":", 1)[1].strip()
                break
        assert session_id is not None

        # 2. List processes
        list_res = await process_manage(action="list")
        assert "Active & Tracked Background Processes" in list_res
        assert session_id in list_res

        # 3. Poll process
        await asyncio.sleep(0.1)
        poll_res = await process_manage(action="poll", session_id=session_id)
        assert "Process Status:" in poll_res

        # 4. Wait for completion
        wait_res = await process_manage(action="wait", session_id=session_id, timeout=5)
        assert f"Process session '{session_id}' finished with exit code 0" in wait_res
        assert "STEP_2_DONE" in wait_res

        # 5. Log process lines with pagination
        log_res = await process_manage(action="log", session_id=session_id, offset=1, limit=10)
        assert f"Logs for Session '{session_id}'" in log_res
        assert "STEP_1_START" in log_res

    @pytest.mark.asyncio
    async def test_kill_process(self):
        # Start a long-running process
        cmd = f'{sys.executable} -c "import time; time.sleep(60)"'
        start_res = await process_manage(action="submit", command=cmd)
        session_id = None
        for line in start_res.splitlines():
            if "Session ID:" in line:
                session_id = line.split(":", 1)[1].strip()
                break
        assert session_id is not None

        # Kill the process
        kill_res = await process_manage(action="kill", session_id=session_id)
        assert f"Process '{session_id}'" in kill_res
        assert "successfully terminated" in kill_res

        # Verify status is killed/not running
        poll_res = await process_manage(action="poll", session_id=session_id)
        assert "KILLED" in poll_res or "FAILED" in poll_res

    @pytest.mark.asyncio
    async def test_submit_stdin_data(self):
        # Start interactive process that echoes input
        cmd = f'{sys.executable} -c "import sys; line = sys.stdin.readline(); print(f\'RECEIVED: {{line.strip()}}\')"'
        start_res = await process_manage(action="submit", command=cmd)
        session_id = None
        for line in start_res.splitlines():
            if "Session ID:" in line:
                session_id = line.split(":", 1)[1].strip()
                break
        assert session_id is not None

        # Send data to stdin
        submit_res = await process_manage(action="submit", session_id=session_id, data="PING_MESSAGE_123")
        assert "Successfully sent" in submit_res

        # Wait for completion
        wait_res = await process_manage(action="wait", session_id=session_id, timeout=5)
        assert "RECEIVED: PING_MESSAGE_123" in wait_res

    @pytest.mark.asyncio
    async def test_error_handling(self):
        # Missing session_id
        res_no_session = await process_manage(action="poll")
        assert "Error: 'session_id' is required" in res_no_session

        # Non-existent session_id
        res_not_found = await process_manage(action="log", session_id="proc_does_not_exist")
        assert "not found" in res_not_found

        # Unknown action
        res_unknown = await process_manage(action="fly_to_moon", session_id="proc_123")
        assert "Error: Unknown action 'fly_to_moon'" in res_unknown

    @pytest.mark.asyncio
    async def test_registry_execution(self):
        registry = ToolRegistry()
        register_process_manager_tools(registry)

        assert "process_manage" in registry.list_tools()
        result = await registry.execute_async(
            "process_manage",
            {"action": "list"},
        )
        assert isinstance(result, ToolResult)
        assert result.is_error is False


# =====================================================================
# 6. Combined Tier 1 Tools and Whitelist Filtering Tests
# =====================================================================


class TestTier1ToolsIntegration:
    """Tests for registering all Tier 1 tools and filtering by whitelist."""

    def test_create_and_register_all_tier1_tools(self):
        tier1_tools = create_tier1_tools()
        assert len(tier1_tools) == 15
        names = [getattr(t, "_tool_name", t.__name__) for t in tier1_tools]
        assert "confirmation_required" in names
        assert "clarify" in names
        assert "analyze_image" in names
        assert "vision_analyze" in names
        assert "execute_code" in names
        assert "delegate_task" in names
        assert "subagent_manage" in names
        assert "process_manage" in names
        assert "skill_view" in names
        assert "skill_list" in names
        assert "web_scrape" in names
        assert "web_search" in names
        assert "web_crawl_domain" in names
        assert "web_content_diff" in names
        assert "web_feed_poll" in names

        registry = ToolRegistry()
        register_tier1_tools(registry)
        registered_names = registry.list_tools()
        for name in names:
            assert name in registered_names

    def test_filter_by_whitelist(self):
        registry = ToolRegistry()
        register_developer_tools(registry)
        register_tier1_tools(registry)

        assert len(registry.list_tools()) >= 10

        # Whitelist developer subset
        dev_whitelist = ["read_file", "search_files", "execute_code"]
        scoped_registry = registry.filter_by_whitelist(dev_whitelist)

        assert set(scoped_registry.list_tools()) == set(dev_whitelist)
        assert scoped_registry.get("read_file") is not None
        assert scoped_registry.get("execute_code") is not None
        assert scoped_registry.get("process_manage") is None
        assert scoped_registry.get("confirmation_required") is None

        # Verify schemas generated for scoped registry
        schemas = scoped_registry.get_schemas()
        assert len(schemas) == 3
        schema_names = [s["name"] for s in schemas]
        assert set(schema_names) == set(dev_whitelist)
