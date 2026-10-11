"""Unit tests for Phase 4 Milestone 2: Profile Auto-Injection and Real Subagent Delegation.

Tests:
1. Profile resolution from disk (~/.valstorm/profiles) and built-in fallbacks.
2. WorkspaceContextManager dynamic system prompt assembly with profile + attached skills.
3. Tool whitelist scoping per profile definition.
4. delegate_task execution with isolated session state and tool scoping.
5. Asynchronous non-blocking subagent delegation (background=True), polling, and parallel waiting.
6. subagent_manage actions (list, poll, wait, kill).
"""

import asyncio
import json
import os
from pathlib import Path
import tempfile
import pytest

from core.context import (
    BUILTIN_PROFILES,
    WorkspaceContextManager,
    format_attached_skills_context,
    list_available_profiles,
    load_profile,
)
from core.models import Message, SessionState, ToolCall, UsageMetadata
from core.tools import ToolRegistry, get_default_registry
from tools import create_tier1_tools, delegate_task, register_tier1_tools, subagent_manage
from tools.delegation import SubagentRegistry, set_subagent_provider_override


class MockProvider:
    """Mock LLM provider returning predetermined assistant responses."""

    def __init__(self, responses: list[Message]):
        self.responses = list(responses)
        self.provider_name = "mock"

    async def generate(self, messages, tools=None, model=None, **kwargs):
        if not self.responses:
            return Message(role="assistant", content="Subagent completed mock task."), UsageMetadata(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        resp = self.responses.pop(0)
        usage = resp.usage or UsageMetadata(prompt_tokens=15, completion_tokens=10, total_tokens=25)
        return resp, usage


def test_load_builtin_profiles():
    """Verify built-in profiles can be loaded by slug."""
    for slug in ["developer", "researcher", "backend-tester", "writer", "orchestrator", "valstorm-assistant"]:
        prof = load_profile(slug)
        assert prof is not None
        assert "name" in prof
        assert "model" in prof
        assert "provider" in prof
        assert "system_prompt" in prof
        assert "allowed_tools" in prof
        assert len(prof["allowed_tools"]) > 0


def test_list_available_profiles():
    """Verify list_available_profiles returns built-in and user profiles."""
    profiles = list_available_profiles()
    assert len(profiles) >= len(BUILTIN_PROFILES)
    profile_slugs = [p.get("api_name") for p in profiles]
    for slug in BUILTIN_PROFILES:
        assert slug in profile_slugs


def test_custom_profile_from_disk():
    """Verify custom profile loading from ~/.valstorm/profiles/<slug>.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prof_dir = Path(tmpdir) / ".valstorm" / "profiles"
        prof_dir.mkdir(parents=True, exist_ok=True)
        custom_file = prof_dir / "sec-auditor.json"
        custom_data = {
            "name": "Security Auditor",
            "description": "Specialized in SAST and vulnerability audits.",
            "model": "gemini-flash-latest",
            "provider": "gemini",
            "system_prompt": "Audit code for security vulnerabilities.",
            "allowed_tools": ["read_file", "search_files", "execute_code"],
            "tag": ["Security & Compliance", "SOP"],
            "scoped_paths": ["apps/api/**", "k8s/**"],
            "max_turns": 15,
        }
        custom_file.write_text(json.dumps(custom_data))

        old_env = os.environ.get("VALSTORM_PROFILES_DIR")
        try:
            os.environ["VALSTORM_PROFILES_DIR"] = str(prof_dir)
            loaded = load_profile("sec-auditor")
            assert loaded["name"] == "Security Auditor"
            assert loaded["allowed_tools"] == ["read_file", "search_files", "execute_code"]
            assert loaded["tag"] == ["Security & Compliance", "SOP"]
            assert loaded["scoped_paths"] == ["apps/api/**", "k8s/**"]

            # Verify resolution by id or alias
            custom_data_with_id = dict(custom_data)
            custom_data_with_id["id"] = "aia_test_sec_123"
            custom_data_with_id["api_name"] = "sec-auditor"
            (prof_dir / "sec-auditor.json").write_text(json.dumps(custom_data_with_id))
            loaded_by_id = load_profile("aia_test_sec_123")
            assert loaded_by_id["name"] == "Security Auditor"
        finally:
            if old_env:
                os.environ["VALSTORM_PROFILES_DIR"] = old_env
            else:
                os.environ.pop("VALSTORM_PROFILES_DIR", None)


def test_workspace_context_manager_profile_prompt_assembly():
    """Verify context manager builds dynamic system prompt with profile instructions."""
    ctx_mgr = WorkspaceContextManager()
    prof = load_profile("developer")
    prompt = ctx_mgr.build_system_prompt(profile=prof)

    assert "Active Persona: Developer" in prompt or "Developer" in prompt
    assert "Attached Specialized Procedural Skills" in prompt


def test_format_attached_skills_context():
    """Verify format_attached_skills_context generates clean markdown guide index."""
    slugs = ["test-driven-development", "systematic-debugging", "custom-guide"]
    formatted = format_attached_skills_context(slugs)

    assert "# Attached Specialized Procedural Skills:" in formatted
    assert "test-driven-development" in formatted
    assert "systematic-debugging" in formatted
    assert "skill_view" in formatted


def test_format_persona_scope_context():
    """Verify format_persona_scope_context outputs structured domain and knowledge boundaries."""
    from core.context import format_persona_scope_context

    # Case 1: Empty or None profile
    assert format_persona_scope_context(None) == ""
    assert format_persona_scope_context({}) == ""

    # Case 2: Full persona scoping with Knowledge Graph tags
    prof = {
        "name": "Marketing Specialist",
        "api_name": "marketing-specialist",
        "tag": ["Marketing", "SOP", "Playbook"],
        "knowledge_vaults": ["vaul_brand_01", {"id": "vaul_campaigns_02", "name": "Q4 Campaigns"}],
        "knowledge_files": ["file_sop_1", {"id": "file_sop_2", "name": "Copywriting_SOP.md"}],
        "scoped_paths": ["apps/marketing-v3/**", "packages/ui/**"],
    }
    formatted = format_persona_scope_context(prof)

    assert "# 🧭 Persona Domain & Scoped Knowledge:" in formatted
    assert "Marketing Specialist (`marketing-specialist`)" in formatted
    assert "Knowledge Graph Tags: `Marketing`, `SOP`, `Playbook`" in formatted
    assert "Primary Knowledge Vaults:" in formatted
    assert "`vaul_brand_01`" in formatted
    assert "Q4 Campaigns (`vaul_campaigns_02`)" in formatted
    assert "Pinned Knowledge Files:" in formatted
    assert "`file_sop_1`" in formatted
    assert "Copywriting_SOP.md (`file_sop_2`)" in formatted
    assert "Prioritized Repository Paths:" in formatted
    assert "`apps/marketing-v3/**`" in formatted
    assert "`packages/ui/**`" in formatted
    assert "Dynamic Knowledge Graph Directive:" in formatted
    assert "Tool Scoping Directive:" in formatted


def test_workspace_orientation_and_scoped_prompt_assembly(tmp_path):
    """Verify WorkspaceContextManager injects git orientation and persona scope into system prompt."""
    from core.context import WorkspaceContextManager

    # Setup mock git repo & packages
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/feature/context-aware\n")
    (tmp_path / "apps").mkdir()
    (tmp_path / "packages").mkdir()

    ctx_mgr = WorkspaceContextManager(workdir=str(tmp_path))
    orientation = ctx_mgr.get_workspace_orientation()
    assert orientation["git_branch"] == "feature/context-aware"
    assert "apps" in orientation["key_directories"]
    assert "packages" in orientation["key_directories"]

    # Assemble prompt with custom scoped profile
    prof = {
        "name": "DevOps Engineer",
        "api_name": "devops",
        "system_prompt": "You are a DevOps engineer.",
        "knowledge_vaults": ["vaul_infra_99"],
        "scoped_paths": ["k8s/**"],
    }
    prompt = ctx_mgr.build_system_prompt(profile=prof)

    assert "Git Branch: feature/context-aware" in prompt
    assert "Project Structure: apps, packages" in prompt
    assert "Active Profile: DevOps Engineer (devops)" in prompt
    assert "# 🧭 Persona Domain & Scoped Knowledge:" in prompt
    assert "vaul_infra_99" in prompt
    assert "k8s/**" in prompt


def test_tool_whitelist_scoping():
    """Verify filtering registry by profile allowed_tools isolates tools."""
    full_registry = get_default_registry()
    register_tier1_tools(full_registry)

    prof = load_profile("researcher")
    allowed = prof["allowed_tools"]
    scoped = full_registry.filter_by_whitelist(allowed)

    for t_name in scoped.list_tools():
        assert t_name in allowed

    # Ensure dangerous/delegated tools are excluded if not in whitelist
    if "terminal_exec" not in allowed:
        assert scoped.get("terminal_exec") is None


@pytest.mark.asyncio
async def test_delegate_task_subagent_run():
    """Verify delegate_task generates structured child session summary."""
    mock_prov = MockProvider([Message(role="assistant", content="Found log analysis: root cause identified.")])
    set_subagent_provider_override(mock_prov)
    try:
        goal = "Investigate error logs and summarize root causes"
        res = await delegate_task(profile="researcher", goal=goal, context="Log file at /tmp/err.log")

        assert "SUBAGENT DELEGATION COMPLETE - PROFILE: RESEARCHER" in res
        assert "Child Session ID: aich_sub_" in res
        assert "Goal:" in res or "--- GOAL ---" in res
        assert "Investigate error logs" in res
        assert "Scoped Tools:" in res
        assert "Tokens:" in res
        assert "Status: COMPLETED" in res
        assert "Found log analysis: root cause identified." in res
    finally:
        set_subagent_provider_override(None)


@pytest.mark.asyncio
async def test_delegate_task_disallows_infinite_recursion():
    """Verify delegate_task strips delegate_task from child allowed tools to prevent recursion loops."""
    mock_prov = MockProvider([Message(role="assistant", content="Plan created successfully.")])
    set_subagent_provider_override(mock_prov)
    try:
        prof = load_profile("orchestrator")
        allowed = prof["allowed_tools"]
        assert "delegate_task" in allowed

        res = await delegate_task(profile="orchestrator", goal="Plan migration")
        assert "SUBAGENT DELEGATION COMPLETE" in res
        assert "delegate_task" not in res.split("Scoped Tools: [")[1].split("]")[0]
    finally:
        set_subagent_provider_override(None)


@pytest.mark.asyncio
async def test_delegate_task_empty_inputs():
    """Verify delegate_task handles missing arguments gracefully."""
    res1 = await delegate_task(profile="", goal="Do work")
    assert "Error: A target profile name must be provided" in res1

    res2 = await delegate_task(profile="developer", goal="")
    assert "Error: A goal must be provided" in res2


@pytest.mark.asyncio
async def test_delegate_task_custom_turns_and_timeout():
    """Verify delegate_task accepts custom max_turns and timeout_sec overrides."""
    mock_prov = MockProvider([Message(role="assistant", content="Subagent with custom turns completed.")])
    set_subagent_provider_override(mock_prov)
    try:
        res = await delegate_task(
            profile="developer",
            goal="Refactor auth tests",
            max_turns=45,
            timeout_sec=60.0,
        )
        assert "SUBAGENT DELEGATION COMPLETE - PROFILE: DEVELOPER" in res
        assert "Subagent with custom turns completed." in res
    finally:
        set_subagent_provider_override(None)


@pytest.mark.asyncio
async def test_delegate_task_background_async_and_poll_wait():
    """Verify delegate_task with background=True runs asynchronously and can be polled and waited on."""
    async def delayed_generate(messages, tools=None, model=None, **kwargs):
        await asyncio.sleep(0.05)
        return Message(role="assistant", content="Async background task finished."), UsageMetadata(prompt_tokens=20, completion_tokens=10, total_tokens=30)

    mock_prov = MockProvider([])
    mock_prov.generate = delayed_generate
    set_subagent_provider_override(mock_prov)

    try:
        # 1. Spawn in background
        spawn_res = await delegate_task(
            profile="developer",
            goal="Implement Turnstile verification in backend",
            background=True,
        )

        assert "SUBAGENT DELEGATION STARTED IN BACKGROUND" in spawn_res
        assert "Task ID: subtask_" in spawn_res
        assert "Status: RUNNING" in spawn_res

        import re
        match = re.search(r"Task ID:\s*(subtask_[a-f0-9]+)", spawn_res)
        assert match is not None
        task_id = match.group(1)

        # 2. Poll the running subagent
        poll_res = await subagent_manage(action="poll", task_id=task_id)
        assert "subtask_" in poll_res

        # 3. Wait for the subagent to complete
        wait_res = await subagent_manage(action="wait", task_id=task_id)
        assert "SUBAGENT DELEGATION COMPLETE - PROFILE: DEVELOPER" in wait_res
        assert "Async background task finished." in wait_res
        assert "Status: COMPLETED" in wait_res
    finally:
        set_subagent_provider_override(None)


@pytest.mark.asyncio
async def test_parallel_multiple_background_subagents():
    """Verify spawning multiple background subagents in parallel and waiting for all of them."""
    async def delayed_generate_backend(messages, tools=None, model=None, **kwargs):
        await asyncio.sleep(0.05)
        return Message(role="assistant", content="Backend Turnstile logic implemented."), UsageMetadata(prompt_tokens=20, completion_tokens=10, total_tokens=30)

    mock_prov = MockProvider([])
    mock_prov.generate = delayed_generate_backend
    set_subagent_provider_override(mock_prov)

    try:
        # Spawn Agent 1 (Backend)
        res1 = await delegate_task(
            profile="developer",
            goal="Backend spam protection",
            background=True,
        )
        # Spawn Agent 2 (Frontend)
        res2 = await delegate_task(
            profile="developer",
            goal="Frontend Turnstile widget",
            background=True,
        )

        assert "SUBAGENT DELEGATION STARTED IN BACKGROUND" in res1
        assert "SUBAGENT DELEGATION STARTED IN BACKGROUND" in res2

        # Wait for all running subagents
        wait_all_res = await subagent_manage(action="wait", task_id="all")
        assert "Backend spam protection" in wait_all_res
        assert "Frontend Turnstile widget" in wait_all_res
        assert "Backend Turnstile logic implemented." in wait_all_res
    finally:
        set_subagent_provider_override(None)


@pytest.mark.asyncio
async def test_subagent_manage_tool():
    """Verify subagent_manage tool actions (spawn, list, poll, wait, kill)."""
    async def delayed_generate(messages, tools=None, model=None, **kwargs):
        await asyncio.sleep(0.5)
        return Message(role="assistant", content="Long running subagent task done."), UsageMetadata(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    mock_prov = MockProvider([])
    mock_prov.generate = delayed_generate
    set_subagent_provider_override(mock_prov)

    try:
        # Spawn via subagent_manage
        spawn_res = await subagent_manage(
            action="spawn",
            profile="researcher",
            goal="Analyze auth architecture",
            background=True,
        )
        assert "SUBAGENT DELEGATION STARTED IN BACKGROUND" in spawn_res

        import re
        task_id = re.search(r"Task ID:\s*(subtask_[a-f0-9]+)", spawn_res).group(1)

        # List tasks
        list_res = await subagent_manage(action="list")
        assert "ACTIVE & RECENT SUBAGENT SESSIONS" in list_res
        assert task_id in list_res

        # Kill task
        kill_res = await subagent_manage(action="kill", task_id=task_id)
        assert "successfully cancelled" in kill_res

        # Poll killed task
        poll_res = await subagent_manage(action="poll", task_id=task_id)
        assert "SUBAGENT DELEGATION KILLED" in poll_res
    finally:
        set_subagent_provider_override(None)
