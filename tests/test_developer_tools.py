"""Unit tests for Developer Toolbelt and Workspace Context Manager."""

import asyncio
import pytest
from pathlib import Path

from core.context import WorkspaceContextManager
from core.tools import ToolRegistry
from tools.developer_tools import (
    patch_file,
    read_file,
    register_developer_tools,
    search_files,
    terminal_exec,
    write_file,
)


@pytest.mark.asyncio
async def test_terminal_exec_success(tmp_path):
    result = await terminal_exec("echo 'Hello Valstorm'", workdir=str(tmp_path))
    assert "Hello Valstorm" in result


@pytest.mark.asyncio
async def test_terminal_exec_nonzero_exit(tmp_path):
    result = await terminal_exec("exit 42", workdir=str(tmp_path))
    assert "Command exited with code 42" in result


@pytest.mark.asyncio
async def test_terminal_exec_timeout(tmp_path):
    result = await terminal_exec("sleep 2", timeout=1, workdir=str(tmp_path))
    assert "timed out after 1 seconds" in result


@pytest.mark.asyncio
async def test_terminal_exec_invalid_dir():
    result = await terminal_exec("ls", workdir="/non/existent/path/for/test/12345")
    assert "Error: Working directory does not exist" in result


@pytest.mark.asyncio
async def test_write_file_and_read_file(tmp_path):
    test_file = tmp_path / "sub" / "dir" / "sample.txt"
    content = "Line 1: Alpha\nLine 2: Beta\nLine 3: Gamma\nLine 4: Delta\n"

    # Write file with auto-created parent dirs
    write_res = await write_file(str(test_file), content)
    assert "Successfully wrote" in write_res
    assert test_file.is_file()

    # Read full file with line numbers
    read_res = await read_file(str(test_file), offset=1, limit=10)
    assert "1| Line 1: Alpha" in read_res
    assert "2| Line 2: Beta" in read_res
    assert "3| Line 3: Gamma" in read_res
    assert "4| Line 4: Delta" in read_res

    # Read with pagination
    paginated_res = await read_file(str(test_file), offset=2, limit=2)
    assert "2| Line 2: Beta" in paginated_res
    assert "3| Line 3: Gamma" in paginated_res
    assert "1| Line 1: Alpha" not in paginated_res
    assert "truncated" in paginated_res


@pytest.mark.asyncio
async def test_patch_file_unique_and_diff(tmp_path):
    test_file = tmp_path / "code.py"
    initial_code = "def greet():\n    return 'hello world'\n"
    test_file.write_text(initial_code, encoding="utf-8")

    # Successful patch
    patch_res = await patch_file(str(test_file), old_string="'hello world'", new_string="'hello valstorm'")
    assert "Successfully patched" in patch_res
    assert "hello valstorm" in test_file.read_text(encoding="utf-8")
    assert "```diff" in patch_res


@pytest.mark.asyncio
async def test_patch_file_ambiguity_and_replace_all(tmp_path):
    test_file = tmp_path / "repeated.txt"
    test_file.write_text("item = 10\nitem = 10\nitem = 10\n", encoding="utf-8")

    # Ambiguous match without replace_all should fail safely
    patch_err = await patch_file(str(test_file), old_string="item = 10", new_string="item = 20")
    assert "matches 3 occurrences" in patch_err

    # With replace_all=True
    patch_ok = await patch_file(str(test_file), old_string="item = 10", new_string="item = 20", replace_all=True)
    assert "Successfully patched" in patch_ok
    assert test_file.read_text(encoding="utf-8") == "item = 20\nitem = 20\nitem = 20\n"


@pytest.mark.asyncio
async def test_patch_file_missing_text(tmp_path):
    test_file = tmp_path / "empty.txt"
    test_file.write_text("sample content", encoding="utf-8")

    res = await patch_file(str(test_file), old_string="non_existent_text", new_string="new")
    assert "was not found" in res


@pytest.mark.asyncio
async def test_search_files_content_and_file_modes(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def run_calculation():\n    pass\n", encoding="utf-8")
    (tmp_path / "src" / "util.ts").write_text("export const runCalculation = () => {};\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored.py").write_text("run_calculation in node_modules\n", encoding="utf-8")

    # Content regex search
    content_res = await search_files("run_calculation", target="content", path=str(tmp_path))
    assert "main.py:1:" in content_res
    assert "node_modules" not in content_res

    # File glob search
    file_res = await search_files("*.ts", target="files", path=str(tmp_path))
    assert "util.ts" in file_res
    assert "main.py" not in file_res


@pytest.mark.asyncio
async def test_search_files_prioritizes_scoped_paths(tmp_path):
    from core.context import set_current_profile
    from core.sandbox import HostSandbox, set_current_sandbox

    # Setup directories: apps/marketing and apps/backend
    (tmp_path / "apps" / "marketing").mkdir(parents=True)
    (tmp_path / "apps" / "backend").mkdir(parents=True)

    (tmp_path / "apps" / "marketing" / "copy.txt").write_text("launch winter marketing campaign\n")
    (tmp_path / "apps" / "backend" / "worker.py").write_text("process campaign database records\n")

    set_current_sandbox(HostSandbox(base_dir=str(tmp_path)))

    # 1. Without profile: searches root
    try:
        res_global = await search_files("campaign", target="content")
        assert "copy.txt" in res_global
        assert "worker.py" in res_global

        # 2. With scoped_paths profile: prioritizes scoped path
        set_current_profile({"name": "Marketer", "scoped_paths": ["apps/marketing/**"]})
        res_scoped = await search_files("campaign", target="content")
        assert "Scoped: apps/marketing/**" in res_scoped
        assert "copy.txt" in res_scoped
        assert "worker.py" not in res_scoped
    finally:
        set_current_profile(None)
        set_current_sandbox(None)


def test_register_developer_tools_in_registry():
    registry = ToolRegistry()
    register_developer_tools(registry)
    tool_names = registry.list_tools()
    assert "terminal_exec" in tool_names
    assert "patch_file" in tool_names
    assert "write_file" in tool_names
    assert "read_file" in tool_names
    assert "search_files" in tool_names


def test_workspace_context_manager(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("# Project Guide\nFollow testing rules.", encoding="utf-8")
    (tmp_path / "valstorm.json").write_text('{"env": "dev", "profile": "admin"}', encoding="utf-8")

    mgr = WorkspaceContextManager(workdir=str(tmp_path))
    prompt = mgr.build_system_prompt()
    assert "Working Directory:" in prompt
    assert "Environment='dev'" in prompt
    assert "Follow testing rules." in prompt
