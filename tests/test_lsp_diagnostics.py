"""Tests for the LSP and Compiler Diagnostics subsystem."""

import asyncio
import os
from pathlib import Path
import subprocess
import sys
import pytest

from core.lsp.diagnostics import (
    SingleFlightGate,
    VALSTORM_DIAGNOSTIC_ENV_VAR,
    _find_bin,
    _run_command,
    check_go_diagnostics,
    check_json_syntax,
    check_python_syntax,
    check_rust_diagnostics,
    check_typescript_syntax,
    check_yaml_syntax,
    cleanup_stale_diagnostic_processes,
    get_file_diagnostics,
    kill_process_tree,
)
from tools.developer_tools import patch_file, write_file


def test_check_python_syntax_valid():
    code = "def hello():\n    return 'world'\n"
    assert check_python_syntax(code) is None


def test_check_python_syntax_invalid():
    code = "def hello(:\n    return 'world'\n"
    errors = check_python_syntax(code, filename="test.py")
    assert errors is not None
    assert len(errors) == 1
    assert "SyntaxError at line 1" in errors[0]


def test_check_json_syntax_valid():
    content = '{"key": "value", "list": [1, 2, 3]}'
    assert check_json_syntax(content) is None


def test_check_json_syntax_invalid():
    content = '{"key": "value", "list": [1, 2, }'
    errors = check_json_syntax(content)
    assert errors is not None
    assert len(errors) == 1
    assert "JSONDecodeError" in errors[0]


def test_check_yaml_syntax_valid():
    content = """
    name: Valstorm
    services:
      - web
      - api
    version: 2
    """
    assert check_yaml_syntax(content) is None


def test_check_yaml_syntax_invalid():
    content = """
    name: Valstorm
    services: [unclosed list
    version: 2
    """
    errors = check_yaml_syntax(content)
    assert errors is not None
    assert len(errors) >= 1
    assert "YAMLError" in errors[0]


@pytest.mark.asyncio
async def test_get_file_diagnostics_clean_python(tmp_path: Path):
    py_file = tmp_path / "clean.py"
    py_file.write_text("x = 1\ny = 2\nz = x + y\n", encoding="utf-8")

    diag = await get_file_diagnostics(py_file)
    assert diag is None


@pytest.mark.asyncio
async def test_get_file_diagnostics_broken_python(tmp_path: Path):
    py_file = tmp_path / "broken.py"
    py_file.write_text("def broken_func(\n    pass\n", encoding="utf-8")

    diag = await get_file_diagnostics(py_file)
    assert diag is not None
    assert "[LSP Compiler Diagnostics]:" in diag
    assert "SyntaxError" in diag


@pytest.mark.asyncio
async def test_get_file_diagnostics_broken_yaml(tmp_path: Path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("key: [bad yaml\n", encoding="utf-8")

    diag = await get_file_diagnostics(yaml_file)
    assert diag is not None
    assert "[LSP Compiler Diagnostics]:" in diag
    assert "YAMLError" in diag


@pytest.mark.asyncio
async def test_write_file_with_lsp_diagnostics(tmp_path: Path):
    target = tmp_path / "bad.py"

    # Write broken syntax
    result = await write_file(str(target), "def test(\n    return 1\n")
    assert "Successfully wrote" in result
    assert "[LSP Compiler Diagnostics]:" in result
    assert "SyntaxError" in result

    # Overwrite with clean syntax
    result_clean = await write_file(str(target), "def test():\n    return 1\n")
    assert "Successfully wrote" in result_clean
    assert "[LSP Compiler Diagnostics]" not in result_clean


@pytest.mark.asyncio
async def test_patch_file_with_lsp_diagnostics(tmp_path: Path):
    target = tmp_path / "module.py"
    await write_file(str(target), "def calculate(a, b):\n    return a + b\n")

    # Patch in broken syntax
    patch_res = await patch_file(str(target), "def calculate(a, b):", "def calculate(a, b:")
    assert "Successfully patched" in patch_res
    assert "[LSP Compiler Diagnostics]:" in patch_res
    assert "SyntaxError" in patch_res

    # Fix it via patch
    fix_res = await patch_file(str(target), "def calculate(a, b:", "def calculate(a, b):")
    assert "Successfully patched" in fix_res
    assert "[LSP Compiler Diagnostics]" not in fix_res


@pytest.mark.asyncio
async def test_lsp_disabled_flag(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VALSTORM_LSP_ENABLED", "false")
    target = tmp_path / "disabled.py"

    result = await write_file(str(target), "def broken(\n")
    assert "Successfully wrote" in result
    assert "[LSP Compiler Diagnostics]" not in result


# ====================================================================
# Comprehensive Stale TSC & Process Exhaustion Prevention Tests
# ====================================================================


@pytest.mark.asyncio
async def test_run_command_kills_process_on_timeout(tmp_path: Path):
    """Verify that when a compiler/check times out, the process is killed immediately."""
    pid_file = tmp_path / "child.pid"
    script = (
        "import os, time; "
        f"open(r'{pid_file}', 'w').write(str(os.getpid())); "
        "time.sleep(30)"
    )

    out = await _run_command([sys.executable, "-c", script], timeout_sec=0.3)
    assert out is None

    await asyncio.sleep(0.2)
    assert pid_file.exists()
    child_pid = int(pid_file.read_text().strip())

    # Verify process was terminated and is no longer running
    alive = False
    try:
        os.kill(child_pid, 0)
        alive = True
    except OSError:
        alive = False

    assert not alive, f"Subprocess {child_pid} was not killed on timeout and remained orphaned!"


@pytest.mark.asyncio
async def test_run_command_kills_process_on_cancellation(tmp_path: Path):
    """Verify that when the outer coroutine is cancelled (e.g. user aborts turn), the process is killed."""
    pid_file = tmp_path / "child_cancel.pid"
    script = (
        "import os, time; "
        f"open(r'{pid_file}', 'w').write(str(os.getpid())); "
        "time.sleep(30)"
    )

    task = asyncio.create_task(_run_command([sys.executable, "-c", script], timeout_sec=10.0))
    await asyncio.sleep(0.25)
    assert pid_file.exists()
    child_pid = int(pid_file.read_text().strip())

    # Cancel task externally
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.sleep(0.2)
    alive = False
    try:
        os.kill(child_pid, 0)
        alive = True
    except OSError:
        alive = False

    assert not alive, f"Subprocess {child_pid} was not killed on cancellation!"


@pytest.mark.asyncio
async def test_single_flight_gate_preempts_older_checks():
    """Verify SingleFlightGate cancels prior in-flight checks for the same workspace."""
    gate = SingleFlightGate()
    log = []

    async def mock_check(name: str, delay: float):
        log.append(f"start_{name}")
        await asyncio.sleep(delay)
        log.append(f"finish_{name}")
        return f"result_{name}"

    # Start check 1 (slow)
    t1 = asyncio.create_task(gate.run("workspace_a", mock_check, "1", 0.4))
    await asyncio.sleep(0.05)

    # Start check 2 for same workspace (should cancel check 1 without raising CancelledError to t1 caller)
    t2 = asyncio.create_task(gate.run("workspace_a", mock_check, "2", 0.05))

    res1, res2 = await asyncio.gather(t1, t2)
    assert res1 is None, "Preempted check 1 must return None cleanly instead of raising CancelledError!"
    assert res2 == "result_2"

    assert "start_1" in log
    assert "finish_1" not in log, "Check 1 should have been cancelled before completing!"
    assert "start_2" in log
    assert "finish_2" in log


@pytest.mark.asyncio
async def test_cleanup_stale_diagnostic_processes(tmp_path: Path):
    """Verify cleanup_stale_diagnostic_processes reaps orphaned processes with diagnostic env marker."""
    env = os.environ.copy()
    env[VALSTORM_DIAGNOSTIC_ENV_VAR] = "1"
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", "import time; time.sleep(30)",
        env=env,
        start_new_session=True,
    )
    pid = proc.pid

    try:
        await asyncio.sleep(0.15)
        reaped = cleanup_stale_diagnostic_processes()
        assert pid in reaped

        await asyncio.sleep(0.15)
        alive = False
        try:
            os.kill(pid, 0)
            alive = True
        except OSError:
            alive = False
        assert not alive, f"Reaper did not terminate stale diagnostic process {pid}!"
    finally:
        try:
            proc.kill()
        except OSError:
            pass


def test_find_bin_traverses_parent_directories(tmp_path: Path):
    """Verify _find_bin traverses monorepo parent directories to find hoisted binaries."""
    root = tmp_path / "monorepo"
    bin_dir = root / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    fake_esbuild = bin_dir / "esbuild"
    fake_esbuild.write_text("#!/bin/sh\necho 1\n")
    fake_esbuild.chmod(0o755)

    deep_subpkg = root / "apps" / "valstorm-desktop" / "src" / "components"
    deep_subpkg.mkdir(parents=True)

    found = _find_bin("esbuild", deep_subpkg)
    assert found == str(fake_esbuild)


@pytest.mark.asyncio
async def test_check_typescript_syntax_with_mock_esbuild(tmp_path: Path):
    """Verify check_typescript_syntax correctly identifies errors using esbuild."""
    bin_dir = tmp_path / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    fake_esbuild = bin_dir / "esbuild"
    fake_esbuild.write_text(
        "#!/bin/sh\n"
        "echo 'X [ERROR] Unexpected token: Unclosed JSX element in Component.tsx:14:1'\n"
        "exit 1\n"
    )
    fake_esbuild.chmod(0o755)

    comp_file = tmp_path / "Component.tsx"
    comp_file.write_text("export const Component = () => <div>;\n", encoding="utf-8")

    errors = await check_typescript_syntax(comp_file)
    assert errors is not None
    assert len(errors) >= 1
    assert "Unexpected token" in errors[0]


@pytest.mark.asyncio
async def test_check_typescript_syntax_clean_file(tmp_path: Path):
    """Verify check_typescript_syntax returns None on clean file."""
    bin_dir = tmp_path / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    fake_esbuild = bin_dir / "esbuild"
    fake_esbuild.write_text("#!/bin/sh\nexit 0\n")
    fake_esbuild.chmod(0o755)

    comp_file = tmp_path / "Clean.tsx"
    comp_file.write_text("export const Clean = () => <div>Hello</div>;\n", encoding="utf-8")

    errors = await check_typescript_syntax(comp_file)
    assert errors is None


@pytest.mark.asyncio
async def test_parallel_patch_file_typescript_single_flight_does_not_cancel_run(tmp_path: Path):
    """Regression test: parallel patch_file calls on .ts files in the same workspace
    must not leak asyncio.CancelledError when SingleFlightGate preempts the first tsc check."""
    bin_dir = tmp_path / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    fake_tsc = bin_dir / "tsc"
    fake_tsc.write_text("#!/bin/sh\nsleep 0.4\nexit 0\n")
    fake_tsc.chmod(0o755)

    tsconfig = tmp_path / "tsconfig.json"
    tsconfig.write_text('{"compilerOptions": {"noEmit": true}}', encoding="utf-8")

    file_a = tmp_path / "a.ts"
    file_b = tmp_path / "b.ts"
    file_a.write_text("export const a = 1;\n", encoding="utf-8")
    file_b.write_text("export const b = 1;\n", encoding="utf-8")

    res_a, res_b = await asyncio.gather(
        patch_file(str(file_a), "export const a = 1;", "export const a = 2;"),
        patch_file(str(file_b), "export const b = 1;", "export const b = 2;"),
    )
    assert "Successfully patched" in res_a
    assert "Successfully patched" in res_b
    assert file_a.read_text(encoding="utf-8") == "export const a = 2;\n"
    assert file_b.read_text(encoding="utf-8") == "export const b = 2;\n"
