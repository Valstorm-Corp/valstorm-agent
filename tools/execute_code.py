"""In-Process Safe Python Code Execution Tool for Valstorm Agent Runtime.

Provides:
- execute_code: In-process Python script execution with tool helper bindings,
  stdout/stderr capture, and timeout management.
"""

import asyncio
import contextlib
import glob
import io
import inspect
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

from core.models import ToolResult
from core.sandbox import HostSandbox, get_current_sandbox
from core.tools import ToolRegistry, calculator, mock_db_lookup, read_local_file, tool
from tools.developer_tools import (
    _truncate_output,
    patch_file,
    read_file,
    search_files,
    terminal_exec,
    write_file,
)
from tools.skill_tool import skill_list, skill_view
from tools.vision_tool import analyze_image, vision_analyze


def _wrap_sync(fn: Callable) -> Callable:
    """Wraps sync or async tool functions so they can be called synchronously in scripts."""
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            res = fn(*args, **kwargs)
        except Exception as e:
            return f"Error calling {getattr(fn, '__name__', 'tool')}: {e}"

        if inspect.isawaitable(res):
            async def _coro_wrapper() -> Any:
                return await res

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        return pool.submit(asyncio.run, _coro_wrapper()).result()
                else:
                    return loop.run_until_complete(res)
            except RuntimeError:
                return asyncio.run(_coro_wrapper())
        return res

    wrapper.__name__ = getattr(fn, "__name__", "tool_wrapper")
    wrapper.__doc__ = getattr(fn, "__doc__", "")
    return wrapper


def _build_execution_environment(
    extra_bindings: Optional[Dict[str, Any]] = None,
    platform: Optional[Any] = None,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Constructs the standard global bindings for execute_code scripts."""
    env: Dict[str, Any] = {
        # Standard libraries
        "json": json,
        "re": re,
        "math": math,
        "os": os,
        "sys": sys,
        "time": time,
        "shlex": shlex,
        "Path": Path,
        "subprocess": subprocess,
        "glob": glob,
        # Standard Tool bindings
        "read_file": _wrap_sync(read_file),
        "write_file": _wrap_sync(write_file),
        "patch_file": _wrap_sync(patch_file),
        "search_files": _wrap_sync(search_files),
        "terminal": _wrap_sync(terminal_exec),
        "terminal_exec": _wrap_sync(terminal_exec),
        "calculator": _wrap_sync(calculator),
        "mock_db_lookup": _wrap_sync(mock_db_lookup),
        "read_local_file": _wrap_sync(read_local_file),
        "skill_view": _wrap_sync(skill_view),
        "skill_list": _wrap_sync(skill_list),
        "analyze_image": _wrap_sync(analyze_image),
        "vision_analyze": _wrap_sync(vision_analyze),
    }

    # Bind Valstorm Platform Context if available
    effective_platform = platform
    if not effective_platform:
        try:
            from tools.valstorm_platform_client import RemotePlatformContext
            effective_platform = RemotePlatformContext(client=client)
        except Exception:
            pass

    if effective_platform:
        env["platform"] = effective_platform
        env["sql_query"] = effective_platform.query.sql
        env["vfs_search"] = effective_platform.query.vfs_search
        env["records_create"] = effective_platform.records.create
        env["records_update"] = effective_platform.records.update
        env["records_delete"] = effective_platform.records.delete
        if hasattr(effective_platform, "slack"):
            env["slack_post_message"] = effective_platform.slack.post_message
            env["slack_list_channels"] = effective_platform.slack.list_channels
            env["slack_get_channel_history"] = effective_platform.slack.get_channel_history
            env["slack_list_users"] = effective_platform.slack.list_users
            env["slack_get_user_profile"] = effective_platform.slack.get_user_profile
            env["slack_add_reaction"] = effective_platform.slack.add_reaction
            env["slack_update_message"] = effective_platform.slack.update_message
            env["slack_delete_message"] = effective_platform.slack.delete_message
            env["slack_get_auth_status"] = effective_platform.slack.get_auth_status

    if extra_bindings:
        for k, v in extra_bindings.items():
            if callable(v):
                env[k] = _wrap_sync(v)
            else:
                env[k] = v

    return env


def _exec_script_worker(code_str: str, global_scope: Dict[str, Any]) -> tuple[str, Optional[str]]:
    """Executes a Python code block capturing stdout and stderr."""
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    local_scope: Dict[str, Any] = {}

    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            # Parse AST to check if last statement is an expression
            parsed = compile(code_str, "<execute_code>", "exec")
            exec(parsed, global_scope, local_scope)

        out = stdout_buf.getvalue()
        err = stderr_buf.getvalue()

        output_parts = []
        if out:
            output_parts.append(out)
        if err:
            output_parts.append(f"--- STDERR ---\n{err}")

        final_output = "\n".join(output_parts).strip()
        return (final_output if final_output else "(Python script executed successfully with no output)", None)

    except Exception:
        err_msg = traceback.format_exc()
        out = stdout_buf.getvalue()
        combined = f"{out}\n\nExecution Error:\n{err_msg}".strip() if out else f"Execution Error:\n{err_msg}"
        return (combined, err_msg)


@tool
async def execute_code(
    code: str,
    timeout_sec: int = 60,
) -> str:
    """Executes an in-process Python script with pre-bound tool helpers and captured output.

    Use when you need multi-step logic (loops, data filtering, batch reads/writes, conditional branches)
    to avoid multiple back-and-forth tool roundtrips.

    Available pre-bound tool helpers:
      - read_file(path, offset=1, limit=2000)
      - write_file(path, content)
      - patch_file(path, old_string, new_string, replace_all=False)
      - search_files(pattern, path='.', target='content', file_glob=None, limit=50)
      - terminal(command, timeout=120, workdir=None)
      - calculator(expression)
      - mock_db_lookup(query)
      - Standard modules: json, re, math, os, sys, time, shlex, Path

    Args:
        code: The Python script string to execute.
        timeout_sec: Maximum execution time in seconds (default: 60).

    Returns:
        The captured stdout/stderr output or error traceback.
    """
    clean_code = (code or "").strip()
    if not clean_code:
        return "Error: No Python code provided to execute."

    # Delegate to isolated sandbox if running in microVM or Docker
    sandbox = get_current_sandbox()
    from core.sandbox import HostSandbox
    if not isinstance(sandbox, HostSandbox):
        return await sandbox.exec_python(clean_code, timeout_sec=timeout_sec)

    globals_dict = _build_execution_environment()

    try:
        result, error = await asyncio.wait_for(
            asyncio.to_thread(_exec_script_worker, clean_code, globals_dict),
            timeout=float(timeout_sec),
        )
        return _truncate_output(result)
    except asyncio.TimeoutError:
        return f"Error: Python script execution timed out after {timeout_sec} seconds."
    except Exception as e:
        return f"Error executing Python code: {type(e).__name__}: {e}"


def create_execute_code_tools(
    client: Optional[Any] = None,
    platform: Optional[Any] = None,
) -> List[Any]:
    """Returns list of execute_code tool functions with bound platform context."""

    @tool
    async def execute_code(
        code: str,
        timeout_sec: int = 60,
    ) -> str:
        """Executes an in-process Python script with pre-bound tool helpers and captured output.

        Use when you need multi-step logic (loops, data filtering, batch reads/writes, conditional branches)
        to avoid multiple back-and-forth tool roundtrips.

        Available pre-bound tool helpers:
          - platform (RemotePlatformContext with .records, .query, .schema, .files)
          - sql_query(query: str, limit=50) -> list[dict]
          - records_create(api_name, records) -> dict/list
          - records_update(api_name, records) -> dict/list
          - records_delete(api_name, ids) -> dict
          - vfs_search(query: str, limit=10) -> list[dict]
          - read_file(path, offset=1, limit=2000)
          - write_file(path, content)
          - patch_file(path, old_string, new_string, replace_all=False)
          - search_files(pattern, path='.', target='content', file_glob=None, limit=50)
          - terminal(command, timeout=120, workdir=None)
          - calculator(expression)
          - mock_db_lookup(query)
          - analyze_image(path, prompt=None, detail='high')
          - Standard modules: json, re, math, os, sys, time, shlex, Path

        Args:
            code: The Python script string to execute.
            timeout_sec: Maximum execution time in seconds (default: 60).

        Returns:
            The captured stdout/stderr output or error traceback.
        """
        clean_code = (code or "").strip()
        if not clean_code:
            return "Error: No Python code provided to execute."

        # Delegate to isolated sandbox if running in microVM or Docker
        sandbox = get_current_sandbox()
        from core.sandbox import HostSandbox
        if not isinstance(sandbox, HostSandbox):
            return await sandbox.exec_python(clean_code, timeout_sec=timeout_sec)

        globals_dict = _build_execution_environment(platform=platform, client=client)

        try:
            result, error = await asyncio.wait_for(
                asyncio.to_thread(_exec_script_worker, clean_code, globals_dict),
                timeout=float(timeout_sec),
            )
            return _truncate_output(result)
        except asyncio.TimeoutError:
            return f"Error: Python script execution timed out after {timeout_sec} seconds."
        except Exception as e:
            return f"Error executing Python code: {type(e).__name__}: {e}"

    return [execute_code]


def register_execute_code_tools(
    registry: ToolRegistry,
    client: Optional[Any] = None,
    platform: Optional[Any] = None,
) -> ToolRegistry:
    """Registers execute_code tools into the provided ToolRegistry."""
    for tool_fn in create_execute_code_tools(client=client, platform=platform):
        registry.register(tool_fn)
    return registry
