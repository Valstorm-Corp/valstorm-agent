"""Execution Sandbox Abstraction Interface for Valstorm Agent Runtime.

Provides:
- BaseSandbox: Abstract interface for isolated command, python, and filesystem operations.
- HostSandbox: Direct local host execution for Desktop Mode (Electron on Port 8650).
- ContextVar helpers: get_current_sandbox(), set_current_sandbox() for async context scoping.
"""

from abc import ABC, abstractmethod
import asyncio
import contextvars
import difflib
import io
import os
from pathlib import Path
import re
import shlex
from typing import Any, Dict, List, Optional

MAX_OUTPUT_BYTES = 16 * 1024  # 16 KB output truncation budget

# Marker appended to every shell command so the sandbox can learn the directory the
# command finished in. This makes `cd` persist across terminal_exec calls (like a real
# shell session) instead of silently resetting, which previously caused the agent to
# believe it was in one directory while file tools edited another.
_CWD_MARKER = "__VSAGENT_CWD__"


def _truncate_output(text: str, max_bytes: int = MAX_OUTPUT_BYTES) -> str:
    """Truncates oversized output preserving beginning and end."""
    text_bytes = text.encode("utf-8")
    if len(text_bytes) <= max_bytes:
        return text

    head_bytes = text_bytes[: max_bytes // 2]
    tail_bytes = text_bytes[-max_bytes // 2 :]

    head_str = head_bytes.decode("utf-8", errors="ignore")
    tail_str = tail_bytes.decode("utf-8", errors="ignore")
    omitted = len(text_bytes) - (len(head_bytes) + len(tail_bytes))

    return (
        f"{head_str}\n\n"
        f"--- [OUTPUT TRUNCATED: {omitted} bytes omitted to protect context window] ---\n\n"
        f"{tail_str}"
    )


class BaseSandbox(ABC):
    """Abstract interface for executing commands and managing files in an isolated environment."""

    @abstractmethod
    async def start(self) -> None:
        """Provisions the sandbox container or execution environment."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Tears down and completely destroys the ephemeral sandbox environment."""
        pass

    @abstractmethod
    async def exec_command(
        self,
        command: str,
        timeout_sec: int = 120,
        workdir: Optional[str] = None,
    ) -> str:
        """Executes a shell command inside the sandbox."""
        pass

    @abstractmethod
    async def exec_python(
        self,
        code: str,
        timeout_sec: int = 60,
    ) -> str:
        """Executes a Python script inside the sandbox."""
        pass

    @abstractmethod
    async def read_file(
        self,
        path: str,
        offset: int = 1,
        limit: int = 2000,
    ) -> str:
        """Reads a file from the sandbox filesystem with line numbering and budget pagination."""
        pass

    @abstractmethod
    async def write_file(
        self,
        path: str,
        content: str,
    ) -> str:
        """Writes or overwrites a file in the sandbox filesystem."""
        pass

    @abstractmethod
    async def patch_file(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Performs precise find-and-replace text replacement in a file."""
        pass

    def resolve_path(self, path: Optional[str]) -> Path:
        """Resolves a (possibly relative) path against the sandbox's current working directory."""
        cwd = getattr(self, "cwd", None)
        p = Path(path or ".").expanduser()
        if not p.is_absolute() and cwd is not None:
            p = Path(cwd) / p
        return p.resolve()


class HostSandbox(BaseSandbox):
    """Local host execution environment for Valstorm Desktop and local developer mode."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir).expanduser().resolve() if base_dir else Path.cwd().resolve()
        # Session working directory: starts at base_dir and follows `cd` in terminal_exec.
        self.cwd = self.base_dir
        self._is_active = True

    async def start(self) -> None:
        """No-op for local host process."""
        self._is_active = True

    async def close(self) -> None:
        """No-op for local host process."""
        self._is_active = False

    async def exec_command(
        self,
        command: str,
        timeout_sec: int = 120,
        workdir: Optional[str] = None,
    ) -> str:
        cwd = self.resolve_path(workdir) if workdir else Path(self.cwd)
        if not cwd.is_dir():
            return f"Error: Working directory does not exist: {cwd}"

        # Append a trailer that reports the final working directory while preserving the exit code.
        wrapped = (
            f"{command}\n"
            f"__vs_rc=$?; printf '\\n{_CWD_MARKER}%s\\n' \"$(pwd)\"; exit $__vs_rc"
        )

        try:
            proc = await asyncio.create_subprocess_shell(
                wrapped,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=float(timeout_sec),
                )
            except asyncio.TimeoutError:
                try:
                    proc.terminate()
                    await asyncio.sleep(0.5)
                    if proc.returncode is None:
                        proc.kill()
                except Exception:
                    pass
                return f"Error: Command timed out after {timeout_sec} seconds: {command}"

            stdout_str = stdout_bytes.decode("utf-8", errors="replace")
            stderr_str = stderr_bytes.decode("utf-8", errors="replace")
            exit_code = proc.returncode or 0

            # Extract and strip the cwd trailer; persist `cd` into the session.
            cwd_note = ""
            marker_idx = stdout_str.rfind(_CWD_MARKER)
            if marker_idx != -1:
                final_cwd_str = stdout_str[marker_idx + len(_CWD_MARKER):].strip().splitlines()[0:1]
                stdout_str = stdout_str[:marker_idx].rstrip("\n")
                if final_cwd_str:
                    final_cwd = Path(final_cwd_str[0])
                    if final_cwd.is_dir() and final_cwd.resolve() != cwd.resolve():
                        self.cwd = final_cwd.resolve()
                        cwd_note = f"\n[Working directory is now {self.cwd} — later terminal_exec and file tool calls with relative paths resolve from here]"

            output_parts = []
            if stdout_str:
                output_parts.append(stdout_str)
            if stderr_str:
                output_parts.append(f"--- STDERR ---\n{stderr_str}")

            combined_output = "\n".join(output_parts).strip()
            truncated = _truncate_output(combined_output)

            if exit_code == 0:
                body = truncated if truncated else "(Command finished with exit code 0 and no output)"
                return body + cwd_note
            else:
                return f"Command exited with code {exit_code}\n\n{truncated}{cwd_note}"
        except Exception as e:
            return f"Error executing command: {type(e).__name__}: {e}"

    async def exec_python(
        self,
        code: str,
        timeout_sec: int = 60,
    ) -> str:
        # Import lazily to avoid circular dependencies
        from tools.execute_code import execute_code
        return await execute_code(code=code, timeout_sec=timeout_sec)

    async def read_file(
        self,
        path: str,
        offset: int = 1,
        limit: int = 2000,
    ) -> str:
        target = self.resolve_path(path)
        if not target.is_file():
            return f"Error: File not found: {target}"

        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error reading file {target}: {e}"

        lines = content.splitlines()
        total_lines = len(lines)

        start_idx = max(1, offset) - 1
        end_idx = min(total_lines, start_idx + limit)

        selected_lines = lines[start_idx:end_idx]
        numbered_lines = [
            f"{idx + 1:5d}| {line}" for idx, line in enumerate(selected_lines, start=start_idx)
        ]

        header = f"[{target} (Lines {start_idx + 1}-{end_idx} of {total_lines})]"
        output = header + "\n" + "\n".join(numbered_lines)

        if end_idx < total_lines:
            output += f"\n\n--- [File truncated: Use offset={end_idx + 1} to continue reading] ---"

        return output

    async def write_file(
        self,
        path: str,
        content: str,
    ) -> str:
        file_path = self.resolve_path(path)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            temp_file = file_path.with_suffix(".tmp_write")
            temp_file.write_text(content, encoding="utf-8")
            temp_file.replace(file_path)

            lines = len(content.splitlines())
            bytes_count = len(content.encode("utf-8"))
            return f"Successfully wrote {bytes_count} bytes ({lines} lines) to {file_path}"
        except Exception as e:
            return f"Error writing to file {file_path}: {type(e).__name__}: {e}"

    async def patch_file(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        file_path = self.resolve_path(path)
        if not file_path.is_file():
            return f"Error: File not found: {file_path}"

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file {file_path}: {e}"

        # 1. Try Native Rust patcher first if available
        try:
            from core.native_bridge import native_patch_content
            native_res = native_patch_content(content, old_string, new_string, replace_all)
            if native_res is not None:
                new_content, diff_text = native_res
                temp_file = file_path.with_suffix(".tmp_patch")
                temp_file.write_text(new_content, encoding="utf-8")
                temp_file.replace(file_path)
                return f"Successfully patched {file_path}:\n\n```diff\n{diff_text}```"
        except Exception:
            pass

        # 2. Pure Python fallback
        if old_string not in content:
            return (
                f"Error: `old_string` was not found in {file_path}.\n"
                f"Please verify exact whitespace, line breaks, and indentation."
            )

        count = content.count(old_string)
        if count > 1 and not replace_all:
            return (
                f"Error: `old_string` matches {count} occurrences in {file_path}. "
                f"Provide more surrounding context to make it unique, or set `replace_all=True`."
            )

        if replace_all:
            new_content = content.replace(old_string, new_string)
        else:
            new_content = content.replace(old_string, new_string, 1)

        try:
            temp_file = file_path.with_suffix(".tmp_patch")
            temp_file.write_text(new_content, encoding="utf-8")
            temp_file.replace(file_path)
        except Exception as e:
            return f"Error writing patch to {file_path}: {e}"

        diff_lines = list(
            difflib.unified_diff(
                content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"a/{file_path.name}",
                tofile=f"b/{file_path.name}",
                n=3,
            )
        )
        diff_text = "".join(diff_lines)
        return f"Successfully patched {file_path} ({count if not replace_all else count} occurrence{'s' if count != 1 else ''} replaced):\n\n```diff\n{diff_text}```"


# ContextVar for async task / request sandbox propagation
_current_sandbox: contextvars.ContextVar[Optional[BaseSandbox]] = contextvars.ContextVar(
    "current_sandbox", default=None
)


_default_host_sandbox: Optional[HostSandbox] = None


def get_current_sandbox() -> BaseSandbox:
    """Gets the currently active execution sandbox in async context.

    Falls back to a process-wide HostSandbox singleton so session state (the working
    directory that follows `cd`) persists across tool calls instead of being recreated
    on every call.
    """
    global _default_host_sandbox
    sb = _current_sandbox.get()
    if sb is None:
        if _default_host_sandbox is None:
            _default_host_sandbox = HostSandbox()
        sb = _default_host_sandbox
    return sb


def resolve_agent_path(path: Optional[str]) -> Path:
    """Resolves a tool-supplied path against the active sandbox's working directory."""
    return get_current_sandbox().resolve_path(path)


def set_current_sandbox(sandbox: Optional[BaseSandbox]) -> contextvars.Token:
    """Sets the active execution sandbox for the current async task context."""
    return _current_sandbox.set(sandbox)
