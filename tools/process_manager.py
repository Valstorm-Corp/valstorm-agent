"""Persistent Background Process Manager Tool for Valstorm Agent Runtime.

Provides:
- process_manage: Manages long-running background processes (servers, watchers, builds)
  with list, poll, log, wait, kill, and submit/start actions.
"""

import asyncio
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import time
from typing import Any, Dict, List, Literal, Optional
import uuid

from core.tools import ToolRegistry, tool
from tools.developer_tools import _truncate_output


class ProcessSession:
    """Represents a tracked background process session."""

    def __init__(
        self,
        session_id: str,
        command: str,
        process: asyncio.subprocess.Process,
        workdir: Optional[str] = None,
    ):
        self.session_id = session_id
        self.command = command
        self.process = process
        self.pid = process.pid
        self.workdir = workdir or str(Path.cwd())
        self.status = "running"  # running, completed, failed, killed
        self.exit_code: Optional[int] = None
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.stdout_lines: List[str] = []
        self.stderr_lines: List[str] = []
        self.last_poll_stdout_idx = 0
        self.last_poll_stderr_idx = 0
        self._reader_tasks: List[asyncio.Task] = []

    def start_readers(self) -> None:
        """Starts asynchronous line readers for process stdout and stderr."""
        if self.process.stdout:
            self._reader_tasks.append(
                asyncio.create_task(self._read_stream(self.process.stdout, self.stdout_lines))
            )
        if self.process.stderr:
            self._reader_tasks.append(
                asyncio.create_task(self._read_stream(self.process.stderr, self.stderr_lines))
            )
        # Background task to monitor completion
        asyncio.create_task(self._monitor_exit())

    async def _read_stream(self, stream: asyncio.StreamReader, target_list: List[str]) -> None:
        try:
            while not stream.at_eof():
                line = await stream.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip("\r\n")
                target_list.append(decoded)
        except Exception:
            pass

    async def _monitor_exit(self) -> None:
        try:
            return_code = await self.process.wait()
            self.exit_code = return_code
            self.end_time = time.time()
            if self.status == "running":
                self.status = "completed" if return_code == 0 else "failed"
        except Exception:
            pass

    def check_status(self) -> str:
        """Polls current status and updates exit_code if finished."""
        if self.process.returncode is not None:
            self.exit_code = self.process.returncode
            if self.end_time is None:
                self.end_time = time.time()
            if self.status == "running":
                self.status = "completed" if self.exit_code == 0 else "failed"
        return self.status

    @property
    def duration_sec(self) -> float:
        end = self.end_time if self.end_time else time.time()
        return round(end - self.start_time, 2)


class ProcessRegistry:
    """In-memory registry managing all background process sessions."""

    _instance: Optional["ProcessRegistry"] = None

    def __init__(self) -> None:
        self.sessions: Dict[str, ProcessSession] = {}

    @classmethod
    def get_instance(cls) -> "ProcessRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, session: ProcessSession) -> None:
        self.sessions[session.session_id] = session

    def get(self, session_id: str) -> Optional[ProcessSession]:
        return self.sessions.get(session_id)

    def list_all(self) -> List[ProcessSession]:
        return list(self.sessions.values())


@tool
async def process_manage(
    action: Literal["list", "poll", "log", "wait", "kill", "submit"],
    session_id: Optional[str] = None,
    data: Optional[str] = None,
    command: Optional[str] = None,
    timeout: Optional[int] = None,
    limit: Optional[int] = 100,
    offset: Optional[int] = 1,
) -> str:
    """Manages long-running background processes (development servers, test watchers, background jobs).

    Actions:
      - 'submit': Starts a new background process if 'command' is provided, or sends input data to stdin if 'session_id' and 'data' are provided.
      - 'list': Lists all tracked background process sessions with their status, PID, and duration.
      - 'poll': Checks the current status and returns incremental new output since last poll for a session_id.
      - 'log': Returns line-numbered logs for a session_id with offset and limit pagination.
      - 'wait': Blocks until the process terminates (up to timeout seconds) and returns final exit code and output.
      - 'kill': Gracefully terminates or forcefully kills the running process.

    Args:
        action: The management action ('list', 'poll', 'log', 'wait', 'kill', 'submit').
        session_id: The unique identifier of the process session (required for poll, log, wait, kill).
        data: Text string to send to process stdin (used with 'submit').
        command: Shell command string to start a new background process (used with 'submit').
        timeout: Maximum seconds to wait (used with 'wait', default: 60).
        limit: Maximum log lines to return (used with 'log', default: 100).
        offset: Line number offset starting from 1 (used with 'log', default: 1).

    Returns:
        Formatted process status, output logs, or confirmation message.
    """
    registry = ProcessRegistry.get_instance()
    norm_action = (action or "").strip().lower()

    valid_actions = {"list", "poll", "log", "wait", "kill", "submit", "start"}
    if norm_action not in valid_actions:
        return f"Error: Unknown action '{norm_action}'. Valid actions are: 'list', 'poll', 'log', 'wait', 'kill', 'submit'."

    # 1. Action: LIST
    if norm_action == "list":
        all_procs = registry.list_all()
        if not all_procs:
            return "No active or recorded background processes."

        lines = [f"Active & Tracked Background Processes ({len(all_procs)}):", ""]
        lines.append(f"{'SESSION ID':<20} | {'PID':<8} | {'STATUS':<10} | {'DURATION':<10} | {'COMMAND'}")
        lines.append("-" * 75)
        for p in all_procs:
            status = p.check_status()
            lines.append(
                f"{p.session_id:<20} | {str(p.pid):<8} | {status.upper():<10} | {p.duration_sec}s{'':<5} | {p.command[:35]}"
            )
        return "\n".join(lines)

    # 2. Action: SUBMIT / START
    if norm_action == "submit" or (norm_action == "start" if hasattr(norm_action, "__str__") else False):
        if command:
            new_session_id = f"proc_{uuid.uuid4().hex[:12]}"
            try:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                session = ProcessSession(session_id=new_session_id, command=command, process=proc)
                session.start_readers()
                registry.register(session)

                return (
                    f"🚀 Started background process:\n"
                    f"Session ID: {new_session_id}\n"
                    f"PID: {proc.pid}\n"
                    f"Command: {command}\n"
                    f"Status: RUNNING\n\n"
                    f"Use process_manage(action='poll', session_id='{new_session_id}') to check status and output."
                )
            except Exception as e:
                return f"Error starting background process '{command}': {e}"

        if session_id:
            session = registry.get(session_id)
            if not session:
                return f"Error: Process session '{session_id}' not found."
            if session.check_status() != "running":
                return f"Error: Process '{session_id}' is not running (status: {session.status})."
            if data is None:
                return f"Error: No data provided to send to process '{session_id}'."

            try:
                if session.process.stdin:
                    input_bytes = (data if data.endswith("\n") else data + "\n").encode("utf-8")
                    session.process.stdin.write(input_bytes)
                    await session.process.stdin.drain()
                    return f"Successfully sent {len(input_bytes)} bytes to stdin of session '{session_id}'."
                return f"Error: Stdin is not available for session '{session_id}'."
            except Exception as e:
                return f"Error sending data to session '{session_id}': {e}"

        return "Error: For action 'submit', either provide 'command' to start a process, or 'session_id' and 'data' to write to stdin."

    # For actions requiring session_id
    if not session_id:
        return f"Error: 'session_id' is required for action '{norm_action}'."

    session = registry.get(session_id)
    if not session:
        return f"Error: Process session '{session_id}' not found."

    status = session.check_status()

    # 3. Action: POLL
    if norm_action == "poll":
        new_stdout = session.stdout_lines[session.last_poll_stdout_idx :]
        new_stderr = session.stderr_lines[session.last_poll_stderr_idx :]
        session.last_poll_stdout_idx = len(session.stdout_lines)
        session.last_poll_stderr_idx = len(session.stderr_lines)

        output_parts = [
            f"Process Status: {status.upper()} (PID: {session.pid}, Duration: {session.duration_sec}s)"
        ]
        if session.exit_code is not None:
            output_parts.append(f"Exit Code: {session.exit_code}")

        if new_stdout or new_stderr:
            output_parts.append("\n--- Incremental Output ---")
            if new_stdout:
                output_parts.append("\n".join(new_stdout))
            if new_stderr:
                output_parts.append("--- STDERR ---")
                output_parts.append("\n".join(new_stderr))
        else:
            output_parts.append("\n(No new output since last poll)")

        return _truncate_output("\n".join(output_parts))

    # 4. Action: LOG
    if norm_action == "log":
        all_lines = []
        for line in session.stdout_lines:
            all_lines.append(f"[stdout] {line}")
        for line in session.stderr_lines:
            all_lines.append(f"[stderr] {line}")

        total_lines = len(all_lines)
        start_idx = max(0, (offset or 1) - 1)
        max_lines = limit or 100
        slice_lines = all_lines[start_idx : start_idx + max_lines]

        if not slice_lines:
            return (
                f"Session '{session_id}' ({status.upper()}): No logs available at offset {offset} "
                f"(total lines: {total_lines})."
            )

        formatted = [
            f"Logs for Session '{session_id}' (Showing lines {start_idx + 1}-{start_idx + len(slice_lines)} of {total_lines}):",
            "",
        ]
        for idx, line in enumerate(slice_lines, start=start_idx + 1):
            formatted.append(f"{idx:4d} | {line}")

        return _truncate_output("\n".join(formatted))

    # 5. Action: WAIT
    if norm_action == "wait":
        wait_timeout = timeout or 60
        try:
            if session.check_status() == "running":
                await asyncio.wait_for(session.process.wait(), timeout=float(wait_timeout))
            
            session.check_status()
            all_out = "\n".join(session.stdout_lines)
            all_err = "\n".join(session.stderr_lines)
            output_body = []
            if all_out:
                output_body.append(all_out)
            if all_err:
                output_body.append(f"--- STDERR ---\n{all_err}")

            return (
                f"Process session '{session_id}' finished with exit code {session.exit_code} "
                f"(Duration: {session.duration_sec}s):\n\n"
                f"{_truncate_output(chr(10).join(output_body) if output_body else '(No output)')}"
            )
        except asyncio.TimeoutError:
            return (
                f"Session '{session_id}' is still running after {wait_timeout}s timeout. "
                f"Recent output lines: {len(session.stdout_lines)}."
            )

    # 6. Action: KILL
    if norm_action == "kill":
        if session.check_status() != "running":
            return f"Process '{session_id}' is already {session.status} (exit code: {session.exit_code})."

        try:
            session.process.terminate()
            await asyncio.sleep(0.3)
            if session.process.returncode is None:
                session.process.kill()
                await asyncio.sleep(0.1)

            session.status = "killed"
            session.end_time = time.time()
            return f"🛑 Process '{session_id}' (PID {session.pid}) has been successfully terminated."
        except Exception as e:
            return f"Error killing process '{session_id}': {e}"

    return f"Error: Unknown action '{norm_action}'. Valid actions are: 'list', 'poll', 'log', 'wait', 'kill', 'submit'."


def create_process_manager_tools() -> List[Any]:
    """Returns list of process manager tool functions."""
    return [process_manage]


def register_process_manager_tools(registry: ToolRegistry) -> ToolRegistry:
    """Registers process manager tools into the provided ToolRegistry."""
    for tool_fn in create_process_manager_tools():
        registry.register(tool_fn)
    return registry
