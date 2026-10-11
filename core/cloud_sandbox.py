"""Serverless Cloud MicroVM Sandbox Adapter for Valstorm Agent Runtime.

Provides:
- CloudMicroVMSandbox (E2BSandbox): Production microVM execution adapter using E2B
  for multi-tenant cloud deployments where local Docker is unavailable or untrusted.
- MicroVM boot in <200ms with hard hypervisor isolation.
- Integrated file management, shell execution, and Python kernel interpretation.
"""

import asyncio
import difflib
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from core.keystore import KeyStore
from core.sandbox import BaseSandbox, _truncate_output
from core.security_guard import SecurityGuard, SecurityQuarantineError

logger = logging.getLogger("valstorm.cloud_sandbox")


class CloudMicroVMSandbox(BaseSandbox):
    """Production Serverless Cloud MicroVM Sandbox Adapter (E2B / Managed MicroVM)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        template: Optional[str] = None,
        timeout_sec: int = 300,
        client: Optional[Any] = None,
    ):
        self.api_key = api_key
        self.template = template
        self.timeout_sec = timeout_sec
        self._custom_client = client
        self._sandbox: Optional[Any] = None
        self.sandbox_id: Optional[str] = None
        self._is_active = False

    def _resolve_api_key(self) -> Optional[str]:
        """Resolves E2B API Key from parameter, keystore, or environment."""
        if self.api_key and self.api_key.strip():
            return self.api_key.strip()
        env_key = os.environ.get("E2B_API_KEY") or os.environ.get("VALSTORM_E2B_API_KEY")
        if env_key and env_key.strip():
            return env_key.strip()
        try:
            keystore = KeyStore()
            ks_key = keystore.get_api_key("e2b")
            if ks_key and ks_key.strip():
                return ks_key.strip()
        except Exception:
            pass
        return None

    async def start(self) -> None:
        """Provisions and boots the serverless microVM in <200ms."""
        if self._custom_client is not None:
            self._sandbox = self._custom_client
            self.sandbox_id = getattr(self._custom_client, "sandbox_id", "mock_vm_id")
            self._is_active = True
            return

        api_key = self._resolve_api_key()
        if not api_key:
            raise ValueError(
                "E2B API key not found. Please set E2B_API_KEY in environment or Valstorm KeyStore."
            )

        try:
            from e2b_code_interpreter import AsyncSandbox

            kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": self.timeout_sec}
            if self.template:
                kwargs["template"] = self.template

            self._sandbox = await AsyncSandbox.create(**kwargs)
            self.sandbox_id = getattr(self._sandbox, "sandbox_id", None)
            self._is_active = True
        except Exception as e:
            self._is_active = False
            raise RuntimeError(f"Failed to provision serverless microVM sandbox: {e}")

    async def close(self) -> None:
        """Tears down and terminates the remote microVM."""
        if not self._sandbox:
            self._is_active = False
            return

        try:
            if hasattr(self._sandbox, "kill"):
                res = self._sandbox.kill()
                if asyncio.iscoroutine(res):
                    await res
        except Exception as e:
            logger.warning(f"Error terminating cloud microVM {self.sandbox_id}: {e}")
        finally:
            self._sandbox = None
            self._is_active = False

    async def exec_command(
        self,
        command: str,
        timeout_sec: int = 120,
        workdir: Optional[str] = None,
    ) -> str:
        """Executes a shell command inside the cloud microVM."""
        if not self._sandbox:
            return "Error: Cloud microVM sandbox is not running."

        cwd = workdir or "/home/user"
        try:
            cmd_runner = self._sandbox.commands
            res = cmd_runner.run(command, timeout=timeout_sec, cwd=cwd)
            if asyncio.iscoroutine(res):
                res = await res

            exit_code = getattr(res, "exit_code", 0) or 0
            stdout_str = getattr(res, "stdout", "") or ""
            stderr_str = getattr(res, "stderr", "") or ""

            parts = []
            if stdout_str:
                parts.append(stdout_str)
            if stderr_str:
                parts.append(f"--- STDERR ---\n{stderr_str}")

            combined = "\n".join(parts).strip()
            truncated = _truncate_output(combined)

            if exit_code == 0:
                return truncated if truncated else "(Command finished with exit code 0 and no output)"
            else:
                return f"Command exited with code {exit_code}\n\n{truncated}"
        except asyncio.TimeoutError:
            return f"Error: Command timed out after {timeout_sec} seconds: {command}"
        except Exception as e:
            return f"Error executing command in cloud microVM: {type(e).__name__}: {e}"

    async def exec_python(
        self,
        code: str,
        timeout_sec: int = 60,
    ) -> str:
        """Executes Python code directly in the microVM Python kernel."""
        if not self._sandbox:
            return "Error: Cloud microVM sandbox is not running."

        try:
            if hasattr(self._sandbox, "run_code"):
                execution = self._sandbox.run_code(code, timeout=timeout_sec)
                if asyncio.iscoroutine(execution):
                    execution = await execution

                output_parts: List[str] = []

                # Collect stdout lines
                logs = getattr(execution, "logs", None)
                if logs:
                    stdout_logs = getattr(logs, "stdout", []) or []
                    stderr_logs = getattr(logs, "stderr", []) or []
                    if stdout_logs:
                        output_parts.append("\n".join(stdout_logs))
                    if stderr_logs:
                        output_parts.append(f"--- STDERR ---\n" + "\n".join(stderr_logs))

                # Collect rich results (e.g. text/plain from expressions)
                results = getattr(execution, "results", []) or []
                for r in results:
                    text = getattr(r, "text", None)
                    if text and text not in output_parts:
                        output_parts.append(text)

                # Collect execution errors
                error = getattr(execution, "error", None)
                if error:
                    err_name = getattr(error, "name", "ExecutionError")
                    err_val = getattr(error, "value", str(error))
                    traceback_str = getattr(error, "traceback", "")
                    output_parts.append(f"Error ({err_name}): {err_val}\n{traceback_str}".strip())

                combined = "\n".join(output_parts).strip()
                truncated = _truncate_output(combined)
                return truncated if truncated else "(Python script executed successfully with no output)"
            else:
                # Fallback to command execution
                return await self.exec_command(f'python3 -c "{code}"', timeout_sec=timeout_sec)
        except Exception as e:
            return f"Error executing Python in cloud microVM: {type(e).__name__}: {e}"

    async def read_file(
        self,
        path: str,
        offset: int = 1,
        limit: int = 2000,
    ) -> str:
        """Reads a file from the microVM filesystem."""
        if not self._sandbox:
            return "Error: Cloud microVM sandbox is not running."

        try:
            files = self._sandbox.files
            read_op = files.read(path)
            if asyncio.iscoroutine(read_op):
                read_op = await read_op
            content = read_op if isinstance(read_op, str) else read_op.decode("utf-8", errors="replace")
        except Exception as e:
            return f"Error reading file {path} from cloud microVM: {e}"

        lines = content.splitlines()
        total_lines = len(lines)
        file_name = Path(path).name

        start_idx = max(1, offset) - 1
        end_idx = min(total_lines, start_idx + limit)

        selected_lines = lines[start_idx:end_idx]
        numbered_lines = [
            f"{idx + 1:5d}| {line}" for idx, line in enumerate(selected_lines, start=start_idx)
        ]

        header = f"[{file_name} (Lines {start_idx + 1}-{end_idx} of {total_lines})]"
        output = header + "\n" + "\n".join(numbered_lines)

        if end_idx < total_lines:
            output += f"\n\n--- [File truncated: Use offset={end_idx + 1} to continue reading] ---"

        return output

    async def write_file(
        self,
        path: str,
        content: str,
    ) -> str:
        """Writes or overwrites a file in the microVM filesystem."""
        if not self._sandbox:
            return "Error: Cloud microVM sandbox is not running."

        # Enforce quarantine security check
        try:
            SecurityGuard.assert_safe_file(path, content, enforce_whitelist=False)
        except SecurityQuarantineError as q_err:
            return f"Security Quarantine Block: {q_err}"

        try:
            files = self._sandbox.files
            write_op = files.write(path, content)
            if asyncio.iscoroutine(write_op):
                await write_op
            lines = len(content.splitlines())
            bytes_count = len(content.encode("utf-8"))
            return f"Successfully wrote {bytes_count} bytes ({lines} lines) to {path}"
        except Exception as e:
            return f"Error writing to file {path} in cloud microVM: {e}"

    async def patch_file(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Patches text in a file inside the cloud microVM and returns a unified diff."""
        if not self._sandbox:
            return "Error: Cloud microVM sandbox is not running."

        try:
            files = self._sandbox.files
            read_op = files.read(path)
            if asyncio.iscoroutine(read_op):
                read_op = await read_op
            content = read_op if isinstance(read_op, str) else read_op.decode("utf-8", errors="replace")
        except Exception as e:
            return f"Error reading file {path} from cloud microVM: {e}"

        if old_string not in content:
            return (
                f"Error: `old_string` was not found in {path}.\n"
                f"Please verify exact whitespace, line breaks, and indentation."
            )

        count = content.count(old_string)
        if count > 1 and not replace_all:
            return (
                f"Error: `old_string` matches {count} occurrences in {path}. "
                f"Provide more surrounding context to make it unique, or set `replace_all=True`."
            )

        if replace_all:
            new_content = content.replace(old_string, new_string)
        else:
            new_content = content.replace(old_string, new_string, 1)

        try:
            write_op = files.write(path, new_content)
            if asyncio.iscoroutine(write_op):
                await write_op
        except Exception as e:
            return f"Error saving patch to {path} in cloud microVM: {e}"

        diff_lines = list(
            difflib.unified_diff(
                content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"a/{Path(path).name}",
                tofile=f"b/{Path(path).name}",
                n=3,
            )
        )
        diff_text = "".join(diff_lines)
        return f"Successfully patched {Path(path).name} ({count} occurrence{'s' if count != 1 else ''} replaced):\n\n```diff\n{diff_text}```"


# Alias for clarity
E2BSandbox = CloudMicroVMSandbox
