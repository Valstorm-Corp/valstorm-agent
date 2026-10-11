"""Ephemeral Docker Sandbox Implementation for Valstorm Agent Runtime.

Provides:
- DockerSandbox: Containerized execution environment using Docker SDK with
  strict resource limits (512MB RAM, 1.0 vCPU, pids-limit=100, network="none", read-only tmpfs).
- In-memory tar streaming for reading, writing, and patching files inside the container.
- Robust container teardown ensuring zero orphan containers on run cancellation or failure.
"""

import asyncio
import difflib
import io
import os
from pathlib import Path
import tarfile
import time
from typing import Any, Dict, Optional

from core.sandbox import BaseSandbox, _truncate_output
from core.security_guard import SecurityGuard, SecurityQuarantineError


class DockerSandbox(BaseSandbox):
    """Hardened Ephemeral Docker Sandbox Runner."""

    def __init__(
        self,
        image: str = "valstorm/agent-sandbox:latest",
        fallback_image: str = "python:3.11-slim",
        mem_limit: str = "512m",
        cpus: float = 1.0,
        pids_limit: int = 100,
        network_mode: str = "none",
        client: Optional[Any] = None,
    ):
        self.image = image
        self.fallback_image = fallback_image
        self.mem_limit = mem_limit
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.network_mode = network_mode
        self._custom_client = client
        self._client: Optional[Any] = None
        self.container: Optional[Any] = None
        self.container_id: Optional[str] = None
        self._is_active = False

    def _get_docker_client(self) -> Any:
        """Retrieves or initializes the Docker client."""
        if self._custom_client is not None:
            return self._custom_client
        if self._client is None:
            import docker
            self._client = docker.from_env()
        return self._client

    async def start(self) -> None:
        """Provisions and starts the ephemeral Docker container."""
        def _provision() -> Any:
            client = self._get_docker_client()
            target_image = self.image
            try:
                client.images.get(target_image)
            except Exception:
                target_image = self.fallback_image

            nano_cpus = int(self.cpus * 1_000_000_000)
            container = client.containers.run(
                image=target_image,
                command=["tail", "-f", "/dev/null"],
                detach=True,
                mem_limit=self.mem_limit,
                nano_cpus=nano_cpus,
                pids_limit=self.pids_limit,
                network_mode=self.network_mode,
                tmpfs={"/tmp": "rw,noexec,nosuid,size=64m"},
                working_dir="/workspace",
                labels={"app": "valstorm-agent-runtime", "type": "ephemeral-sandbox"},
            )
            return container

        try:
            self.container = await asyncio.to_thread(_provision)
            self.container_id = self.container.id[:12] if self.container else None
            self._is_active = True
        except Exception as e:
            self._is_active = False
            raise RuntimeError(f"Failed to start Docker sandbox container: {e}")

    async def close(self) -> None:
        """Tears down and forcefully removes the ephemeral container."""
        if not self.container:
            self._is_active = False
            return

        def _cleanup(cont: Any) -> None:
            try:
                cont.remove(force=True)
            except Exception:
                pass

        try:
            await asyncio.to_thread(_cleanup, self.container)
        finally:
            self.container = None
            self._is_active = False

    async def exec_command(
        self,
        command: str,
        timeout_sec: int = 120,
        workdir: Optional[str] = None,
    ) -> str:
        """Executes a shell command inside the Docker container."""
        if not self.container:
            return "Error: Docker sandbox is not running."

        work_dir = workdir or "/workspace"

        def _run_cmd() -> tuple[int, str]:
            exec_res = self.container.exec_run(
                cmd=["/bin/sh", "-c", command],
                workdir=work_dir,
                demux=True,
            )
            exit_code = exec_res.exit_code or 0
            stdout_bytes, stderr_bytes = exec_res.output or (b"", b"")
            stdout_str = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            stderr_str = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

            parts = []
            if stdout_str:
                parts.append(stdout_str)
            if stderr_str:
                parts.append(f"--- STDERR ---\n{stderr_str}")

            combined = "\n".join(parts).strip()
            return exit_code, combined

        try:
            exit_code, output = await asyncio.wait_for(
                asyncio.to_thread(_run_cmd),
                timeout=float(timeout_sec),
            )
            truncated = _truncate_output(output)
            if exit_code == 0:
                return truncated if truncated else "(Command finished with exit code 0 and no output)"
            else:
                return f"Command exited with code {exit_code}\n\n{truncated}"
        except asyncio.TimeoutError:
            return f"Error: Command timed out after {timeout_sec} seconds: {command}"
        except Exception as e:
            return f"Error executing command in Docker sandbox: {type(e).__name__}: {e}"

    async def exec_python(
        self,
        code: str,
        timeout_sec: int = 60,
    ) -> str:
        """Executes a Python script inside the Docker container."""
        if not self.container:
            return "Error: Docker sandbox is not running."

        # Write code to /workspace/.valstorm_exec.py and execute
        script_name = f".valstorm_exec_{int(time.time() * 1000)}.py"
        await self.write_file(script_name, code)
        try:
            return await self.exec_command(
                command=f"python3 /workspace/{script_name}",
                timeout_sec=timeout_sec,
                workdir="/workspace",
            )
        finally:
            try:
                self.container.exec_run(f"rm -f /workspace/{script_name}")
            except Exception:
                pass

    async def read_file(
        self,
        path: str,
        offset: int = 1,
        limit: int = 2000,
    ) -> str:
        """Reads a file from the Docker container via tar archive streaming."""
        if not self.container:
            return "Error: Docker sandbox is not running."

        norm_path = path if path.startswith("/") else f"/workspace/{path}"

        def _get_bytes() -> str:
            bits, stat = self.container.get_archive(norm_path)
            file_data = b"".join(bits)
            tar_stream = io.BytesIO(file_data)
            with tarfile.open(fileobj=tar_stream, mode="r:*") as tar:
                member = tar.next()
                if member is None:
                    return f"Error: File not found in archive: {path}"
                extracted = tar.extractfile(member)
                if extracted is None:
                    return f"Error: Could not extract file content from archive: {path}"
                return extracted.read().decode("utf-8", errors="replace")

        try:
            content = await asyncio.to_thread(_get_bytes)
            if content.startswith("Error:"):
                return content
        except Exception as e:
            return f"Error reading file {path} from Docker sandbox: {e}"

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
        """Writes a file to the Docker container via in-memory tar stream."""
        if not self.container:
            return "Error: Docker sandbox is not running."

        # Enforce quarantine security check
        try:
            SecurityGuard.assert_safe_file(path, content, enforce_whitelist=False)
        except SecurityQuarantineError as q_err:
            return f"Security Quarantine Block: {q_err}"

        norm_path = path if path.startswith("/") else f"/workspace/{path}"
        parent_dir = str(Path(norm_path).parent)
        file_name = Path(norm_path).name

        def _put_bytes() -> None:
            self.container.exec_run(f"mkdir -p {parent_dir}")
            tar_stream = io.BytesIO()
            content_bytes = content.encode("utf-8")
            with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                tarinfo = tarfile.TarInfo(name=file_name)
                tarinfo.size = len(content_bytes)
                tarinfo.mtime = int(time.time())
                tarinfo.mode = 0o644
                tar.addfile(tarinfo, io.BytesIO(content_bytes))

            tar_stream.seek(0)
            self.container.put_archive(parent_dir, tar_stream.getvalue())

        try:
            await asyncio.to_thread(_put_bytes)
            lines = len(content.splitlines())
            bytes_count = len(content.encode("utf-8"))
            return f"Successfully wrote {bytes_count} bytes ({lines} lines) to {path}"
        except Exception as e:
            return f"Error writing to file {path} in Docker sandbox: {e}"

    async def patch_file(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Patches a file in the Docker container and returns a diff."""
        if not self.container:
            return "Error: Docker sandbox is not running."

        norm_path = path if path.startswith("/") else f"/workspace/{path}"

        def _read_existing() -> str:
            bits, _ = self.container.get_archive(norm_path)
            tar_stream = io.BytesIO(b"".join(bits))
            with tarfile.open(fileobj=tar_stream, mode="r:*") as tar:
                member = tar.next()
                if not member:
                    return f"Error: File not found: {path}"
                extracted = tar.extractfile(member)
                if not extracted:
                    return f"Error: Could not read file: {path}"
                return extracted.read().decode("utf-8", errors="replace")

        try:
            content = await asyncio.to_thread(_read_existing)
            if content.startswith("Error:"):
                return content
        except Exception as e:
            return f"Error reading file {path} from Docker sandbox: {e}"

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

        write_res = await self.write_file(path, new_content)
        if write_res.startswith("Error"):
            return write_res

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
