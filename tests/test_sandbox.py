"""Unit and Mock Integration Tests for Valstorm Sandbox Execution Subsystem.

Verifies:
1. HostSandbox: Subprocess command execution, timeouts, and local filesystem operations.
2. DockerSandbox: Mocked container lifecycle, resource caps (512MB RAM, 1 vCPU, pids-limit, network="none"),
   tar streaming file I/O, and clean container teardown.
3. ContextVar Sandbox Routing: Toolbelt calls (terminal_exec, write_file, read_file, patch_file)
   dynamically route to the active scoped sandbox.
"""

import asyncio
import io
from pathlib import Path
import tarfile
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from core.sandbox import (
    BaseSandbox,
    HostSandbox,
    get_current_sandbox,
    set_current_sandbox,
)
from core.docker_sandbox import DockerSandbox
from tools.developer_tools import patch_file, read_file, terminal_exec, write_file


# =====================================================================
# 1. HostSandbox Unit Tests
# =====================================================================


@pytest.mark.asyncio
async def test_host_sandbox_command_execution(tmp_path):
    sandbox = HostSandbox(base_dir=str(tmp_path))
    await sandbox.start()

    # Success command
    res = await sandbox.exec_command("echo 'Hello Sandbox'", workdir=str(tmp_path))
    assert "Hello Sandbox" in res

    # Non-zero exit code
    err_res = await sandbox.exec_command("exit 42", workdir=str(tmp_path))
    assert "Command exited with code 42" in err_res

    # Timeout
    timeout_res = await sandbox.exec_command("sleep 2", timeout_sec=1, workdir=str(tmp_path))
    assert "timed out after 1 seconds" in timeout_res

    await sandbox.close()


@pytest.mark.asyncio
async def test_host_sandbox_filesystem_operations(tmp_path):
    sandbox = HostSandbox(base_dir=str(tmp_path))
    target_file = tmp_path / "test_file.txt"

    # Write file
    write_res = await sandbox.write_file(str(target_file), "Line 1: Alpha\nLine 2: Beta\nLine 3: Gamma\n")
    assert "Successfully wrote" in write_res
    assert target_file.is_file()

    # Read file with pagination
    read_res = await sandbox.read_file(str(target_file), offset=1, limit=2)
    assert "1| Line 1: Alpha" in read_res
    assert "2| Line 2: Beta" in read_res
    assert "truncated" in read_res

    # Patch file
    patch_res = await sandbox.patch_file(str(target_file), old_string="Beta", new_string="Valstorm")
    assert "Successfully patched" in patch_res
    assert "Line 2: Valstorm" in target_file.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_host_sandbox_exec_python():
    sandbox = HostSandbox()
    python_code = "print(sum([10, 20, 30]))"
    res = await sandbox.exec_python(python_code)
    assert "60" in res


# =====================================================================
# 2. DockerSandbox Mocked Unit Tests
# =====================================================================


@pytest.mark.asyncio
async def test_docker_sandbox_lifecycle_and_resource_caps():
    mock_client = MagicMock()
    mock_container = MagicMock()
    mock_container.id = "mock_container_123456789abc"
    mock_client.containers.run.return_value = mock_container

    sandbox = DockerSandbox(
        image="valstorm/agent-sandbox:latest",
        mem_limit="512m",
        cpus=1.0,
        pids_limit=100,
        network_mode="none",
        client=mock_client,
    )

    # Start container
    await sandbox.start()
    assert sandbox._is_active is True
    assert sandbox.container_id == "mock_contain"

    # Verify security parameters passed to Docker
    mock_client.containers.run.assert_called_once()
    _, kwargs = mock_client.containers.run.call_args
    assert kwargs["mem_limit"] == "512m"
    assert kwargs["nano_cpus"] == 1_000_000_000
    assert kwargs["pids_limit"] == 100
    assert kwargs["network_mode"] == "none"
    assert kwargs["working_dir"] == "/workspace"
    assert kwargs["tmpfs"] == {"/tmp": "rw,noexec,nosuid,size=64m"}

    # Close container
    await sandbox.close()
    assert sandbox._is_active is False
    mock_container.remove.assert_called_once_with(force=True)


@pytest.mark.asyncio
async def test_docker_sandbox_exec_command():
    mock_client = MagicMock()
    mock_container = MagicMock()
    mock_container.exec_run.return_value = MagicMock(
        exit_code=0,
        output=(b"Docker execution success\n", b""),
    )
    mock_client.containers.run.return_value = mock_container

    sandbox = DockerSandbox(client=mock_client)
    await sandbox.start()

    res = await sandbox.exec_command("python3 --version")
    assert "Docker execution success" in res

    # Verify exec_run invocation
    mock_container.exec_run.assert_called_once_with(
        cmd=["/bin/sh", "-c", "python3 --version"],
        workdir="/workspace",
        demux=True,
    )

    await sandbox.close()


@pytest.mark.asyncio
async def test_docker_sandbox_tar_streaming_file_io():
    mock_client = MagicMock()
    mock_container = MagicMock()
    mock_client.containers.run.return_value = mock_container

    sandbox = DockerSandbox(client=mock_client)
    sandbox.container = mock_container
    sandbox._is_active = True

    # 1. Test write_file creates tar archive and calls put_archive
    write_res = await sandbox.write_file("data/test.csv", "id,name\n1,Valstorm\n")
    assert "Successfully wrote" in write_res
    mock_container.put_archive.assert_called_once()
    call_args = mock_container.put_archive.call_args
    assert call_args[0][0] == "/workspace/data"

    # 2. Test read_file parses tar stream from get_archive
    tar_buf = io.BytesIO()
    sample_content = b"header1,header2\nval1,val2\nval3,val4\n"
    with tarfile.open(fileobj=tar_buf, mode="w") as tar:
        tarinfo = tarfile.TarInfo(name="test.csv")
        tarinfo.size = len(sample_content)
        tar.addfile(tarinfo, io.BytesIO(sample_content))

    tar_buf.seek(0)
    mock_container.get_archive.return_value = ([tar_buf.getvalue()], {})

    read_res = await sandbox.read_file("data/test.csv", offset=1, limit=2)
    assert "1| header1,header2" in read_res
    assert "2| val1,val2" in read_res
    assert "test.csv" in read_res


# =====================================================================
# 3. ContextVar Dynamic Sandbox Routing Tests
# =====================================================================


@pytest.mark.asyncio
async def test_toolbelt_routes_to_active_context_sandbox():
    mock_sandbox = MagicMock(spec=BaseSandbox)
    mock_sandbox.exec_command = AsyncMock(return_value="Output from Mock Sandbox")
    mock_sandbox.write_file = AsyncMock(return_value="Successfully wrote to Mock Sandbox")
    mock_sandbox.read_file = AsyncMock(return_value="[mock.txt (Lines 1-1 of 1)]\n    1| Mock Data")

    token = set_current_sandbox(mock_sandbox)
    try:
        # terminal_exec routes to mock_sandbox
        cmd_out = await terminal_exec("echo 'test'")
        assert cmd_out == "Output from Mock Sandbox"
        mock_sandbox.exec_command.assert_called_once()

        # write_file routes to mock_sandbox
        w_out = await write_file("mock.txt", "data")
        assert w_out == "Successfully wrote to Mock Sandbox"
        mock_sandbox.write_file.assert_called_once_with(path="mock.txt", content="data")

        # read_file routes to mock_sandbox
        r_out = await read_file("mock.txt")
        assert "Mock Data" in r_out
        mock_sandbox.read_file.assert_called_once_with(path="mock.txt", offset=1, limit=2000)

    finally:
        set_current_sandbox(None)

    # After reset, get_current_sandbox returns HostSandbox
    default_sb = get_current_sandbox()
    assert isinstance(default_sb, HostSandbox)
