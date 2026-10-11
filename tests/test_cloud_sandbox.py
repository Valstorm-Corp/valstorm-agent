"""Unit tests for Serverless Cloud MicroVM Sandbox Adapter (E2B / Managed MicroVM)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from core.cloud_sandbox import CloudMicroVMSandbox, E2BSandbox


def _make_mock_e2b_sandbox():
    mock_vm = MagicMock()
    mock_vm.sandbox_id = "e2b_test_microvm_12345"
    mock_vm.kill = AsyncMock()

    # Mock commands runner
    mock_commands = MagicMock()
    mock_res = MagicMock()
    mock_res.exit_code = 0
    mock_res.stdout = "Hello from Cloud MicroVM\n"
    mock_res.stderr = ""
    mock_commands.run = AsyncMock(return_value=mock_res)
    mock_vm.commands = mock_commands

    # Mock code interpreter
    mock_exec = MagicMock()
    mock_exec.logs = MagicMock(stdout=["Result: 42"], stderr=[])
    mock_exec.results = [MagicMock(text="42")]
    mock_exec.error = None
    mock_vm.run_code = AsyncMock(return_value=mock_exec)

    # Mock files manager
    mock_files = MagicMock()
    mock_files.read = MagicMock(return_value="line 1: Alpha\nline 2: Beta\nline 3: Gamma\n")
    mock_files.write = MagicMock()
    mock_vm.files = mock_files

    return mock_vm


@pytest.mark.asyncio
async def test_cloud_sandbox_lifecycle():
    mock_vm = _make_mock_e2b_sandbox()
    sandbox = CloudMicroVMSandbox(client=mock_vm)

    await sandbox.start()
    assert sandbox._is_active is True
    assert sandbox.sandbox_id == "e2b_test_microvm_12345"

    await sandbox.close()
    assert sandbox._is_active is False
    mock_vm.kill.assert_awaited_once()


@pytest.mark.asyncio
async def test_cloud_sandbox_exec_command():
    mock_vm = _make_mock_e2b_sandbox()
    sandbox = CloudMicroVMSandbox(client=mock_vm)
    await sandbox.start()

    res = await sandbox.exec_command("echo 'test'")
    assert "Hello from Cloud MicroVM" in res
    mock_vm.commands.run.assert_awaited_once_with("echo 'test'", timeout=120, cwd="/home/user")

    await sandbox.close()


@pytest.mark.asyncio
async def test_cloud_sandbox_exec_python():
    mock_vm = _make_mock_e2b_sandbox()
    sandbox = CloudMicroVMSandbox(client=mock_vm)
    await sandbox.start()

    res = await sandbox.exec_python("print(42)")
    assert "42" in res
    mock_vm.run_code.assert_awaited_once_with("print(42)", timeout=60)

    await sandbox.close()


@pytest.mark.asyncio
async def test_cloud_sandbox_files_read_write_patch():
    mock_vm = _make_mock_e2b_sandbox()
    mock_vm.files.read = AsyncMock(return_value="line 1: Alpha\nline 2: Beta\nline 3: Gamma\n")
    mock_vm.files.write = AsyncMock()

    sandbox = CloudMicroVMSandbox(client=mock_vm)
    sandbox._sandbox = mock_vm
    sandbox._is_active = True

    # 1. Read file with pagination
    read_res = await sandbox.read_file("data.csv", offset=1, limit=2)
    assert "1| line 1: Alpha" in read_res
    assert "2| line 2: Beta" in read_res
    assert "truncated" in read_res

    # 2. Write file
    write_res = await sandbox.write_file("data.csv", "new content")
    assert "Successfully wrote" in write_res
    mock_vm.files.write.assert_called_with("data.csv", "new content")

    # 3. Patch file
    patch_res = await sandbox.patch_file("data.csv", old_string="line 2: Beta", new_string="line 2: Valstorm")
    assert "Successfully patched" in patch_res
    assert "```diff" in patch_res


@pytest.mark.asyncio
async def test_cloud_sandbox_missing_api_key():
    sandbox = CloudMicroVMSandbox(api_key=None)
    with patch.dict("os.environ", {}, clear=True), patch("core.cloud_sandbox.KeyStore") as mock_ks:
        mock_ks.return_value.get_api_key.return_value = None
        with pytest.raises(ValueError, match="E2B API key not found"):
            await sandbox.start()
