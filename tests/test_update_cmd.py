"""Unit tests for vsagent update command and external dependency resolution."""

from pathlib import Path
import re
import subprocess
from unittest.mock import MagicMock, patch
import pytest
import typer

from cli.cmds import status_cmd


AGENT_RUNTIME_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = AGENT_RUNTIME_ROOT / "pyproject.toml"


def test_pyproject_declares_git_urls_for_internal_packages():
    """Ensure valstorm-sanitizer and valstorm-client have public GitHub URLs for external users."""
    content = PYPROJECT_PATH.read_text()
    # When [tool.uv.sources] is stripped for external/public installs, dependencies must still resolve via GitHub
    stripped = re.sub(r"(?ms)\n\[tool\.uv\.sources\].*?(?=\n\[|\Z)", "", content)

    assert "valstorm-sanitizer @ git+https://github.com/Valstorm-Corp/valstorm-sanitizer.git" in stripped
    assert "valstorm-client @ git+https://github.com/Valstorm-Corp/valstorm-client-python.git" in stripped
    assert "allow-direct-references = true" in stripped


def test_update_command_uv_tool_upgrade_success(monkeypatch):
    """When installed as a uv tool, uv tool upgrade valstorm-agent succeeds."""
    monkeypatch.setattr(status_cmd, "__file__", "/Users/test/.local/share/uv/tools/valstorm-agent/lib/python3.12/site-packages/cli/cmds/status_cmd.py")
    monkeypatch.setattr(status_cmd.sys, "executable", "/Users/test/.local/share/uv/tools/valstorm-agent/bin/python")
    monkeypatch.setattr(status_cmd.shutil, "which", lambda cmd: "/usr/local/bin/uv" if cmd == "uv" else None)

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(status_cmd.subprocess, "run", fake_run)
    status_cmd.update_command()

    assert calls == [["uv", "tool", "upgrade", "valstorm-agent"]]


def test_update_command_uv_tool_upgrade_fallback_on_failure(monkeypatch):
    """When uv tool upgrade fails (e.g. missing sanitizer/client), falls back to uv tool install --from public git with --with deps."""
    monkeypatch.setattr(status_cmd, "__file__", "/Users/test/.local/share/uv/tools/valstorm-agent/lib/python3.12/site-packages/cli/cmds/status_cmd.py")
    monkeypatch.setattr(status_cmd.sys, "executable", "/Users/test/.local/share/uv/tools/valstorm-agent/bin/python")
    monkeypatch.setattr(status_cmd.shutil, "which", lambda cmd: "/usr/local/bin/uv" if cmd == "uv" else None)

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["uv", "tool", "upgrade", "valstorm-agent"]:
            raise subprocess.CalledProcessError(1, cmd, stderr=b"valstorm-sanitizer was not found")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(status_cmd.subprocess, "run", fake_run)
    status_cmd.update_command()

    assert len(calls) == 2
    assert calls[0] == ["uv", "tool", "upgrade", "valstorm-agent"]
    fallback_cmd = calls[1]
    assert fallback_cmd[:3] == ["uv", "tool", "install"]
    assert "git+https://github.com/Valstorm-Corp/valstorm-agent.git" in fallback_cmd
    assert "git+https://github.com/Valstorm-Corp/valstorm-sanitizer.git" in fallback_cmd
    assert "git+https://github.com/Valstorm-Corp/valstorm-client-python.git" in fallback_cmd
    assert "--force" in fallback_cmd
    assert "--upgrade" in fallback_cmd


def test_update_command_pip_fallback_uses_public_repos(monkeypatch):
    """When not a uv tool, pip upgrade uses public GitHub repos and never private monorepo.git."""
    monkeypatch.setattr(status_cmd, "__file__", "/usr/local/lib/python3.12/site-packages/cli/cmds/status_cmd.py")
    monkeypatch.setattr(status_cmd.sys, "executable", "/usr/local/bin/python3")
    monkeypatch.setattr(status_cmd.shutil, "which", lambda cmd: None)

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(status_cmd.subprocess, "run", fake_run)
    status_cmd.update_command()

    assert len(calls) == 1
    cmd_str = " ".join(calls[0])
    assert "monorepo.git" not in cmd_str
    assert "git+https://github.com/Valstorm-Corp/valstorm-sanitizer.git" in calls[0]
    assert "git+https://github.com/Valstorm-Corp/valstorm-client-python.git" in calls[0]
    assert "git+https://github.com/Valstorm-Corp/valstorm-agent.git" in calls[0]
