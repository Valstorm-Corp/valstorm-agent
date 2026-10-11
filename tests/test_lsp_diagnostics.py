"""Tests for the LSP and Compiler Diagnostics subsystem."""

import os
from pathlib import Path
import pytest

from core.lsp.diagnostics import (
    check_go_diagnostics,
    check_json_syntax,
    check_python_syntax,
    check_rust_diagnostics,
    check_yaml_syntax,
    get_file_diagnostics,
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
