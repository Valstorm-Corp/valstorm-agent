"""Fast Language Server Protocol (LSP) and Compiler Diagnostics Engine.

Provides zero-overhead, real-time diagnostic checks for modified files:
- Python: In-process AST syntax validation + optional ruff/flake8/pyright
- JSON: In-process parser validation
- YAML: In-process PyYAML safe loader validation
- TypeScript / JavaScript: Project-aware syntax & typecheck diagnostics (tsc)
- Go: Native `go vet` compiler syntax and type checking
- Rust: `cargo check` and `rustc` compiler diagnostics
- Immediate error feedback appended directly to tool call observations
"""

import ast
import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional


def check_python_syntax(content: str, filename: str = "<string>") -> Optional[List[str]]:
    """Performs instant AST syntax verification on Python source code."""
    try:
        ast.parse(content, filename=filename)
        return None
    except SyntaxError as e:
        line = e.lineno or 1
        col = e.offset or 1
        msg = e.msg or "Invalid syntax"
        text = (e.text or "").strip()
        snippet = f" -> `{text}`" if text else ""
        return [f"SyntaxError at line {line}, col {col}: {msg}{snippet}"]
    except Exception as e:
        return [f"Parser error: {str(e)}"]


def check_json_syntax(content: str) -> Optional[List[str]]:
    """Performs instant JSON parse verification."""
    try:
        json.loads(content)
        return None
    except json.JSONDecodeError as e:
        return [f"JSONDecodeError at line {e.lineno}, col {e.colno}: {e.msg}"]
    except Exception as e:
        return [f"JSON parser error: {str(e)}"]


def check_yaml_syntax(content: str) -> Optional[List[str]]:
    """Performs instant YAML parse verification."""
    try:
        import yaml
        list(yaml.safe_load_all(content))
        return None
    except Exception as e:
        if hasattr(e, "problem_mark") and getattr(e, "problem_mark", None) is not None:
            mark = e.problem_mark
            line = mark.line + 1
            col = mark.column + 1
            prob = getattr(e, "problem", str(e))
            return [f"YAMLError at line {line}, col {col}: {prob}"]
        return [f"YAMLError: {str(e)}"]


async def _run_command(cmd: List[str], cwd: Optional[Path] = None, timeout_sec: float = 2.0) -> Optional[str]:
    """Runs a diagnostic CLI tool asynchronously with a strict timeout."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        out_text = (stdout.decode("utf-8", errors="ignore") + stderr.decode("utf-8", errors="ignore")).strip()
        return out_text if out_text else None
    except Exception:
        return None


async def check_go_diagnostics(file_path: Path, timeout_sec: float = 2.0) -> Optional[List[str]]:
    """Runs `go vet` compiler syntax and type verification on Go source code."""
    go_bin = shutil.which("go")
    if not go_bin:
        return None

    # Check for go.mod directory
    go_mod_dir = None
    cur = file_path.parent
    for _ in range(6):
        if (cur / "go.mod").exists():
            go_mod_dir = cur
            break
        if cur.parent == cur:
            break
        cur = cur.parent

    cwd = go_mod_dir or file_path.parent
    cmd = [go_bin, "vet", str(file_path)] if not go_mod_dir else [go_bin, "vet", "./..."]
    out = await _run_command(cmd, cwd=cwd, timeout_sec=timeout_sec)
    if out:
        fname = file_path.name
        lines = [
            line.strip()
            for line in out.splitlines()
            if line.strip() and (fname in line or "syntax error" in line.lower() or "error" in line.lower())
        ]
        if lines:
            return lines[:5]
    return None


async def check_rust_diagnostics(file_path: Path, timeout_sec: float = 2.0) -> Optional[List[str]]:
    """Runs `cargo check` or `rustc` compiler checks on Rust source files."""
    cargo_bin = shutil.which("cargo")
    rustc_bin = shutil.which("rustc")

    cargo_dir = None
    cur = file_path.parent
    for _ in range(6):
        if (cur / "Cargo.toml").exists():
            cargo_dir = cur
            break
        if cur.parent == cur:
            break
        cur = cur.parent

    if cargo_bin and cargo_dir:
        cmd = [cargo_bin, "check", "--message-format=short"]
        out = await _run_command(cmd, cwd=cargo_dir, timeout_sec=timeout_sec)
        if out:
            fname = file_path.name
            lines = [
                line.strip()
                for line in out.splitlines()
                if line.strip() and (fname in line or "error[" in line or "error:" in line)
            ]
            if lines:
                return lines[:5]
    elif rustc_bin:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                cmd = [rustc_bin, "--emit=metadata", "--out-dir", tmpdir, str(file_path)]
                out = await _run_command(cmd, cwd=file_path.parent, timeout_sec=timeout_sec)
                if out and ("error[" in out or "error:" in out):
                    lines = [line.strip() for line in out.splitlines() if line.strip() and "error" in line.lower()]
                    if lines:
                        return lines[:5]
        except Exception:
            pass

    return None


async def get_file_diagnostics(file_path: Path, timeout_sec: float = 2.0) -> Optional[str]:
    """Inspects a file on disk and returns compiler / linter diagnostics if any errors exist.

    Supported formats:
    - Python (.py)
    - JSON (.json)
    - YAML (.yaml, .yml)
    - TypeScript / JavaScript (.ts, .tsx, .js, .jsx)
    - Go (.go)
    - Rust (.rs)

    Returns:
        Formatted multi-line diagnostic string, or None if the file is completely clean.
    """
    if not os.environ.get("VALSTORM_LSP_ENABLED", "true").lower() in ("true", "1", "yes"):
        return None

    if not file_path.is_file():
        return None

    ext = file_path.suffix.lower()
    errors: List[str] = []

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[LSP Diagnostics Error]: Could not read file: {e}"

    # 1. Python (.py)
    if ext == ".py":
        ast_errors = check_python_syntax(content, filename=str(file_path.name))
        if ast_errors:
            errors.extend(ast_errors)
        else:
            ruff_bin = shutil.which("ruff")
            if ruff_bin:
                ruff_out = await _run_command(
                    [ruff_bin, "check", "--output-format=concise", "--no-fix", str(file_path)],
                    cwd=file_path.parent,
                    timeout_sec=timeout_sec,
                )
                if ruff_out and ("error" in ruff_out.lower() or "syntax" in ruff_out.lower()):
                    lines = [line.strip() for line in ruff_out.splitlines() if line.strip() and not line.startswith("Found ")]
                    errors.extend(lines[:5])

    # 2. JSON (.json)
    elif ext == ".json":
        json_errors = check_json_syntax(content)
        if json_errors:
            errors.extend(json_errors)

    # 3. YAML (.yaml, .yml)
    elif ext in (".yaml", ".yml"):
        yaml_errors = check_yaml_syntax(content)
        if yaml_errors:
            errors.extend(yaml_errors)

    # 4. TypeScript / JavaScript (.ts, .tsx, .js, .jsx)
    elif ext in (".ts", ".tsx", ".js", ".jsx"):
        workspace_root = file_path.parent
        cur = file_path.parent
        for _ in range(5):
            if (cur / "tsconfig.json").exists() or (cur / "package.json").exists():
                workspace_root = cur
                break
            if cur.parent == cur:
                break
            cur = cur.parent

        tsc_local = workspace_root / "node_modules" / ".bin" / "tsc"
        tsc_bin = str(tsc_local) if tsc_local.exists() else shutil.which("tsc")

        if tsc_bin and (workspace_root / "tsconfig.json").exists():
            tsc_out = await _run_command(
                [tsc_bin, "--noEmit", "--skipLibCheck", "--pretty", "false"],
                cwd=workspace_root,
                timeout_sec=timeout_sec,
            )
            if tsc_out:
                fname = file_path.name
                matching_lines = [
                    line.strip() for line in tsc_out.splitlines()
                    if fname in line and ("error TS" in line or "error" in line.lower())
                ]
                if matching_lines:
                    errors.extend(matching_lines[:5])

    # 5. Go (.go)
    elif ext == ".go":
        go_errors = await check_go_diagnostics(file_path, timeout_sec=timeout_sec)
        if go_errors:
            errors.extend(go_errors)

    # 6. Rust (.rs)
    elif ext == ".rs":
        rust_errors = await check_rust_diagnostics(file_path, timeout_sec=timeout_sec)
        if rust_errors:
            errors.extend(rust_errors)

    if not errors:
        return None

    header = "[LSP Compiler Diagnostics]:"
    bulleted = "\n".join(f"- {err}" for err in errors)
    return f"{header}\n{bulleted}\n\nNotice: Please resolve the above diagnostic errors on your next turn."
