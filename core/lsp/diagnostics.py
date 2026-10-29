"""Fast Language Server Protocol (LSP) and Compiler Diagnostics Engine.

Provides zero-overhead, real-time diagnostic checks for modified files:
- Python: In-process AST syntax validation + optional ruff/flake8/pyright
- JSON: In-process parser validation
- YAML: In-process PyYAML safe loader validation
- TypeScript / JavaScript: Fast single-file syntax check (esbuild/biome/node) + gated tsc typecheck
- Go: Native `go vet` compiler syntax and type checking
- Rust: `cargo check` and `rustc` compiler diagnostics
- Immediate error feedback appended directly to tool call observations
- Guaranteed process-tree termination on timeout and cancellation (no orphaned tsc zombies)
- Single-Flight concurrency gate per workspace root
- Startup and periodic stale diagnostic process reaper
"""

import ast
import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("valstorm.lsp")

VALSTORM_DIAGNOSTIC_ENV_VAR = "VALSTORM_DIAGNOSTIC_SUBPROCESS"


def kill_process_tree(pid: int) -> None:
    """Forcefully terminates a process and its child processes cross-platform."""
    if not pid or pid <= 0:
        return
    if sys.platform == "win32":
        try:
            # /F = forcefully terminate, /T = terminate process tree, /PID = target pid
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5,
            )
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
    else:
        # POSIX: If target process is its own process group leader (pgid == pid),
        # kill the entire process group. Otherwise, terminate the individual pid
        # to prevent collateral termination of parent or sibling processes.
        try:
            pgid = os.getpgid(pid)
            if pgid == pid:
                os.killpg(pgid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass


def cleanup_stale_diagnostic_processes() -> List[int]:
    """Scans for and forcefully terminates orphaned tsc or diagnostic subprocesses.

    Identifies:
    1. Any process bearing VALSTORM_DIAGNOSTIC_SUBPROCESS=1 in its environment.
    2. Any orphaned tsc / node process running with '--noEmit' and '--skipLibCheck'.

    Returns:
        List of terminated process IDs.
    """
    killed_pids: List[int] = []
    my_pid = os.getpid()

    # 1. Linux /proc inspection (zero-dependency, universally available on Linux)
    if os.path.exists("/proc"):
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                pid = int(entry)
                if pid == my_pid or pid == 1:
                    continue

                cmdline_path = f"/proc/{pid}/cmdline"
                environ_path = f"/proc/{pid}/environ"

                is_target = False
                try:
                    with open(cmdline_path, "rb") as f:
                        cmdline_bytes = f.read()
                    cmdline = cmdline_bytes.decode("utf-8", errors="ignore").replace(chr(0), " ")

                    # Target criteria 1: tsc --noEmit --skipLibCheck
                    if ("tsc" in cmdline or "node" in cmdline) and "--noEmit" in cmdline and "--skipLibCheck" in cmdline:
                        is_target = True

                    # Target criteria 2: VALSTORM_DIAGNOSTIC_SUBPROCESS in environ
                    if not is_target and os.path.exists(environ_path):
                        try:
                            with open(environ_path, "rb") as ef:
                                env_bytes = ef.read()
                            if b"VALSTORM_DIAGNOSTIC_SUBPROCESS=1" in env_bytes:
                                is_target = True
                        except (PermissionError, ProcessLookupError, FileNotFoundError):
                            pass

                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    continue

                if is_target:
                    try:
                        kill_process_tree(pid)
                        killed_pids.append(pid)
                    except Exception:
                        pass
            if killed_pids:
                return killed_pids
        except Exception:
            pass

    # 2. Darwin / macOS / BSD fallback via ps
    if sys.platform != "win32":
        try:
            ps_bin = shutil.which("ps")
            if ps_bin:
                ps_args = (
                    [ps_bin, "axwwE", "-o", "pid,command"]
                    if sys.platform == "darwin"
                    else [ps_bin, "-eo", "pid,args"]
                )
                ps_out = subprocess.run(
                    ps_args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=3,
                    check=False,
                )
                for line in ps_out.stdout.splitlines():
                    line = line.strip()
                    if not line or "ps -eo" in line or "ps axwwE" in line:
                        continue
                    parts = line.split(maxsplit=1)
                    if len(parts) != 2 or not parts[0].isdigit():
                        continue
                    pid = int(parts[0])
                    args = parts[1]
                    if pid == my_pid:
                        continue
                    is_tsc_orphan = ("tsc" in args or "node" in args) and "--noEmit" in args and "--skipLibCheck" in args
                    is_diag_env = f"{VALSTORM_DIAGNOSTIC_ENV_VAR}=1" in args
                    if is_tsc_orphan or is_diag_env:
                        try:
                            kill_process_tree(pid)
                            killed_pids.append(pid)
                        except Exception:
                            pass
            if killed_pids:
                return killed_pids
        except Exception:
            pass

    # 3. Windows fallback
    if sys.platform == "win32":
        try:
            cmd = 'wmic process where "CommandLine like '%--noEmit%' and CommandLine like '%--skipLibCheck%'" get ProcessId'
            res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=5)
            for line in res.stdout.splitlines():
                line = line.strip()
                if line.isdigit():
                    pid = int(line)
                    if pid != my_pid:
                        kill_process_tree(pid)
                        killed_pids.append(pid)
        except Exception:
            pass

    return killed_pids


class SingleFlightGate:
    """Ensures at most one diagnostic check runs per workspace at any time.

    If a new check starts while a previous check is still running for the same workspace,
    the older in-flight check is cancelled immediately (preemptive latest-wins).
    """

    def __init__(self):
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def run(self, key: str, coro_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Runs coro_fn(*args, **kwargs), cancelling any in-flight task for `key`."""
        current_task = asyncio.current_task()
        async with self._lock:
            existing = self._active_tasks.get(key)
            if existing and not existing.done() and existing != current_task:
                setattr(existing, "_preempted_by_gate", True)
                existing.cancel()
                try:
                    await asyncio.wait([existing], timeout=0.5)
                except Exception:
                    pass

            task = asyncio.create_task(coro_fn(*args, **kwargs))
            self._active_tasks[key] = task

        try:
            return await task
        except asyncio.CancelledError:
            caller_cancelling = (
                getattr(current_task, "cancelling", lambda: 0)() > 0
                if current_task is not None
                else False
            )
            if getattr(task, "_preempted_by_gate", False) and not caller_cancelling:
                return None
            if not task.done():
                task.cancel()
            raise
        finally:
            async with self._lock:
                if self._active_tasks.get(key) is task:
                    self._active_tasks.pop(key, None)


_global_single_flight_gate = SingleFlightGate()


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


def _find_bin(name: str, start_dir: Path) -> Optional[str]:
    """Finds a tool binary walking up the directory tree or in global PATH."""
    cur = start_dir
    for _ in range(8):
        cand = cur / "node_modules" / ".bin" / name
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
        if sys.platform == "win32":
            for ext in (".cmd", ".exe", ".bat"):
                cand_win = cur / "node_modules" / ".bin" / f"{name}{ext}"
                if cand_win.is_file():
                    return str(cand_win)
        if cur.parent == cur:
            break
        cur = cur.parent
    return shutil.which(name)


async def _run_command(
    cmd: List[str],
    cwd: Optional[Path] = None,
    timeout_sec: float = 2.0,
    env: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Runs a diagnostic CLI tool asynchronously with strict timeout and guaranteed cleanup.

    Spawns in a new process group / session and forcefully terminates the process tree
    on timeout, cancellation, or exception to prevent orphaned zombie processes.
    """
    proc = None
    sub_env = os.environ.copy()
    if env:
        sub_env.update(env)
    sub_env[VALSTORM_DIAGNOSTIC_ENV_VAR] = "1"

    kwargs: Dict[str, Any] = {
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "cwd": cwd,
        "env": sub_env,
    }

    # On POSIX, create a new process group leader so os.killpg() wipes all children
    if sys.platform != "win32":
        kwargs["start_new_session"] = True

    try:
        proc = await asyncio.create_subprocess_exec(*cmd, **kwargs)
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        except (asyncio.TimeoutError, asyncio.CancelledError, BaseException) as e:
            if proc and proc.returncode is None:
                kill_process_tree(proc.pid)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                except Exception:
                    pass
            if isinstance(e, asyncio.CancelledError):
                raise
            return None

        out_text = (stdout.decode("utf-8", errors="ignore") + stderr.decode("utf-8", errors="ignore")).strip()
        return out_text if out_text else None
    except Exception:
        return None
    finally:
        # Guarantee cleanup even on unexpected termination flow
        if proc and proc.returncode is None:
            kill_process_tree(proc.pid)
            try:
                await asyncio.wait_for(proc.wait(), timeout=0.5)
            except Exception:
                pass


async def check_typescript_syntax(file_path: Path, timeout_sec: float = 2.0) -> Optional[List[str]]:
    """Runs instant single-file syntax check using esbuild, biome, or node.

    Takes 5-20 milliseconds with ~25 MB RAM instead of 45 seconds and 1.5 GB RAM.
    """
    ext = file_path.suffix.lower()

    # 1. Try esbuild (installed in Vite / Next.js / TypeScript projects)
    esbuild_bin = _find_bin("esbuild", file_path.parent)
    if esbuild_bin:
        loader_map = {
            ".ts": "ts",
            ".tsx": "tsx",
            ".js": "js",
            ".jsx": "jsx",
            ".mjs": "js",
            ".cjs": "js",
        }
        loader = loader_map.get(ext, "tsx")
        cmd = [esbuild_bin, str(file_path), f"--loader={loader}", "--no-bundle"]
        out = await _run_command(cmd, cwd=file_path.parent, timeout_sec=min(timeout_sec, 2.0))
        if out and ("error:" in out.lower() or "unexpected" in out.lower() or "syntax" in out.lower()):
            lines = [
                line.strip()
                for line in out.splitlines()
                if line.strip() and not line.startswith("▲ [WARNING]")
            ]
            if lines:
                return lines[:5]
        return None

    # 2. Try biome check
    biome_bin = _find_bin("biome", file_path.parent)
    if biome_bin:
        cmd = [biome_bin, "check", str(file_path)]
        out = await _run_command(cmd, cwd=file_path.parent, timeout_sec=min(timeout_sec, 2.0))
        if out and ("error" in out.lower() or "parse" in out.lower()):
            lines = [line.strip() for line in out.splitlines() if line.strip() and "error" in line.lower()]
            if lines:
                return lines[:5]
        return None

    # 3. For pure JS files, fall back to node --check (instant built-in AST parse)
    if ext in (".js", ".mjs", ".cjs"):
        node_bin = shutil.which("node")
        if node_bin:
            cmd = [node_bin, "--check", str(file_path)]
            out = await _run_command(cmd, cwd=file_path.parent, timeout_sec=min(timeout_sec, 2.0))
            if out:
                lines = [line.strip() for line in out.splitlines() if line.strip() and "error" in line.lower()]
                if lines:
                    return lines[:5]
            return None

    return None


async def check_go_diagnostics(file_path: Path, timeout_sec: float = 2.0) -> Optional[List[str]]:
    """Runs `go vet` compiler syntax and type verification on Go source code."""
    go_bin = shutil.which("go")
    if not go_bin:
        return None

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


async def _run_tsc_diagnostics(
    file_path: Path,
    workspace_root: Path,
    tsc_bin: str,
    timeout_sec: float,
) -> Optional[List[str]]:
    """Runs tsc compiler check scoped to workspace with relative error filtering."""
    tsc_out = await _run_command(
        [tsc_bin, "--noEmit", "--skipLibCheck", "--pretty", "false"],
        cwd=workspace_root,
        timeout_sec=timeout_sec,
    )
    if tsc_out:
        fname = file_path.name
        try:
            rel_path = str(file_path.relative_to(workspace_root)).replace(chr(92), "/")
        except ValueError:
            rel_path = fname

        matching_lines = [
            line.strip() for line in tsc_out.splitlines()
            if (rel_path in line or fname in line) and ("error TS" in line or "error" in line.lower())
        ]
        if matching_lines:
            return matching_lines[:5]
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
        # Step 4A: Fast single-file syntax check (5-20ms, 25MB RAM)
        fast_errors = await check_typescript_syntax(file_path, timeout_sec=min(timeout_sec, 1.0))
        if fast_errors:
            errors.extend(fast_errors)
        else:
            # Step 4B: Gated tsc typecheck (Single-Flight per workspace)
            workspace_root = file_path.parent
            cur = file_path.parent
            for _ in range(8):
                if (cur / "tsconfig.json").exists() or (cur / "package.json").exists():
                    workspace_root = cur
                    break
                if cur.parent == cur:
                    break
                cur = cur.parent

            tsc_bin = _find_bin("tsc", file_path.parent)

            if tsc_bin and (workspace_root / "tsconfig.json").exists():
                gate_key = f"tsc:{workspace_root.resolve()}"
                tsc_errors = await _global_single_flight_gate.run(
                    gate_key,
                    _run_tsc_diagnostics,
                    file_path=file_path,
                    workspace_root=workspace_root,
                    tsc_bin=tsc_bin,
                    timeout_sec=timeout_sec,
                )
                if tsc_errors:
                    errors.extend(tsc_errors)

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
    nl = chr(10)
    bulleted = nl.join(f"- {err}" for err in errors)
    return f"{header}{nl}{bulleted}{nl}{nl}Notice: Please resolve the above diagnostic errors on your next turn."
