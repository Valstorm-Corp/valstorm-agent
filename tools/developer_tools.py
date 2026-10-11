"""Developer Toolbelt for Valstorm Agent Runtime.

Provides high-performance, safe software development tools:
- terminal_exec: Async subprocess execution with timeout and output windowing
- patch_file: Targeted find-and-replace edits with diff output and live LSP diagnostics
- write_file: Atomic file authoring and overwrite with parent directory creation and live LSP diagnostics
- read_file: Line-numbered file reading with budget pagination
- search_files: Fast regex grep and file search ignoring build artifacts
"""

import asyncio
import difflib
import fnmatch
import os
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from core.tools import ToolRegistry, tool
from core.sandbox import get_current_sandbox, resolve_agent_path

IGNORED_DIRECTORIES = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".next",
    ".turbo",
    ".cache",
    ".hermes",
}

MAX_OUTPUT_BYTES = 16 * 1024  # 16 KB output truncation budget


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


# =====================================================================
# 1. terminal_exec
# =====================================================================


@tool
async def terminal_exec(
    command: str,
    timeout: int = 120,
    workdir: Optional[str] = None,
) -> str:
    """Executes a shell command asynchronously and returns stdout, stderr, and exit code.

    Args:
        command: The shell command to execute (e.g. 'pytest tests/', 'git status').
        timeout: Maximum seconds to allow the process to run (default: 120).
        workdir: Directory to execute the command from (defaults to the session working directory).
            A `cd` inside the command persists for later terminal_exec and file tool calls,
            like a normal shell session.
    """
    sandbox = get_current_sandbox()
    return await sandbox.exec_command(command=command, timeout_sec=timeout, workdir=workdir)


# =====================================================================
# 2. patch_file
# =====================================================================


@tool
async def patch_file(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """Replaces unique text within an existing file with new text and returns a unified diff.

    Args:
        path: Path to the target file to edit.
        old_string: Exact text to find in the file.
        new_string: Replacement text.
        replace_all: If True, replaces all occurrences. If False, requires old_string to match exactly once.
    """
    sandbox = get_current_sandbox()
    res = await sandbox.patch_file(path=path, old_string=old_string, new_string=new_string, replace_all=replace_all)
    if "Successfully patched" in res:
        try:
            from core.lsp.diagnostics import get_file_diagnostics
            file_path = resolve_agent_path(path)
            diagnostics = await get_file_diagnostics(file_path)
            if diagnostics:
                res += f"\n\n{diagnostics}"
        except Exception:
            pass
    return res


# =====================================================================
# 3. write_file
# =====================================================================


@tool
async def write_file(path: str, content: str) -> str:
    """Creates a new file or completely overwrites an existing file with the given content.

    Args:
        path: Path to the file to create or overwrite.
        content: Complete content to write.
    """
    sandbox = get_current_sandbox()
    res = await sandbox.write_file(path=path, content=content)
    if "Successfully wrote" in res:
        try:
            from core.lsp.diagnostics import get_file_diagnostics
            file_path = resolve_agent_path(path)
            diagnostics = await get_file_diagnostics(file_path)
            if diagnostics:
                res += f"\n\n{diagnostics}"
        except Exception:
            pass
    return res


# =====================================================================
# 4. read_file
# =====================================================================


@tool
async def read_file(path: str, offset: int = 1, limit: int = 2000) -> str:
    """Reads a file with 1-indexed line numbers and pagination.

    Args:
        path: Path to the file to read.
        offset: Line number to start reading from (1-indexed, default: 1).
        limit: Maximum number of lines to return (default: 2000).
    """
    sandbox = get_current_sandbox()
    return await sandbox.read_file(path=path, offset=offset, limit=limit)


# =====================================================================
# 5. search_files
# =====================================================================


def _search_files_sync(
    pattern: str,
    target: Literal["content", "files"] = "content",
    path: str = ".",
    file_glob: Optional[str] = None,
    limit: int = 50,
) -> str:
    import shutil
    import subprocess

    root_dir = Path(path).expanduser().resolve()
    if not root_dir.is_dir():
        return f"Error: Search directory not found: {root_dir}"

    # 0. Blazing-fast Native Rust search if available
    try:
        from core.native_bridge import native_search_files
        native_matches = native_search_files(pattern=pattern, target=target, path=str(root_dir), file_glob=file_glob, limit=limit)
        if native_matches is not None:
            if not native_matches:
                if target == "files":
                    return f"No files found matching pattern '{pattern}' in {root_dir}"
                else:
                    return f"No content matches found for regex '{pattern}' in {root_dir}"
            if target == "files":
                return f"Found {len(native_matches)} files matching '{pattern}':\n" + "\n".join(native_matches)
            else:
                return f"Found {len(native_matches)} matches for '{pattern}':\n" + "\n".join(native_matches)
    except Exception:
        pass

    # 1. High-speed Ripgrep path if installed
    rg_bin = shutil.which("rg")
    if rg_bin:
        try:
            if target == "files":
                cmd = [rg_bin, "--files", "--hidden"]
                for ign in IGNORED_DIRECTORIES:
                    cmd.extend(["-g", f"!{ign}"])
                if pattern and pattern != "*":
                    cmd.extend(["-g", f"*{pattern}*" if not any(c in pattern for c in "*?[]") else pattern])

                res = subprocess.run(cmd, cwd=str(root_dir), capture_output=True, text=True, timeout=10)
                if res.returncode == 0 or res.stdout:
                    file_lines = [l.strip() for l in res.stdout.splitlines() if l.strip()][:limit]
                    if not file_lines:
                        return f"No files found matching pattern '{pattern}' in {root_dir}"
                    return f"Found {len(file_lines)} files matching '{pattern}':\n" + "\n".join(file_lines)

            elif target == "content":
                cmd = [rg_bin, "-n", "-i", "--max-count", str(limit)]
                for ign in IGNORED_DIRECTORIES:
                    cmd.extend(["-g", f"!{ign}"])
                if file_glob:
                    cmd.extend(["-g", file_glob])
                cmd.extend([pattern, "."])

                res = subprocess.run(cmd, cwd=str(root_dir), capture_output=True, text=True, timeout=10)
                if res.returncode in (0, 1):
                    lines = [l.strip() for l in res.stdout.splitlines() if l.strip()][:limit]
                    if not lines:
                        return f"No content matches found for regex '{pattern}' in {root_dir}"
                    return f"Found {len(lines)} matches for '{pattern}':\n" + "\n".join(lines)
        except Exception:
            pass  # Fallback to python traversal if rg hits edge-case error

    # 2. Python fallback traversal
    matches: List[str] = []

    if target == "files":
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRECTORIES and not d.startswith(".")]
            for file_name in files:
                if file_name.startswith("."):
                    continue
                full_path = Path(root) / file_name
                rel_path = full_path.relative_to(root_dir)
                if fnmatch.fnmatch(file_name, pattern) or (pattern in file_name):
                    matches.append(str(rel_path))
                    if len(matches) >= limit:
                        break
            if len(matches) >= limit:
                break

        if not matches:
            return f"No files found matching pattern '{pattern}' in {root_dir}"
        return f"Found {len(matches)} files matching '{pattern}':\n" + "\n".join(matches)

    elif target == "content":
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Error: Invalid regular expression '{pattern}': {e}"

        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRECTORIES and not d.startswith(".")]
            for file_name in files:
                if file_name.startswith("."):
                    continue
                if file_glob and not fnmatch.fnmatch(file_name, file_glob):
                    continue

                file_path = Path(root) / file_name
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        for line_idx, line in enumerate(f, 1):
                            if regex.search(line):
                                rel_path = file_path.relative_to(root_dir)
                                clean_line = line.strip()
                                if len(clean_line) > 200:
                                    clean_line = clean_line[:197] + "..."
                                matches.append(f"{rel_path}:{line_idx}: {clean_line}")
                                if len(matches) >= limit:
                                    break
                except Exception:
                    continue
                if len(matches) >= limit:
                    break
            if len(matches) >= limit:
                break

        if not matches:
            return f"No content matches found for regex '{pattern}' in {root_dir}"
        return f"Found {len(matches)} matches for '{pattern}':\n" + "\n".join(matches)

    return f"Error: Unsupported search target '{target}'. Use 'content' or 'files'."


@tool
async def search_files(
    pattern: str,
    target: Literal["content", "files"] = "content",
    path: str = ".",
    file_glob: Optional[str] = None,
    limit: int = 50,
) -> str:
    """Searches file contents using regex or finds file paths by name matching a pattern.

    Args:
        pattern: Regex pattern to search inside file contents, or wildcard/name to find files.
        target: 'content' to search inside files (grep), or 'files' to search for file paths by name.
        path: Directory to search in (defaults to current working directory).
        file_glob: Optional filter for filenames when target='content' (e.g. '*.py', '*.ts').
        limit: Maximum number of matches to return (default: 50).
    """
    resolved_path = resolve_agent_path(path)

    # Scoped path prioritization when searching root/current working directory
    if path in (".", "", "./"):
        try:
            from core.context import get_current_profile
            prof = get_current_profile()
            scoped_paths = prof.get("scoped_paths") if prof else None
            if scoped_paths and isinstance(scoped_paths, list):
                scoped_hits = []
                for sp in scoped_paths:
                    clean_sp = sp.split("/*")[0].strip()
                    cand_dir = resolved_path / clean_sp
                    if cand_dir.is_dir():
                        sub_res = await asyncio.to_thread(_search_files_sync, pattern, target, str(cand_dir), file_glob, limit)
                        if sub_res and not sub_res.startswith("No ") and not sub_res.startswith("Error:"):
                            scoped_hits.append(f"### [Scoped: {sp}]\n{sub_res}")

                if scoped_hits:
                    return "\n\n".join(scoped_hits)
        except Exception:
            pass

    return await asyncio.to_thread(_search_files_sync, pattern, target, str(resolved_path), file_glob, limit)


# =====================================================================
# 5. code_outline (AST structural skeleton)
# =====================================================================


@tool
async def code_outline(path: str) -> str:
    """Generates a compact structural outline (signatures, interfaces, types, exports) of a code file.

    Saves 80-90% context tokens compared to reading the entire file when discovering codebase architecture.

    Args:
        path: Path to the code file (e.g. 'packages/components/App.tsx' or 'apps/api/main.py').
    """
    # New logic (User Request): Skip token-saving compaction/outlining and fall back to a full file read.
    # The performance benefit is retained in other native tools (search, symbols, patch).
    # This ensures the LLM receives the full context ("smart") at the expense of tokens ("skip compacting").
    try:
        sandbox = get_current_sandbox()
        # Using a very large limit (50k lines) to ensure the entire file is read,
        # providing the full context with line numbers.
        return await sandbox.read_file(path=path, offset=1, limit=50000)
    except Exception as e:
        return f"Error reading file {path} fully (skipping outline): {e}"


# =====================================================================
# 6. find_symbols (Symbol graph search)
# =====================================================================


@tool
async def find_symbols(
    query: str = "",
    kind: Optional[str] = None,
    path: str = ".",
    limit: int = 50,
) -> str:
    """Finds exact declarations (functions, classes, interfaces, types, structs, enums) across the project in milliseconds.

    Args:
        query: Symbol name or substring to search (e.g. 'useUnifiedRouter' or 'Customer').
        kind: Optional symbol filter: 'function', 'class', 'interface', 'type', 'struct', 'enum'.
        path: Directory to search in (defaults to current working directory).
        limit: Maximum matches to return (default: 50).
    """
    root_dir = resolve_agent_path(path)
    if not root_dir.is_dir():
        return f"Error: Directory not found: {root_dir}"

    try:
        from core.native_bridge import native_find_symbols
        matches = native_find_symbols(str(root_dir), query=query, kind=kind, limit=limit)
        if matches is not None:
            if not matches:
                return f"No symbols found matching query '{query}' in {root_dir}"
            lines = [f"{m['file']}:{m['line']} [{m['kind']}] {m['name']} -> {m['signature']}" for m in matches]
            return f"Found {len(matches)} symbols matching '{query}':\n" + "\n".join(lines)
    except Exception:
        pass

    # Fallback to search_files
    return await search_files(pattern=query, target="content", path=str(root_dir), limit=limit)


# =====================================================================
# 7. package_blast_radius (Monorepo dependency topology)
# =====================================================================


@tool
async def package_blast_radius(path: str, root_dir: str = ".") -> str:
    """Calculates the monorepo blast radius for a given file or package.

    Determines which internal packages and dependents are affected by modifying this file.

    Args:
        path: The file path being modified (e.g. 'packages/components/Button.tsx').
        root_dir: Workspace root directory (defaults to current working directory).
    """
    root = resolve_agent_path(root_dir)
    try:
        import json
        from core.native_bridge import native_get_blast_radius
        report = native_get_blast_radius(str(root), path)
        if report:
            owning = report.get("owning_package") or "Unknown"
            direct = report.get("direct_dependents") or []
            affected = report.get("affected_package_paths") or []
            return (
                f"### Monorepo Blast Radius for `{path}`\n"
                f"- **Owning Package**: `{owning}`\n"
                f"- **Direct Dependents ({len(direct)})**: {', '.join(f'`{d}`' for d in direct) if direct else 'None'}\n"
                f"- **Affected Package Directories**: {', '.join(f'`{p}`' for p in affected) if affected else 'None'}\n"
            )
    except Exception as e:
        return f"Error calculating blast radius: {e}"

    return "Blast radius engine unavailable (requires native acceleration)."


# =====================================================================
# Registration helper
# =====================================================================


def create_developer_tools() -> List[Any]:
    """Returns the list of developer tool functions."""
    return [
        terminal_exec,
        patch_file,
        write_file,
        read_file,
        search_files,
        code_outline,
        find_symbols,
        package_blast_radius,
    ]


def register_developer_tools(registry: ToolRegistry) -> ToolRegistry:
    """Registers all developer tools into the provided ToolRegistry."""
    for tool_fn in create_developer_tools():
        registry.register(tool_fn)
    return registry
