"""Native Rust Acceleration Bridge for Valstorm Agent Runtime.

Provides high-performance, in-process C/Rust extensions via vsagent_native:
- Blazing-fast file name and regex content search (zero subprocess overhead)
- Robust, whitespace-tolerant AST/content patcher with unified diffs
- Structural code outlining (stubs) for 90% context window savings
- Monorepo package dependency and blast radius calculation
- Compiled microsecond secret & credential redaction
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_NATIVE_AVAILABLE = False
_native_mod = None

try:
    import vsagent_native  # type: ignore

    _native_mod = vsagent_native
    _NATIVE_AVAILABLE = True
except ImportError:
    logger.debug("[vsagent] Native Rust accelerator (vsagent_native) not loaded. Falling back to pure Python.")


def is_native_available() -> bool:
    """Returns True if the Rust native acceleration module is loaded."""
    return _NATIVE_AVAILABLE


def native_search_files(
    pattern: str,
    target: str = "content",
    path: str = ".",
    file_glob: Optional[str] = None,
    limit: int = 50,
) -> Optional[List[str]]:
    """Runs fast in-process search using the Rust engine."""
    if not _NATIVE_AVAILABLE or _native_mod is None:
        return None
    try:
        return _native_mod.search_files(path, pattern, target, file_glob, limit)
    except Exception as e:
        logger.debug(f"[vsagent] Native search_files error: {e}")
        return None


def native_patch_content(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> Optional[Tuple[str, str]]:
    """Patches content using the Rust engine.

    Returns (patched_content, unified_diff) or None on failure/unavailability.
    """
    if not _NATIVE_AVAILABLE or _native_mod is None:
        return None
    try:
        return _native_mod.patch_content(content, old_string, new_string, replace_all)
    except Exception as e:
        logger.debug(f"[vsagent] Native patch_content error: {e}")
        return None


def native_generate_outline(file_path: str, content: str) -> Optional[str]:
    """Generates a structural code outline using the Rust engine."""
    if not _NATIVE_AVAILABLE or _native_mod is None:
        return None
    try:
        return _native_mod.generate_outline(file_path, content)
    except Exception as e:
        logger.debug(f"[vsagent] Native generate_outline error: {e}")
        return None


def native_find_symbols(
    root_dir: str = ".",
    query: str = "",
    kind: Optional[str] = None,
    limit: int = 50,
) -> Optional[List[Dict[str, Any]]]:
    """Finds symbol declarations across files using the Rust engine."""
    if not _NATIVE_AVAILABLE or _native_mod is None:
        return None
    try:
        return _native_mod.find_symbols(root_dir, query, kind, limit)
    except Exception as e:
        logger.debug(f"[vsagent] Native find_symbols error: {e}")
        return None


def native_scan_workspace(root_dir: str = ".") -> Optional[Dict[str, Any]]:
    """Scans monorepo workspace structure using the Rust engine."""
    if not _NATIVE_AVAILABLE or _native_mod is None:
        return None
    try:
        import json
        raw = _native_mod.scan_workspace(root_dir)
        return json.loads(raw)
    except Exception as e:
        logger.debug(f"[vsagent] Native scan_workspace error: {e}")
        return None


def native_get_blast_radius(root_dir: str, target_file: str) -> Optional[Dict[str, Any]]:
    """Calculates package blast radius using the Rust engine."""
    if not _NATIVE_AVAILABLE or _native_mod is None:
        return None
    try:
        import json
        raw = _native_mod.get_blast_radius(root_dir, target_file)
        return json.loads(raw)
    except Exception as e:
        logger.debug(f"[vsagent] Native get_blast_radius error: {e}")
        return None


def native_sanitize_text(text: str) -> Optional[str]:
    """Sanitizes sensitive tokens using the compiled Rust regex engine."""
    if not _NATIVE_AVAILABLE or _native_mod is None:
        return None
    try:
        return _native_mod.fast_sanitize_text(text)
    except Exception as e:
        logger.debug(f"[vsagent] Native sanitize_text error: {e}")
        return None
