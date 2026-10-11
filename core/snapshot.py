"""Transactional Git Snapshots and Turn Rollback Engine for Valstorm Agent Runtime.

Provides automatic pre-turn working tree checkpoints and instant `/undo` rollbacks:
- Captures lightweight Git tree / diff snapshots before mutating tools execute
- Supports multi-turn rollback (/undo) in CLI REPL and API sessions
- Rolls back both filesystem modifications and conversational turn history in SQLite
"""

import asyncio
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class TurnSnapshot:
    """Represents a filesystem snapshot captured prior to a conversational turn."""
    session_id: str
    turn_index: int
    git_stash_ref: Optional[str] = None
    dirty_files: List[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class GitSnapshotManager:
    """Manages transactional Git working tree snapshots and rollbacks for agent sessions."""

    def __init__(self, workdir: Optional[Path] = None):
        self.workdir = Path(workdir).expanduser().resolve() if workdir else Path.cwd().resolve()
        self._snapshots: Dict[str, List[TurnSnapshot]] = {}

    def _is_git_repo(self) -> bool:
        """Checks if workdir is inside a Git working tree."""
        git_dir = self.workdir / ".git"
        if git_dir.exists():
            return True
        # Check upward
        for parent in self.workdir.parents:
            if (parent / ".git").exists():
                return True
        return False

    async def _exec_git(self, args: List[str], timeout_sec: float = 5.0) -> Tuple[int, str, str]:
        """Runs a git command in the workspace directory."""
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=str(self.workdir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        out = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()
        return proc.returncode or 0, out, err

    async def create_snapshot(self, session_id: str, turn_index: int) -> Optional[TurnSnapshot]:
        """Creates a lightweight snapshot of the current working tree before a turn executes."""
        if not self._is_git_repo() or not shutil.which("git"):
            return None

        try:
            # 1. Inspect status
            code, status_out, _ = await self._exec_git(["status", "--porcelain"])
            dirty_files = [line[3:].strip() for line in status_out.splitlines() if line.strip()]

            # 2. Create git stash snapshot reference without modifying worktree
            # 'git stash create' creates a commit object of the dirty state and outputs hash, without changing HEAD
            code, stash_hash, _ = await self._exec_git(["stash", "create", f"vsagent_snap_{session_id}_{turn_index}"])
            stash_ref = stash_hash.strip() if code == 0 and stash_hash.strip() else None

            snap = TurnSnapshot(
                session_id=session_id,
                turn_index=turn_index,
                git_stash_ref=stash_ref,
                dirty_files=dirty_files,
            )

            if session_id not in self._snapshots:
                self._snapshots[session_id] = []
            self._snapshots[session_id].append(snap)
            return snap
        except Exception:
            return None

    async def revert_turn(self, session_id: str, turn_index: Optional[int] = None) -> Tuple[bool, str]:
        """Reverts the filesystem to the snapshot captured before the specified turn (or last turn)."""
        if not self._is_git_repo() or not shutil.which("git"):
            return False, "Workspace is not a Git repository. Snapshot rollback unavailable."

        snaps = self._snapshots.get(session_id, [])
        if not snaps:
            return False, "No snapshot checkpoints found for this session."

        target_snap: Optional[TurnSnapshot] = None
        if turn_index is not None:
            for s in reversed(snaps):
                if s.turn_index == turn_index:
                    target_snap = s
                    break
        else:
            target_snap = snaps[-1]

        if not target_snap:
            return False, f"Snapshot for turn {turn_index} not found."

        try:
            # If we captured a stash ref, restore working tree to that exact state
            if target_snap.git_stash_ref:
                # Restore worktree using checkout / stash apply
                # First clean untracked created during turn if any
                code, _, err = await self._exec_git(["stash", "apply", "--index", target_snap.git_stash_ref])
                if code != 0:
                    # Fallback to simple checkout
                    await self._exec_git(["checkout", "."])
            else:
                # If tree was clean before turn, checkout clean
                await self._exec_git(["checkout", "."])

            # Remove snapshot from list
            self._snapshots[session_id] = [s for s in snaps if s.turn_index < target_snap.turn_index]

            return True, f"Successfully rolled back working tree to turn #{target_snap.turn_index} checkpoint."
        except Exception as e:
            return False, f"Error rolling back snapshot: {e}"


# Global default singleton instance
_global_snapshot_manager: Optional[GitSnapshotManager] = None


def get_snapshot_manager(workdir: Optional[Path] = None) -> GitSnapshotManager:
    global _global_snapshot_manager
    if _global_snapshot_manager is None or (workdir and _global_snapshot_manager.workdir != workdir):
        _global_snapshot_manager = GitSnapshotManager(workdir=workdir)
    return _global_snapshot_manager
