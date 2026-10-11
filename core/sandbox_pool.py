"""Warm Ephemeral Sandbox Pooling and Lifecycle Manager for Valstorm Agent Runtime.

Provides:
- SandboxPool: Standby pool of pre-booted DockerSandbox micro-containers providing
  sub-50ms instant container acquisition for cloud and user runs.
- Async Background Replenishment: Automatically provisions replacement containers
  in the background without blocking active ReAct agent turns.
- Non-Blocking Release & Auto-Reap: Asynchronously closes and removes containers upon run completion.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Set

from core.sandbox import BaseSandbox
from core.docker_sandbox import DockerSandbox

logger = logging.getLogger("valstorm.sandbox_pool")


class SandboxPool:
    """Manages a pool of pre-warmed, ephemeral sandbox execution containers."""

    def __init__(
        self,
        target_size: int = 2,
        max_size: int = 5,
        factory: Optional[Callable[[], BaseSandbox]] = None,
        auto_replenish: bool = True,
    ):
        self.target_size = target_size
        self.max_size = max_size
        self._factory = factory or (lambda: DockerSandbox())
        self.auto_replenish = auto_replenish
        self._idle_pool: List[BaseSandbox] = []
        self._in_use: Set[BaseSandbox] = set()
        self._lock = asyncio.Lock()
        self._is_running = False
        self._replenish_tasks: Set[asyncio.Task] = set()

    @property
    def idle_count(self) -> int:
        """Returns number of pre-warmed idle sandboxes ready for instant acquisition."""
        return len(self._idle_pool)

    @property
    def in_use_count(self) -> int:
        """Returns number of sandboxes currently active in agent runs."""
        return len(self._in_use)

    async def warm_up(self, count: Optional[int] = None) -> int:
        """Pre-boots sandboxes concurrently to reach the target pool size."""
        self._is_running = True
        needed = (count if count is not None else self.target_size) - len(self._idle_pool)
        if needed <= 0:
            return 0

        async def _spawn_one() -> Optional[BaseSandbox]:
            sb = self._factory()
            try:
                await sb.start()
                return sb
            except Exception as e:
                logger.warning(f"Failed to pre-warm sandbox container: {e}")
                return None

        tasks = [_spawn_one() for _ in range(needed)]
        results = await asyncio.gather(*tasks)

        spawned = 0
        async with self._lock:
            for sb in results:
                if sb is not None:
                    self._idle_pool.append(sb)
                    spawned += 1

        return spawned

    async def acquire(self) -> BaseSandbox:
        """Acquires an ephemeral sandbox with near-zero latency (<50ms).

        If a pre-warmed container is available in the pool, returns it immediately
        and triggers background replenishment. Otherwise, creates one on demand.
        """
        sandbox: Optional[BaseSandbox] = None

        async with self._lock:
            if self._idle_pool:
                sandbox = self._idle_pool.pop(0)
                self._in_use.add(sandbox)

        # If pool was empty, provision synchronously on-demand
        if sandbox is None:
            sandbox = self._factory()
            await sandbox.start()
            async with self._lock:
                self._in_use.add(sandbox)

        # Trigger background replenishment to keep warm standby capacity
        if self.auto_replenish and self._is_running:
            self._schedule_replenish()

        return sandbox

    def _schedule_replenish(self) -> None:
        """Schedules a non-blocking background task to warm up replacement sandboxes."""
        needed = self.target_size - len(self._idle_pool)
        if needed > 0 and len(self._idle_pool) + len(self._in_use) < self.max_size:
            task = asyncio.create_task(self._replenish_worker(needed))
            self._replenish_tasks.add(task)
            task.add_done_callback(self._replenish_tasks.discard)

    async def _replenish_worker(self, count: int) -> None:
        """Background worker provisioning standby containers."""
        for _ in range(count):
            if not self._is_running:
                break
            async with self._lock:
                if len(self._idle_pool) >= self.target_size:
                    break
            sb = self._factory()
            try:
                await sb.start()
                async with self._lock:
                    if self._is_running and len(self._idle_pool) < self.target_size:
                        self._idle_pool.append(sb)
                    else:
                        # Pool full or shutting down, clean up
                        asyncio.create_task(sb.close())
            except Exception as e:
                logger.warning(f"Error during background sandbox replenishment: {e}")

    async def release(self, sandbox: BaseSandbox) -> None:
        """Releases a sandbox, scheduling container destruction in the background."""
        async with self._lock:
            self._in_use.discard(sandbox)

        # Destroy the used container asynchronously so client turns don't block
        async def _reap(sb: BaseSandbox) -> None:
            try:
                await sb.close()
            except Exception as e:
                logger.warning(f"Error closing used sandbox container: {e}")

        reap_task = asyncio.create_task(_reap(sandbox))
        self._replenish_tasks.add(reap_task)
        reap_task.add_done_callback(self._replenish_tasks.discard)

        if self.auto_replenish and self._is_running:
            self._schedule_replenish()

    async def shutdown(self) -> None:
        """Gracefully shuts down the pool and terminates all idling and active containers."""
        self._is_running = False

        # Cancel any active replenishment tasks
        for t in list(self._replenish_tasks):
            t.cancel()

        to_close: List[BaseSandbox] = []
        async with self._lock:
            to_close.extend(self._idle_pool)
            to_close.extend(self._in_use)
            self._idle_pool.clear()
            self._in_use.clear()

        async def _safe_close(sb: BaseSandbox) -> None:
            try:
                await sb.close()
            except Exception:
                pass

        if to_close:
            await asyncio.gather(*[_safe_close(sb) for sb in to_close])


# Global Singleton Pool
_global_pool: Optional[SandboxPool] = None


def get_global_sandbox_pool() -> Optional[SandboxPool]:
    """Returns the global warm sandbox pool if initialized."""
    return _global_pool


async def init_global_sandbox_pool(
    target_size: int = 2,
    max_size: int = 5,
    factory: Optional[Callable[[], BaseSandbox]] = None,
) -> SandboxPool:
    """Initializes and pre-warms the global sandbox pool."""
    global _global_pool
    if _global_pool is None:
        _global_pool = SandboxPool(
            target_size=target_size,
            max_size=max_size,
            factory=factory,
            auto_replenish=True,
        )
        await _global_pool.warm_up()
    return _global_pool


async def shutdown_global_sandbox_pool() -> None:
    """Tears down the global sandbox pool."""
    global _global_pool
    if _global_pool is not None:
        await _global_pool.shutdown()
        _global_pool = None
