"""Unit tests for Warm Ephemeral Sandbox Pooling and Lifecycle Subsystem."""

import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest

from core.sandbox import BaseSandbox
from core.sandbox_pool import SandboxPool, init_global_sandbox_pool, shutdown_global_sandbox_pool


def _make_mock_sandbox():
    sb = MagicMock(spec=BaseSandbox)
    sb.start = AsyncMock()
    sb.close = AsyncMock()
    sb.exec_command = AsyncMock(return_value="mock output")
    return sb


@pytest.mark.asyncio
async def test_pool_warm_up():
    created_sandboxes = []

    def mock_factory():
        sb = _make_mock_sandbox()
        created_sandboxes.append(sb)
        return sb

    pool = SandboxPool(target_size=3, max_size=5, factory=mock_factory, auto_replenish=False)
    spawned = await pool.warm_up()

    assert spawned == 3
    assert pool.idle_count == 3
    assert pool.in_use_count == 0
    assert len(created_sandboxes) == 3
    for sb in created_sandboxes:
        sb.start.assert_awaited_once()

    await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_instant_acquire():
    def mock_factory():
        return _make_mock_sandbox()

    pool = SandboxPool(target_size=2, max_size=5, factory=mock_factory, auto_replenish=False)
    await pool.warm_up()

    assert pool.idle_count == 2

    # Acquire should be instant from warm pool
    sb1 = await pool.acquire()
    assert pool.idle_count == 1
    assert pool.in_use_count == 1

    sb2 = await pool.acquire()
    assert pool.idle_count == 0
    assert pool.in_use_count == 2

    await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_empty_fallback():
    created = []

    def mock_factory():
        sb = _make_mock_sandbox()
        created.append(sb)
        return sb

    # Pool with 0 warm up
    pool = SandboxPool(target_size=2, max_size=5, factory=mock_factory, auto_replenish=False)
    assert pool.idle_count == 0

    sb = await pool.acquire()
    assert pool.in_use_count == 1
    sb.start.assert_awaited_once()

    await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_auto_replenish():
    created = []

    def mock_factory():
        sb = _make_mock_sandbox()
        created.append(sb)
        return sb

    pool = SandboxPool(target_size=2, max_size=5, factory=mock_factory, auto_replenish=True)
    await pool.warm_up()
    assert pool.idle_count == 2

    # Acquire one - should trigger background replenishment
    sb = await pool.acquire()
    assert pool.in_use_count == 1

    # Allow async background task to run
    await asyncio.sleep(0.05)

    assert pool.idle_count == 2
    await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_release_and_reap():
    def mock_factory():
        return _make_mock_sandbox()

    pool = SandboxPool(target_size=2, max_size=5, factory=mock_factory, auto_replenish=False)
    await pool.warm_up()

    sb = await pool.acquire()
    assert pool.in_use_count == 1

    await pool.release(sb)
    assert pool.in_use_count == 0

    # Wait for non-blocking close task
    await asyncio.sleep(0.05)
    sb.close.assert_awaited_once()

    await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_shutdown():
    sandboxes = []

    def mock_factory():
        sb = _make_mock_sandbox()
        sandboxes.append(sb)
        return sb

    pool = SandboxPool(target_size=2, max_size=5, factory=mock_factory, auto_replenish=False)
    await pool.warm_up()
    in_use_sb = await pool.acquire()

    await pool.shutdown()
    assert pool.idle_count == 0
    assert pool.in_use_count == 0

    # All sandboxes must be closed
    for sb in sandboxes:
        sb.close.assert_awaited()
