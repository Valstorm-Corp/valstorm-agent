"""Retry utilities and exponential backoff helpers for LLM providers."""

import asyncio
import logging
import random
from typing import Any, AsyncIterator, Callable, Coroutine, Optional, Set

logger = logging.getLogger("vsagent.retry")

RETRYABLE_STATUS_CODES: Set[int] = {408, 429, 500, 502, 503, 504, 529}
NON_RETRYABLE_STATUS_CODES: Set[int] = {400, 401, 403, 404, 422}

RETRYABLE_PHRASES = (
    "503",
    "429",
    "502",
    "504",
    "500",
    "529",
    "408",
    "unavailable",
    "resource_exhausted",
    "resource exhausted",
    "deadline_exceeded",
    "deadline exceeded",
    "high demand",
    "spikes in demand",
    "rate limit",
    "rate_limit",
    "ratelimit",
    "quota exceeded",
    "quota",
    "overloaded",
    "over capacity",
    "try again later",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "connection reset",
    "connection error",
    "connection refused",
    "timed out",
    "timeout",
)


def extract_status_code(exc: BaseException) -> Optional[int]:
    """Attempts to extract an HTTP or API status code from an exception object."""
    for attr in ("status_code", "code", "http_status", "status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
        if isinstance(val, str) and val.isdigit():
            return int(val)

    resp = getattr(exc, "response", None)
    if resp is not None:
        for attr in ("status_code", "code", "status"):
            val = getattr(resp, attr, None)
            if isinstance(val, int):
                return val
            if isinstance(val, str) and val.isdigit():
                return int(val)
    return None


def is_retryable_error(exc: BaseException) -> bool:
    """Determines whether an exception is a transient or retryable API error."""
    if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError, SystemExit)):
        return False

    # Check for known standard timeout and connection error classes
    if isinstance(exc, (TimeoutError, ConnectionError, ConnectionResetError, ConnectionRefusedError, BrokenPipeError)):
        return True

    # Check for httpx timeout/network errors if httpx is present
    exc_type_name = type(exc).__name__
    if any(k in exc_type_name for k in ("Timeout", "ConnectError", "NetworkError", "RemoteProtocolError", "RateLimitError", "InternalServerError")):
        return True

    status_code = extract_status_code(exc)
    if status_code is not None:
        if status_code in RETRYABLE_STATUS_CODES:
            return True
        if status_code in NON_RETRYABLE_STATUS_CODES:
            err_str = (str(exc) + " " + repr(exc)).lower()
            if any(p in err_str for p in ("high demand", "spikes in demand", "rate limit", "resource_exhausted", "unavailable")):
                return True
            return False

    # Check error message strings
    err_str = (str(exc) + " " + repr(exc)).lower()
    return any(phrase in err_str for phrase in RETRYABLE_PHRASES)


def compute_backoff_delay(
    attempt: int,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 30.0,
    jitter: bool = True,
) -> float:
    """Computes exponential backoff delay with optional jitter for attempt index (0-indexed)."""
    delay = initial_delay * (backoff_factor ** attempt)
    delay = min(delay, max_delay)
    if jitter:
        # Add jitter up to 25% of delay or max 0.5s
        jitter_val = random.uniform(0, min(0.5, delay * 0.25))
        delay += jitter_val
    return delay


async def execute_with_retry(
    coro_fn: Callable[[], Coroutine[Any, Any, Any]],
    provider_name: str = "llm",
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> Any:
    """Executes an async operation with exponential backoff retries on transient errors."""
    attempt = 0
    while True:
        try:
            return await coro_fn()
        except Exception as e:
            if attempt < max_retries and is_retryable_error(e):
                delay = compute_backoff_delay(
                    attempt=attempt,
                    initial_delay=initial_delay,
                    backoff_factor=backoff_factor,
                    max_delay=max_delay,
                    jitter=jitter,
                )
                logger.warning(
                    f"[{provider_name}] API call failed with retryable error ({e}). "
                    f"Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})..."
                )
                if on_retry:
                    try:
                        on_retry(attempt + 1, e, delay)
                    except Exception:
                        pass
                await asyncio.sleep(delay)
                attempt += 1
            else:
                raise


async def execute_stream_with_retry(
    stream_fn: Callable[[], AsyncIterator[Any]],
    provider_name: str = "llm",
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> AsyncIterator[Any]:
    """Wraps an async generator with retry and exponential backoff on transient errors."""
    attempt = 0
    while True:
        try:
            gen = stream_fn()
            async for item in gen:
                yield item
            return
        except Exception as e:
            if attempt < max_retries and is_retryable_error(e):
                delay = compute_backoff_delay(
                    attempt=attempt,
                    initial_delay=initial_delay,
                    backoff_factor=backoff_factor,
                    max_delay=max_delay,
                    jitter=jitter,
                )
                logger.warning(
                    f"[{provider_name}] Stream failed with retryable error ({e}). "
                    f"Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})..."
                )
                if on_retry:
                    try:
                        on_retry(attempt + 1, e, delay)
                    except Exception:
                        pass
                await asyncio.sleep(delay)
                attempt += 1
            else:
                raise
