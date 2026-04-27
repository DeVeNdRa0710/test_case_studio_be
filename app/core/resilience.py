"""
Centralized retry + circuit-breaker policy for external dependencies
(Pinecone, Neo4j, LLM/embedding providers).

- `retryable_external` — tenacity decorator. Retries only on classes considered
  transient (network errors, 429, 5xx, neo4j ServiceUnavailable). Auth/validation
  failures (401/403/422 etc.) raise immediately so we don't waste retries.
- `pinecone_breaker`, `neo4j_breaker`, `llm_breaker` — per-dep circuit breakers.
  When open, calls fail fast with `ExternalServiceUnavailable` instead of
  cascading 500s.
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any, Awaitable, Callable, TypeVar

import pybreaker
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.core.exceptions import ExternalServiceUnavailable
from app.core.logging import logger


F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


_RETRYABLE_NAMES = {
    "ServiceUnavailable",
    "SessionExpired",
    "TransientError",
    "ConnectionError",
    "TimeoutError",
    "ReadTimeout",
    "WriteTimeout",
    "PoolTimeout",
    "ConnectTimeout",
    "RemoteDisconnected",
    "ResourceExhausted",
    "DeadlineExceeded",
    "Aborted",
    "Unavailable",
    "InternalServerError",
    "ServerError",
    "RetryError",
    "APIConnectionError",
    "APITimeoutError",
    "RateLimitError",
}


_NON_RETRYABLE_NAMES = {
    "AuthenticationError",
    "PermissionDeniedError",
    "BadRequestError",
    "NotFoundError",
    "InvalidArgument",
    "InvalidRequestError",
    "Unauthorized",
    "ValueError",
    "TypeError",
    "AssertionError",
}


def _is_retryable(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in _NON_RETRYABLE_NAMES:
        return False
    if name in _RETRYABLE_NAMES:
        return True

    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int):
        if status in {408, 425, 429} or 500 <= status < 600:
            return True
        if 400 <= status < 500:
            return False

    text = str(exc).lower()
    if any(k in text for k in ("timeout", "temporarily unavailable", "connection reset", "service unavailable")):
        return True

    return False


def retryable_external(
    *,
    attempts: int = 4,
    initial: float = 1.0,
    maximum: float = 8.0,
) -> Callable[[F], F]:
    """Decorator: retry transient errors, fail fast on permanent ones."""

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(attempts),
                    wait=wait_exponential_jitter(initial=initial, max=maximum),
                    retry=retry_if_exception(_is_retryable),
                    reraise=True,
                ):
                    with attempt:
                        return await fn(*args, **kwargs)
            except RetryError as exc:
                raise exc.last_attempt.exception()  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


pinecone_breaker = pybreaker.CircuitBreaker(
    fail_max=5, reset_timeout=30, name="pinecone"
)
neo4j_breaker = pybreaker.CircuitBreaker(
    fail_max=5, reset_timeout=30, name="neo4j"
)
llm_breaker = pybreaker.CircuitBreaker(
    fail_max=5, reset_timeout=30, name="llm"
)


def with_breaker(breaker: pybreaker.CircuitBreaker, dep_label: str) -> Callable[[F], F]:
    """
    Wrap an async callable with a circuit breaker. When the breaker is open,
    raises ExternalServiceUnavailable (mapped to 503 by the exception handlers).
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            if breaker.current_state == "open":
                raise ExternalServiceUnavailable(
                    f"{dep_label} circuit open; failing fast",
                )
            try:
                result = await fn(*args, **kwargs)
                if breaker.fail_counter:
                    breaker.call(lambda: None)
                return result
            except Exception as exc:
                if _is_retryable(exc) or isinstance(exc, ExternalServiceUnavailable):
                    try:
                        await asyncio.shield(asyncio.sleep(0))
                        breaker.call(lambda: (_ for _ in ()).throw(exc))
                    except pybreaker.CircuitBreakerError as cb_exc:
                        logger.warning(f"{dep_label} circuit opened: {cb_exc}")
                        raise ExternalServiceUnavailable(
                            f"{dep_label} unavailable: {exc}"
                        ) from exc
                    except Exception:
                        pass
                raise

        return wrapper  # type: ignore[return-value]

    return decorator
