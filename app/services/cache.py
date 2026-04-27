"""
Pluggable cache: in-memory by default, Redis if REDIS_URL is set.

Backends store JSON-serialized values under string keys with TTL.
Caching is best-effort: any backend error is logged and treated as a miss
so the request path is never blocked by cache infra.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any, Awaitable, Callable, Protocol

from app.core.config import settings
from app.core.logging import logger


class CacheBackend(Protocol):
    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any, ttl: int) -> None: ...


class InMemoryCache:
    def __init__(self, max_entries: int = 5000) -> None:
        self._store: dict[str, tuple[float, Any]] = {}
        self._max = max_entries
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        item = self._store.get(key)
        if item is None:
            return None
        expires_at, value = item
        if expires_at < time.monotonic():
            self._store.pop(key, None)
            return None
        return value

    async def set(self, key: str, value: Any, ttl: int) -> None:
        async with self._lock:
            if len(self._store) >= self._max:
                now = time.monotonic()
                expired = [k for k, (exp, _) in self._store.items() if exp < now]
                for k in expired:
                    self._store.pop(k, None)
                while len(self._store) >= self._max:
                    self._store.pop(next(iter(self._store)))
            self._store[key] = (time.monotonic() + ttl, value)


class RedisCache:
    def __init__(self, url: str) -> None:
        import redis.asyncio as redis_async

        self._client = redis_async.from_url(url, decode_responses=True)

    async def get(self, key: str) -> Any | None:
        try:
            raw = await self._client.get(key)
        except Exception as exc:
            logger.debug(f"Redis GET failed: {exc}")
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    async def set(self, key: str, value: Any, ttl: int) -> None:
        try:
            await self._client.set(key, json.dumps(value), ex=ttl)
        except Exception as exc:
            logger.debug(f"Redis SET failed: {exc}")


_backend: CacheBackend | None = None


def get_cache() -> CacheBackend:
    global _backend
    if _backend is None:
        if settings.redis_url:
            try:
                _backend = RedisCache(settings.redis_url)
                logger.info(f"Cache backend: Redis ({settings.redis_url})")
            except Exception as exc:
                logger.warning(f"Redis init failed, falling back to in-memory: {exc}")
                _backend = InMemoryCache()
        else:
            _backend = InMemoryCache()
            logger.info("Cache backend: in-memory")
    return _backend


def make_key(namespace: str, *parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"{namespace}:{digest}"


async def get_or_set(
    key: str,
    ttl: int,
    producer: Callable[[], Awaitable[Any]],
) -> tuple[Any, bool]:
    """
    Returns (value, was_cache_hit). Falls through to `producer()` on miss
    or any backend error.
    """
    if not settings.cache_enabled:
        return await producer(), False

    cache = get_cache()
    try:
        cached = await cache.get(key)
    except Exception as exc:
        logger.debug(f"cache.get raised: {exc}")
        cached = None
    if cached is not None:
        return cached, True

    value = await producer()
    try:
        await cache.set(key, value, ttl)
    except Exception as exc:
        logger.debug(f"cache.set raised: {exc}")
    return value, False
