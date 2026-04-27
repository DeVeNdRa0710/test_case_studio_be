"""
Rate limiter built on slowapi. Keyed by API key header (`X-API-Key`) when present,
otherwise by client IP. Disabled when settings.rate_limit_enabled is False — in that
case the decorators become no-ops so route handlers behave exactly as before.
"""

from __future__ import annotations

from fastapi import Request
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

from app.core.config import settings


def _key(request: Request) -> str:
    api_key = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
    if api_key:
        return f"key:{api_key.strip()}"
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(
    key_func=_key,
    enabled=settings.rate_limit_enabled,
    storage_uri=settings.redis_url or "memory://",
    default_limits=[],
)


async def rate_limit_exceeded_handler(_: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "code": "rate_limited",
                "message": f"Rate limit exceeded: {exc.detail}",
            }
        },
        headers={"Retry-After": "60"},
    )
