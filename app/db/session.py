"""
Async SQLAlchemy engine + session. Defaults to SQLite-WAL on disk; flip to
Postgres by setting DATABASE_URL=postgresql+asyncpg://...
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.core.logging import logger


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def _ensure_sqlite_dir(url: str) -> None:
    if not _is_sqlite(url):
        return
    if ":///" not in url:
        return
    rel = url.split(":///", 1)[1]
    if not rel or rel.startswith(":"):
        return
    p = Path(rel)
    if not p.is_absolute():
        p = Path(os.getcwd()) / p
    p.parent.mkdir(parents=True, exist_ok=True)


_ensure_sqlite_dir(settings.database_url)

engine_kwargs: dict = {"echo": False, "future": True}
if _is_sqlite(settings.database_url):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_async_engine(settings.database_url, **engine_kwargs)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_db() -> None:
    """Apply migrations / set SQLite pragmas. Called from FastAPI lifespan."""
    from sqlalchemy import text

    if _is_sqlite(settings.database_url):
        async with engine.begin() as conn:
            await conn.execute(text("PRAGMA journal_mode=WAL"))
            await conn.execute(text("PRAGMA synchronous=NORMAL"))
            await conn.execute(text("PRAGMA foreign_keys=ON"))

    from app.db.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info(f"DB ready: {settings.database_url}")


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
