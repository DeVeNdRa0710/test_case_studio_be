"""
One-shot import of legacy data/projects.json into the new DB.
Idempotent: if a row already exists, it is left alone. Renames the JSON to
.bak after a successful import so the old file is never silently abandoned.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from app.core.logging import logger
from app.db.models import Project
from app.db.session import SessionLocal


_LEGACY_PATH = Path(__file__).resolve().parents[2] / "data" / "projects.json"


def _parse_iso(s: str | None) -> datetime:
    if not s:
        return datetime.now(timezone.utc)
    try:
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime.now(timezone.utc)


async def import_legacy_projects() -> int:
    if not _LEGACY_PATH.exists():
        return 0
    try:
        data = json.loads(_LEGACY_PATH.read_text())
    except Exception as exc:
        logger.warning(f"Legacy projects.json unreadable, skipping import: {exc}")
        return 0

    projects = data.get("projects") if isinstance(data, dict) else None
    if not isinstance(projects, list) or not projects:
        _LEGACY_PATH.rename(_LEGACY_PATH.with_suffix(".json.bak"))
        return 0

    inserted = 0
    async with SessionLocal() as session:
        for p in projects:
            name = (p.get("name") or "").strip().lower()
            if not name:
                continue
            existing = await session.scalar(select(Project).where(Project.name == name))
            if existing:
                continue
            session.add(
                Project(
                    name=name,
                    description=(p.get("description") or None),
                    created_at=_parse_iso(p.get("created_at")),
                )
            )
            inserted += 1
        await session.commit()

    backup = _LEGACY_PATH.with_suffix(".json.bak")
    try:
        _LEGACY_PATH.rename(backup)
    except Exception as exc:
        logger.warning(f"Failed to archive legacy projects.json: {exc}")
    if inserted:
        logger.info(f"Imported {inserted} legacy project(s) from projects.json → {backup.name}")
    return inserted
