"""
In-process async job tracker.

POST /generate-testcases/async returns immediately with {job_id}; the actual
LLM work runs as an asyncio.Task. State is persisted in the `jobs` table so
the job survives a process check (and is visible to other workers if you flip
to Postgres). Tasks themselves don't survive a restart — that requires a real
broker (Arq/Celery), which is the Tier-B path.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from sqlalchemy import select

from app.core.logging import logger
from app.db.models import Job
from app.db.session import SessionLocal


JobRunner = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


async def _set_status(
    job_id: str,
    *,
    status: str,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    async with SessionLocal() as s:
        job = await s.scalar(select(Job).where(Job.id == job_id))
        if not job:
            return
        job.status = status
        job.updated_at = datetime.now(timezone.utc)
        if result is not None:
            job.result = result
        if error is not None:
            job.error = error
        await s.commit()


async def enqueue(
    *,
    kind: str,
    project: str | None,
    request_payload: dict[str, Any],
    runner: JobRunner,
) -> str:
    job_id = f"job_{uuid.uuid4().hex[:16]}"
    async with SessionLocal() as s:
        s.add(
            Job(
                id=job_id,
                kind=kind,
                project_name=project,
                status="queued",
                request=request_payload,
            )
        )
        await s.commit()

    async def _run():
        await _set_status(job_id, status="running")
        try:
            result = await runner(request_payload)
            await _set_status(job_id, status="done", result=result)
            logger.info(f"job {job_id} ({kind}) done")
        except Exception as exc:
            logger.exception(f"job {job_id} ({kind}) failed")
            await _set_status(job_id, status="error", error=str(exc))

    asyncio.create_task(_run())
    return job_id


async def get_job(job_id: str) -> dict[str, Any] | None:
    async with SessionLocal() as s:
        job = await s.scalar(select(Job).where(Job.id == job_id))
        if not job:
            return None
        return {
            "job_id": job.id,
            "kind": job.kind,
            "project": job.project_name,
            "status": job.status,
            "result": job.result,
            "error": job.error,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
        }
