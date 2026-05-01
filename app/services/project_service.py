"""
Project registry + cascade deletion (DB-backed).

A project is a tenant-like scope. Its name is used as:
  - the Pinecone namespace
  - a `project` property stamped on every Neo4j node
  - a metadata field on every embedded chunk
so deletion can cascade cleanly across stores.

Public API (sync wrappers around the async DB) is preserved so callers do
not need to change.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from app.core.exceptions import IngestionError
from app.core.logging import logger
from app.db.models import AuditEvent, Project as ProjectModel
from app.db.session import SessionLocal
from app.rag.graph.graph_store import get_graph_store
from app.rag.vector.vector_store import get_vector_store

DEFAULT_PROJECT = "default"
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,48}[a-z0-9]$|^[a-z0-9]$")


class ProjectError(IngestionError):
    pass


class ProjectNotFoundError(ProjectError):
    pass


class ProjectAlreadyExistsError(ProjectError):
    pass


def normalize_project_name(raw: str) -> str:
    if not isinstance(raw, str):
        raise ProjectError("Project name must be a string")
    name = raw.strip().lower().replace(" ", "-").replace("_", "-")
    name = re.sub(r"-+", "-", name).strip("-")
    if not name or not _NAME_RE.match(name):
        raise ProjectError(
            "Project name must be 1–50 chars, lowercase alphanumerics/dash only "
            "(e.g. 'project-one')"
        )
    return name


def _to_dict(p: ProjectModel) -> dict[str, Any]:
    created = p.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return {
        "name": p.name,
        "description": p.description,
        "created_at": created.isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


class ProjectService:
    """All public methods are async; FastAPI handlers await them directly."""

    @staticmethod
    async def list_projects_async() -> list[dict[str, Any]]:
        async with SessionLocal() as s:
            rows = (await s.execute(select(ProjectModel).order_by(ProjectModel.created_at))).scalars().all()
            return [_to_dict(p) for p in rows]

    @staticmethod
    async def get_async(name: str) -> dict[str, Any]:
        n = normalize_project_name(name)
        async with SessionLocal() as s:
            p = await s.scalar(select(ProjectModel).where(ProjectModel.name == n))
            if not p:
                raise ProjectNotFoundError(f"Project '{n}' does not exist")
            return _to_dict(p)

    @staticmethod
    async def exists_async(name: str) -> bool:
        try:
            await ProjectService.get_async(name)
            return True
        except ProjectNotFoundError:
            return False

    @staticmethod
    async def create_async(name: str, description: str | None = None) -> dict[str, Any]:
        n = normalize_project_name(name)
        desc = (description or "").strip() or None
        async with SessionLocal() as s:
            existing = await s.scalar(select(ProjectModel).where(ProjectModel.name == n))
            if existing:
                raise ProjectAlreadyExistsError(f"Project '{n}' already exists")
            p = ProjectModel(name=n, description=desc)
            s.add(p)
            s.add(AuditEvent(project_name=n, action="project.create", target=n, payload={"description": desc}))
            await s.commit()
            await s.refresh(p)
            logger.info(f"Created project '{n}'")
            return _to_dict(p)

    async def delete(self, name: str) -> dict[str, int]:
        n = normalize_project_name(name)
        async with SessionLocal() as s:
            p = await s.scalar(select(ProjectModel).where(ProjectModel.name == n))
            if not p:
                raise ProjectNotFoundError(f"Project '{n}' does not exist")

        stats = {"vectors_deleted": 0, "nodes_deleted": 0, "rels_deleted": 0}

        try:
            stats["vectors_deleted"] = await get_vector_store().delete_namespace(n)
        except Exception as exc:
            logger.warning(f"Pinecone delete for project '{n}' failed: {exc}")

        try:
            nodes, rels = await get_graph_store().delete_project(n)
            stats["nodes_deleted"] = nodes
            stats["rels_deleted"] = rels
        except Exception as exc:
            logger.warning(f"Neo4j delete for project '{n}' failed: {exc}")

        async with SessionLocal() as s:
            p = await s.scalar(select(ProjectModel).where(ProjectModel.name == n))
            if p:
                await s.delete(p)
            s.add(AuditEvent(project_name=n, action="project.delete", target=n, payload=stats))
            await s.commit()

        logger.info(
            f"Deleted project '{n}': vectors={stats['vectors_deleted']} "
            f"nodes={stats['nodes_deleted']} rels={stats['rels_deleted']}"
        )
        return stats

    # ----------- async require helper (called from FastAPI handlers) -----------

    async def require_async(self, name: str) -> str:
        n = normalize_project_name(name)
        if not await self.exists_async(n):
            raise ProjectNotFoundError(f"Project '{n}' does not exist")
        return n


_project_service: ProjectService | None = None


def get_project_service() -> ProjectService:
    global _project_service
    if _project_service is None:
        _project_service = ProjectService()
    return _project_service
