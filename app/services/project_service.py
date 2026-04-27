"""
Project registry + cascade deletion.

A project is a tenant-like scope. Its name is used as:
  - the Pinecone namespace
  - a `project` property stamped on every Neo4j node
  - a metadata field on every embedded chunk
so deletion can cascade cleanly across stores.
"""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.exceptions import IngestionError
from app.core.logging import logger
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


class ProjectService:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or Path(__file__).resolve().parents[2] / "data" / "projects.json"
        self._lock = threading.Lock()
        self._ensure_file()

    def _ensure_file(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text(json.dumps({"projects": []}, indent=2))

    def _read(self) -> dict[str, Any]:
        try:
            return json.loads(self._path.read_text())
        except Exception:
            return {"projects": []}

    def _write(self, data: dict[str, Any]) -> None:
        self._path.write_text(json.dumps(data, indent=2))

    def list_projects(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._read().get("projects", []))

    def get(self, name: str) -> dict[str, Any]:
        n = normalize_project_name(name)
        with self._lock:
            for p in self._read().get("projects", []):
                if p["name"] == n:
                    return p
        raise ProjectNotFoundError(f"Project '{n}' does not exist")

    def exists(self, name: str) -> bool:
        try:
            self.get(name)
            return True
        except ProjectNotFoundError:
            return False

    def require(self, name: str) -> str:
        """Return the normalized name and raise if the project doesn't exist."""
        n = normalize_project_name(name)
        if not self.exists(n):
            raise ProjectNotFoundError(f"Project '{n}' does not exist")
        return n

    def create(self, name: str, description: str | None = None) -> dict[str, Any]:
        n = normalize_project_name(name)
        with self._lock:
            data = self._read()
            if any(p["name"] == n for p in data.get("projects", [])):
                raise ProjectAlreadyExistsError(f"Project '{n}' already exists")
            project = {
                "name": n,
                "description": (description or "").strip() or None,
                "created_at": _now_iso(),
            }
            data.setdefault("projects", []).append(project)
            self._write(data)
            logger.info(f"Created project '{n}'")
            return project

    async def delete(self, name: str) -> dict[str, int]:
        """Cascade delete: Pinecone namespace + Neo4j nodes + registry entry."""
        n = normalize_project_name(name)
        if not self.exists(n):
            raise ProjectNotFoundError(f"Project '{n}' does not exist")

        stats = {"vectors_deleted": 0, "nodes_deleted": 0, "rels_deleted": 0}

        vector_store = get_vector_store()
        try:
            stats["vectors_deleted"] = await vector_store.delete_namespace(n)
        except Exception as exc:
            logger.warning(f"Pinecone delete for project '{n}' failed: {exc}")

        graph_store = get_graph_store()
        try:
            nodes, rels = await graph_store.delete_project(n)
            stats["nodes_deleted"] = nodes
            stats["rels_deleted"] = rels
        except Exception as exc:
            logger.warning(f"Neo4j delete for project '{n}' failed: {exc}")

        with self._lock:
            data = self._read()
            data["projects"] = [p for p in data.get("projects", []) if p["name"] != n]
            self._write(data)

        logger.info(
            f"Deleted project '{n}': vectors={stats['vectors_deleted']} "
            f"nodes={stats['nodes_deleted']} rels={stats['rels_deleted']}"
        )
        return stats


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


_project_service: ProjectService | None = None


def get_project_service() -> ProjectService:
    global _project_service
    if _project_service is None:
        _project_service = ProjectService()
    return _project_service
