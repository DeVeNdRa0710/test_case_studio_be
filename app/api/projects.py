from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.core.logging import logger
from app.db.models import Document as DocumentRow
from app.db.session import SessionLocal
from app.services.project_service import (
    ProjectAlreadyExistsError,
    ProjectError,
    ProjectNotFoundError,
    get_project_service,
)

router = APIRouter(prefix="/projects", tags=["projects"])


class Project(BaseModel):
    name: str
    description: str | None = None
    created_at: str


class ProjectListResponse(BaseModel):
    ok: bool = True
    projects: list[Project]


class CreateProjectRequest(BaseModel):
    name: str = Field(
        ...,
        description="Project identifier. Must be 1–50 chars, lowercase alphanumerics + dashes.",
        examples=["project-one"],
    )
    description: str | None = Field(
        default=None,
        description="Optional short description shown in the Projects list.",
    )


class DeleteProjectResponse(BaseModel):
    ok: bool = True
    name: str
    vectors_deleted: int
    nodes_deleted: int
    rels_deleted: int


class IngestedDocument(BaseModel):
    id: str
    project_name: str
    kind: str
    module: str | None = None
    title: str | None = None
    source: str | None = None
    chunks_indexed: int
    nodes_upserted: int
    relationships_upserted: int
    created_at: datetime
    extra: dict[str, Any] | None = None


class DocumentListResponse(BaseModel):
    ok: bool = True
    project: str
    total: int
    documents: list[IngestedDocument]


@router.get(
    "",
    response_model=ProjectListResponse,
    summary="List all projects",
)
async def list_projects() -> ProjectListResponse:
    svc = get_project_service()
    rows = await svc.list_projects_async()
    return ProjectListResponse(projects=[Project(**p) for p in rows])


@router.post(
    "",
    response_model=Project,
    status_code=201,
    summary="Create a new project",
    description=(
        "Creates an isolated tenant. The project's name becomes the Pinecone "
        "namespace and a `project` property on every Neo4j node created "
        "under it, enabling clean deletion later."
    ),
)
async def create_project(payload: CreateProjectRequest) -> Project:
    svc = get_project_service()
    try:
        project = await svc.create_async(payload.name, payload.description)
    except ProjectAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ProjectError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Project(**project)


@router.get(
    "/{name}/documents",
    response_model=DocumentListResponse,
    summary="List all ingested documents in a project",
    description=(
        "Returns every requirement / figma / api_spec document recorded for the project, "
        "newest first. Use the optional `kind` query param to filter by source type."
    ),
)
async def list_project_documents(
    name: str,
    kind: str | None = None,
    limit: int = 200,
) -> DocumentListResponse:
    project = await get_project_service().require_async(name)
    limit = max(1, min(limit, 500))

    async with SessionLocal() as s:
        stmt = select(DocumentRow).where(DocumentRow.project_name == project)
        if kind:
            stmt = stmt.where(DocumentRow.kind == kind)
        stmt = stmt.order_by(DocumentRow.created_at.desc()).limit(limit)
        result = await s.execute(stmt)
        rows = result.scalars().all()

    docs = [
        IngestedDocument(
            id=r.id,
            project_name=r.project_name,
            kind=r.kind,
            module=r.module,
            title=r.title,
            source=r.source,
            chunks_indexed=r.chunks_indexed,
            nodes_upserted=r.nodes_upserted,
            relationships_upserted=r.relationships_upserted,
            created_at=r.created_at,
            extra=r.extra,
        )
        for r in rows
    ]
    return DocumentListResponse(project=project, total=len(docs), documents=docs)


@router.delete(
    "/{name}",
    response_model=DeleteProjectResponse,
    summary="Delete a project and ALL its data (cascade)",
    description=(
        "Cascade-deletes the project:\n"
        "- Pinecone: deletes the entire namespace (all embeddings for this project).\n"
        "- Neo4j: `MATCH (n {project: $name}) DETACH DELETE n`.\n"
        "- Registry: removes the project entry.\n\n"
        "The `default` project cannot be deleted."
    ),
)
async def delete_project(name: str) -> DeleteProjectResponse:
    svc = get_project_service()
    try:
        stats = await svc.delete(name)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ProjectError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Project delete failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}") from exc
    return DeleteProjectResponse(
        name=name,
        vectors_deleted=stats.get("vectors_deleted", 0),
        nodes_deleted=stats.get("nodes_deleted", 0),
        rels_deleted=stats.get("rels_deleted", 0),
    )
