from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.logging import logger
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
