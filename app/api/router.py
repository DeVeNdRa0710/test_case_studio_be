from fastapi import APIRouter

from app.api import generate, health, ingest, projects

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(projects.router)
api_router.include_router(ingest.router)
api_router.include_router(generate.router)
