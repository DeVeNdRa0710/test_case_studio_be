import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, logger
from app.rag.graph.neo4j_client import get_neo4j_client
from app.rag.vector.pinecone_client import get_pinecone_client


def allowed_origins() -> list[str]:
    """
    Read FRONTEND_URL from env. Accepts a single URL or a comma-separated list.
    Falls back to common local dev origins.
    """
    raw = os.getenv("FRONTEND_URL", "").strip()
    extra = [o.strip() for o in raw.split(",") if o.strip()]
    defaults = [
        "http://localhost:5173",
        "https://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # Dedupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for o in extra + defaults:
        if o not in seen:
            seen.add(o)
            out.append(o)
    return out


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    logger.info(f"Starting {settings.app_name} ({settings.app_env})")
    pc = get_pinecone_client()
    neo = get_neo4j_client()
    try:
        await pc.connect()
    except Exception as exc:
        logger.error(f"Pinecone init failed: {exc}")
    try:
        await neo.connect()
    except Exception as exc:
        logger.error(f"Neo4j init failed: {exc}")

    yield

    await pc.close()
    await neo.close()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        lifespan=lifespan,
    )

    origins = allowed_origins()
    logger.info(f"CORS allowed origins: {origins}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    app.include_router(api_router)
    return app

app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
    )

