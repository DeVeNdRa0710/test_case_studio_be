from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logging import logger


class AppError(Exception):
    status_code: int = 500
    code: str = "app_error"

    def __init__(self, message: str, *, status_code: int | None = None, code: str | None = None):
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code


class IngestionError(AppError):
    status_code = 400
    code = "ingestion_error"


class RetrievalError(AppError):
    status_code = 500
    code = "retrieval_error"


class GenerationError(AppError):
    status_code = 500
    code = "generation_error"


class ExternalServiceUnavailable(AppError):
    status_code = 503
    code = "external_service_unavailable"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        logger.error(f"AppError [{exc.code}]: {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(Exception)
    async def handle_unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "Internal server error"}},
        )
