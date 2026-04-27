import sys

from loguru import logger

from app.core.config import settings


def configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.log_level.upper(),
        backtrace=False,
        diagnose=False,
        enqueue=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
            "| <level>{level:<8}</level> "
            "| <cyan>{name}:{function}:{line}</cyan> "
            "- <level>{message}</level>"
        ),
    )


__all__ = ["configure_logging", "logger"]
