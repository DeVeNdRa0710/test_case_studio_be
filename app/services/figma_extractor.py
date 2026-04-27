from typing import Any

from app.core.exceptions import GenerationError
from app.core.logging import logger
from app.prompts import FIGMA_EXTRACTION_SYSTEM, FIGMA_EXTRACTION_USER
from app.services.vision import get_vision
from app.utils.json_io import extract_json


class FigmaScreenshotExtractor:
    def __init__(self) -> None:
        self._vision = get_vision()

    async def extract(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        screen_name: str,
        module: str,
    ) -> dict[str, Any]:
        system = FIGMA_EXTRACTION_SYSTEM
        user = FIGMA_EXTRACTION_USER.format(
            screen_name=screen_name or "Screen",
            module=module or "(unspecified)",
        )
        raw = await self._vision.complete_with_image(
            system=system,
            user=user,
            image_bytes=image_bytes,
            mime_type=mime_type,
            json_mode=True,
        )
        try:
            data = extract_json(raw)
        except Exception as exc:
            logger.error(f"Vision returned non-JSON Figma tree: {exc}\nRaw: {raw[:500]}")
            raise GenerationError("Could not parse UI tree from screenshot") from exc

        if not isinstance(data, dict) or "name" not in data or "type" not in data:
            raise GenerationError("Extracted tree missing required keys (name, type)")

        # Normalize: force root name = provided screen_name and type = FRAME
        if screen_name:
            data["name"] = screen_name
        data.setdefault("type", "FRAME")
        data.setdefault("children", [])
        return data


_extractor: FigmaScreenshotExtractor | None = None


def get_figma_screenshot_extractor() -> FigmaScreenshotExtractor:
    global _extractor
    if _extractor is None:
        _extractor = FigmaScreenshotExtractor()
    return _extractor
