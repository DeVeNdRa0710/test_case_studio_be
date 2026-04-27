from app.core.logging import logger
from app.models.graph import GraphExtraction
from app.prompts import GRAPH_EXTRACTION_SYSTEM, GRAPH_EXTRACTION_USER
from app.services.llm import get_llm
from app.utils.json_io import extract_json


class GraphExtractor:
    def __init__(self) -> None:
        self._llm = get_llm()

    async def extract(
        self, *, text: str, source_type: str, modules: list[str] | None = None
    ) -> GraphExtraction:
        if not text.strip():
            return GraphExtraction()

        system = GRAPH_EXTRACTION_SYSTEM
        user = GRAPH_EXTRACTION_USER.format(
            modules=", ".join(modules or []) or "(none)",
            source_type=source_type,
            text=text[:12000],
        )
        raw = await self._llm.complete(system, user, json_mode=True)
        try:
            data = extract_json(raw)
            if isinstance(data, list):
                data = {"nodes": [], "relationships": data}
            return GraphExtraction(**data)
        except Exception as exc:
            logger.warning(f"Graph extraction parse failed: {exc}")
            return GraphExtraction()


_graph_extractor: GraphExtractor | None = None


def get_graph_extractor() -> GraphExtractor:
    global _graph_extractor
    if _graph_extractor is None:
        _graph_extractor = GraphExtractor()
    return _graph_extractor
