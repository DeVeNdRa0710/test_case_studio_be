import json
import uuid
from typing import Any

from app.core.exceptions import IngestionError
from app.core.logging import logger
from app.models.chunk import Chunk
from app.models.graph import GraphExtraction, GraphNode, GraphRelationship, NodeLabel, RelType
from app.rag.graph.graph_store import get_graph_store
from app.rag.vector.vector_store import get_vector_store
from app.schemas.ingestion import (
    ApiSpecIngestRequest,
    FigmaIngestRequest,
    RequirementIngestRequest,
)
from app.services.graph_extractor import get_graph_extractor
from app.utils.chunker import chunk_text


class IngestionService:
    def __init__(self) -> None:
        self._vector = get_vector_store()
        self._graph = get_graph_store()
        self._extractor = get_graph_extractor()

    async def ingest_requirement(
        self, payload: RequirementIngestRequest
    ) -> tuple[str, int, int, int]:
        if not payload.content.strip():
            raise IngestionError("Requirement content is empty")

        doc_id = f"req_{uuid.uuid4().hex[:10]}"
        metadata = {
            "type": "requirement",
            "project": payload.project,
            "module": payload.module,
            "title": payload.title,
            "source": payload.source or "api",
            **payload.metadata,
        }
        chunks = chunk_text(payload.content, doc_id=doc_id, metadata=metadata)
        indexed = await self._vector.index_chunks(chunks, namespace=payload.project)
        logger.info(f"[{payload.project}] Indexed {indexed} chunks for {doc_id}")

        extraction = await self._extractor.extract(
            text=payload.content, source_type="requirement", modules=[payload.module]
        )
        extraction = _ensure_module(extraction, payload.module)
        nodes, rels = await self._graph.upsert(extraction, project=payload.project)
        return doc_id, indexed, nodes, rels

    async def ingest_requirement_text(
        self,
        *,
        project: str,
        module: str,
        title: str,
        content: str,
        source: str | None = None,
    ) -> tuple[str, int, int, int]:
        return await self.ingest_requirement(
            RequirementIngestRequest(
                project=project, module=module, title=title, content=content, source=source
            )
        )

    async def ingest_figma(self, payload: FigmaIngestRequest) -> tuple[str, int, int, int]:
        flat = _flatten_figma(payload.figma_json)
        body = (
            f"UI Screen: {payload.screen_name}\n"
            f"Module: {payload.module}\n\n"
            f"Elements:\n{flat}"
        )
        doc_id = f"ui_{uuid.uuid4().hex[:10]}"
        metadata = {
            "type": "ui",
            "project": payload.project,
            "module": payload.module,
            "screen_name": payload.screen_name,
            **payload.metadata,
        }
        chunks = chunk_text(body, doc_id=doc_id, metadata=metadata)
        indexed = await self._vector.index_chunks(chunks, namespace=payload.project)

        extraction = await self._extractor.extract(
            text=body, source_type="figma", modules=[payload.module]
        )
        extraction = _ensure_module(extraction, payload.module)
        extraction.nodes.append(
            GraphNode(
                label=NodeLabel.UI_SCREEN,
                name=payload.screen_name,
                properties={"module": payload.module},
            )
        )
        extraction.relationships.append(
            GraphRelationship(
                type=RelType.DEPENDS_ON,
                from_label=NodeLabel.UI_SCREEN,
                from_name=payload.screen_name,
                to_label=NodeLabel.MODULE,
                to_name=payload.module,
            )
        )
        nodes, rels = await self._graph.upsert(extraction, project=payload.project)
        return doc_id, indexed, nodes, rels

    async def ingest_api_spec(self, payload: ApiSpecIngestRequest) -> tuple[str, int, int, int]:
        spec = payload.spec or {}
        doc_id = f"api_{uuid.uuid4().hex[:10]}"
        endpoints = _flatten_openapi(spec)
        body = (
            f"API Specification for module: {payload.module}\n"
            f"Description: {payload.description or ''}\n\n"
            f"Endpoints:\n{endpoints}"
        )
        metadata = {
            "type": "api_spec",
            "project": payload.project,
            "module": payload.module,
            **payload.metadata,
        }
        chunks = chunk_text(body, doc_id=doc_id, metadata=metadata)
        indexed = await self._vector.index_chunks(chunks, namespace=payload.project)

        extraction = await self._extractor.extract(
            text=body, source_type="api_spec", modules=[payload.module]
        )
        extraction = _ensure_module(extraction, payload.module)
        for ep in _iter_endpoints(spec):
            api_name = f"{ep['method']} {ep['path']}"
            extraction.nodes.append(
                GraphNode(
                    label=NodeLabel.API,
                    name=api_name,
                    properties={
                        "method": ep["method"],
                        "path": ep["path"],
                        "module": payload.module,
                    },
                )
            )
            extraction.relationships.append(
                GraphRelationship(
                    type=RelType.DEPENDS_ON,
                    from_label=NodeLabel.API,
                    from_name=api_name,
                    to_label=NodeLabel.MODULE,
                    to_name=payload.module,
                )
            )
        nodes, rels = await self._graph.upsert(extraction, project=payload.project)
        return doc_id, indexed, nodes, rels


def _ensure_module(extraction: GraphExtraction, module: str) -> GraphExtraction:
    if not any(
        n.label == NodeLabel.MODULE and n.name.lower() == module.lower()
        for n in extraction.nodes
    ):
        extraction.nodes.append(GraphNode(label=NodeLabel.MODULE, name=module))
    return extraction


def _flatten_figma(node: Any, depth: int = 0, lines: list[str] | None = None) -> str:
    if lines is None:
        lines = []
    if isinstance(node, dict):
        name = node.get("name")
        ntype = node.get("type")
        if name or ntype:
            lines.append(f"{'  ' * depth}- {ntype or 'node'}: {name or ''}")
        for key in ("children", "components", "frames"):
            child = node.get(key)
            if isinstance(child, list):
                for c in child:
                    _flatten_figma(c, depth + 1, lines)
    elif isinstance(node, list):
        for c in node:
            _flatten_figma(c, depth, lines)
    return "\n".join(lines) if depth == 0 else ""


def _iter_endpoints(spec: dict[str, Any]):
    paths = spec.get("paths") or {}
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, op in methods.items():
            if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                continue
            yield {
                "method": method.upper(),
                "path": path,
                "summary": (op or {}).get("summary", ""),
                "operationId": (op or {}).get("operationId", ""),
            }


def _flatten_openapi(spec: dict[str, Any]) -> str:
    parts: list[str] = []
    for ep in _iter_endpoints(spec):
        parts.append(
            f"- {ep['method']} {ep['path']} — {ep['summary']} ({ep['operationId']})"
        )
    if not parts:
        parts.append(json.dumps(spec)[:2000])
    return "\n".join(parts)


_ingestion_service: IngestionService | None = None


def get_ingestion_service() -> IngestionService:
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = IngestionService()
    return _ingestion_service
