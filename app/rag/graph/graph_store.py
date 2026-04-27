from typing import Any

from app.models.graph import GraphExtraction
from app.rag.graph.neo4j_client import get_neo4j_client


class GraphStore:
    def __init__(self) -> None:
        self._client = get_neo4j_client()

    async def upsert(self, extraction: GraphExtraction, project: str) -> tuple[int, int]:
        return await self._client.upsert_extraction(extraction, project=project)

    async def dependencies(
        self,
        project: str,
        modules: list[str] | None = None,
        keywords: list[str] | None = None,
        depth: int = 2,
        limit: int = 50,
    ) -> dict[str, Any]:
        return await self._client.fetch_dependencies(
            project=project, modules=modules, keywords=keywords, depth=depth, limit=limit
        )

    async def delete_project(self, project: str) -> tuple[int, int]:
        return await self._client.delete_project(project)


_graph_store: GraphStore | None = None


def get_graph_store() -> GraphStore:
    global _graph_store
    if _graph_store is None:
        _graph_store = GraphStore()
    return _graph_store
