from typing import Any

from app.core.config import settings
from app.core.logging import logger
from app.models.chunk import RetrievedChunk
from app.rag.graph.graph_store import get_graph_store
from app.rag.vector.vector_store import get_vector_store


class HybridRetriever:
    def __init__(self) -> None:
        self._vector = get_vector_store()
        self._graph = get_graph_store()

    async def retrieve(
        self,
        query: str,
        *,
        project: str,
        modules: list[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        k = top_k or settings.top_k
        filter_: dict | None = None
        if modules:
            filter_ = {"module": {"$in": modules}}

        try:
            docs = await self._vector.search(
                query, top_k=k, namespace=project, filter_=filter_
            )
        except Exception as exc:
            logger.warning(f"Vector search failed: {exc}")
            docs = []

        keywords = _extract_keywords(query)
        try:
            graph = await self._graph.dependencies(
                project=project, modules=modules, keywords=keywords, depth=2
            )
        except Exception as exc:
            logger.warning(f"Graph lookup failed: {exc}")
            graph = {"nodes": [], "relationships": []}

        return {
            "documents": docs,
            "graph": graph,
            "merged_context": _merge_context(docs, graph),
        }


def _extract_keywords(query: str) -> list[str]:
    import re

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", query)
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        low = t.lower()
        if low in seen or low in _STOPWORDS:
            continue
        seen.add(low)
        out.append(t)
    return out[:8]


_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "that", "this",
    "should", "must", "will", "when", "what", "which", "about", "generate",
    "create", "please", "test", "cases", "case", "flow",
}


def _merge_context(docs: list[RetrievedChunk], graph: dict[str, Any]) -> str:
    doc_section = "\n\n".join(
        f"[doc {i + 1} | score={d.score:.3f} | module={d.metadata.get('module', '?')}]\n{d.text}"
        for i, d in enumerate(docs)
    ) or "(no documents)"

    node_lines = [
        f"- ({', '.join(n.get('labels', []))}) {n.get('name')}"
        for n in graph.get("nodes", [])
    ] or ["(no nodes)"]
    rel_lines = [
        f"- {r.get('start')} -[{r.get('type')} {_fmt_rel_props(r.get('properties', {}))}]-> {r.get('end')}"
        for r in graph.get("relationships", [])
    ] or ["(no relationships)"]

    graph_section = "Nodes:\n" + "\n".join(node_lines) + "\n\nRelationships:\n" + "\n".join(rel_lines)
    return f"## Documents\n{doc_section}\n\n## Graph\n{graph_section}"


def _fmt_rel_props(props: dict[str, Any]) -> str:
    kept = {k: v for k, v in props.items() if k in {"condition", "action", "trigger_point"} and v}
    if not kept:
        return ""
    return "{" + ", ".join(f"{k}={v!r}" for k, v in kept.items()) + "}"


_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
