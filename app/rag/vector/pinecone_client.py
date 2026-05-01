from __future__ import annotations

import asyncio
from typing import Any, Sequence

from pinecone import Pinecone, ServerlessSpec

from app.core.config import settings
from app.core.logging import logger
from app.core.resilience import pinecone_breaker, retryable_external, with_breaker
from app.models.chunk import Chunk, RetrievedChunk


def _sanitize_metadata(md: dict[str, Any]) -> dict[str, Any]:
    """Pinecone rejects null values and nested objects. Drop None and stringify
    anything that's not a primitive / list of strings."""
    out: dict[str, Any] = {}
    for k, v in md.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, list):
            cleaned = [str(x) for x in v if x is not None]
            if cleaned:
                out[k] = cleaned
        else:
            out[k] = str(v)
    return out


class PineconeClient:
    """Thin async-friendly wrapper over Pinecone's sync SDK."""

    def __init__(self) -> None:
        self._pc: Pinecone | None = None
        self._index = None

    def _ensure_client(self) -> None:
        if self._pc is None:
            if not settings.pinecone_api_key:
                raise RuntimeError("PINECONE_API_KEY is not configured")
            self._pc = Pinecone(api_key=settings.pinecone_api_key)

    async def connect(self) -> None:
        await asyncio.to_thread(self._connect_sync)

    def _connect_sync(self) -> None:
        self._ensure_client()
        assert self._pc is not None
        existing = {i["name"] for i in self._pc.list_indexes()}
        if settings.pinecone_index not in existing:
            logger.info(f"Creating Pinecone index '{settings.pinecone_index}'")
            self._pc.create_index(
                name=settings.pinecone_index,
                dimension=settings.pinecone_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.pinecone_cloud,
                    region=settings.pinecone_region,
                ),
            )
        self._index = self._pc.Index(settings.pinecone_index)
        logger.info(f"Pinecone index '{settings.pinecone_index}' ready")

    async def close(self) -> None:
        self._index = None
        self._pc = None

    @retryable_external()
    @with_breaker(pinecone_breaker, "Pinecone")
    async def upsert_chunks(
        self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]], namespace: str = "default"
    ) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")

        vectors = [
            {
                "id": c.id,
                "values": list(vec),
                "metadata": _sanitize_metadata({**c.metadata, "text": c.text}),
            }
            for c, vec in zip(chunks, embeddings)
        ]

        def _upsert() -> int:
            if self._index is None:
                raise RuntimeError("Pinecone index not initialized")
            batch = 100
            count = 0
            for i in range(0, len(vectors), batch):
                self._index.upsert(vectors=vectors[i : i + batch], namespace=namespace)
                count += len(vectors[i : i + batch])
            return count

        return await asyncio.to_thread(_upsert)

    @retryable_external()
    @with_breaker(pinecone_breaker, "Pinecone")
    async def delete_namespace(self, namespace: str) -> int:
        """
        Delete every vector in a namespace. Returns best-effort count of vectors
        removed (0 if the namespace didn't exist).
        """
        def _delete() -> int:
            if self._index is None:
                raise RuntimeError("Pinecone index not initialized")
            count = 0
            try:
                stats = self._index.describe_index_stats() or {}
                ns_stats = (stats.get("namespaces") or {}).get(namespace) or {}
                count = int(ns_stats.get("vector_count", 0))
            except Exception as exc:
                logger.debug(f"describe_index_stats failed: {exc}")
            try:
                self._index.delete(delete_all=True, namespace=namespace)
            except Exception as exc:
                msg = str(exc)
                if "Namespace not found" in msg or "404" in msg:
                    return 0
                raise
            return count

        return await asyncio.to_thread(_delete)

    @retryable_external()
    @with_breaker(pinecone_breaker, "Pinecone")
    async def query(
        self,
        vector: Sequence[float],
        top_k: int = 6,
        namespace: str = "default",
        filter_: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        def _query() -> list[RetrievedChunk]:
            if self._index is None:
                raise RuntimeError("Pinecone index not initialized")
            res = self._index.query(
                vector=list(vector),
                top_k=top_k,
                include_metadata=True,
                namespace=namespace,
                filter=filter_ or None,
            )
            out: list[RetrievedChunk] = []
            for match in res.get("matches", []):
                md = match.get("metadata") or {}
                text = md.pop("text", "")
                out.append(
                    RetrievedChunk(
                        id=str(match.get("id")),
                        text=text,
                        score=float(match.get("score", 0.0)),
                        metadata=md,
                    )
                )
            return out

        return await asyncio.to_thread(_query)


_pinecone_client: PineconeClient | None = None


def get_pinecone_client() -> PineconeClient:
    global _pinecone_client
    if _pinecone_client is None:
        _pinecone_client = PineconeClient()
    return _pinecone_client
