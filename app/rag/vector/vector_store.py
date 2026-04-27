from typing import Sequence

from app.core.config import settings
from app.core.logging import logger
from app.models.chunk import Chunk, RetrievedChunk
from app.rag.vector.pinecone_client import get_pinecone_client
from app.services.cache import get_or_set, make_key
from app.services.embeddings import get_embedding_service


class VectorStore:
    def __init__(self) -> None:
        self._pc = get_pinecone_client()
        self._embed = get_embedding_service()

    async def index_chunks(self, chunks: Sequence[Chunk], namespace: str) -> int:
        if not chunks:
            return 0
        vectors = await self._embed.embed([c.text for c in chunks])
        return await self._pc.upsert_chunks(chunks, vectors, namespace=namespace)

    async def search(
        self,
        query: str,
        top_k: int = 6,
        namespace: str = "default",
        filter_: dict | None = None,
    ) -> list[RetrievedChunk]:
        vec, hit = await get_or_set(
            make_key("embed", settings.embeddings_provider, settings.gemini_embed_model, query),
            settings.cache_embed_ttl,
            lambda: self._embed.embed_one(query),
        )
        if hit:
            logger.debug("embed cache hit")
        if not vec:
            return []
        return await self._pc.query(vec, top_k=top_k, namespace=namespace, filter_=filter_)

    async def delete_namespace(self, namespace: str) -> int:
        return await self._pc.delete_namespace(namespace)


_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
