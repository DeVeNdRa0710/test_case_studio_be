import asyncio
from typing import Protocol, Sequence

from app.core.config import settings
from app.core.logging import logger
from app.core.resilience import llm_breaker, retryable_external, with_breaker


EMBED_BATCH_SIZE = 100
EMBED_MAX_CONCURRENCY = 5


class Embedder(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
    async def embed_one(self, text: str) -> list[float]: ...


async def _fan_out(
    texts: Sequence[str],
    embed_one_batch,
    *,
    batch_size: int = EMBED_BATCH_SIZE,
    max_concurrency: int = EMBED_MAX_CONCURRENCY,
) -> list[list[float]]:
    if not texts:
        return []
    if len(texts) <= batch_size:
        return await embed_one_batch(list(texts))

    sem = asyncio.Semaphore(max_concurrency)

    async def _run(slice_: list[str]) -> list[list[float]]:
        async with sem:
            return await embed_one_batch(slice_)

    slices = [list(texts[i : i + batch_size]) for i in range(0, len(texts), batch_size)]
    results = await asyncio.gather(*[_run(s) for s in slices])
    out: list[list[float]] = []
    for r in results:
        out.extend(r)
    return out


class GeminiEmbedder:
    def __init__(self) -> None:
        from google import genai
        from google.genai import types

        if not settings.gemini_api_key:
            logger.warning("GEMINI_API_KEY is not set; embeddings will fail at runtime.")
        self._client = genai.Client(api_key=settings.gemini_api_key or "missing")
        self._model = settings.gemini_embed_model
        self._config = types.EmbedContentConfig(
            output_dimensionality=settings.pinecone_dimension
        )

    @retryable_external()
    @with_breaker(llm_breaker, "Embeddings")
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.aio.models.embed_content(
            model=self._model,
            contents=texts,
            config=self._config,
        )
        return [list(e.values) for e in resp.embeddings]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return await _fan_out(texts, self._embed_batch)

    async def embed_one(self, text: str) -> list[float]:
        vectors = await self._embed_batch([text])
        return vectors[0] if vectors else []


class OpenAIEmbedder:
    def __init__(self) -> None:
        from openai import AsyncOpenAI

        if not settings.openai_api_key:
            logger.warning("OPENAI_API_KEY is not set; embeddings will fail at runtime.")
        self._client = AsyncOpenAI(api_key=settings.openai_api_key or "missing")
        self._model = settings.openai_embed_model

    @retryable_external()
    @with_breaker(llm_breaker, "Embeddings")
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return await _fan_out(texts, self._embed_batch)

    async def embed_one(self, text: str) -> list[float]:
        vectors = await self._embed_batch([text])
        return vectors[0] if vectors else []


_embedding_service: Embedder | None = None


def get_embedding_service() -> Embedder:
    global _embedding_service
    if _embedding_service is None:
        provider = settings.embeddings_provider.lower()
        if provider == "openai":
            _embedding_service = OpenAIEmbedder()
        else:
            _embedding_service = GeminiEmbedder()
        logger.info(f"Embeddings provider initialized: {provider}")
    return _embedding_service


# Backwards-compatible alias
EmbeddingService = Embedder
