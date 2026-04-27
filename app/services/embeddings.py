from typing import Protocol, Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import logger


class Embedder(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
    async def embed_one(self, text: str) -> list[float]: ...


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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await self._client.aio.models.embed_content(
            model=self._model,
            contents=list(texts),
            config=self._config,
        )
        return [list(e.values) for e in resp.embeddings]

    async def embed_one(self, text: str) -> list[float]:
        vectors = await self.embed([text])
        return vectors[0] if vectors else []


class OpenAIEmbedder:
    def __init__(self) -> None:
        from openai import AsyncOpenAI

        if not settings.openai_api_key:
            logger.warning("OPENAI_API_KEY is not set; embeddings will fail at runtime.")
        self._client = AsyncOpenAI(api_key=settings.openai_api_key or "missing")
        self._model = settings.openai_embed_model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await self._client.embeddings.create(model=self._model, input=list(texts))
        return [d.embedding for d in resp.data]

    async def embed_one(self, text: str) -> list[float]:
        vectors = await self.embed([text])
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
