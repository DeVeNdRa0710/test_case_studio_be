from __future__ import annotations

from typing import Protocol

from app.core.config import settings
from app.core.logging import logger
from app.core.resilience import llm_breaker, retryable_external, with_breaker


class LLMProvider(Protocol):
    async def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str: ...


class GeminiProvider:
    def __init__(self) -> None:
        from google import genai

        if not settings.gemini_api_key:
            logger.warning("GEMINI_API_KEY is not set")
        self._client = genai.Client(api_key=settings.gemini_api_key or "missing")
        self._model = settings.gemini_model or settings.llm_model

    @retryable_external()
    @with_breaker(llm_breaker, "LLM")
    async def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature if temperature is not None else settings.llm_temperature,
            max_output_tokens=max_tokens or settings.llm_max_tokens,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        resp = await self._client.aio.models.generate_content(
            model=self._model,
            contents=user,
            config=cfg,
        )
        return getattr(resp, "text", "") or ""


class OpenAIProvider:
    def __init__(self) -> None:
        from openai import AsyncOpenAI

        if not settings.openai_api_key:
            logger.warning("OPENAI_API_KEY is not set")
        self._client = AsyncOpenAI(api_key=settings.openai_api_key or "missing")
        self._model = settings.llm_model

    @retryable_external()
    @with_breaker(llm_breaker, "LLM")
    async def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        kwargs: dict = {
            "model": self._model,
            "temperature": temperature if temperature is not None else settings.llm_temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


class GroqProvider:
    def __init__(self) -> None:
        from groq import AsyncGroq

        if not settings.groq_api_key:
            logger.warning("GROQ_API_KEY is not set")
        self._client = AsyncGroq(api_key=settings.groq_api_key or "missing")
        self._model = settings.groq_model

    @retryable_external()
    @with_breaker(llm_breaker, "LLM")
    async def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        kwargs: dict = {
            "model": self._model,
            "temperature": temperature if temperature is not None else settings.llm_temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


_llm: LLMProvider | None = None


def get_llm() -> LLMProvider:
    global _llm
    if _llm is None:
        provider = settings.llm_provider.lower()
        if provider == "gemini":
            _llm = GeminiProvider()
        elif provider == "groq":
            _llm = GroqProvider()
        else:
            _llm = OpenAIProvider()
        logger.info(f"LLM provider initialized: {provider}")
    return _llm
