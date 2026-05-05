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
        seed: int | None = None,
        top_p: float | None = None,
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
        seed: int | None = None,
        top_p: float | None = None,
    ) -> str:
        from google.genai import types

        cfg_kwargs: dict = {
            "system_instruction": system,
            "temperature": temperature if temperature is not None else settings.llm_temperature,
            "max_output_tokens": max_tokens or settings.llm_max_tokens,
            "response_mime_type": "application/json" if json_mode else "text/plain",
        }
        if seed is not None:
            cfg_kwargs["seed"] = seed
        if top_p is not None:
            cfg_kwargs["top_p"] = top_p
        cfg = types.GenerateContentConfig(**cfg_kwargs)
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
        seed: int | None = None,
        top_p: float | None = None,
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
        if seed is not None:
            kwargs["seed"] = seed
        if top_p is not None:
            kwargs["top_p"] = top_p
        resp = await self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


class BedrockProvider:
    """AWS Bedrock LLM provider.

    Uses the Converse API so the same code works across Anthropic Claude,
    Meta Llama, Mistral, Amazon Nova, etc. Auth via standard AWS credential
    chain (env vars, ~/.aws/credentials, IAM role).
    """

    def __init__(self) -> None:
        import boto3
        from botocore.config import Config as BotoConfig

        if not settings.aws_region:
            logger.warning("AWS_REGION is not set; Bedrock calls will fail.")

        client_kwargs: dict = {
            "service_name": "bedrock-runtime",
            "region_name": settings.aws_region or "us-east-1",
            "config": BotoConfig(
                retries={"max_attempts": 3, "mode": "standard"},
                read_timeout=120,
                connect_timeout=10,
            ),
        }
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = settings.aws_access_key_id
            client_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
            if settings.aws_session_token:
                client_kwargs["aws_session_token"] = settings.aws_session_token

        self._client = boto3.client(**client_kwargs)
        self._model_id = settings.bedrock_model_id or settings.llm_model

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
        seed: int | None = None,
        top_p: float | None = None,
    ) -> str:
        import asyncio

        user_content = user
        if json_mode:
            user_content = (
                f"{user}\n\nRespond ONLY with valid JSON. No prose, no markdown fences."
            )

        inference_config: dict = {
            "temperature": temperature if temperature is not None else settings.llm_temperature,
            "maxTokens": max_tokens or settings.llm_max_tokens,
        }
        if top_p is not None:
            inference_config["topP"] = top_p

        kwargs: dict = {
            "modelId": self._model_id,
            "messages": [{"role": "user", "content": [{"text": user_content}]}],
            "inferenceConfig": inference_config,
        }
        if system:
            kwargs["system"] = [{"text": system}]

        # boto3 client is sync; offload to a thread to keep the event loop free.
        resp = await asyncio.to_thread(self._client.converse, **kwargs)

        try:
            blocks = resp["output"]["message"]["content"]
            return "".join(b.get("text", "") for b in blocks)
        except (KeyError, TypeError):
            return ""


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
        seed: int | None = None,
        top_p: float | None = None,
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
        if seed is not None:
            kwargs["seed"] = seed
        if top_p is not None:
            kwargs["top_p"] = top_p
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
        elif provider == "bedrock":
            _llm = BedrockProvider()
        else:
            _llm = OpenAIProvider()
        logger.info(f"LLM provider initialized: {provider}")
    return _llm
