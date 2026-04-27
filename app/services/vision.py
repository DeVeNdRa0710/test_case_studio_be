from __future__ import annotations

import base64
import re
from typing import Protocol

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import logger


class LLMQuotaError(RuntimeError):
    """Raised when the LLM provider reports quota exhaustion (429)."""


class LLMModelNotFoundError(RuntimeError):
    """Raised when the configured model is missing/deprecated (404)."""


def _status_code(exc: BaseException) -> int | None:
    """Extract HTTP status code from common SDK exception shapes."""
    for attr in ("status_code", "code", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    resp = getattr(exc, "response", None)
    if resp is not None:
        val = getattr(resp, "status_code", None)
        if isinstance(val, int):
            return val
    return None


def _is_quota_error(exc: BaseException) -> bool:
    """Detect real quota/rate-limit errors without false positives."""
    if isinstance(exc, LLMQuotaError):
        return True
    if _status_code(exc) == 429:
        return True
    msg = str(exc)
    # Require explicit status markers, not substrings like "quota" that may
    # appear incidentally in unrelated error text.
    if re.search(r"\bRESOURCE_EXHAUSTED\b", msg):
        return True
    if re.search(r"\brate[_ -]?limit\b", msg, re.IGNORECASE):
        return True
    return False


def _is_auth_error(exc: BaseException) -> bool:
    code = _status_code(exc)
    if code in (401, 403):
        return True
    msg = str(exc)
    return bool(re.search(r"\b(UNAUTHENTICATED|PERMISSION_DENIED)\b", msg))


def _is_not_found_error(exc: BaseException) -> bool:
    """Detect model-not-found / deprecated-model errors."""
    if isinstance(exc, LLMModelNotFoundError):
        return True
    if _status_code(exc) == 404:
        return True
    msg = str(exc)
    return bool(re.search(r"\bNOT_FOUND\b", msg))


def _not_quota(exc: BaseException) -> bool:
    """Retry predicate: skip retries for permanent errors (quota/auth/404)."""
    if _is_quota_error(exc) or _is_auth_error(exc) or _is_not_found_error(exc):
        return False
    return True


class VisionProvider(Protocol):
    async def complete_with_image(
        self,
        system: str,
        user: str,
        image_bytes: bytes,
        mime_type: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = True,
    ) -> str: ...


class GeminiVisionProvider:
    def __init__(self) -> None:
        from google import genai

        if not settings.gemini_api_key:
            logger.warning("GEMINI_API_KEY is not set")
        self._client = genai.Client(api_key=settings.gemini_api_key or "missing")
        self._model = settings.gemini_model or settings.llm_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception(_not_quota),
    )
    async def complete_with_image(
        self,
        system: str,
        user: str,
        image_bytes: bytes,
        mime_type: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = True,
    ) -> str:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature if temperature is not None else settings.llm_temperature,
            max_output_tokens=max_tokens or settings.llm_max_tokens,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            user,
        ]
        try:
            resp = await self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=cfg,
            )
        except Exception as exc:
            logger.error(
                f"Gemini vision call failed (model={self._model}, "
                f"status={_status_code(exc)}, type={type(exc).__name__}): {exc}"
            )
            if _is_quota_error(exc):
                raise LLMQuotaError(
                    "Gemini quota/rate limit reached. Wait and retry, or switch "
                    "LLM_PROVIDER=openai in the backend .env."
                ) from exc
            if _is_not_found_error(exc):
                raise LLMModelNotFoundError(
                    f"Gemini model '{self._model}' is unavailable or deprecated. "
                    "Update GEMINI_MODEL in the backend .env to a current model "
                    "(e.g. 'gemini-2.5-flash' or 'gemini-flash-latest')."
                ) from exc
            raise
        return getattr(resp, "text", "") or ""


class OpenAIVisionProvider:
    def __init__(self) -> None:
        from openai import AsyncOpenAI

        if not settings.openai_api_key:
            logger.warning("OPENAI_API_KEY is not set")
        self._client = AsyncOpenAI(api_key=settings.openai_api_key or "missing")
        self._model = settings.llm_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception(_not_quota),
    )
    async def complete_with_image(
        self,
        system: str,
        user: str,
        image_bytes: bytes,
        mime_type: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = True,
    ) -> str:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_uri = f"data:{mime_type};base64,{b64}"

        kwargs: dict = {
            "model": self._model,
            "temperature": temperature if temperature is not None else settings.llm_temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            logger.error(
                f"OpenAI vision call failed (model={self._model}, "
                f"status={_status_code(exc)}, type={type(exc).__name__}): {exc}"
            )
            if _is_quota_error(exc):
                raise LLMQuotaError(
                    "OpenAI quota/rate limit reached. Check your plan & billing, "
                    "or switch LLM_PROVIDER in the backend .env."
                ) from exc
            if _is_not_found_error(exc):
                raise LLMModelNotFoundError(
                    f"OpenAI model '{self._model}' is unavailable. Update LLM_MODEL "
                    "in the backend .env to a model your key can access."
                ) from exc
            raise
        return resp.choices[0].message.content or ""


_vision: VisionProvider | None = None


def get_vision() -> VisionProvider:
    """
    Vision provider. Groq currently has no stable vision-capable model in this
    codebase's config, so we fall back to Gemini or OpenAI. If LLM_PROVIDER is
    set to 'groq', we use Gemini vision when a Gemini key is present, else
    OpenAI.
    """
    global _vision
    if _vision is not None:
        return _vision

    provider = settings.llm_provider.lower()
    if provider == "openai":
        _vision = OpenAIVisionProvider()
    elif provider == "groq":
        _vision = (
            GeminiVisionProvider()
            if settings.gemini_api_key
            else OpenAIVisionProvider()
        )
    else:
        _vision = GeminiVisionProvider()

    logger.info(f"Vision provider initialized: {_vision.__class__.__name__}")
    return _vision
