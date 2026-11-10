"""Explicit Gemini embeddings and a separately labelled offline test model."""

from __future__ import annotations

import hashlib
import math
import os

from llama_index.core.base.embeddings.base import BaseEmbedding

from config import Settings
from src.service_errors import classify_service_error, provider_status_code


OFFLINE_MODEL = "kingdomai-offline-hash-v1"
OFFLINE_DIMENSION = 64


class ProviderError(ValueError):
    """Sanitized provider/configuration errors that never include SDK payloads."""

    def __init__(self, message: str, *, code: int | None = None, kind: str | None = None) -> None:
        super().__init__(message)
        self.code = code if type(code) is int else None
        self.kind = kind


def provider_error_message(error: Exception, *, operation: str = "Embedding") -> str:
    code = provider_status_code(error)
    kind = classify_service_error(error)
    if code in (401, 403):
        return "Gemini rejected access. Check the local API key and account permissions."
    if code == 404:
        return "The configured Gemini model is unavailable for this account; it was not replaced."
    if code == 429:
        from src.rate_limits import retry_advice

        advice = retry_advice(error)
        if advice.quota_zero:
            return "Gemini reports a zero quota for this request. Check the project's active model limits in AI Studio."
        if advice.kind == "daily":
            return "Gemini daily quota is exhausted. Wait for its reset or check the project's active limits in AI Studio."
        return "Gemini quota or rate limit exceeded. No billing or plan change was made."
    if kind == "service_timeout":
        return "Gemini did not complete the request within its time limit. Try again later."
    if kind == "connection_error":
        return "Could not connect to Gemini. Check the connection and try again."
    if kind == "service_unavailable":
        suffix = f" ({code})" if code is not None else ""
        return f"Gemini service is temporarily unavailable{suffix}. Try again later."
    return f"{operation} request failed. Check the connection, model availability and account quota."


def create_embedding_model(settings: Settings) -> BaseEmbedding:
    if not settings.has_api_key:
        raise ProviderError("GEMINI_API_KEY is missing. Set it locally in .env; no offline fallback is automatic.")
    # This installed adapter interprets any nonempty value, even 'false', as
    # Vertex mode and would discard the explicitly supplied API key.
    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI"):
        raise ProviderError("Remove GOOGLE_GENAI_USE_VERTEXAI to use the configured Gemini API provider.")
    from google.genai import types
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

    try:
        return GoogleGenAIEmbedding(
            model_name=settings.embedding_model,
            api_key=settings.gemini_api_key,
            embedding_config={"output_dimensionality": settings.embedding_dimension},
            embed_batch_size=10,
            retries=1,
            retry_min_seconds=1,
            retry_max_seconds=2,
            http_options=types.HttpOptions(
                timeout=60_000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
    except Exception as exc:
        raise ProviderError(provider_error_message(exc), code=provider_status_code(exc), kind=classify_service_error(exc)) from None


def create_llm(settings: Settings):
    """Construct the explicit chat adapter without a model-metadata API call."""
    if not settings.has_api_key:
        raise ProviderError("GEMINI_API_KEY is missing. Set it locally in .env before live questions.")
    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI"):
        raise ProviderError("Remove GOOGLE_GENAI_USE_VERTEXAI to use the configured Gemini API provider.")
    from google.genai import types
    from llama_index.llms.google_genai import GoogleGenAI
    from src.prompts import RESPONSE_SCHEMA

    try:
        return GoogleGenAI(
            model=settings.llm_model,
            api_key=settings.gemini_api_key,
            max_tokens=4096,
            context_window=16384,
            max_retries=1,
            file_mode="inline",
            built_in_tool=None,
            is_function_calling_model=False,
            http_options=types.HttpOptions(
                timeout=round(settings.llm_request_timeout_seconds * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
            generation_config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
                max_output_tokens=4096,
                thinking_config=types.ThinkingConfig(
                    thinking_level=settings.llm_thinking_level,
                    include_thoughts=False,
                ),
                tools=[],
                automatic_continuation=False,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True,
                    maximum_remote_calls=None,
                ),
            ),
        )
    except Exception as exc:
        raise ProviderError(provider_error_message(exc, operation="Gemini"), code=provider_status_code(exc), kind=classify_service_error(exc)) from None


class OfflineEmbedding(BaseEmbedding):
    """Deterministic test vectors. They do not measure semantic relevance."""

    model_name: str = OFFLINE_MODEL

    def _get_text_embedding(self, text: str) -> list[float]:
        raw = hashlib.sha512(text.encode("utf-8")).digest()
        values = [(value - 127.5) / 127.5 for value in raw]
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)


def create_offline_embedding() -> BaseEmbedding:
    return OfflineEmbedding(embed_batch_size=10)
