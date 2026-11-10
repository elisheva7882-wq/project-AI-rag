"""Classify provider failures using structured fields and transport types only.

Messages, request payloads, and exception chains are deliberately not inspected.
Sanitized wrappers may retain one of the explicitly supported public kinds.
"""

from __future__ import annotations

import httpx
import requests


_PUBLIC_KINDS = frozenset((
    "access_denied",
    "model_unavailable",
    "rate_limit",
    "service_timeout",
    "connection_error",
    "service_unavailable",
))
_STATUS_KINDS = {
    401: "access_denied",
    403: "access_denied",
    404: "model_unavailable",
    429: "rate_limit",
    408: "service_timeout",
    504: "service_timeout",
    500: "service_unavailable",
    502: "service_unavailable",
    503: "service_unavailable",
}


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def provider_status_code(error: Exception | None) -> int | None:
    """Read an integer provider/HTTP code without coercing arbitrary values."""
    if error is None:
        return None
    code = getattr(error, "code", None)
    if _integer(code):
        return code
    response_code = getattr(getattr(error, "response", None), "status_code", None)
    return response_code if _integer(response_code) else None


def classify_service_error(error: Exception | None) -> str | None:
    """Return a safe known category, or None when no structured signal exists."""
    if error is None:
        return None
    kind = getattr(error, "kind", None)
    if isinstance(kind, str) and kind in _PUBLIC_KINDS:
        return kind
    status_kind = _STATUS_KINDS.get(provider_status_code(error))
    if status_kind is not None:
        return status_kind
    # requests.ConnectTimeout also inherits ConnectionError; timeout comes first.
    if isinstance(error, (TimeoutError, httpx.TimeoutException, requests.exceptions.Timeout)):
        return "service_timeout"
    if isinstance(error, (ConnectionError, httpx.NetworkError, requests.exceptions.ConnectionError)):
        return "connection_error"
    return None
