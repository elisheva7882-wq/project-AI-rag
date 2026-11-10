"""Whitelisted quota diagnostics and bounded retry timing, without payload logs."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Literal


@dataclass(frozen=True)
class RetryAdvice:
    kind: Literal["daily", "rate", "unknown"] = "unknown"
    retry_after: float | None = None
    quota_zero: bool = False


def _duration(value) -> float | None:
    if isinstance(value, str):
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)s", value.strip())
        if match:
            seconds = float(match[1])
            return seconds if math.isfinite(seconds) else None
    return None


def retry_advice(error: Exception) -> RetryAdvice:
    """Read only structured quota IDs/delays. Never return message or project IDs."""
    details = getattr(error, "details", None)
    kind: Literal["daily", "rate", "unknown"] = "unknown"
    delays: list[float] = []
    quota_zero = False
    stack = [(details, 0)]
    visited = 0
    while stack and visited < 256:
        value, depth = stack.pop()
        visited += 1
        if depth > 8:
            continue
        if isinstance(value, dict):
            if str(value.get("@type", "")).endswith("RetryInfo"):
                delay = _duration(value.get("retryDelay"))
                if delay is not None:
                    delays.append(delay)
            for field in ("quotaId", "quotaMetric"):
                identifier = value.get(field)
                if isinstance(identifier, str):
                    normalized = identifier.lower().replace("_", "").replace("-", "")
                    if "perday" in normalized or "daily" in normalized:
                        kind = "daily"
                    elif kind != "daily" and "perminute" in normalized:
                        kind = "rate"
            quota_value = value.get("quotaValue")
            if not isinstance(quota_value, bool) and quota_value in (0, "0"):
                quota_zero = True
            stack.extend((child, depth + 1) for child in value.values() if isinstance(child, (dict, list)))
        elif isinstance(value, list):
            stack.extend((child, depth + 1) for child in value[:128])
    headers = getattr(getattr(error, "response", None), "headers", {})
    if hasattr(headers, "get"):
        value = headers.get("Retry-After") or headers.get("retry-after")
        if value is not None:
            try:
                delay = float(value)
            except (TypeError, ValueError):
                try:
                    date = parsedate_to_datetime(value)
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=timezone.utc)
                    delay = max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    delay = -1.0
            if math.isfinite(delay) and delay >= 0:
                delays.append(delay)
    return RetryAdvice(kind, max(delays) if delays else None, quota_zero)


def next_retry_delay(error: Exception, *, retry_number: int, max_wait_seconds: float) -> float | None:
    """None means stop, rather than retry sooner than the provider permits."""
    advice = retry_advice(error)
    if advice.kind == "daily" or advice.quota_zero:
        return None
    delay = max(5 * 2 ** min(max(retry_number - 1, 0), 8) + random.uniform(0, 1), advice.retry_after or 0)
    return delay if delay <= max_wait_seconds else None
