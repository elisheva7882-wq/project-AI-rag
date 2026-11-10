"""Validated local settings; loading them never calls an API or builds an index."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from ipaddress import ip_address
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parent


class ConfigurationError(ValueError):
    """An invalid setting that can be shown safely in the user interface."""


@dataclass(frozen=True)
class Settings:
    project_root: Path
    gemini_api_key: str = field(repr=False)
    llm_model: str
    embedding_model: str
    chroma_path: Path
    chroma_collection: str
    knowledge_base_path: Path
    eval_path: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    similarity_threshold: float | None
    embedding_dimension: int = 3072
    embedding_request_interval_seconds: float = 6.0
    embedding_max_retries: int = 2
    embedding_max_retry_wait_seconds: float = 60.0
    llm_request_timeout_seconds: float = 120.0
    llm_thinking_level: str = "medium"

    @property
    def has_api_key(self) -> bool:
        return bool(self.gemini_api_key)

    def setup_issues(self) -> tuple[str, ...]:
        """Report missing prerequisites without preventing the future UI from loading.

        File counts here are informational. Full corpus/ID/content validation is
        performed by the knowledge preparation step, not by this settings module.
        """
        issues: list[str] = []
        if not self.has_api_key:
            issues.append("חסר GEMINI_API_KEY תקין בקובץ .env או במשתני הסביבה.")
        if not self.knowledge_base_path.is_dir():
            issues.append("תיקיית knowledge_base עדיין אינה קיימת.")
        elif not any(self.knowledge_base_path.glob("*.md")):
            issues.append("תיקיית knowledge_base עדיין אינה מכילה מסמכי Markdown.")
        if not (self.project_root / "storage" / "index_manifest.json").is_file():
            issues.append("לא נמצא מניפסט אינדקס; מוכנות האינדקס טרם אומתה.")
        return tuple(issues)


def _integer(values: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(values.get(name, str(default)))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be an integer.") from exc


def _nonempty(values: Mapping[str, str], name: str, default: str) -> str:
    value = values.get(name, default).strip()
    if not value:
        raise ConfigurationError(f"{name} must not be empty.")
    return value


def _bounded_float(
    values: Mapping[str, str], name: str, default: float, maximum: float, *, minimum: float = 0.0
) -> float:
    try:
        value = float(values.get(name, str(default)))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a number.") from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} must be finite and between {minimum} and {maximum}.")
    return value


def _path(values: Mapping[str, str], name: str, default: str, root: Path) -> Path:
    path = Path(_nonempty(values, name, default)).expanduser()
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if path.exists() and not path.is_dir():
        raise ConfigurationError(f"{name} must point to a directory.")
    return path


def _overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _collection_name(values: Mapping[str, str]) -> str:
    name = _nonempty(values, "CHROMA_COLLECTION", "kingdomai_altheria_docs")
    if re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]", name) is None or ".." in name:
        raise ConfigurationError("CHROMA_COLLECTION must be 3-512 ASCII name characters, with alphanumeric ends and no consecutive periods.")
    try:
        ip_address(name)
    except ValueError:
        return name
    raise ConfigurationError("CHROMA_COLLECTION must not be an IP address.")


def load_settings(
    *,
    project_root: Path = PROJECT_ROOT,
    env_file: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Load project .env then process environment, with environment taking priority.

    Relative data paths always resolve against the project, even if the caller
    starts from another working directory. This function has no write side effects.
    """
    root = project_root.resolve()
    dotenv_path = root / ".env" if env_file is None else env_file
    values = {
        key: value
        for key, value in dotenv_values(
            dotenv_path=dotenv_path, encoding="utf-8-sig", interpolate=False
        ).items()
        if value is not None
    }
    values.update(os.environ if environ is None else environ)

    chunk_size = _integer(values, "CHUNK_SIZE", 512)
    chunk_overlap = _integer(values, "CHUNK_OVERLAP", 64)
    top_k = _integer(values, "TOP_K", 5)
    embedding_dimension = _integer(values, "EMBEDDING_DIMENSION", 3072)
    embedding_max_retries = _integer(values, "EMBEDDING_MAX_RETRIES", 2)
    if not 0 <= embedding_max_retries <= 5:
        raise ConfigurationError("EMBEDDING_MAX_RETRIES must be between zero and five.")
    embedding_interval = _bounded_float(values, "EMBEDDING_REQUEST_INTERVAL_SECONDS", 6.0, 60.0)
    embedding_retry_wait = _bounded_float(values, "EMBEDDING_MAX_RETRY_WAIT_SECONDS", 60.0, 60.0)
    llm_timeout = _bounded_float(values, "LLM_REQUEST_TIMEOUT_SECONDS", 120.0, 300.0, minimum=10.0)
    llm_thinking_level = _nonempty(values, "LLM_THINKING_LEVEL", "medium")
    if llm_thinking_level not in {"low", "medium", "high"}:
        raise ConfigurationError("LLM_THINKING_LEVEL must be low, medium or high.")
    if not 1 <= embedding_dimension <= 3072:
        raise ConfigurationError("EMBEDDING_DIMENSION must be between 1 and 3072.")
    if chunk_size <= 0:
        raise ConfigurationError("CHUNK_SIZE must be positive.")
    if not 0 <= chunk_overlap < chunk_size:
        raise ConfigurationError("CHUNK_OVERLAP must be between zero and CHUNK_SIZE - 1.")
    if not 1 <= top_k <= 5:
        raise ConfigurationError("TOP_K must be between 1 and 5 for this MVP.")

    raw_threshold = values.get("SIMILARITY_THRESHOLD", "").strip()
    threshold: float | None = None
    if raw_threshold:
        try:
            threshold = float(raw_threshold)
        except ValueError as exc:
            raise ConfigurationError("SIMILARITY_THRESHOLD must be a number or empty.") from exc
        if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
            raise ConfigurationError("SIMILARITY_THRESHOLD must be a finite cosine similarity in [-1, 1].")

    knowledge = _path(values, "KNOWLEDGE_BASE_PATH", "./knowledge_base", root)
    evaluation = _path(values, "EVAL_PATH", "./eval", root)
    chroma = _path(values, "CHROMA_PATH", "./storage/chroma", root)
    if _overlap(knowledge, evaluation):
        raise ConfigurationError("KNOWLEDGE_BASE_PATH and EVAL_PATH must be separate, non-nested directories.")
    if _overlap(chroma, knowledge) or _overlap(chroma, evaluation):
        raise ConfigurationError("CHROMA_PATH must be separate from knowledge and evaluation directories.")

    api_key = values.get("GEMINI_API_KEY", "").strip()
    if api_key.lower() in {"your_gemini_api_key_here", "your_api_key_here", "changeme"}:
        api_key = ""

    return Settings(
        project_root=root,
        gemini_api_key=api_key,
        llm_model=_nonempty(values, "LLM_MODEL", "gemini-3.8-flash"),
        embedding_model=_nonempty(values, "EMBEDDING_MODEL", "gemini-embedding-001"),
        chroma_path=chroma,
        chroma_collection=_collection_name(values),
        knowledge_base_path=knowledge,
        eval_path=evaluation,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        similarity_threshold=threshold,
        embedding_dimension=embedding_dimension,
        embedding_request_interval_seconds=embedding_interval,
        embedding_max_retries=embedding_max_retries,
        embedding_max_retry_wait_seconds=embedding_retry_wait,
        llm_request_timeout_seconds=llm_timeout,
        llm_thinking_level=llm_thinking_level,
    )
