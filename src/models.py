"""Explicit data boundaries between retrieval, grounded answers and the UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


MAX_QUESTION_CHARS = 2000


class QuestionError(ValueError):
    """A question must be nonempty and fit the application input limit."""


def validate_question(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise QuestionError("יש להזין שאלה שאינה ריקה.")
    if len(question) > MAX_QUESTION_CHARS:
        raise QuestionError(f"השאלה ארוכה מדי. ניתן להזין עד {MAX_QUESTION_CHARS} תווים.")
    if "\x00" in question:
        raise QuestionError("השאלה מכילה תו לא תקין.")
    return question  # Preserve the original question; no query expansion.


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    source_file: str
    title: str
    section: str
    text: str
    similarity: float
    document_type: str = ""
    period: str = ""
    topics: str = ""


@dataclass(frozen=True)
class SourceRef:
    document_id: str
    source_file: str
    title: str
    document_type: str
    period: str
    section: str
    chunk_id: str
    quote: str


@dataclass(frozen=True)
class Answer:
    text: str
    status: Literal["answered", "partial", "no_answer", "error"]
    sources: tuple[SourceRef, ...] = ()
    retrieved_chunks: int = 0
    offline: bool = False
    error_code: str | None = None
    no_answer_reason: Literal["no_retrieved_evidence", "model_refusal", "response_rejected"] | None = None
