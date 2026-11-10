"""Grounded Hebrew answers with strict source/evidence validation.

Exact citation checks cannot prove that a paraphrase logically follows from a
quote. Live semantic evaluation is still required; unsafe structure is refused.
"""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any

from llama_index.core.llms import ChatMessage, MessageRole

from config import Settings
from src.file_loader import EXPECTED_DOCUMENT_IDS
from src.gemini_models import create_llm
from src.indexer import IndexNotReadyError, IndexingError
from src.models import Answer, RetrievedChunk, SourceRef, validate_question
from src.prompts import NO_ANSWER_TEXT, SYSTEM_PROMPT
from src.retriever import ArchiveRetriever
from src.service_errors import classify_service_error, provider_status_code


MAX_RESPONSE_CHARS = 24000
_ALT_ID = re.compile(r"\bALT[-־‐‑–—]([0-9]{1,3})\b", re.IGNORECASE)
_HEBREW = re.compile(r"[א-ת]")
_PRELIMINARY = re.compile(r"ראשוני|מוקדם|אינו סופי")
_QUALIFIER = re.compile(r"ראשוני|בתחילה|מוקדם|לא סופי|אינו סופי")
_NUMBERS = re.compile(r"[0-9]+(?:[.,][0-9]+)*")


class QAError(ValueError):
    """Invalid service configuration; mock answers must be explicitly isolated."""


class _InvalidAnswer(ValueError):
    pass


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidAnswer("Duplicate JSON keys.")
        result[key] = value
    return result


def _source_ref(chunk: RetrievedChunk, quote: str) -> SourceRef:
    return SourceRef(
        document_id=chunk.document_id,
        source_file=chunk.source_file,
        title=chunk.title,
        document_type=chunk.document_type,
        period=chunk.period,
        section=chunk.section,
        chunk_id=chunk.chunk_id,
        quote=quote,
    )


def _error_answer(error: Exception | None, *, offline: bool, retrieved: int = 0, missing_key: bool = False) -> Answer:
    code = provider_status_code(error)
    kind = classify_service_error(error)
    if missing_key:
        error_code = "missing_api_key"
        text = "כדי לבנות אינדקס Gemini ולשאול שאלות בחיבור חי, יש להגדיר GEMINI_API_KEY בקובץ .env המקומי."
    elif isinstance(error, IndexNotReadyError):
        error_code = "index_not_ready"
        text = "האינדוקס עדיין לא הושלם ואין אינדקס Gemini מוכן לשאלות. יש להריץ python -m scripts.ingest --rebuild; התקדמות שמורה תמשיך מאותה נקודה."
    elif isinstance(error, IndexingError):
        error_code = "index_invalid"
        text = "האינדקס אינו תואם למקורות או להגדרות. יש לבדוק אותו ולהריץ python -m scripts.ingest --rebuild לפי הצורך."
    elif kind == "access_denied":
        error_code = "access_denied"
        text = "Gemini דחה את הגישה. יש לבדוק את המפתח המקומי ואת הרשאות החשבון."
    elif kind == "model_unavailable":
        error_code = "model_unavailable"
        text = "המודל שהוגדר אינו זמין בחשבון. לא בוצעה החלפה אוטומטית של המודל."
    elif kind == "rate_limit":
        error_code = "rate_limit"
        text = "הבקשה נעצרה עקב מכסה או הגבלת קצב של Gemini. אפשר לנסות מאוחר יותר; לא בוצע שינוי במסלול החיוב."
    elif kind == "service_timeout":
        error_code = "service_timeout"
        detail = f" (קוד {code})" if code is not None else ""
        text = f"שירות Gemini לא השלים את הבקשה בזמן{detail}. אפשר לנסות שוב את השאלה מאוחר יותר."
    elif kind == "connection_error":
        error_code = "connection_error"
        text = "לא ניתן להתחבר לשירות Gemini. יש לבדוק את החיבור לאינטרנט ולנסות שוב."
    elif kind == "service_unavailable":
        error_code = "service_unavailable"
        detail = f" (שגיאת שירות {code})" if code is not None else ""
        text = f"שירות Gemini אינו זמין כרגע{detail}. אפשר לנסות שוב את השאלה מאוחר יותר."
    else:
        error_code = "service_unavailable"
        text = "אירעה תקלה בזמן הטיפול בשאלה. אפשר לנסות שוב מאוחר יותר."
    return Answer(text=text, status="error", retrieved_chunks=retrieved, offline=offline, error_code=error_code)


class QAService:
    def __init__(self, settings: Settings, *, retriever=None, llm=None, offline: bool = False) -> None:
        if not offline and (retriever is not None or llm is not None):
            raise QAError("Injected retrievers and models are allowed only in explicit offline tests.")
        if offline and llm is None:
            raise QAError("Offline answers require an explicitly supplied test model; no mock answer fallback is automatic.")
        self.settings = settings
        self.offline = offline
        self._retriever = retriever
        self._llm = llm
        self._last_retrieval: tuple[RetrievedChunk, ...] = ()

    @property
    def last_retrieval(self) -> tuple[RetrievedChunk, ...]:
        """Exact evidence from the latest ask; diagnostic state, never gold data.

        A service belongs to one session and handles questions sequentially.
        Evaluators inspect this after ask instead of issuing a second retrieval.
        """
        return self._last_retrieval

    def _no_answer(self, retrieved: int = 0, *, reason="response_rejected") -> Answer:
        return Answer(text=NO_ANSWER_TEXT, status="no_answer", retrieved_chunks=retrieved, offline=self.offline, no_answer_reason=reason)

    def _validate_chunks(self, chunks) -> tuple[RetrievedChunk, ...]:
        if not isinstance(chunks, (tuple, list)) or len(chunks) > self.settings.top_k or len(chunks) > 5:
            raise _InvalidAnswer("Too many or malformed retrieved chunks.")
        seen = set()
        for chunk in chunks:
            if (
                not isinstance(chunk, RetrievedChunk)
                or chunk.document_id not in EXPECTED_DOCUMENT_IDS
                or not chunk.chunk_id
                or chunk.chunk_id in seen
                or not chunk.text.strip()
                or not chunk.title
                or not math.isfinite(chunk.similarity)
                or not -1 <= chunk.similarity <= 1
            ):
                raise _InvalidAnswer("Invalid retrieved source identity or metadata.")
            seen.add(chunk.chunk_id)
        return tuple(chunks)

    def _messages(self, question: str, chunks: tuple[RetrievedChunk, ...]):
        payload = {
            "question": question,
            "sources": [{
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "title": chunk.title,
                "document_type": chunk.document_type,
                "period": chunk.period,
                "section": chunk.section,
                "text": chunk.text,
            } for chunk in chunks],
        }
        return [
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=json.dumps(payload, ensure_ascii=False)),
        ]

    def _parse_answer(self, content: Any, chunks: tuple[RetrievedChunk, ...]) -> Answer:
        if not isinstance(content, str) or not content or len(content) > MAX_RESPONSE_CHARS:
            raise _InvalidAnswer("Missing or oversized model output.")
        try:
            response = json.loads(content, object_pairs_hook=_unique_object)
        except (ValueError, RecursionError):
            raise _InvalidAnswer("Invalid JSON output.") from None
        if not isinstance(response, dict) or set(response) != {"status", "claims"}:
            raise _InvalidAnswer("Unexpected response schema.")
        status, claims = response["status"], response["claims"]
        if status not in ("answered", "partial", "no_answer") or not isinstance(claims, list) or len(claims) > 10:
            raise _InvalidAnswer("Invalid response status or claims.")
        if status == "no_answer":
            if claims:
                raise _InvalidAnswer("Refusal must contain no factual claims.")
            return self._no_answer(len(chunks), reason="model_refusal")
        if not claims:
            raise _InvalidAnswer("An answer must contain grounded claims.")
        available = {chunk.chunk_id: chunk for chunk in chunks}
        sources: list[SourceRef] = []
        paragraphs: list[str] = []
        used_chunks: list[RetrievedChunk] = []
        for claim in claims:
            if not isinstance(claim, dict) or set(claim) != {"text", "evidence"}:
                raise _InvalidAnswer("Invalid claim schema.")
            text, evidence = claim["text"], claim["evidence"]
            if not isinstance(text, str) or not text.strip() or len(text) > 1200 or not _HEBREW.search(text):
                raise _InvalidAnswer("Claim must be bounded Hebrew text.")
            if re.search(r"[A-Za-z]{2,}", _ALT_ID.sub("", text)):
                raise _InvalidAnswer("Claims must be written in Hebrew.")
            if not isinstance(evidence, list) or not 1 <= len(evidence) <= 3:
                raise _InvalidAnswer("Every claim requires source evidence.")
            cited: list[RetrievedChunk] = []
            numeric_evidence: list[str] = []
            for item in evidence:
                if not isinstance(item, dict) or set(item) != {"chunk_id", "quote"}:
                    raise _InvalidAnswer("Invalid evidence schema.")
                chunk_id, quote = item["chunk_id"], item["quote"]
                if not isinstance(chunk_id, str) or chunk_id not in available:
                    raise _InvalidAnswer("Evidence references an unretrieved chunk.")
                chunk = available[chunk_id]
                if not isinstance(quote, str) or not 12 <= len(quote) <= 1000 or not quote.strip() or quote not in chunk.text:
                    raise _InvalidAnswer("Quote is not an exact source excerpt.")
                cited.append(chunk)
                numeric_evidence.extend((quote, chunk.title, chunk.period))
                reference = _source_ref(chunk, quote)
                if reference not in sources:
                    sources.append(reference)
            cited_ids = {chunk.document_id for chunk in cited}
            mentioned_ids = {f"ALT-{int(match[1]):02d}" for match in _ALT_ID.finditer(text)}
            if not mentioned_ids <= cited_ids:
                raise _InvalidAnswer("Claim mentions an uncited archive document.")
            if not set(_NUMBERS.findall(_ALT_ID.sub("", text))) <= set(_NUMBERS.findall("\n".join(numeric_evidence))):
                raise _InvalidAnswer("Claim introduces numbers absent from its cited evidence.")
            if all(_PRELIMINARY.search(chunk.document_type) for chunk in cited) and not _QUALIFIER.search(text):
                raise _InvalidAnswer("A preliminary report cannot be stated as a final finding.")
            used_chunks.extend(cited)
            labels = list(dict.fromkeys(
                f"{chunk.document_id}, סעיף: {chunk.section}" if chunk.section else chunk.document_id
                for chunk in cited
            ))
            paragraphs.append(f"{text.strip()} [{' ; '.join(labels)}]")
        if all(_PRELIMINARY.search(chunk.document_type) for chunk in used_chunks):
            status = "partial"
        if status == "partial":
            paragraphs.append("זו תשובה חלקית לפי המקטעים שאוחזרו.")
        return Answer(text="\n\n".join(paragraphs), status=status, sources=tuple(sources), retrieved_chunks=len(chunks), offline=self.offline)

    def ask(self, question: str) -> Answer:
        self._last_retrieval = ()
        question = validate_question(question)
        if not self.offline and not self.settings.has_api_key:
            return _error_answer(None, offline=False, missing_key=True)
        try:
            if self._retriever is None:
                self._retriever = ArchiveRetriever(self.settings, offline=self.offline)
            chunks = self._validate_chunks(self._retriever.retrieve(question))
            self._last_retrieval = chunks
        except Exception as exc:
            return _error_answer(exc, offline=self.offline)
        if not chunks:
            return self._no_answer(reason="no_retrieved_evidence")
        try:
            if self._llm is None:
                self._llm = create_llm(self.settings)
            messages = self._messages(question, chunks)
            # Retry only transient HTTP 503 errors, never quota, auth, or bad requests.
            # Three total requests, with 1s then 2s delay; preserve the last error.
            for attempt in range(3):
                try:
                    response = self._llm.chat(messages)
                    break
                except Exception as exc:
                    if self.offline or provider_status_code(exc) != 503 or attempt == 2:
                        raise
                    print(f"KingdomAI: Gemini 503; retry {attempt + 1}/2", flush=True)
                    time.sleep(2 ** attempt)
        except Exception as exc:
            return _error_answer(exc, offline=self.offline, retrieved=len(chunks))
        try:
            # Tool/function output and empty blocked responses have no answer.
            message = response.message
            if getattr(message, "role", MessageRole.ASSISTANT) not in (MessageRole.ASSISTANT, "assistant"):
                return self._no_answer(len(chunks))
            additional = getattr(message, "additional_kwargs", {})
            if additional.get("tool_calls") or additional.get("function_call"):
                return self._no_answer(len(chunks))
            if any(getattr(block, "block_type", None) != "text" for block in getattr(message, "blocks", ())):
                return self._no_answer(len(chunks))
            return self._parse_answer(message.content, chunks)
        except (ValueError, TypeError, AttributeError, KeyError, RecursionError):
            return self._no_answer(len(chunks))
