"""Observe one QA attempt without a second retrieval or an answer-key prompt."""

from __future__ import annotations

import json
import socket
from dataclasses import FrozenInstanceError, replace

import pytest

from src.models import QuestionError
from src.prompts import NO_ANSWER_TEXT
from src.qa_service import QAService
from src.retriever import RetrievalError

from qa_helpers import ControlledLLM, StaticRetriever, passage, response_for, settings_for


@pytest.fixture(autouse=True)
def forbid_external_connections(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("QA diagnostics must not open an external connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def _service(tmp_path, *, chunks=(), response=None, error=None):
    retriever = StaticRetriever(chunks)
    model = ControlledLLM(response, error=error)
    service = QAService(settings_for(tmp_path), retriever=retriever, llm=model, offline=True)
    return service, retriever, model


@pytest.mark.parametrize("case", ["no_retrieved_evidence", "model_refusal", "response_rejected"])
def test_refusal_diagnostic_distinguishes_absent_evidence_model_choice_and_rejection(tmp_path, case):
    chunk = passage()
    chunks = () if case == "no_retrieved_evidence" else (chunk,)
    response = "invalid JSON" if case == "response_rejected" else {"status": "no_answer", "claims": []}
    service, retriever, model = _service(tmp_path, chunks=chunks, response=response)

    answer = service.ask("שאלת בדיקה טכנית")

    assert answer.status == "no_answer"
    assert answer.no_answer_reason == case
    assert answer.text == NO_ANSWER_TEXT
    assert not answer.sources
    assert answer.retrieved_chunks == len(chunks)
    assert service.last_retrieval == chunks
    assert retriever.queries == ["שאלת בדיקה טכנית"]
    assert len(model.calls) == (0 if not chunks else 1)


def test_rejected_fabricated_quote_is_distinct_from_valid_model_refusal(tmp_path):
    chunk = passage()
    invalid = response_for(chunk, quote="ציטוט בדיקה מומצא שאינו חלק מהמקור שנשלף.")
    service, _, model = _service(tmp_path, chunks=(chunk,), response=invalid)
    first = service.ask("שאלה טכנית ראשונה")
    model.response = {"status": "no_answer", "claims": []}
    second = service.ask("שאלה טכנית שנייה")

    assert first.text == second.text == NO_ANSWER_TEXT
    assert first.no_answer_reason == "response_rejected"
    assert second.no_answer_reason == "model_refusal"
    assert service.last_retrieval == (chunk,)


def test_trace_is_exact_evidence_used_for_same_original_question_and_model_payload(tmp_path):
    first, second = passage(), passage(2, text="המסמך השני כולל נתוני בדיקה נפרדים לצורך אימות התוכנה.")
    service, retriever, model = _service(tmp_path, chunks=(second, first), response=response_for(first))
    question = "  שאלה מקורית לבדיקה, עם רווחים?  "

    answer = service.ask(question)
    trace = service.last_retrieval
    payload = json.loads(model.calls[0][1].content)

    assert answer.status == "answered"
    assert answer.no_answer_reason is None
    assert retriever.queries == [question]
    assert len(model.calls) == 1
    assert payload["question"] == question
    assert trace[0] is second and trace[1] is first
    assert payload["sources"] == [
        {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "title": chunk.title,
            "document_type": chunk.document_type,
            "period": chunk.period,
            "section": chunk.section,
            "text": chunk.text,
        }
        for chunk in trace
    ]
    assert {source.chunk_id for source in answer.sources} <= {chunk.chunk_id for chunk in trace}
    assert set(payload) == {"question", "sources"}


def test_trace_survives_llm_timeout_and_error_is_not_counted_as_refusal(tmp_path):
    class Timeout(RuntimeError):
        code = 504

    chunk = passage()
    service, retriever, model = _service(tmp_path, chunks=(chunk,), error=Timeout("private SDK data"))

    answer = service.ask("שאלה טכנית")

    assert answer.status == "error"
    assert answer.error_code == "service_timeout"
    assert answer.no_answer_reason is None
    assert answer.retrieved_chunks == 1
    assert service.last_retrieval == (chunk,)
    assert len(retriever.queries) == len(model.calls) == 1
    assert "private SDK data" not in answer.text


def test_next_retrieval_failure_clears_old_trace_before_returning_error(tmp_path):
    chunk = passage()
    service, retriever, model = _service(tmp_path, chunks=(chunk,), response=response_for(chunk))
    assert service.ask("שאלה ראשונה").status == "answered"
    assert service.last_retrieval == (chunk,)
    retriever.error = RetrievalError("private failure", code=429)

    answer = service.ask("שאלה שנייה")

    assert answer.status == "error" and answer.error_code == "rate_limit"
    assert answer.no_answer_reason is None
    assert answer.retrieved_chunks == 0
    assert service.last_retrieval == ()
    assert retriever.queries == ["שאלה ראשונה", "שאלה שנייה"]
    assert len(model.calls) == 1


def test_invalid_retrieved_source_does_not_replace_trace_with_unvalidated_evidence(tmp_path):
    chunk = passage()
    service, retriever, model = _service(tmp_path, chunks=(chunk,), response=response_for(chunk))
    service.ask("שאלה ראשונה")
    retriever.chunks = (replace(chunk, document_id="ALT-99"),)

    answer = service.ask("שאלה שנייה")

    assert answer.status == "error"
    assert answer.no_answer_reason is None
    assert service.last_retrieval == ()
    assert len(model.calls) == 1


@pytest.mark.parametrize("invalid_question", ["", "   ", "א" * 2001, "שאלה\x00"])
def test_invalid_next_question_clears_prior_trace_without_retrieval_or_llm(tmp_path, invalid_question):
    chunk = passage()
    service, retriever, model = _service(tmp_path, chunks=(chunk,), response=response_for(chunk))
    service.ask("שאלה תקינה")
    assert service.last_retrieval == (chunk,)

    with pytest.raises(QuestionError):
        service.ask(invalid_question)

    assert service.last_retrieval == ()
    assert retriever.queries == ["שאלה תקינה"]
    assert len(model.calls) == 1


def test_missing_key_check_clears_prior_trace_without_reusing_cached_models(tmp_path):
    chunk = passage()
    service, retriever, model = _service(tmp_path, chunks=(chunk,), response=response_for(chunk))
    service.ask("שאלה במצב בדיקה")
    assert service.last_retrieval == (chunk,)
    # Exercise a mode transition with explicit offline doubles still installed.
    # Missing-key validation must finish before any of those clients are called.
    service.offline = False
    assert not service.settings.has_api_key

    answer = service.ask("שאלה ללא מפתח")

    assert answer.status == "error" and answer.error_code == "missing_api_key"
    assert answer.no_answer_reason is None
    assert service.last_retrieval == ()
    assert retriever.queries == ["שאלה במצב בדיקה"]
    assert len(model.calls) == 1


def test_trace_is_read_only_immutable_snapshot_and_next_ask_does_not_modify_old_snapshot(tmp_path):
    first, second = passage(), passage(2)
    service, retriever, model = _service(tmp_path, chunks=(first,), response=response_for(first))
    assert service.last_retrieval == ()
    service.ask("שאלה ראשונה")
    old_trace = service.last_retrieval

    with pytest.raises(AttributeError):
        service.last_retrieval = (second,)
    with pytest.raises(AttributeError):
        old_trace.append(second)
    with pytest.raises(FrozenInstanceError):
        old_trace[0].text = "טקסט ששונה מחוץ לשירות"

    retriever.chunks = (second,)
    model.response = response_for(second)
    service.ask("שאלה שנייה")

    assert isinstance(service.last_retrieval, tuple)
    assert service.last_retrieval == (second,)
    assert old_trace == (first,)
    assert retriever.queries == ["שאלה ראשונה", "שאלה שנייה"]


@pytest.mark.parametrize("status", ["answered", "partial"])
def test_successful_answer_has_no_refusal_diagnostic(tmp_path, status):
    chunk = passage()
    service, _, _ = _service(tmp_path, chunks=(chunk,), response=response_for(chunk, status=status))

    answer = service.ask("שאלה טכנית")

    assert answer.status == status
    assert answer.no_answer_reason is None
    assert service.last_retrieval == (chunk,)
