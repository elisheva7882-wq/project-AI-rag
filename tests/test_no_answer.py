"""Missing evidence and provider failures produce explicit safe responses."""

from __future__ import annotations

import socket

import pytest

from src.models import MAX_QUESTION_CHARS, QuestionError
from src.prompts import NO_ANSWER_TEXT
from src.qa_service import QAError, QAService
from src.retriever import RetrievalError

from qa_helpers import ControlledLLM, StaticRetriever, passage, response_for, settings_for


def test_empty_retrieval_returns_fixed_refusal_without_calling_llm(tmp_path):
    retriever = StaticRetriever()
    model = ControlledLLM(error=AssertionError("No LLM call is needed without source evidence"))
    service = QAService(settings_for(tmp_path), retriever=retriever, llm=model, offline=True)
    answer = service.ask("מה שם החתול שלא תועד במקור?")
    assert answer.status == "no_answer"
    assert answer.text == NO_ANSWER_TEXT
    assert answer.retrieved_chunks == 0
    assert not answer.sources
    assert not model.calls


def test_model_can_report_insufficient_evidence_after_retrieval(tmp_path):
    model = ControlledLLM({"status": "no_answer", "claims": []})
    service = QAService(settings_for(tmp_path), retriever=StaticRetriever((passage(),)), llm=model, offline=True)
    answer = service.ask("מהי סיסמה שאינה מופיעה במקור?")
    assert answer.status == "no_answer"
    assert answer.text == NO_ANSWER_TEXT
    assert answer.retrieved_chunks == 1
    assert not answer.sources
    assert len(model.calls) == 1


@pytest.mark.parametrize("question", ["", " \n\t", None, 123, "א" * (MAX_QUESTION_CHARS + 1), "שאלה\x00לא תקינה"])
def test_invalid_question_is_rejected_before_retrieval_or_model(tmp_path, question):
    retriever = StaticRetriever((passage(),))
    model = ControlledLLM(response_for(passage()))
    service = QAService(settings_for(tmp_path), retriever=retriever, llm=model, offline=True)
    with pytest.raises(QuestionError):
        service.ask(question)
    assert not retriever.queries
    assert not model.calls


def test_offline_mode_requires_an_explicit_test_llm(tmp_path):
    with pytest.raises(QAError):
        QAService(settings_for(tmp_path), retriever=StaticRetriever(), offline=True)


def test_injected_test_components_cannot_be_used_as_production_qa(tmp_path):
    with pytest.raises(QAError):
        QAService(settings_for(tmp_path), retriever=StaticRetriever(), llm=ControlledLLM())


def test_missing_key_is_reported_before_any_live_retrieval_or_model(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Missing API key must not result in network access")

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    answer = QAService(settings_for(tmp_path)).ask("מה כתוב בארכיון?")
    assert answer.status == "error"
    assert answer.error_code == "missing_api_key"
    assert not answer.sources
    assert "GEMINI_API_KEY" in answer.text or "מפתח" in answer.text


@pytest.mark.parametrize("code", [401, 403, 404, 429, 503, 504])
def test_llm_api_failure_does_not_become_an_archive_answer(tmp_path, code):
    class ProviderFailure(RuntimeError):
        def __init__(self):
            super().__init__("SDK payload contains secret-that-must-not-be-displayed")
            self.code = code

    model = ControlledLLM(error=ProviderFailure())
    service = QAService(settings_for(tmp_path), retriever=StaticRetriever((passage(),)), llm=model, offline=True)
    answer = service.ask("מה כתוב במקור?")
    assert answer.status == "error"
    assert answer.error_code
    assert not answer.sources
    assert "secret-that-must-not-be-displayed" not in answer.text


def test_gemini_gateway_timeout_has_specific_guidance_without_automatic_retries(tmp_path):
    class GatewayTimeout(RuntimeError):
        code = 504

    model = ControlledLLM(error=GatewayTimeout("private-sdk-payload"))
    service = QAService(settings_for(tmp_path), retriever=StaticRetriever((passage(),)), llm=model, offline=True)
    answer = service.ask("שאלת בדיקה טכנית")
    assert answer.status == "error" and answer.error_code == "service_timeout"
    assert "504" in answer.text and "מאוחר יותר" in answer.text
    assert "private-sdk-payload" not in answer.text
    assert not answer.sources
    assert len(model.calls) == 1


def test_retrieval_failure_does_not_call_model_or_leak_sdk_payload(tmp_path):
    retriever = StaticRetriever(error=RetrievalError("internal-sensitive-error-must-not-be-echoed", code=429))
    model = ControlledLLM()
    service = QAService(settings_for(tmp_path), retriever=retriever, llm=model, offline=True)
    answer = service.ask("מה כתוב במקור?")
    assert answer.status == "error"
    assert not answer.sources
    assert "internal-sensitive-error-must-not-be-echoed" not in answer.text
    assert not model.calls


def test_ask_cli_tells_when_the_api_key_is_needed_without_mock_fallback(tmp_path, monkeypatch, capsys):
    import scripts.ask as command

    monkeypatch.setattr(command, "load_settings", lambda: settings_for(tmp_path))
    assert command.main(["--question", "מה כתוב בארכיון?"]) == 2
    output = capsys.readouterr()
    assert "GEMINI_API_KEY" in output.out
    assert ".env" in output.out
    assert "מקורות מהארכיון" not in output.out
    assert not output.err


def test_ask_cli_rejects_empty_question_before_live_service_use(tmp_path, monkeypatch, capsys):
    import scripts.ask as command

    monkeypatch.setattr(command, "load_settings", lambda: settings_for(tmp_path))
    assert command.main(["--question", " "]) == 2
    output = capsys.readouterr()
    assert "ERROR" in output.err
    assert not output.out
