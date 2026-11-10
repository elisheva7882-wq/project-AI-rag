"""Offline regressions for sanitized provider failures from SDK to chat UI."""

from __future__ import annotations

import socket
from ipaddress import ip_address
from pathlib import Path
from typing import Any

import httpx
import pytest
import requests
from llama_index.core.bridge.pydantic import Field
from streamlit.testing.v1 import AppTest

from config import load_settings
from src.gemini_models import (
    OfflineEmbedding,
    ProviderError,
    create_embedding_model,
    create_llm,
)
from src.indexer import get_manifest_path, ingest
from src.qa_service import QAService
from src.retriever import ArchiveRetriever, RetrievalError

from qa_helpers import ControlledLLM, StaticRetriever, passage


SECRET = "runtime-test-private-key-and-SDK-payload"
QUESTION = "  שאלת בדיקת תוכנה מקורית, ללא היסטוריית שיחה?  "
APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class PrivateSDKError(RuntimeError):
    """A structured SDK error whose message is private."""

    def __init__(self, code: int) -> None:
        super().__init__(SECRET)
        self.code = code

class UnformattableSDKError(PrivateSDKError):
    """Application wrappers must not need to format a provider exception."""

    def __str__(self) -> str:
        raise AssertionError("Private SDK errors must not be formatted")


def _failure(case: str) -> Exception:
    if case in {"500", "502", "503", "504"}:
        return PrivateSDKError(int(case))
    if case == "http_timeout":
        return httpx.ReadTimeout(SECRET, request=httpx.Request("POST", f"https://invalid.test/{SECRET}"))
    if case == "http_connection":
        return httpx.ConnectError(SECRET, request=httpx.Request("POST", f"https://invalid.test/{SECRET}"))
    if case == "requests_timeout":
        return requests.exceptions.ConnectTimeout(SECRET)
    raise AssertionError(f"Unknown controlled test case: {case}")


@pytest.fixture(autouse=True)
def offline_transports_only(monkeypatch):
    """Allow Windows asyncio's local socket pair, but no external transport."""
    original_connect = socket.socket.connect
    original_create_connection = socket.create_connection

    def require_loopback(address):
        host = address[0] if isinstance(address, tuple) else ""
        if host == "localhost":
            return
        try:
            if ip_address(host).is_loopback:
                return
        except ValueError:
            pass
        raise AssertionError("Runtime failure regression tests must stay offline")

    def loopback_connect(stream, address):
        require_loopback(address)
        return original_connect(stream, address)

    def loopback_create_connection(address, *args, **kwargs):
        require_loopback(address)
        return original_create_connection(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", loopback_connect)
    monkeypatch.setattr(socket, "create_connection", loopback_create_connection)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)


@pytest.mark.parametrize("factory", [create_llm, create_embedding_model], ids=["llm", "embedding"])
@pytest.mark.parametrize(
    "case,code,kind",
    [
        ("503", 503, "service_unavailable"),
        ("http_timeout", None, "service_timeout"),
        ("http_connection", None, "connection_error"),
    ],
)
def test_real_adapter_constructor_preserves_safe_failure_category(
    tmp_path, monkeypatch, factory, case, code, kind
):
    import google.genai

    settings = load_settings(project_root=tmp_path, environ={"GEMINI_API_KEY": SECRET})
    calls = []

    def fail_client(**kwargs):
        calls.append(kwargs)
        raise UnformattableSDKError(503) if case == "503" else _failure(case)

    monkeypatch.setattr(google.genai, "Client", fail_client)

    with pytest.raises(ProviderError) as captured:
        factory(settings)

    assert captured.value.code == code
    assert captured.value.kind == kind
    assert SECRET not in str(captured.value)
    assert len(calls) == 1
    assert calls[0]["api_key"] == SECRET
    assert captured.value.__suppress_context__ is True


class FailingQueryEmbedding(OfflineEmbedding):
    failure: Any = Field(exclude=True)
    queries: list[str] = Field(default_factory=list, exclude=True)

    def _get_query_embedding(self, query):
        self.queries.append(query)
        raise self.failure


@pytest.fixture
def isolated_index(tmp_path, corpus_directory):
    root = tmp_path / "runtime-failure-project"
    root.mkdir()
    settings = load_settings(project_root=root, environ={"KNOWLEDGE_BASE_PATH": str(corpus_directory)})
    ingest(settings, rebuild=True, offline=True)
    return settings


@pytest.mark.parametrize(
    "case,code,kind",
    [
        ("503", 503, "service_unavailable"),
        ("http_timeout", None, "service_timeout"),
        ("http_connection", None, "connection_error"),
    ],
)
def test_query_embedding_failure_preserves_category_and_existing_vectors(
    isolated_index, case, code, kind
):
    model = FailingQueryEmbedding(failure=_failure(case))
    retriever = ArchiveRetriever(isolated_index, offline=True, embed_model=model)
    manifest = get_manifest_path(isolated_index, offline=True)
    before = (manifest.read_bytes(), retriever._collection.id, retriever._collection.count())

    with pytest.raises(RetrievalError) as captured:
        retriever.retrieve(QUESTION)

    assert captured.value.code == code
    assert captured.value.kind == kind
    assert SECRET not in str(captured.value)
    assert model.queries == [QUESTION]
    assert before == (manifest.read_bytes(), retriever._collection.id, retriever._collection.count())


@pytest.mark.parametrize(
    "case,error_code",
    [
        ("503", "service_unavailable"),
        ("http_timeout", "service_timeout"),
        ("http_connection", "connection_error"),
    ],
)
def test_query_failure_reaches_qa_without_generation_retry_or_index_rebuild_advice(
    isolated_index, case, error_code
):
    model = FailingQueryEmbedding(failure=_failure(case))
    retriever = ArchiveRetriever(isolated_index, offline=True, embed_model=model)
    llm = ControlledLLM()
    service = QAService(isolated_index, retriever=retriever, llm=llm, offline=True)

    answer = service.ask(QUESTION)

    assert answer.status == "error"
    assert answer.error_code == error_code
    assert answer.retrieved_chunks == 0
    assert not answer.sources
    assert answer.no_answer_reason is None
    assert service.last_retrieval == ()
    assert model.queries == [QUESTION]
    assert llm.calls == []
    assert SECRET not in answer.text
    assert "אינדקס" not in answer.text
    assert "ingest" not in answer.text
    if case == "503":
        assert "Gemini" in answer.text and "503" in answer.text


@pytest.mark.parametrize(
    "case,error_code",
    [
        ("500", "service_unavailable"),
        ("502", "service_unavailable"),
        ("503", "service_unavailable"),
        ("504", "service_timeout"),
        ("http_timeout", "service_timeout"),
        ("requests_timeout", "service_timeout"),
        ("http_connection", "connection_error"),
    ],
)
def test_generation_failure_keeps_exact_evidence_and_does_not_retry(
    tmp_path, case, error_code
):
    settings = load_settings(project_root=tmp_path, environ={})
    chunks = (passage(),)
    retriever = StaticRetriever(chunks)
    llm = ControlledLLM(error=_failure(case))
    service = QAService(settings, retriever=retriever, llm=llm, offline=True)

    answer = service.ask(QUESTION)

    assert answer.status == "error" and answer.error_code == error_code
    assert answer.retrieved_chunks == 1
    assert service.last_retrieval == chunks
    assert answer.no_answer_reason is None
    assert not answer.sources
    assert retriever.queries == [QUESTION]
    assert len(llm.calls) == 1
    assert SECRET not in answer.text
    assert "אינדקס" not in answer.text and "ingest" not in answer.text
    if case in {"500", "502", "503", "504"}:
        assert case in answer.text


def test_llm_sdk_constructor_failure_reaches_live_qa_as_503(tmp_path, monkeypatch):
    import google.genai
    from src import qa_service

    settings = load_settings(project_root=tmp_path, environ={"GEMINI_API_KEY": SECRET})
    retriever = StaticRetriever((passage(),))
    construction_calls = []

    def fail_client(**kwargs):
        construction_calls.append(kwargs)
        raise _failure("503")

    monkeypatch.setattr(google.genai, "Client", fail_client)
    monkeypatch.setattr(qa_service, "ArchiveRetriever", lambda *args, **kwargs: retriever)
    service = QAService(settings)

    answer = service.ask(QUESTION)

    assert answer.status == "error" and answer.error_code == "service_unavailable"
    assert "503" in answer.text and "Gemini" in answer.text
    assert SECRET not in answer.text and "אינדקס" not in answer.text
    assert retriever.queries == [QUESTION]
    assert len(construction_calls) == 1
    assert service.last_retrieval == retriever.chunks


@pytest.mark.parametrize(
    "case,error_code",
    [("503", "service_unavailable"), ("http_timeout", "service_timeout"), ("http_connection", "connection_error")],
)
def test_chat_renders_specific_failure_and_reruns_do_not_repeat_request(
    tmp_path, monkeypatch, case, error_code
):
    from src import ui

    settings = load_settings(project_root=tmp_path, environ={"GEMINI_API_KEY": SECRET})
    retriever = StaticRetriever((passage(),))
    llm = ControlledLLM(error=_failure(case))
    service = QAService(settings, retriever=retriever, llm=llm, offline=True)
    monkeypatch.setattr(ui, "load_settings", lambda: settings)
    monkeypatch.setattr(
        ui,
        "check_readiness",
        lambda selected_settings: ui.SetupStatus(True, documents=19, chunks=137, generation_id="runtime-test"),
    )
    monkeypatch.setattr(ui, "QAService", lambda selected_settings: service)

    app = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not app.exception
    app.chat_input[0].set_value(QUESTION).run()
    assert not app.exception

    turn = app.session_state["turns"][0]
    assert turn.answer.error_code == error_code
    assert app.error
    banner = "\n".join(error.value for error in app.error)
    displayed_answer = "\n".join(str(element.value) for element in app.text)
    expected_banner = {
        "service_unavailable": "לא התקבלה תשובה מהשירות.",
        "service_timeout": "תם זמן ההמתנה לתשובה.",
        "connection_error": "החיבור לשירות Gemini לא הושלם.",
    }
    assert expected_banner[error_code] in banner
    assert "אינדקס" not in banner and "אינדקס" not in displayed_answer
    assert SECRET not in banner and SECRET not in displayed_answer
    assert turn.answer.text in displayed_answer
    if case == "503":
        assert "503" in displayed_answer
    app.run()
    app.run()
    assert not app.exception
    assert retriever.queries == [QUESTION]
    assert len(llm.calls) == 1
    assert len(app.session_state["turns"]) == 1
