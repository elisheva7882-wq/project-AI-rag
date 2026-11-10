"""Offline Streamlit interaction tests; no real key, Gemini or index is used."""

from __future__ import annotations

import socket
from dataclasses import replace
from ipaddress import ip_address
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from config import ConfigurationError, Settings
from src.models import MAX_QUESTION_CHARS, Answer, SourceRef
from src.prompts import NO_ANSWER_TEXT


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
TEST_SECRET = "test-only-key-must-never-be-rendered"
QUESTION = "  שאלה טכנית לבדיקת הממשק?  "
SOURCE = SourceRef(
    document_id="ALT-01",
    source_file="ALT-01_בדיקה.md",
    title="כותרת מקור טכני לבדיקת ממשק",
    document_type="מסמך בדיקת תוכנה",
    period="תקופת בדיקה טכנית",
    section="סעיף בדיקת תצוגה",
    chunk_id="ALT-01-ui-test",
    quote="זהו ציטוט טכני לצורך בדיקת הממשק בלבד.",
)
ANSWER = Answer(
    text="תשובה טכנית שנבדקת בממשק [ALT-01].",
    status="answered",
    sources=(SOURCE,),
    retrieved_chunks=1,
)


def _widget(elements, key):
    return next(element for element in elements if element.key == key)


def _visible_text(app):
    values = []
    for kind in ("title", "text", "caption", "info", "warning", "error", "markdown", "metric"):
        values.extend(str(element.value) for element in app.get(kind))
    return "\n".join(values).replace("\u2066", "").replace("\u2069", "")


def _source_expanders(app):
    return [expander for expander in app.expander if "מקורות מהארכיון" in expander.label]


def _submit(app, question=QUESTION):
    _widget(app.chat_input, "question").set_value(question).run()
    assert not app.exception
    return app


@pytest.fixture
def ui_harness(tmp_path, monkeypatch):
    from src import ui

    settings = Settings(
        project_root=tmp_path,
        gemini_api_key=TEST_SECRET,
        llm_model="gemini-ui-test",
        embedding_model="embedding-ui-test",
        chroma_path=tmp_path / "storage" / "chroma",
        chroma_collection="ui_test_collection",
        knowledge_base_path=tmp_path / "knowledge",
        eval_path=tmp_path / "evaluation",
        chunk_size=512,
        chunk_overlap=64,
        top_k=5,
        similarity_threshold=None,
    )
    state = SimpleNamespace(
        settings=settings,
        status=ui.SetupStatus(True, documents=19, chunks=137, generation_id="test-generation"),
        answer=ANSWER,
        failure=None,
        service_count=0,
        queries=[],
        readiness_calls=0,
        configuration_failure=None,
    )

    def load_test_settings():
        if state.configuration_failure is not None:
            raise state.configuration_failure
        return state.settings

    def readiness(test_settings):
        assert test_settings is state.settings
        state.readiness_calls += 1
        return state.status

    class RecordingService:
        def __init__(self, test_settings):
            assert test_settings is state.settings
            state.service_count += 1

        def ask(self, question):
            state.queries.append(question)
            if state.failure is not None:
                raise state.failure
            return state.answer

    original_connect = socket.socket.connect
    original_create_connection = socket.create_connection

    def require_loopback(address):
        # Windows asyncio implements its internal wakeup pipe with a loopback
        # socket pair. Permit that local mechanism, while blocking Gemini and
        # every other external destination before any connection is made.
        host = address[0] if isinstance(address, tuple) else ""
        if host == "localhost":
            return
        try:
            if ip_address(host).is_loopback:
                return
        except ValueError:
            pass
        raise AssertionError("Streamlit interaction tests must not use external network access")

    def loopback_connect(sock, address):
        require_loopback(address)
        return original_connect(sock, address)

    def loopback_create_connection(address, *args, **kwargs):
        require_loopback(address)
        return original_create_connection(address, *args, **kwargs)

    monkeypatch.setattr(ui, "load_settings", load_test_settings)
    monkeypatch.setattr(ui, "check_readiness", readiness)
    monkeypatch.setattr(ui, "QAService", RecordingService)
    monkeypatch.setattr(socket, "create_connection", loopback_create_connection)
    monkeypatch.setattr(socket.socket, "connect", loopback_connect)
    state.new_app = lambda: AppTest.from_file(APP_PATH, default_timeout=15).run()
    return state


def test_initial_render_has_hebrew_rtl_readiness_and_models_without_creating_service(ui_harness):
    from src.ui import PAGE_TITLE

    app = ui_harness.new_app()
    assert not app.exception
    assert app.title[0].value == PAGE_TITLE
    rendered = _visible_text(app)
    assert "rtl" in rendered.lower()
    assert "19" in rendered and "137" in rendered
    assert ui_harness.settings.llm_model in rendered
    assert ui_harness.settings.embedding_model in rendered
    assert TEST_SECRET not in rendered
    assert not _widget(app.chat_input, "question").disabled
    assert ui_harness.service_count == 0
    assert ui_harness.queries == []


def test_question_displays_grounded_answer_and_full_source_metadata(ui_harness):
    app = _submit(ui_harness.new_app())
    assert ui_harness.queries == [QUESTION]
    assert ui_harness.service_count == 1
    assert [message.name for message in app.chat_message] == ["user", "assistant"]
    rendered = _visible_text(app)
    assert QUESTION.strip() in rendered
    assert ANSWER.text in rendered
    for value in (SOURCE.document_id, SOURCE.title, SOURCE.document_type, SOURCE.period, SOURCE.section, SOURCE.quote):
        assert value in rendered
    assert any("מקורות מהארכיון" in expander.label for expander in app.expander)
    assert TEST_SECRET not in rendered


def test_reruns_keep_history_and_do_not_repeat_question_or_construct_service(ui_harness):
    app = _submit(ui_harness.new_app())
    app.run()
    app.run()
    assert not app.exception
    assert ui_harness.queries == [QUESTION]
    assert ui_harness.service_count == 1
    assert len(app.chat_message) == 2
    assert len(app.session_state["turns"]) == 1


def test_followup_reuses_service_and_passes_only_the_original_current_question(ui_harness):
    app = _submit(ui_harness.new_app())
    followup = "שאלה טכנית נוספת ללא היסטוריה?"
    _submit(app, followup)
    assert ui_harness.queries == [QUESTION, followup]
    assert ui_harness.service_count == 1
    assert len(app.chat_message) == 4


def test_clear_removes_session_history_and_allows_a_fresh_question_without_rebuilding_service(ui_harness):
    app = _submit(ui_harness.new_app())
    _widget(app.button, "clear_chat").click().run()
    assert not app.exception
    assert not app.chat_message
    assert app.session_state["turns"] == []
    assert ui_harness.queries == [QUESTION]
    _submit(app, "שאלה לאחר ניקוי?")
    assert ui_harness.queries == [QUESTION, "שאלה לאחר ניקוי?"]
    assert ui_harness.service_count == 1


def test_suggested_question_is_submitted_once_and_removed_when_history_exists(ui_harness):
    from src.ui import EXAMPLE_QUESTIONS

    app = ui_harness.new_app()
    example = _widget(app.button, "example_0")
    assert example.label == EXAMPLE_QUESTIONS[0]
    example.click().run()
    assert not app.exception
    assert ui_harness.queries == [EXAMPLE_QUESTIONS[0]]
    assert len(app.chat_message) == 2
    assert not any(button.key and button.key.startswith("example_") for button in app.button)
    app.run()
    assert ui_harness.queries == [EXAMPLE_QUESTIONS[0]]


def test_no_evidence_displays_refusal_without_sources(ui_harness):
    ui_harness.answer = Answer(text=NO_ANSWER_TEXT, status="no_answer")
    app = _submit(ui_harness.new_app())
    assert app.info
    assert NO_ANSWER_TEXT in _visible_text(app)
    assert not _source_expanders(app)
    assert SOURCE.title not in _visible_text(app)


def test_partial_answer_is_labelled_and_retains_its_source(ui_harness):
    ui_harness.answer = replace(ANSWER, status="partial")
    app = _submit(ui_harness.new_app())
    assert any("חלקית" in warning.value for warning in app.warning)
    assert ANSWER.text in _visible_text(app)
    assert SOURCE.quote in _visible_text(app)


@pytest.mark.parametrize(
    "code,message",
    [
        ("rate_limit", "הבקשה נעצרה עקב מכסה או הגבלת קצב של Gemini."),
        ("service_timeout", "שירות Gemini לא השלים את הבקשה בזמן (שגיאת שרת 504)."),
    ],
)
def test_provider_failure_is_displayed_as_an_error_without_sources_or_automatic_repeat(ui_harness, code, message):
    ui_harness.answer = Answer(text=message, status="error", error_code=code)
    app = _submit(ui_harness.new_app())
    assert app.error
    assert message in _visible_text(app)
    assert not _source_expanders(app)
    app.run()
    assert ui_harness.queries == [QUESTION]


@pytest.mark.parametrize("code", ["missing_api_key", "missing_documents", "index_not_ready", "index_invalid"])
def test_unready_archive_blocks_questions_and_service_creation(ui_harness, code):
    from src.ui import SetupStatus

    message = f"מצב בדיקה טכני: {code}"
    ui_harness.status = SetupStatus(False, code=code, message=message)
    app = ui_harness.new_app()
    assert not app.exception
    assert message in _visible_text(app)
    assert _widget(app.chat_input, "question").disabled
    assert all(button.disabled for button in app.button if button.key and button.key.startswith("example_"))
    # Explicitly bypass the test client's disabled-widget check to simulate a
    # crafted event. The server-side readiness guard must still prevent a call.
    _widget(app.chat_input, "question").proto.disabled = False
    _submit(app)
    assert ui_harness.service_count == 0
    assert ui_harness.queries == []


def test_configuration_error_does_not_create_service_or_expose_secret(ui_harness):
    ui_harness.configuration_failure = ConfigurationError("CHUNK_SIZE must be positive.")
    app = ui_harness.new_app()
    assert not app.exception
    assert app.error or app.warning
    assert ui_harness.service_count == 0
    assert ui_harness.queries == []
    assert TEST_SECRET not in _visible_text(app)


def test_unexpected_service_exception_is_sanitized_and_not_retried_on_rerun(ui_harness):
    ui_harness.failure = RuntimeError(f"private provider payload {TEST_SECRET}")
    app = _submit(ui_harness.new_app())
    assert app.error
    assert TEST_SECRET not in _visible_text(app)
    assert "private provider payload" not in _visible_text(app)
    assert not _source_expanders(app)
    app.run()
    assert ui_harness.queries == [QUESTION]


def test_service_construction_failure_is_sanitized_without_a_crash_or_automatic_retry(ui_harness, monkeypatch):
    from src import ui

    constructors = []

    def fail_construction(settings):
        constructors.append(settings)
        raise RuntimeError(f"private constructor payload {TEST_SECRET}")

    monkeypatch.setattr(ui, "QAService", fail_construction)
    app = _submit(ui_harness.new_app())
    assert app.error
    assert TEST_SECRET not in _visible_text(app)
    assert "private constructor payload" not in _visible_text(app)
    assert not ui_harness.queries
    app.run()
    assert len(constructors) == 1


@pytest.mark.parametrize("question", ["   ", "א" * (MAX_QUESTION_CHARS + 1), "שאלה\x00לא תקינה"])
def test_invalid_question_is_rejected_before_creating_or_calling_service(ui_harness, question):
    app = _submit(ui_harness.new_app(), question)
    assert app.error or app.warning
    assert ui_harness.service_count == 0
    assert ui_harness.queries == []
    assert app.session_state["turns"] == []


def test_distinct_sessions_do_not_share_history_or_service(ui_harness):
    first = _submit(ui_harness.new_app())
    second = ui_harness.new_app()
    assert not second.exception
    assert len(first.chat_message) == 2
    assert not second.chat_message
    assert second.session_state["turns"] == []
    _submit(second, "שאלה בחלון נפרד?")
    assert ui_harness.service_count == 2
    assert len(first.chat_message) == 2


@pytest.mark.parametrize("change", ["generation", "model", "key"])
def test_changed_runtime_identity_clears_old_history_and_replaces_cached_service(ui_harness, change):
    app = _submit(ui_harness.new_app())
    if change == "generation":
        ui_harness.status = replace(ui_harness.status, generation_id="different-generation")
    elif change == "model":
        ui_harness.settings = replace(ui_harness.settings, llm_model="different-test-model")
    else:
        ui_harness.settings = replace(ui_harness.settings, gemini_api_key="different-test-only-key")
    app.run()
    assert not app.exception
    assert not app.chat_message
    assert app.session_state["turns"] == []
    _submit(app, "שאלה לאחר שינוי הגדרות?")
    assert ui_harness.service_count == 2
    assert ui_harness.queries == [QUESTION, "שאלה לאחר שינוי הגדרות?"]


def test_history_is_bounded_and_discarded_old_turns_are_not_submitted_again(ui_harness):
    from src.ui import MAX_HISTORY_TURNS

    app = ui_harness.new_app()
    for number in range(MAX_HISTORY_TURNS + 2):
        _submit(app, f"שאלה טכנית {number}?")
    turns = app.session_state["turns"]
    assert len(turns) == MAX_HISTORY_TURNS
    assert turns[0].question == "שאלה טכנית 2?"
    assert turns[-1].question == f"שאלה טכנית {MAX_HISTORY_TURNS + 1}?"
    assert len(app.chat_message) == MAX_HISTORY_TURNS * 2
    assert len(ui_harness.queries) == MAX_HISTORY_TURNS + 2
    assert ui_harness.service_count == 1


def test_dynamic_user_answer_and_source_strings_are_never_rendered_as_unsafe_html(ui_harness):
    marker = '<img src="bad" onerror="alert(1)">'
    ui_harness.answer = replace(ANSWER, text=f"תשובה טכנית {marker}", sources=(replace(SOURCE, quote=marker),))
    app = _submit(ui_harness.new_app(), f"שאלת בדיקה {marker}")
    assert marker in _visible_text(app)
    # Only fixed application CSS may opt in to HTML; all untrusted values must
    # remain in text elements or in escaped Markdown.
    for markdown in app.markdown:
        assert marker not in markdown.value

