"""Evaluation accounting and isolation checks using synthetic cases only."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from config import load_settings
from src import evaluation
from src.models import Answer, SourceRef
from src.prompts import NO_ANSWER_TEXT

from qa_helpers import passage


QUESTIONS_FILE = "שאלות_בדיקה_בלבד.md"
ANSWERS_FILE = "מפתח_תשובות_ומקורות.jsonl"
KEY_SENTINEL = "PRIVATE_EXPECTED_ANSWER_NEVER_FOR_MODEL"


@pytest.fixture
def evaluation_settings(tmp_path, archive_entries):
    settings = load_settings(project_root=tmp_path, environ={})
    settings.eval_path.mkdir(parents=True)
    for name in (QUESTIONS_FILE, ANSWERS_FILE):
        selected = [value for key, value in archive_entries.items() if key.endswith("/" + name)]
        assert len(selected) == 1
        (settings.eval_path / name).write_bytes(selected[0])
    return settings


@pytest.fixture
def mock_index(monkeypatch):
    manifest = {
        "generation_id": "1" * 32,
        "corpus_version": "2" * 64,
        "documents": 19,
        "chunks": 19,
        "embedding": {"model": "offline-test", "dimension": 64},
    }
    calls = []

    def validate(settings, *, offline=False):
        calls.append((settings, offline))
        return manifest

    monkeypatch.setattr(evaluation, "validate_index", validate)
    return calls


def _rows(settings):
    return [json.loads(line) for line in (settings.eval_path / ANSWERS_FILE).read_text(encoding="utf-8").splitlines()]


def _write_rows(settings, rows):
    (settings.eval_path / ANSWERS_FILE).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )


def _source(chunk):
    return SourceRef(
        document_id=chunk.document_id,
        source_file=chunk.source_file,
        title=chunk.title,
        document_type=chunk.document_type,
        period=chunk.period,
        section=chunk.section,
        chunk_id=chunk.chunk_id,
        quote=chunk.text,
    )


class ControlledService:
    """The deliberately narrow signature catches leaked evaluation fields."""

    def __init__(self, answer=None, chunks=(), *, failure=None):
        self.answer = answer or Answer("תשובה טכנית לבדיקה.", "answered")
        self.chunks = tuple(chunks)
        self.failure = failure
        self.calls = []
        self.last_retrieval = ()

    def ask(self, question):
        assert isinstance(question, str)
        assert KEY_SENTINEL not in question
        self.calls.append(question)
        self.last_retrieval = self.chunks
        if self.failure is not None:
            raise self.failure
        return self.answer


class ControlledRetriever:
    def __init__(self, chunks=(), *, failure=None):
        self.chunks = tuple(chunks)
        self.failure = failure
        self.calls = []

    def retrieve(self, question):
        assert isinstance(question, str)
        assert KEY_SENTINEL not in question
        self.calls.append(question)
        if self.failure is not None:
            raise self.failure
        return self.chunks


def _run(settings, tmp_path, **kwargs):
    return evaluation.run_evaluation(
        settings,
        offline=True,
        output_dir=tmp_path / "reports",
        request_interval_seconds=0,
        **kwargs,
    )


def test_loads_45_cases_with_expected_answers_for_local_scoring_only(evaluation_settings):
    cases = evaluation.load_evaluation_cases(evaluation_settings)
    assert len(cases) == 45
    assert [case.id for case in cases] == [f"Q{number:02d}" for number in range(1, 46)]
    assert cases[0].question == "שאלת בדיקה טכנית 01?"
    assert cases[0].source_doc_ids == ("ALT-01",)
    assert cases[0].expected_answer == "תשובת דמה טכנית 01."
    assert cases[0].answerable is True


def test_question_markdown_and_key_must_match(evaluation_settings):
    rows = _rows(evaluation_settings)
    rows[0]["question"] = "שאלה שונה מזו שבקובץ הבדיקה?"
    _write_rows(evaluation_settings, rows)
    with pytest.raises(evaluation.EvaluationError):
        evaluation.load_evaluation_cases(evaluation_settings)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "Q02"),
        ("expected_answer", ""),
        ("source_doc_ids", ["ALT-99"]),
        ("answerable", "false"),
        ("category", ""),
    ],
)
def test_invalid_case_is_rejected(evaluation_settings, field, value):
    rows = _rows(evaluation_settings)
    rows[0][field] = value
    _write_rows(evaluation_settings, rows)
    with pytest.raises(evaluation.EvaluationError):
        evaluation.load_evaluation_cases(evaluation_settings)


def test_missing_case_is_rejected(evaluation_settings):
    _write_rows(evaluation_settings, _rows(evaluation_settings)[:-1])
    with pytest.raises(evaluation.EvaluationError):
        evaluation.load_evaluation_cases(evaluation_settings)


def test_duplicate_json_keys_are_rejected(evaluation_settings):
    path = evaluation_settings.eval_path / ANSWERS_FILE
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('"id": "Q01"', '"id": "Q00", "id": "Q01"', 1), encoding="utf-8")
    with pytest.raises(evaluation.EvaluationError):
        evaluation.load_evaluation_cases(evaluation_settings)


def test_only_original_question_is_passed_to_answer_service(evaluation_settings, mock_index, tmp_path):
    rows = _rows(evaluation_settings)
    rows[0]["expected_answer"] = KEY_SENTINEL
    _write_rows(evaluation_settings, rows)
    chunk = passage()
    service = ControlledService(Answer("טקסט נתמך לצורך בדיקה.", "answered", sources=(_source(chunk),), offline=True), (chunk,))
    report = _run(evaluation_settings, tmp_path, question_ids=["Q01"], service=service)
    assert service.calls == [rows[0]["question"]]
    assert report["mode"] == "offline_test_answers"
    assert report["rows"][0]["execution"] == "completed"
    assert report["rows"][0]["semantic_review"] == "pending"
    assert report["summary"]["semantic_review_pending"] == 1
    assert mock_index == [(evaluation_settings, True)]


def test_retrieval_mode_does_not_call_answer_service(evaluation_settings, mock_index, tmp_path):
    retriever = ControlledRetriever((passage(),))
    report = _run(evaluation_settings, tmp_path, mode="retrieval", question_ids=["Q01"], retriever=retriever)
    assert retriever.calls == ["שאלת בדיקה טכנית 01?"]
    assert report["mode"] == "offline_retrieval"
    row = report["rows"][0]
    assert row["answer"] is None
    assert row["semantic_review"] == "not_applicable"
    assert row["retrieved"][0]["chunk_id"] == passage().chunk_id
    assert row["retrieved"][0]["rank"] == 1


@pytest.mark.parametrize("component", ["service", "retriever"])
def test_injected_components_require_explicit_offline_mode(evaluation_settings, mock_index, tmp_path, component):
    value = ControlledService() if component == "service" else ControlledRetriever()
    with pytest.raises(evaluation.EvaluationError):
        evaluation.run_evaluation(
            evaluation_settings,
            offline=False,
            output_dir=tmp_path / "rejected",
            question_ids=["Q01"],
            **{component: value},
        )
    assert not value.calls


@pytest.mark.parametrize("ids", [[], ["Q01", "Q01"], ["Q99"]])
def test_question_selection_must_be_nonempty_unique_and_known(evaluation_settings, mock_index, tmp_path, ids):
    service = ControlledService()
    with pytest.raises(evaluation.EvaluationError):
        _run(evaluation_settings, tmp_path, question_ids=ids, service=service)
    assert not service.calls


def test_selected_cases_keep_the_original_dataset_order(evaluation_settings, mock_index, tmp_path):
    service = ControlledService(Answer(NO_ANSWER_TEXT, "no_answer", offline=True))
    report = _run(evaluation_settings, tmp_path, question_ids=["Q03", "Q01"], service=service)
    assert report["selected_ids"] == ["Q01", "Q03"]
    assert service.calls == ["שאלת בדיקה טכנית 01?", "שאלת בדיקה טכנית 03?"]


def test_technical_error_is_not_recorded_as_refusal_or_completed(evaluation_settings, mock_index, tmp_path):
    service = ControlledService(Answer("שגיאת שירות.", "error", offline=True, error_code="service_timeout"))
    report = _run(evaluation_settings, tmp_path, question_ids=["Q01"], service=service)
    row = report["rows"][0]
    assert row["execution"] == "error"
    assert row["answer"]["status"] == "error"
    assert row["semantic_review"] == "not_run"
    assert report["summary"]["completed"] == 0
    assert report["summary"]["errors"] == 1
    assert report["summary"]["semantic_review_pending"] == 0


def test_two_consecutive_errors_stop_and_preserve_not_run_cases(evaluation_settings, mock_index, tmp_path):
    service = ControlledService(failure=RuntimeError("private provider payload must stay private"))
    report = _run(evaluation_settings, tmp_path, question_ids=["Q01", "Q02", "Q03"], service=service)
    assert len(service.calls) == 2
    assert [row["execution"] for row in report["rows"]] == ["error", "error", "not_run"]
    assert report["summary"]["errors"] == 2
    assert report["summary"]["not_run"] == 1
    persisted = Path(report["report_path"]).read_text(encoding="utf-8")
    assert "private provider payload" not in persisted
    assert json.loads(persisted)["rows"] == report["rows"]


def test_source_coverage_uses_all_retrieved_not_only_cited_sources(evaluation_settings, mock_index, tmp_path):
    rows = _rows(evaluation_settings)
    rows[0]["source_doc_ids"] = ["ALT-01", "ALT-02", "ALT-03"]
    _write_rows(evaluation_settings, rows)
    first, second = passage(), passage(2)
    service = ControlledService(Answer("טקסט בדיקה.", "answered", sources=(_source(first),), offline=True), (first, second))
    report = _run(evaluation_settings, tmp_path, question_ids=["Q01"], service=service)
    coverage = report["rows"][0]["source_coverage"]
    assert set(coverage["retrieved_source_ids"]) == {"ALT-01", "ALT-02"}
    assert set(coverage["matched_source_ids"]) == {"ALT-01", "ALT-02"}
    assert coverage["recall"] == pytest.approx(2 / 3)


def test_unanswerable_case_has_no_positive_source_recall(evaluation_settings, mock_index, tmp_path):
    rows = _rows(evaluation_settings)
    rows[0]["answerable"] = False
    rows[0]["source_doc_ids"] = ["ALT-19"]
    _write_rows(evaluation_settings, rows)
    service = ControlledService(Answer(NO_ANSWER_TEXT, "no_answer", offline=True))
    report = _run(evaluation_settings, tmp_path, question_ids=["Q01"], service=service)
    row = report["rows"][0]
    assert row["execution"] == "completed"
    assert row["source_coverage"]["recall"] is None
    assert row["semantic_review"] == "pending"


@pytest.mark.parametrize("violation", ["too_many_chunks", "unretrieved_citation", "inexact_quote"])
def test_structural_failures_remain_visible(evaluation_settings, mock_index, tmp_path, violation):
    first = passage()
    chunks = (first,)
    source = _source(first)
    if violation == "too_many_chunks":
        chunks = tuple(passage(number) for number in range(1, 7))
    elif violation == "unretrieved_citation":
        source = _source(passage(2))
    else:
        source = replace(source, quote="ציטוט מומצא שלא מופיע במקטע שנשלף.")
    service = ControlledService(Answer("טקסט לצורך בדיקה.", "answered", sources=(source,), offline=True), chunks)
    report = _run(evaluation_settings, tmp_path, question_ids=["Q01"], service=service)
    checks = report["rows"][0]["checks"]
    assert False in checks.values()
    assert report["summary"]["structural_failures"] == 1


def test_existing_reports_are_never_overwritten(evaluation_settings, mock_index, tmp_path):
    service = ControlledService(Answer(NO_ANSWER_TEXT, "no_answer", offline=True))
    report = _run(evaluation_settings, tmp_path, question_ids=["Q01"], service=service)
    report_path = Path(report["report_path"])
    original = report_path.read_bytes()
    another = ControlledService()
    with pytest.raises(evaluation.EvaluationError):
        _run(evaluation_settings, tmp_path, question_ids=["Q01"], service=another)
    assert report_path.read_bytes() == original
    assert not another.calls


@pytest.mark.parametrize("status", ["answered", "partial"])
def test_substantive_answer_requires_evidence_even_if_no_citation_was_invented(
    evaluation_settings, mock_index, tmp_path, status
):
    service = ControlledService(Answer("טענה ללא ראיה שנדרש לזהות בבדיקה.", status, offline=True), (passage(),))
    report = _run(evaluation_settings, tmp_path, question_ids=["Q01"], service=service)
    row = report["rows"][0]
    assert row["execution"] == "completed"
    assert report["summary"]["structural_failures"] == 1
    assert row["checks"]["exact_quotes"] is False
    assert row["semantic_review"] == "pending"


def test_rate_limit_stops_immediately_with_other_cases_marked_not_run(evaluation_settings, mock_index, tmp_path):
    service = ControlledService(Answer("המכסה מוצתה.", "error", offline=True, error_code="rate_limit"))
    report = _run(evaluation_settings, tmp_path, question_ids=["Q01", "Q02", "Q03"], service=service)
    assert len(service.calls) == 1
    assert [row["execution"] for row in report["rows"]] == ["error", "not_run", "not_run"]
    assert report["summary"]["errors"] == 1
    assert report["summary"]["not_run"] == 2


def test_result_is_persisted_before_the_next_question(evaluation_settings, mock_index, tmp_path):
    report_path = tmp_path / "reports" / "report.json"

    class ObservingService(ControlledService):
        def ask(self, question):
            snapshot = json.loads(report_path.read_text(encoding="utf-8"))
            if self.calls:
                assert snapshot["rows"][0]["execution"] == "completed"
                assert snapshot["rows"][1]["execution"] == "not_run"
            return super().ask(question)

    service = ObservingService(Answer(NO_ANSWER_TEXT, "no_answer", offline=True))
    report = _run(evaluation_settings, tmp_path, question_ids=["Q01", "Q02"], service=service)
    assert report["summary"]["completed"] == 2


@pytest.mark.parametrize("destination", ["knowledge_base_path", "eval_path", "chroma_path"])
def test_report_output_cannot_overlap_sources_key_or_vector_database(
    evaluation_settings, mock_index, tmp_path, destination
):
    service = ControlledService()
    with pytest.raises(evaluation.EvaluationError):
        evaluation.run_evaluation(
            evaluation_settings,
            offline=True,
            service=service,
            question_ids=["Q01"],
            output_dir=getattr(evaluation_settings, destination),
            request_interval_seconds=0,
        )
    assert not service.calls


@pytest.mark.parametrize("interval", [float("nan"), float("inf"), -1, 61, True])
def test_invalid_intervals_are_rejected_before_service_calls(evaluation_settings, mock_index, tmp_path, interval):
    service = ControlledService()
    with pytest.raises(evaluation.EvaluationError):
        evaluation.run_evaluation(
            evaluation_settings,
            offline=True,
            service=service,
            question_ids=["Q01"],
            output_dir=tmp_path / "invalid-interval",
            request_interval_seconds=interval,
        )
    assert not service.calls


def test_cli_reports_invalid_offline_answer_mode_without_fake_fallback(evaluation_settings, mock_index, monkeypatch, capsys):
    from scripts import evaluate as command

    monkeypatch.setattr(command, "load_settings", lambda: evaluation_settings)
    code = command.main(["--offline", "--ids", "Q01"])
    assert code == 2
    assert "explicit test service" in capsys.readouterr().err
