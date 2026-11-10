"""Explicit, read-only evaluation. Gold answers never enter retrieval or prompts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from config import Settings
from scripts.prepare_knowledge import ANSWERS_FILE, EVALUATION_FILES, _validate_evaluation
from src.file_loader import CorpusError, decode_text, read_bytes, reject_links
from src.indexer import validate_index
from src.models import Answer, validate_question
from src.qa_service import QAService
from src.retriever import ArchiveRetriever
from src.service_errors import classify_service_error


class EvaluationError(ValueError):
    """Safe configuration/input errors, without provider payloads."""


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    question: str
    expected_answer: str
    source_doc_ids: tuple[str, ...]
    category: str
    answerable: bool


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationError("Duplicate evaluation JSON keys are forbidden.")
        result[key] = value
    return result


def _invalid_constant(_):
    raise EvaluationError("Non-finite evaluation JSON numbers are forbidden.")


def load_evaluation_cases(settings: Settings) -> tuple[EvaluationCase, ...]:
    """Validate both supplied files; this module is never imported by the bot."""
    try:
        reject_links(settings.eval_path)
        entries = {name: read_bytes(settings.eval_path / name) for name in EVALUATION_FILES}
        rows = [json.loads(line, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
                for line in decode_text(ANSWERS_FILE, entries[ANSWERS_FILE]).splitlines() if line.strip()]
        _validate_evaluation(entries)
        for row in rows:
            validate_question(row["question"])
        return tuple(EvaluationCase(row["id"], row["question"], row["expected_answer"],
                                    tuple(row["source_doc_ids"]), row["category"], row["answerable"])
                     for row in rows)
    except (CorpusError, OSError, ValueError, KeyError, TypeError) as exc:
        raise EvaluationError("Evaluation files are invalid, unsafe, or do not match the 45 supplied questions.") from exc


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _output_directory(settings: Settings, requested) -> Path:
    path = Path(requested) if requested is not None else (
        settings.project_root / "storage" / "evaluations" /
        (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]))
    try:
        reject_links(path)
        path = path.resolve()
        protected = (settings.knowledge_base_path, settings.eval_path, settings.chroma_path,
                     settings.project_root / "storage" / "offline" / "chroma")
        if any(_overlap(path, target.resolve()) for target in protected):
            raise EvaluationError("Evaluation output must be separate from knowledge, evaluation inputs and Chroma storage.")
        if path.exists() and not path.is_dir():
            raise EvaluationError("Evaluation output must be a directory.")
        for name in ("report.json", "report.md"):
            reject_links(path / name)
            if (path / name).exists():
                raise EvaluationError("An evaluation report already exists here; choose another output directory.")
    except (CorpusError, OSError) as exc:
        raise EvaluationError("Unsafe or unavailable evaluation output directory.") from exc
    return path


def _literal(text: str) -> str:
    # Indentation makes supplied Markdown/HTML literal, including embedded fences.
    return "\n".join("    " + line for line in str(text).splitlines()) + "\n"


def _markdown(report: dict) -> str:
    lines = ["# KingdomAI evaluation", "", f"Mode: {report['mode']}", "",
             "Structural diagnostics only. Semantic correctness requires human review.", "",
             "Reference source coverage is diagnostic; the reference list is not a mandatory citation checklist.", "",
             _literal(json.dumps(report["summary"], ensure_ascii=False, indent=2))]
    for row in report["rows"]:
        lines.extend([f"## {row['id']}", "", f"Execution: {row['execution']}; semantic review: {row['semantic_review']}", "",
                      "Question:", "", _literal(row["question"]), "Expected answer (review only):", "", _literal(row["expected_answer"])])
        if row["answer"] is not None:
            lines.extend(["Actual answer:", "", _literal(row["answer"]["text"])])
        if row["error_code"]:
            lines.extend([f"Error code: {row['error_code']}", ""])
        lines.extend(["Checks and source coverage:", "", _literal(json.dumps(
            {"checks": row["checks"], "source_coverage": row["source_coverage"]}, ensure_ascii=False, indent=2))])
        for chunk in row["retrieved"]:
            lines.extend([f"Retrieved rank {chunk['rank']}:", "", _literal(json.dumps(chunk, ensure_ascii=False, indent=2))])
    return "\n".join(lines)


def _write_atomic(path: Path, text: str) -> None:
    reject_links(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".evaluation-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _persist(report: dict) -> None:
    rows = report["rows"]
    report["updated_at"] = _now()
    report["summary"] = {
        "completed": sum(row["execution"] == "completed" for row in rows),
        "errors": sum(row["execution"] == "error" for row in rows),
        "not_run": sum(row["execution"] == "not_run" for row in rows),
        "structural_failures": sum(row["execution"] == "completed" and False in row["checks"].values() for row in rows),
        "semantic_review_pending": sum(row["semantic_review"] == "pending" for row in rows),
    }
    try:
        _write_atomic(Path(report["report_path"]), json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        _write_atomic(Path(report["markdown_path"]), _markdown(report))
    except (OSError, CorpusError, ValueError) as exc:
        raise EvaluationError("Could not persist the evaluation report; execution stopped.") from exc


_ERROR_CODES = frozenset(("missing_api_key", "access_denied", "model_unavailable", "rate_limit",
                          "service_timeout", "service_unavailable", "connection_error", "index_not_ready", "index_invalid"))
_FATAL_CODES = _ERROR_CODES - {"service_timeout", "service_unavailable", "connection_error"}


def _error_code(error) -> str:
    return classify_service_error(error) or "service_unavailable"


def _diagnostics(row: dict, chunks: tuple, answer: Answer | None, top_k: int) -> None:
    row["retrieved"] = [{**asdict(chunk), "rank": rank} for rank, chunk in enumerate(chunks, 1)]
    row["checks"]["top_k"] = len(chunks) <= top_k
    expected, actual = set(row["expected_source_ids"]), {chunk.document_id for chunk in chunks}
    row["source_coverage"] = {"expected_source_ids": sorted(expected), "retrieved_source_ids": sorted(actual),
                              "matched_source_ids": sorted(expected & actual),
                              "recall": len(expected & actual) / len(expected) if row["answerable"] and expected else None}
    if answer is not None:
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        has_required_sources = answer.status not in {"answered", "partial"} or bool(answer.sources)
        row["checks"]["cited_ids_in_retrieved"] = has_required_sources and all(
            source.chunk_id in by_id and source.document_id == by_id[source.chunk_id].document_id for source in answer.sources)
        row["checks"]["exact_quotes"] = has_required_sources and all(
            source.chunk_id in by_id and bool(source.quote) and source.quote in by_id[source.chunk_id].text for source in answer.sources)


def run_evaluation(settings: Settings, *, mode="answers", offline=False, question_ids=None, output_dir=None,
                   service=None, retriever=None, request_interval_seconds=6.0, max_consecutive_errors=2,
                   progress=None) -> dict:
    """Run selected original questions, saving each result before requesting another."""
    if mode not in {"answers", "retrieval"}:
        raise EvaluationError("Mode must be answers or retrieval.")
    if not isinstance(request_interval_seconds, (int, float)) or isinstance(request_interval_seconds, bool) or not math.isfinite(request_interval_seconds) or not 0 <= request_interval_seconds <= 60:
        raise EvaluationError("Request interval must be finite and between zero and 60 seconds.")
    if not isinstance(max_consecutive_errors, int) or isinstance(max_consecutive_errors, bool) or not 1 <= max_consecutive_errors <= 45:
        raise EvaluationError("Maximum consecutive errors must be between one and 45.")
    if not offline and (service is not None or retriever is not None):
        raise EvaluationError("Injected components require explicit offline mode.")
    if mode == "answers" and offline and service is None:
        raise EvaluationError("Offline answers require an explicit test service; no mock fallback exists.")
    if not offline and not settings.has_api_key:
        raise EvaluationError("A local GEMINI_API_KEY is required for live evaluation.")
    cases = load_evaluation_cases(settings)
    selected_ids = [case.id for case in cases] if question_ids is None else list(question_ids)
    if not selected_ids or any(not isinstance(item, str) for item in selected_ids) or len(selected_ids) != len(set(selected_ids)) or set(selected_ids) - {case.id for case in cases}:
        raise EvaluationError("Question IDs must be unique IDs from Q01 through Q45.")
    selected = [case for case in cases if case.id in selected_ids]
    destination = _output_directory(settings, output_dir)
    try:
        manifest = validate_index(settings, offline=offline)
    except Exception as exc:
        raise EvaluationError("The selected index is not ready or does not match the configuration.") from exc
    try:
        if mode == "answers":
            service = service if service is not None else QAService(settings)
        else:
            retriever = retriever if retriever is not None else ArchiveRetriever(settings, offline=offline)
    except Exception as exc:
        raise EvaluationError("Could not initialize the selected evaluation service.") from exc
    evaluation_hash = hashlib.sha256()
    for name in sorted(EVALUATION_FILES):
        evaluation_hash.update(name.encode("utf-8") + b"\0" + read_bytes(settings.eval_path / name))
    label = ("offline_test_answers" if offline else "live_answers") if mode == "answers" else ("offline_retrieval" if offline else "live_retrieval")
    report = {"schema_version": 1, "mode": label, "started_at": _now(), "dataset_total": len(cases),
              "selected_ids": [case.id for case in selected], "metadata": {
                  "llm_model": settings.llm_model,
                  "llm_request_timeout_seconds": getattr(settings, "llm_request_timeout_seconds", 60.0),
                  "llm_thinking_level": getattr(settings, "llm_thinking_level", "medium"),
                  "embedding": manifest.get("embedding", {}),
                  "chunking": manifest.get("chunking", {}), "versions": manifest.get("versions", {}),
                  "top_k": settings.top_k, "similarity_threshold": settings.similarity_threshold,
                  "generation_id": manifest.get("generation_id"), "corpus_version": manifest.get("corpus_version"),
                  "evaluation_sha256": evaluation_hash.hexdigest(),
                  "request_interval_seconds": request_interval_seconds, "max_consecutive_errors": max_consecutive_errors},
              "report_path": str(destination / "report.json"), "markdown_path": str(destination / "report.md"),
              "rows": [{"id": case.id, "question": case.question, "expected_answer": case.expected_answer,
                        "category": case.category, "answerable": case.answerable, "expected_source_ids": list(case.source_doc_ids),
                        "execution": "not_run", "answer": None, "retrieved": [], "error_code": None, "elapsed_seconds": None,
                        "source_coverage": None, "checks": {"top_k": None, "cited_ids_in_retrieved": None, "exact_quotes": None},
                        "semantic_review": "not_run"} for case in selected]}
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise EvaluationError("Could not create the evaluation output directory.") from exc
    _persist(report)
    consecutive_errors, previous_request = 0, None
    for number, row in enumerate(report["rows"], 1):
        if not offline and previous_request is not None:
            delay = request_interval_seconds - (time.monotonic() - previous_request)
            if delay > 0:
                time.sleep(delay)
        started = time.monotonic()
        previous_request = started
        try:
            answer = service.ask(row["question"]) if mode == "answers" else None
            chunks = tuple(service.last_retrieval) if answer is not None else tuple(retriever.retrieve(row["question"]))
            _diagnostics(row, chunks, answer, settings.top_k)
            if answer is not None:
                row["answer"] = asdict(answer)
            if answer is not None and answer.status == "error":
                row["execution"] = "error"
                row["error_code"] = answer.error_code if answer.error_code in _ERROR_CODES else "service_unavailable"
            else:
                row["execution"] = "completed"
                row["semantic_review"] = "pending" if answer is not None else "not_applicable"
        except Exception as exc:
            row["execution"], row["error_code"] = "error", _error_code(exc)
        row["elapsed_seconds"] = round(time.monotonic() - started, 3)
        _persist(report)
        if progress is not None:
            progress(f"{number}/{len(selected)} {row['id']}: {row['execution']}" + (f" ({row['error_code']})" if row["error_code"] else ""))
        consecutive_errors = consecutive_errors + 1 if row["execution"] == "error" else 0
        if row["error_code"] in _FATAL_CODES or consecutive_errors >= max_consecutive_errors:
            break
    return report
