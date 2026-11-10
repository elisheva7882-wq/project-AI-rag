"""Offline preparation safety and idempotence, using temporary technical fixtures."""

from __future__ import annotations

import json
import socket
import stat
import zipfile
from pathlib import Path

import pytest

from config import load_settings
import scripts.prepare_knowledge as preparation
from scripts.prepare_knowledge import discover_source, prepare_knowledge
from src.file_loader import CorpusError, load_corpus

from conftest import (
    ANSWERS_FILE,
    ARCHIVE_ROOT,
    EVALUATION_DIRECTORY,
    MARKDOWN_DIRECTORY,
    QUESTIONS_FILE,
)


def _destinations(tmp_path):
    return tmp_path / "output" / "knowledge_base", tmp_path / "output" / "eval"


def _snapshot(directory):
    if not directory.exists():
        return {}
    return {path.relative_to(directory).as_posix(): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


def test_zip_copies_only_nineteen_markdown_documents_and_separate_evaluation(make_zip, archive_entries, corpus_entries, tmp_path):
    source = make_zip()
    knowledge, evaluation = _destinations(tmp_path)
    result = prepare_knowledge(source, knowledge, evaluation)
    assert len(result.documents) == 19
    assert result.evaluation_count == 45
    assert result.copied_files == 21
    assert result.unchanged_files == 0
    assert _snapshot(knowledge) == corpus_entries
    assert set(_snapshot(evaluation)) == {QUESTIONS_FILE, ANSWERS_FILE}
    for filename in (QUESTIONS_FILE, ANSWERS_FILE):
        assert (evaluation / filename).read_bytes() == archive_entries[f"{ARCHIVE_ROOT}/{EVALUATION_DIRECTORY}/{filename}"]
    assert len(load_corpus(knowledge)) == 19
    assert not list((tmp_path / "output").rglob("*.txt"))
    assert not list((tmp_path / "output").rglob("manifest.json"))
    assert not list((tmp_path / "output").rglob("00_קראו_תחילה.md"))


@pytest.mark.parametrize("use_parent", [False, True])
def test_extracted_archive_and_its_single_parent_are_supported(extracted_archive, corpus_entries, tmp_path, use_parent):
    knowledge, evaluation = _destinations(tmp_path)
    source = extracted_archive.parent if use_parent else extracted_archive
    result = prepare_knowledge(source, knowledge, evaluation)
    assert len(result.documents) == 19
    assert _snapshot(knowledge) == corpus_entries


def test_repeated_preparation_is_unchanged_and_preserves_modification_times(make_zip, tmp_path):
    source = make_zip()
    knowledge, evaluation = _destinations(tmp_path)
    prepare_knowledge(source, knowledge, evaluation)
    files = list(knowledge.iterdir()) + list(evaluation.iterdir())
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in files}
    result = prepare_knowledge(source, knowledge, evaluation)
    assert result.copied_files == 0
    assert result.unchanged_files == 21
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in files} == before


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_original_utf8_bom_bytes_are_preserved(make_zip, archive_entries, tmp_path, newline):
    document_path = f"{ARCHIVE_ROOT}/{MARKDOWN_DIRECTORY}/ALT-01_בדיקה.md"
    archive_entries[document_path] = b"\xef\xbb\xbf" + archive_entries[document_path].replace(b"\n", newline)
    source = make_zip(archive_entries)
    knowledge, evaluation = _destinations(tmp_path)
    prepare_knowledge(source, knowledge, evaluation)
    assert (knowledge / "ALT-01_בדיקה.md").read_bytes() == archive_entries[document_path]
    assert not load_corpus(knowledge)[0].text.startswith("\ufeff")


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "../escaped.md",
        "/absolute.md",
        "C:/absolute.md",
        "C:\\absolute.md",
        "ignored/../../escaped.md",
        "ignored\\..\\escaped.md",
        "ignored/file.md:stream",
        "ignored/file?.md",
        "ignored/file*.md",
        "ignored/file|.md",
        "ignored/file<.md",
        "ignored/file>.md",
        'ignored/file".md',
    ],
)
def test_zip_slip_is_rejected_even_in_ignored_entries(make_zip, archive_entries, tmp_path, unsafe_name):
    archive_entries[unsafe_name] = b"not a knowledge source"
    source = make_zip(archive_entries)
    knowledge, evaluation = _destinations(tmp_path)
    with pytest.raises(CorpusError):
        prepare_knowledge(source, knowledge, evaluation)
    assert not _snapshot(knowledge)
    assert not _snapshot(evaluation)
    assert not (tmp_path / "escaped.md").exists()


def test_zip_symbolic_links_are_rejected_before_copying(make_zip, archive_entries, tmp_path):
    info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/ignored-link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    source = make_zip([*archive_entries.items(), (info, b"../outside")])
    knowledge, evaluation = _destinations(tmp_path)
    with pytest.raises(CorpusError):
        prepare_knowledge(source, knowledge, evaluation)
    assert not _snapshot(knowledge)
    assert not _snapshot(evaluation)


def test_duplicate_zip_member_cannot_hide_payload(make_zip, archive_entries, tmp_path):
    first_name, content = next(iter(archive_entries.items()))
    with pytest.warns(UserWarning, match="Duplicate name"):
        source = make_zip([*archive_entries.items(), (first_name, content)])
    knowledge, evaluation = _destinations(tmp_path)
    with pytest.raises(CorpusError):
        prepare_knowledge(source, knowledge, evaluation)
    assert not _snapshot(knowledge)


def test_multiple_source_roots_are_ambiguous_and_rejected(make_zip, archive_entries, tmp_path):
    extra_root = {
        name.replace(f"{ARCHIVE_ROOT}/", "second-root/", 1): content
        for name, content in archive_entries.items()
    }
    source = make_zip({**archive_entries, **extra_root})
    knowledge, evaluation = _destinations(tmp_path)
    with pytest.raises(CorpusError):
        prepare_knowledge(source, knowledge, evaluation)
    assert not _snapshot(knowledge)


def test_missing_archive_document_is_rejected_before_any_copy(make_zip, archive_entries, tmp_path):
    del archive_entries[f"{ARCHIVE_ROOT}/{MARKDOWN_DIRECTORY}/ALT-19_בדיקה.md"]
    source = make_zip(archive_entries)
    knowledge, evaluation = _destinations(tmp_path)
    with pytest.raises(CorpusError):
        prepare_knowledge(source, knowledge, evaluation)
    assert not _snapshot(knowledge)
    assert not _snapshot(evaluation)


@pytest.mark.parametrize("destination", ["knowledge", "evaluation"])
def test_conflicting_existing_file_is_preserved_without_partial_copy(make_zip, tmp_path, destination):
    source = make_zip()
    knowledge, evaluation = _destinations(tmp_path)
    directory = knowledge if destination == "knowledge" else evaluation
    directory.mkdir(parents=True)
    filename = "ALT-19_בדיקה.md" if destination == "knowledge" else ANSWERS_FILE
    (directory / filename).write_bytes(b"EXISTING USER CONTENT MUST SURVIVE")
    before = _snapshot(tmp_path / "output")
    with pytest.raises(CorpusError):
        prepare_knowledge(source, knowledge, evaluation)
    assert _snapshot(tmp_path / "output") == before


def test_unexpected_destination_file_is_not_deleted_or_ignored(make_zip, tmp_path):
    source = make_zip()
    knowledge, evaluation = _destinations(tmp_path)
    knowledge.mkdir(parents=True)
    (knowledge / "answers.jsonl").write_bytes(b"FORBIDDEN_EVALUATION_FIXTURE")
    with pytest.raises(CorpusError):
        prepare_knowledge(source, knowledge, evaluation)
    assert _snapshot(knowledge) == {"answers.jsonl": b"FORBIDDEN_EVALUATION_FIXTURE"}
    assert not _snapshot(evaluation)


def test_disk_write_failure_removes_partial_new_files_and_preserves_existing_files(make_zip, corpus_entries, tmp_path, monkeypatch):
    source = make_zip()
    knowledge, evaluation = _destinations(tmp_path)
    knowledge.mkdir(parents=True)
    (knowledge / ".gitkeep").touch()
    existing = knowledge / "ALT-01_בדיקה.md"
    existing.write_bytes(corpus_entries[existing.name])
    before = _snapshot(tmp_path / "output")
    existing_mtime = existing.stat().st_mtime_ns
    real_open = Path.open
    failure_observed = []

    class FailingWriter:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *exception):
            self.stream.close()

        def write(self, data):
            self.stream.write(data[:7])
            failure_observed.append(True)
            raise OSError("Injected disk write failure after a partial file write")

    def open_with_failure(path, mode="r", *args, **kwargs):
        stream = real_open(path, mode, *args, **kwargs)
        if mode == "xb" and path == knowledge / "ALT-03_בדיקה.md":
            return FailingWriter(stream)
        return stream

    monkeypatch.setattr(Path, "open", open_with_failure)
    with pytest.raises(CorpusError):
        prepare_knowledge(source, knowledge, evaluation)
    assert failure_observed == [True]
    assert _snapshot(tmp_path / "output") == before
    assert existing.stat().st_mtime_ns == existing_mtime
    assert not _snapshot(evaluation)


@pytest.mark.parametrize("nested", [False, True])
def test_evaluation_destination_cannot_overlap_knowledge(make_zip, tmp_path, nested):
    source = make_zip()
    knowledge = tmp_path / "knowledge_base"
    evaluation = knowledge / "eval" if nested else knowledge
    with pytest.raises(CorpusError):
        prepare_knowledge(source, knowledge, evaluation)
    assert not _snapshot(knowledge)


@pytest.mark.parametrize("failure", ["invalid_json", "missing_row", "wrong_question", "unknown_source", "duplicate_id"])
def test_invalid_evaluation_is_rejected_before_knowledge_copy(make_zip, archive_entries, tmp_path, failure):
    path = f"{ARCHIVE_ROOT}/{EVALUATION_DIRECTORY}/{ANSWERS_FILE}"
    rows = [json.loads(line) for line in archive_entries[path].decode("utf-8").splitlines()]
    if failure == "invalid_json":
        archive_entries[path] = b"{ definitely not JSON }\n"
    else:
        if failure == "missing_row":
            rows.pop()
        elif failure == "wrong_question":
            rows[0]["question"] = "Does not match the separate questions file"
        elif failure == "unknown_source":
            rows[0]["source_doc_ids"] = ["ALT-99"]
        elif failure == "duplicate_id":
            rows[-1]["id"] = rows[0]["id"]
        archive_entries[path] = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode("utf-8")
    source = make_zip(archive_entries)
    knowledge, evaluation = _destinations(tmp_path)
    with pytest.raises(CorpusError):
        prepare_knowledge(source, knowledge, evaluation)
    assert not _snapshot(knowledge)
    assert not _snapshot(evaluation)


def test_preparation_requires_no_network_or_api_key(make_zip, tmp_path, monkeypatch):
    def forbidden_network(*args, **kwargs):
        raise AssertionError("Step 2 must not access the network")

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(socket, "create_connection", forbidden_network)
    monkeypatch.setattr(socket.socket, "connect", forbidden_network)
    source = make_zip()
    knowledge, evaluation = _destinations(tmp_path)
    assert len(prepare_knowledge(source, knowledge, evaluation).documents) == 19


def test_auto_discovery_requires_an_archive_source(tmp_path):
    with pytest.raises(CorpusError):
        discover_source(tmp_path)


def test_auto_discovery_rejects_multiple_matching_archives(make_zip, tmp_path):
    make_zip(name="first.zip")
    make_zip(name="second.zip")
    with pytest.raises(CorpusError):
        discover_source(tmp_path)


def test_auto_discovery_ignores_unrelated_zip_and_selects_single_corpus(make_zip, tmp_path):
    source = make_zip(name="kingdom.zip")
    make_zip({"unrelated.txt": b"not an Altheria corpus"}, name="unrelated.zip")
    assert discover_source(tmp_path) == source


@pytest.mark.parametrize("ambiguous", [False, True])
def test_cli_reports_missing_or_ambiguous_source_without_creating_documents(make_zip, tmp_path, monkeypatch, capsys, ambiguous):
    if ambiguous:
        make_zip(name="first.zip")
        make_zip(name="second.zip")
    settings = load_settings(project_root=tmp_path, environ={})
    monkeypatch.setattr(preparation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(preparation, "load_settings", lambda: settings)
    assert preparation.main([]) == 2
    assert "ERROR:" in capsys.readouterr().err
    assert not settings.knowledge_base_path.exists()
    assert not settings.eval_path.exists()
