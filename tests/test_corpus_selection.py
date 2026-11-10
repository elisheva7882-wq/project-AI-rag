"""Offline checks for the exact knowledge boundary; evaluation never enters it."""

from __future__ import annotations

import hashlib
import os
from dataclasses import FrozenInstanceError

import pytest

from src.file_loader import (
    CorpusError,
    EXPECTED_DOCUMENT_IDS,
    load_corpus,
    validate_corpus,
    validate_document,
)


def test_valid_corpus_has_exactly_the_nineteen_archive_ids(corpus_entries):
    documents = validate_corpus(dict(reversed(list(corpus_entries.items()))))
    expected = tuple(f"ALT-{number:02d}" for number in range(1, 20))
    assert tuple(document.document_id for document in documents) == expected
    assert tuple(EXPECTED_DOCUMENT_IDS) == expected


def test_document_preserves_text_metadata_and_original_byte_hash(corpus_entries):
    filename, original = next(iter(corpus_entries.items()))
    raw = b"\xef\xbb\xbf" + original
    document = validate_document(filename, raw)
    assert document.document_id == "ALT-01"
    assert document.source_file == filename
    assert document.title == "מסמך טכני לבדיקת טעינה 01"
    assert document.document_type == "קובץ בדיקה בלבד"
    assert document.period == "נתון טכני ללא משמעות היסטורית"
    assert document.topics == "בדיקת תוכנה, קידוד"
    assert document.text == original.decode("utf-8")
    assert document.content_hash == hashlib.sha256(raw).hexdigest()
    with pytest.raises(FrozenInstanceError):
        document.document_id = "ALT-02"


def test_main_heading_is_title_fallback(corpus_entries):
    filename, raw = next(iter(corpus_entries.items()))
    text = raw.decode("utf-8")
    text = "\n".join(line for line in text.splitlines() if not line.startswith("כותרת:"))
    assert validate_document(filename, text.encode("utf-8")).title == "מסמך טכני לבדיקת טעינה 01"


@pytest.mark.parametrize("data", [b"", b" \r\n\t", b"\xef\xbb\xbf", b"\xff\xfe\x00\x00"])
def test_empty_or_invalid_utf8_document_is_rejected(data):
    with pytest.raises(CorpusError):
        validate_document("ALT-01_בדיקה.md", data)


@pytest.mark.parametrize(
    "filename",
    [
        "README.md",
        "manifest.json",
        "ALT-01_duplicate.txt",
        "ALT-00_invalid.md",
        "ALT-20_invalid.md",
        "ALT-1_invalid.md",
        "../ALT-01_escape.md",
        "eval/ALT-01_question.md",
        "שאלות_בדיקה_בלבד.md",
        "ALT-01_bad?.md",
        "ALT-01_bad*.md",
        "ALT-01_bad|.md",
        "ALT-01_bad<.md",
        "ALT-01_bad>.md",
        'ALT-01_bad".md',
        "ALT-01_bad\x00.md",
    ],
)
def test_only_original_alt_markdown_filenames_are_accepted(corpus_entries, filename):
    with pytest.raises(CorpusError):
        validate_document(filename, next(iter(corpus_entries.values())))


def test_filename_id_and_body_id_must_match(corpus_entries):
    raw = corpus_entries["ALT-01_בדיקה.md"]
    with pytest.raises(CorpusError):
        validate_document("ALT-02_בדיקה.md", raw)


def test_document_id_header_cannot_be_missing_or_ambiguous(corpus_entries):
    raw = corpus_entries["ALT-01_בדיקה.md"].decode("utf-8")
    for text in (
        raw.replace("מזהה מסמך: ALT-01\n", ""),
        raw.replace("מזהה מסמך: ALT-01", "מזהה מסמך: ALT-01\nמזהה מסמך: ALT-02"),
    ):
        with pytest.raises(CorpusError):
            validate_document("ALT-01_בדיקה.md", text.encode("utf-8"))


def test_metadata_without_document_body_is_rejected():
    raw = "# בדיקה\n\nמזהה מסמך: ALT-01\nכותרת: בדיקה\nסוג מסמך: בדיקה\n"
    with pytest.raises(CorpusError):
        validate_document("ALT-01_בדיקה.md", raw.encode("utf-8"))


def test_missing_document_prevents_partial_corpus(corpus_entries):
    del corpus_entries["ALT-19_בדיקה.md"]
    with pytest.raises(CorpusError):
        validate_corpus(corpus_entries)


def test_duplicate_document_id_cannot_replace_a_missing_id(corpus_entries):
    del corpus_entries["ALT-19_בדיקה.md"]
    corpus_entries["ALT-01_עותק.md"] = corpus_entries["ALT-01_בדיקה.md"]
    with pytest.raises(CorpusError):
        validate_corpus(corpus_entries)


def test_byte_identical_duplicate_is_rejected(corpus_entries):
    corpus_entries["ALT-19_בדיקה.md"] = corpus_entries["ALT-01_בדיקה.md"]
    with pytest.raises(CorpusError):
        validate_corpus(corpus_entries)


def test_directory_load_is_sorted_and_does_not_rewrite_sources(corpus_directory):
    before = {path.name: path.read_bytes() for path in corpus_directory.iterdir()}
    documents = load_corpus(corpus_directory)
    assert len(documents) == 19
    assert documents[0].document_id == "ALT-01"
    assert documents[-1].document_id == "ALT-19"
    assert {path.name: path.read_bytes() for path in corpus_directory.iterdir()} == before


@pytest.mark.parametrize(
    "extra",
    ["README.md", "manifest.json", "ALT-01_duplicate.txt", "שאלות_בדיקה_בלבד.md", "מפתח_תשובות_ומקורות.jsonl"],
)
def test_contaminated_knowledge_directory_is_rejected(corpus_directory, extra):
    (corpus_directory / extra).write_text("FORBIDDEN_EVALUATION_FIXTURE", encoding="utf-8")
    with pytest.raises(CorpusError):
        load_corpus(corpus_directory)


def test_loader_rejects_subdirectories_instead_of_recursive_loading(corpus_directory):
    (corpus_directory / "eval").mkdir()
    (corpus_directory / "eval" / "answers.md").write_text("FORBIDDEN_EVALUATION_FIXTURE", encoding="utf-8")
    with pytest.raises(CorpusError):
        load_corpus(corpus_directory)


def test_loader_rejects_missing_directory(tmp_path):
    with pytest.raises(CorpusError):
        load_corpus(tmp_path / "absent")


def test_loader_rejects_document_symlink_when_platform_allows_it(corpus_directory, tmp_path):
    original = corpus_directory / "ALT-01_בדיקה.md"
    external = tmp_path / "outside.md"
    external.write_bytes(original.read_bytes())
    original.unlink()
    try:
        os.symlink(external, original)
    except OSError as exc:
        pytest.skip(f"Creating symlinks is unavailable in this Windows environment: {exc.winerror}")
    with pytest.raises(CorpusError):
        load_corpus(corpus_directory)
