"""Synthetic technical fixtures, written only below pytest's temporary directory."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest


ARCHIVE_ROOT = "ממלכת_אלתריה_מאגר_RAG"
MARKDOWN_DIRECTORY = "01_מסמכים_Markdown"
EVALUATION_DIRECTORY = "03_בדיקות_ללא_העלאה_ל_RAG"
QUESTIONS_FILE = "שאלות_בדיקה_בלבד.md"
ANSWERS_FILE = "מפתח_תשובות_ומקורות.jsonl"


@pytest.fixture
def corpus_entries():
    """A complete technical corpus; it contains no fictional kingdom assertions."""
    entries = {}
    for number in range(1, 20):
        text = (
            f"# מסמך טכני לבדיקת טעינה {number:02d}\n\n"
            f"מזהה מסמך: ALT-{number:02d}\n"
            f"כותרת: מסמך טכני לבדיקת טעינה {number:02d}\n"
            "סוג מסמך: קובץ בדיקה בלבד\n"
            "תקופה: נתון טכני ללא משמעות היסטורית\n"
            "נושאים: בדיקת תוכנה, קידוד\n\n"
            "---\n\n"
            "## תוכן בדיקה\n\n"
            f"תוכן טכני ייחודי לבדיקת תוכנה בלבד: {number:02d}.\n"
        )
        entries[f"ALT-{number:02d}_בדיקה.md"] = text.encode("utf-8")
    return entries


@pytest.fixture
def archive_entries(corpus_entries):
    entries = {
        f"{ARCHIVE_ROOT}/{MARKDOWN_DIRECTORY}/{name}": content
        for name, content in corpus_entries.items()
    }
    entries.update(
        {
            f"{ARCHIVE_ROOT}/02_מסמכים_TXT/{Path(name).stem}.txt": content
            for name, content in corpus_entries.items()
        }
    )
    entries[f"{ARCHIVE_ROOT}/00_קראו_תחילה.md"] = b"# Fixture introduction, not knowledge\n"
    entries[f"{ARCHIVE_ROOT}/manifest.json"] = json.dumps(
        {"document_count": 19, "assessment_questions": 45}
    ).encode("utf-8")
    questions = [f"שאלת בדיקה טכנית {number:02d}?" for number in range(1, 46)]
    entries[f"{ARCHIVE_ROOT}/{EVALUATION_DIRECTORY}/{QUESTIONS_FILE}"] = (
        "# שאלות טכניות לבדיקה בלבד\n\n"
        + "\n".join(f"{number}. {question}" for number, question in enumerate(questions, 1))
        + "\n"
    ).encode("utf-8")
    answer_rows = [
        {
            "id": f"Q{number:02d}",
            "question": question,
            "expected_answer": f"תשובת דמה טכנית {number:02d}.",
            "source_doc_ids": [f"ALT-{((number - 1) % 19) + 1:02d}"],
            "category": "בדיקת תוכנה",
            "answerable": True,
        }
        for number, question in enumerate(questions, 1)
    ]
    entries[f"{ARCHIVE_ROOT}/{EVALUATION_DIRECTORY}/{ANSWERS_FILE}"] = (
        "\n".join(json.dumps(row, ensure_ascii=False) for row in answer_rows) + "\n"
    ).encode("utf-8")
    return entries


@pytest.fixture
def make_zip(tmp_path, archive_entries):
    def write(entries=None, name="corpus.zip"):
        archive_path = tmp_path / name
        selected = archive_entries if entries is None else entries
        pairs = selected.items() if isinstance(selected, dict) else selected
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for filename, content in pairs:
                archive.writestr(filename, content)
        return archive_path

    return write


@pytest.fixture
def corpus_directory(tmp_path, corpus_entries):
    directory = tmp_path / "knowledge"
    directory.mkdir()
    (directory / ".gitkeep").touch()
    for name, content in corpus_entries.items():
        (directory / name).write_bytes(content)
    return directory


@pytest.fixture
def extracted_archive(tmp_path, archive_entries):
    source = tmp_path / "input"
    for name, content in archive_entries.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return source / ARCHIVE_ROOT
