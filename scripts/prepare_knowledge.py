"""Safely copy only original Markdown sources; keep evaluation files separate."""

from __future__ import annotations

import argparse
import json
import re
import stat
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Mapping

from config import ConfigurationError, PROJECT_ROOT, load_settings
from src.file_loader import (
    CorpusError,
    EXPECTED_DOCUMENT_IDS,
    MAX_FILE_BYTES,
    SourceDocument,
    decode_text,
    read_bytes,
    reject_links,
    validate_corpus,
)


DOCUMENTS_DIR = "01_מסמכים_Markdown"
EVALUATION_DIR = "03_בדיקות_ללא_העלאה_ל_RAG"
QUESTIONS_FILE = "שאלות_בדיקה_בלבד.md"
ANSWERS_FILE = "מפתח_תשובות_ומקורות.jsonl"
EVALUATION_FILES = frozenset((QUESTIONS_FILE, ANSWERS_FILE))
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class PreparationResult:
    documents: tuple[SourceDocument, ...]
    evaluation_count: int
    copied_files: int
    unchanged_files: int


def _zip_path(info: zipfile.ZipInfo) -> PurePosixPath:
    name = info.filename
    parts = name.removesuffix("/").split("/")
    if (
        not name
        or "\\" in name
        or name.startswith("/")
        or any(part in {"", ".", ".."} for part in parts)
        or any(any(character in '<>:"|?*' for character in part) or part.endswith((".", " ")) for part in parts)
        or any(any(ord(character) < 32 for character in part) for part in parts)
        or any(PureWindowsPath(part).is_reserved() for part in parts)
    ):
        raise CorpusError("Unsafe ZIP member path; archive was not copied.")
    mode = stat.S_IFMT(info.external_attr >> 16)
    if mode not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise CorpusError("ZIP links and special files are not allowed.")
    return PurePosixPath(*parts)


def _read_zip(source: Path) -> tuple[dict[str, bytes], dict[str, bytes]]:
    try:
        with zipfile.ZipFile(source) as archive:
            members: list[tuple[PurePosixPath, zipfile.ZipInfo]] = []
            seen: set[str] = set()
            total_size = 0
            roots: set[PurePosixPath] = set()
            for info in archive.infolist():
                path = _zip_path(info)
                normalized = path.as_posix().casefold()
                if normalized in seen:
                    raise CorpusError("Duplicate ZIP member paths are not allowed.")
                seen.add(normalized)
                total_size += info.file_size
                if total_size > MAX_ARCHIVE_BYTES:
                    raise CorpusError("Archive exceeds the uncompressed size limit.")
                members.append((path, info))
                if DOCUMENTS_DIR in path.parts:
                    if path.parts.count(DOCUMENTS_DIR) != 1:
                        raise CorpusError("Ambiguous Markdown directory in archive.")
                    index = path.parts.index(DOCUMENTS_DIR)
                    roots.add(PurePosixPath(*path.parts[:index]))
            if len(roots) != 1:
                raise CorpusError("Expected exactly one 01_מסמכים_Markdown directory in archive.")
            root = roots.pop()
            selected: dict[str, dict[str, bytes]] = {DOCUMENTS_DIR: {}, EVALUATION_DIR: {}}
            for path, info in members:
                relative = path.relative_to(root) if path.is_relative_to(root) else None
                if relative is None or not relative.parts or relative.parts[0] not in selected:
                    continue
                if len(relative.parts) == 1 and info.is_dir():
                    continue
                if len(relative.parts) != 2 or info.is_dir():
                    raise CorpusError("Source and evaluation directories must contain only direct files.")
                if info.file_size > MAX_FILE_BYTES:
                    raise CorpusError("Selected archive file exceeds the size limit.")
                with archive.open(info) as stream:
                    data = stream.read(MAX_FILE_BYTES + 1)
                if len(data) > MAX_FILE_BYTES:
                    raise CorpusError("Selected archive file exceeds the size limit.")
                selected[relative.parts[0]][relative.name] = data
            return selected[DOCUMENTS_DIR], selected[EVALUATION_DIR]
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError) as exc:
        raise CorpusError("Archive could not be read; it may be corrupt, encrypted or unsupported.") from exc


def _read_folder(source: Path) -> tuple[dict[str, bytes], dict[str, bytes]]:
    if (source / DOCUMENTS_DIR).is_dir():
        root = source
    else:
        candidates = []
        for child in source.iterdir():
            reject_links(child)
            if child.is_dir() and (child / DOCUMENTS_DIR).is_dir():
                candidates.append(child)
        if len(candidates) != 1:
            raise CorpusError("Expected one extracted archive root containing 01_מסמכים_Markdown.")
        root = candidates[0]
    selected = []
    for directory in (root / DOCUMENTS_DIR, root / EVALUATION_DIR):
        reject_links(directory)
        if not directory.is_dir():
            raise CorpusError(f"Required archive directory is missing: {directory.name}")
        selected.append({child.name: read_bytes(child) for child in directory.iterdir()})
    return selected[0], selected[1]


def _validate_evaluation(entries: Mapping[str, bytes]) -> int:
    if set(entries) != EVALUATION_FILES:
        raise CorpusError("Expected exactly the two supplied evaluation files, kept outside knowledge_base.")
    questions_text = decode_text(QUESTIONS_FILE, entries[QUESTIONS_FILE])
    questions = re.findall(r"^[ \t]*(\d+)\.[ \t]+(.+?)[ \t]*\r?$", questions_text, re.MULTILINE)
    if [int(number) for number, _ in questions] != list(range(1, 46)):
        raise CorpusError("Evaluation Markdown must contain exactly 45 numbered questions.")
    try:
        rows = [json.loads(line) for line in decode_text(ANSWERS_FILE, entries[ANSWERS_FILE]).splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise CorpusError("Evaluation answer key must contain valid JSONL records.") from exc
    if len(rows) != 45:
        raise CorpusError("Evaluation answer key must contain exactly 45 records.")
    required_fields = {"id", "question", "expected_answer", "source_doc_ids", "category", "answerable"}
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or not required_fields <= row.keys():
            raise CorpusError("Evaluation record schema is invalid.")
        if row["id"] != f"Q{index:02d}" or row["question"] != questions[index - 1][1].strip():
            raise CorpusError("Evaluation question IDs or question text do not match the numbered questions.")
        if (
            not isinstance(row["expected_answer"], str)
            or not row["expected_answer"].strip()
            or not isinstance(row["category"], str)
            or not row["category"].strip()
            or not isinstance(row["answerable"], bool)
            or not isinstance(row["source_doc_ids"], list)
            or any(not isinstance(doc_id, str) or doc_id not in EXPECTED_DOCUMENT_IDS for doc_id in row["source_doc_ids"])
        ):
            raise CorpusError("Evaluation record values or source IDs are invalid.")
    return len(rows)


def _overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _preflight_destination(destination: Path, entries: Mapping[str, bytes]) -> set[str]:
    reject_links(destination)
    if destination.exists() and not destination.is_dir():
        raise CorpusError("Copy destination must be a directory.")
    unchanged: set[str] = set()
    if destination.is_dir():
        for child in destination.iterdir():
            reject_links(child)
            if child.name == ".gitkeep" and child.is_file():
                continue
            if not child.is_file() or child.name not in entries:
                raise CorpusError(f"Unexpected existing destination entry: {child.name}")
            if read_bytes(child) != entries[child.name]:
                raise CorpusError(f"Existing file differs; no files were copied: {child.name}")
            unchanged.add(child.name)
    return unchanged


def prepare_knowledge(source: Path, knowledge_base: Path, eval_path: Path) -> PreparationResult:
    source, knowledge_base, eval_path = map(Path, (source, knowledge_base, eval_path))
    for path in (source, knowledge_base, eval_path):
        reject_links(path)
    knowledge_base, eval_path = knowledge_base.resolve(), eval_path.resolve()
    if _overlap(knowledge_base, eval_path):
        raise CorpusError("Knowledge and evaluation destinations must be separate, non-nested directories.")
    if source.is_file():
        documents_data, evaluation_data = _read_zip(source)
    elif source.is_dir():
        documents_data, evaluation_data = _read_folder(source)
    else:
        raise CorpusError("Archive source was not found; provide --source with a ZIP or extracted directory.")
    documents = validate_corpus(documents_data)
    evaluation_count = _validate_evaluation(evaluation_data)
    plans = [(knowledge_base, documents_data), (eval_path, evaluation_data)]
    preflight = [(destination, entries, _preflight_destination(destination, entries)) for destination, entries in plans]
    created: list[Path] = []
    try:
        for destination, entries, unchanged in preflight:
            destination.mkdir(parents=True, exist_ok=True)
            for name, data in sorted(entries.items()):
                if name in unchanged:
                    continue
                # Exclusive creation protects existing files even if they appear
                # between preflight and writing. Paths come only from validated
                # direct filenames, never arbitrary ZIP member paths.
                target = destination / name
                with target.open("xb") as stream:
                    created.append(target)
                    stream.write(data)
    except OSError as exc:
        cleanup_failed = False
        for target in reversed(created):
            if target.parent in (knowledge_base, eval_path):
                try:
                    target.unlink(missing_ok=True)
                except OSError:
                    cleanup_failed = True
        if cleanup_failed:
            raise CorpusError("Copy failed; some new files could not be removed. Inspect both output directories before retrying.") from exc
        raise CorpusError("Copy failed; newly created files were rolled back. Existing files were not overwritten.") from exc
    return PreparationResult(tuple(documents), evaluation_count, len(created), sum(len(item[2]) for item in preflight))


def discover_source(project_root: Path) -> Path:
    candidates = []
    for child in project_root.iterdir():
        if child.is_file() and child.suffix.lower() == ".zip":
            reject_links(child)
            try:
                with zipfile.ZipFile(child) as archive:
                    if any(DOCUMENTS_DIR in PurePosixPath(name).parts for name in archive.namelist()):
                        candidates.append(child)
            except zipfile.BadZipFile:
                continue
        elif child.is_dir() and (child / DOCUMENTS_DIR).is_dir():
            reject_links(child)
            candidates.append(child)
    if len(candidates) != 1:
        raise CorpusError("Place exactly one Altheria ZIP/extracted archive in the project root, or specify --source explicitly.")
    return candidates[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the 19 original Altheria Markdown documents and separate evaluation files.")
    parser.add_argument("--source", type=Path, help="ZIP path or extracted archive root; otherwise discover one source in the project root.")
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
        source = args.source if args.source is not None else discover_source(PROJECT_ROOT)
        result = prepare_knowledge(source, settings.knowledge_base_path, settings.eval_path)
    except (CorpusError, ConfigurationError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Validated {len(result.documents)} original Markdown documents (ALT-01 through ALT-19).")
    print(f"Evaluation only: {result.evaluation_count} questions and answers in eval; excluded from knowledge loading.")
    print(f"Copied files: {result.copied_files}; unchanged files: {result.unchanged_files}.")
    for document in result.documents:
        print(f"{document.document_id}: {document.source_file}")
    print("No embeddings, vector database or Gemini API requests were created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
