"""Strict, offline selection of the 19 original Altheria Markdown documents.

This module never reads evaluation material or scans a project recursively.
LlamaIndex document/node creation belongs to the indexing step.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


EXPECTED_DOCUMENT_IDS = tuple(f"ALT-{number:02d}" for number in range(1, 20))
DOCUMENT_FILENAME = re.compile(r'(ALT-\d{2})_[^/\\<>:"|?*\x00-\x1f]+\.md', re.ASCII)
MAX_FILE_BYTES = 5 * 1024 * 1024


class CorpusError(ValueError):
    """Invalid or unsafe corpus input; no partial corpus may be indexed."""


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    source_file: str
    title: str
    document_type: str
    period: str
    topics: str
    text: str
    content_hash: str


def reject_links(path: Path) -> None:
    """Reject symlinks and Windows junctions before resolving a supplied path."""
    for candidate in (path.absolute(), *path.absolute().parents):
        if candidate.is_symlink() or getattr(candidate, "is_junction", lambda: False)():
            raise CorpusError(f"Links and junctions are not allowed: {candidate.name}")


def read_bytes(path: Path) -> bytes:
    reject_links(path)
    if not path.is_file():
        raise CorpusError(f"Expected a regular file: {path.name}")
    with path.open("rb") as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise CorpusError(f"File exceeds the size limit: {path.name}")
    return data


def decode_text(filename: str, data: bytes) -> str:
    if len(data) > MAX_FILE_BYTES:
        raise CorpusError(f"File exceeds the size limit: {filename}")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CorpusError(f"File must use UTF-8 or UTF-8 BOM: {filename}") from exc
    if not text.strip() or "\x00" in text:
        raise CorpusError(f"File is empty or contains NUL characters: {filename}")
    return text


def _metadata(text: str, label: str) -> str:
    matches = re.findall(rf"^{re.escape(label)}:[ \t]*([^\r\n]*)\r?$", text, re.MULTILINE)
    if len(matches) > 1:
        raise CorpusError(f"Repeated document metadata field: {label}")
    return matches[0].strip() if matches else ""


def validate_document(filename: str, data: bytes) -> SourceDocument:
    match = DOCUMENT_FILENAME.fullmatch(filename)
    if match is None or match[1] not in EXPECTED_DOCUMENT_IDS:
        raise CorpusError(f"Unexpected knowledge document filename: {filename}")
    text = decode_text(filename, data)
    document_id = _metadata(text, "מזהה מסמך")
    if document_id != match[1]:
        raise CorpusError(f"Document ID is missing or differs from filename: {filename}")
    title = _metadata(text, "כותרת")
    if not title:
        heading = re.search(r"^#[ \t]+([^\r\n]+)\r?$", text, re.MULTILINE)
        title = heading[1].strip() if heading else ""
    if not title:
        raise CorpusError(f"Document title is missing: {filename}")
    # The supplied documents separate their metadata from source text with ---.
    # Also support ordinary Markdown without this separator, while rejecting a
    # header/metadata-only file rather than mistaking it for a usable document.
    body = re.split(r"^---[ \t]*\r?$", text, maxsplit=1, flags=re.MULTILINE)
    source_lines = body[1].splitlines() if len(body) == 2 else text.splitlines()
    metadata_labels = ("מזהה מסמך:", "כותרת:", "סוג מסמך:", "תקופה:", "נושאים:", "מקור:")
    if not any(line.strip() and not line.lstrip().startswith(("#", *metadata_labels)) for line in source_lines):
        raise CorpusError(f"Document has no source text beyond its metadata: {filename}")
    return SourceDocument(
        document_id=document_id,
        source_file=filename,
        title=title,
        document_type=_metadata(text, "סוג מסמך"),
        period=_metadata(text, "תקופה"),
        topics=_metadata(text, "נושאים"),
        text=text,
        content_hash=hashlib.sha256(data).hexdigest(),
    )


def validate_corpus(entries: Mapping[str, bytes]) -> list[SourceDocument]:
    documents = [validate_document(name, data) for name, data in entries.items()]
    ids = [document.document_id for document in documents]
    if len(ids) != len(set(ids)):
        raise CorpusError("Duplicate ALT document IDs are not allowed.")
    if set(ids) != set(EXPECTED_DOCUMENT_IDS):
        missing = sorted(set(EXPECTED_DOCUMENT_IDS) - set(ids))
        raise CorpusError(f"Expected exactly 19 documents ALT-01 through ALT-19; missing: {', '.join(missing)}")
    if len({document.content_hash for document in documents}) != len(documents):
        raise CorpusError("Duplicate document contents are not allowed.")
    return sorted(documents, key=lambda document: document.document_id)


def load_corpus(path: Path) -> list[SourceDocument]:
    path = Path(path)
    reject_links(path)
    if not path.is_dir():
        raise CorpusError("Knowledge directory does not exist.")
    entries: dict[str, bytes] = {}
    for child in sorted(path.iterdir()):
        reject_links(child)
        if child.name == ".gitkeep" and child.is_file():
            continue
        if not child.is_file() or DOCUMENT_FILENAME.fullmatch(child.name) is None:
            raise CorpusError(f"Unexpected entry in knowledge directory: {child.name}")
        entries[child.name] = read_bytes(child)
    return validate_corpus(entries)
