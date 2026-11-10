"""LlamaIndex chunking, explicit embeddings and recoverable persistent ingestion."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Callable, Sequence
from uuid import uuid4

from llama_index.core import Document
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.core.utils import get_tokenizer

from config import Settings
from src.embedding_checkpoint import CheckpointError, EmbeddingCheckpoint
from src.file_loader import SourceDocument, load_corpus, reject_links
from src.gemini_models import (
    OFFLINE_DIMENSION,
    OFFLINE_MODEL,
    create_embedding_model,
    create_offline_embedding,
    provider_error_message,
)
from src.rate_limits import next_retry_delay
from src.vector_store import (
    GENERATION_METADATA_KEY,
    VectorStoreError,
    as_vector_store,
    create_staging_collection,
    discard_staging_collection,
    get_collection,
    open_client,
    promote_collection,
)


SCHEMA_VERSION = 1
BATCH_SIZE = 10
_DISPLAY_ONLY = ["source_file", "document_type", "period", "topics", "content_hash", "chunk_id"]


class IndexingError(ValueError):
    """An incomplete or incompatible index must never be used silently."""


class IndexNotReadyError(IndexingError):
    """No completed index is available; a partial checkpoint is not an index."""


@dataclass(frozen=True)
class IngestionResult:
    documents: int
    chunks: int
    rebuilt: bool
    offline: bool
    manifest_path: Path
    collection_name: str
    reused_embeddings: int = 0


def get_manifest_path(settings: Settings, *, offline: bool = False) -> Path:
    directory = settings.project_root / "storage"
    return (directory / "offline" if offline else directory) / "index_manifest.json"


def _storage_settings(settings: Settings, offline: bool) -> Settings:
    if not offline:
        return settings
    return replace(
        settings,
        chroma_path=settings.project_root / "storage" / "offline" / "chroma",
        chroma_collection="kingdomai_altheria_docs_offline",
    )


def _validate_output_paths(settings: Settings, offline: bool) -> None:
    manifest = get_manifest_path(settings, offline=offline).resolve()
    checkpoint_directory = manifest.parent / "embedding_checkpoints"
    for source_directory in (settings.knowledge_base_path.resolve(), settings.eval_path.resolve()):
        if manifest.is_relative_to(source_directory):
            raise IndexingError("Index manifest storage must be outside knowledge and evaluation directories.")
        if checkpoint_directory == source_directory or checkpoint_directory in source_directory.parents or source_directory in checkpoint_directory.parents:
            raise IndexingError("Embedding checkpoints must be separate from knowledge and evaluation directories.")
    if offline:
        offline_root = (settings.project_root / "storage" / "offline").resolve()
        for other in (settings.knowledge_base_path, settings.eval_path, settings.chroma_path):
            other = other.resolve()
            if offline_root == other or offline_root in other.parents or other in offline_root.parents:
                raise IndexingError("Offline test storage must be separate from production, knowledge and evaluation directories.")


def _sections(source: SourceDocument):
    # Keep every original section, including the title/metadata preamble.
    headings = list(re.finditer(r"^#{1,6}[ \t]+([^\r\n]+)\r?$", source.text, re.MULTILINE))
    starts = sorted({0, *(match.start() for match in headings)})
    titles = {match.start(): match[1].strip() for match in headings}
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(source.text)
        text = source.text[start:end]
        if text.strip():
            yield index, titles.get(start, source.title), text


def build_nodes(documents: Sequence[SourceDocument], settings: Settings) -> list[TextNode]:
    tokenizer = get_tokenizer()
    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        tokenizer=tokenizer,
        include_prev_next_rel=False,
    )
    nodes: list[TextNode] = []
    for source in documents:
        for section_index, section, text in _sections(source):
            metadata = {
                "document_id": source.document_id,
                "source_file": source.source_file,
                "title": source.title,
                "document_type": source.document_type,
                "period": source.period,
                "topics": source.topics,
                "section": section,
                "content_hash": source.content_hash,
            }
            # SOURCE is the archive ALT ID, even when a document has sections.
            # ChromaVectorStore derives its top-level document_id from this ID.
            document = Document(
                id_=source.document_id,
                text=text,
                metadata=metadata,
                excluded_embed_metadata_keys=list(_DISPLAY_ONLY),
                excluded_llm_metadata_keys=list(_DISPLAY_ONLY),
            )
            try:
                section_nodes = splitter.get_nodes_from_documents([document])
            except ValueError:
                raise IndexingError("Document metadata exceeds the available chunk token budget; reduce metadata or adjust chunk settings.") from None
            for chunk_index, node in enumerate(section_nodes):
                identity = json.dumps(
                    [source.document_id, source.content_hash, settings.chunk_size,
                     settings.chunk_overlap, section_index, chunk_index, node.text],
                    ensure_ascii=False,
                )
                node.id_ = f"{source.document_id}-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"
                node.metadata["chunk_id"] = node.id_
                if len(tokenizer(node.get_content(MetadataMode.EMBED))) > settings.chunk_size:
                    raise IndexingError("A chunk plus its embedding metadata exceeds CHUNK_SIZE; reduce metadata or rebuild settings.")
                nodes.append(node)
    if not nodes or len({node.id_ for node in nodes}) != len(nodes):
        raise IndexingError("Chunking produced no nodes or duplicate chunk IDs.")
    return nodes


def _signature(settings: Settings, documents: Sequence[SourceDocument], offline: bool) -> dict:
    sources = [
        {"document_id": source.document_id, "source_file": source.source_file, "content_hash": source.content_hash}
        for source in documents
    ]
    store = _storage_settings(settings, offline)
    return {
        "schema_version": SCHEMA_VERSION,
        "offline": offline,
        "embedding": {
            "provider": "offline-test" if offline else "google-gemini",
            "model": OFFLINE_MODEL if offline else settings.embedding_model,
            "dimension": OFFLINE_DIMENSION if offline else settings.embedding_dimension,
            "requested_model": settings.embedding_model,
            "requested_dimension": settings.embedding_dimension,
            "document_task": "RETRIEVAL_DOCUMENT",
            "query_task": "RETRIEVAL_QUERY",
        },
        "chunking": {
            "strategy": "SentenceSplitter",
            "tokenizer": "cl100k_base",
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "section_boundaries": True,
        },
        "corpus_version": hashlib.sha256(json.dumps(sources, sort_keys=True).encode("utf-8")).hexdigest(),
        "sources": sources,
        "collection_name": store.chroma_collection,
        "chroma_path": str(store.chroma_path.resolve()),
        "metric": "cosine",
        "versions": {name: version(name) for name in (
            "llama-index-core", "llama-index-vector-stores-chroma", "chromadb",
            "llama-index-embeddings-google-genai", "google-genai", "tiktoken",
        )},
    }


def _validate_collection(collection, nodes: Sequence[TextNode], dimension: int, generation_id: str) -> None:
    if (collection.metadata or {}).get(GENERATION_METADATA_KEY) != generation_id:
        raise IndexingError("Index generation differs from manifest; rebuild or inspect an interrupted promotion.")
    configuration = collection.configuration
    if configuration.get("hnsw", {}).get("space") != "cosine":
        raise IndexingError("Index collection does not use cosine distance.")
    if collection.count() != len(nodes):
        raise IndexingError("Index chunk count differs from the validated corpus.")
    stored = collection.get(include=["metadatas", "documents"])
    expected = {node.id_: node for node in nodes}
    if set(stored["ids"]) != set(expected):
        raise IndexingError("Stored chunk IDs differ from the current corpus/chunk settings; rebuild required.")
    for index, chunk_id in enumerate(stored["ids"]):
        node = expected[chunk_id]
        metadata = stored["metadatas"][index]
        if any(metadata.get(key) != value for key, value in node.metadata.items()):
            raise IndexingError("Stored source metadata differs from the validated source document.")
        if stored["documents"][index] != node.get_content(MetadataMode.NONE):
            raise IndexingError("Stored chunk text differs from its original source.")
    sample = collection.get(limit=1, include=["embeddings"])["embeddings"]
    if sample is None or len(sample) != 1 or len(sample[0]) != dimension:
        raise IndexingError("Stored embedding dimension differs from the configured model; rebuild required.")


def validate_index(settings: Settings, *, offline: bool = False) -> dict:
    _validate_output_paths(settings, offline)
    manifest_path = get_manifest_path(settings, offline=offline)
    reject_links(manifest_path)
    if not manifest_path.is_file():
        raise IndexNotReadyError("Index manifest is missing. Run python -m scripts.ingest --rebuild first.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise IndexingError("Index manifest cannot be read or is invalid JSON; inspect it before rebuilding.") from None
    if not isinstance(manifest, dict):
        raise IndexingError("Index manifest must be a JSON object.")
    documents = load_corpus(settings.knowledge_base_path)
    expected = _signature(settings, documents, offline)
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise IndexingError("Index model, dimension, corpus, storage or chunk configuration changed; explicit --rebuild is required.")
    nodes = build_nodes(documents, settings)
    if manifest.get("documents") != len(documents) or manifest.get("chunks") != len(nodes):
        raise IndexingError("Manifest document/chunk counts are invalid.")
    generation = manifest.get("generation_id")
    if not isinstance(generation, str) or re.fullmatch(r"[0-9a-f]{32}", generation) is None:
        raise IndexingError("Manifest generation ID is invalid.")
    store = _storage_settings(settings, offline)
    reject_links(store.chroma_path)
    if not (store.chroma_path / "chroma.sqlite3").is_file():
        raise IndexNotReadyError("Persistent Chroma storage is missing; rebuild required.")
    collection = get_collection(open_client(store.chroma_path), store.chroma_collection)
    _validate_collection(collection, nodes, expected["embedding"]["dimension"], generation)
    return manifest


@contextmanager
def _build_lock(manifest_path: Path):
    reject_links(manifest_path.parent)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest_path.with_suffix(".lock")
    reject_links(lock_path)
    try:
        stream = lock_path.open("x", encoding="utf-8")
    except FileExistsError:
        raise IndexingError("An index build lock exists. Wait for the active build; inspect a stale lock after a crash.") from None
    try:
        with stream:
            stream.write(str(os.getpid()))
            stream.flush()
            yield
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            warnings.warn("Build finished, but its lock could not be removed; inspect the lock before another rebuild.", RuntimeWarning)


def _write_manifest_atomic(path: Path, manifest: dict) -> None:
    reject_links(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".index_manifest_", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None  # A committed manifest must not report a cleanup failure.
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                warnings.warn("An uncommitted temporary manifest could not be removed.", RuntimeWarning)


def _embed_nodes(
    nodes: Sequence[TextNode], model: BaseEmbedding, dimension: int,
    *, checkpoint: EmbeddingCheckpoint, settings: Settings, offline: bool,
    fresh_embeddings: bool, progress: Callable[[str], None] | None,
) -> int:
    checkpoint.validate_location()
    cached = {} if fresh_embeddings else checkpoint.load()
    reused = len(cached)
    pending = []
    for node in nodes:
        if node.id_ in cached:
            node.embedding = cached[node.id_]
        else:
            pending.append(node)
    if progress:
        progress(f"Embedding progress: {reused}/{len(nodes)} reused from checkpoint; {len(pending)} remaining.")
    last_request_started: float | None = None
    for offset in range(0, len(pending), BATCH_SIZE):
        batch = pending[offset:offset + BATCH_SIZE]
        retries = 0
        while True:
            if not offline and last_request_started is not None:
                wait = settings.embedding_request_interval_seconds - (time.monotonic() - last_request_started)
                if wait > 0:
                    time.sleep(wait)
            last_request_started = time.monotonic()
            try:
                embeddings = model.get_text_embedding_batch([node.get_content(MetadataMode.EMBED) for node in batch])
                break
            except Exception as exc:
                delay = None
                if getattr(exc, "code", None) == 429 and retries < settings.embedding_max_retries:
                    delay = next_retry_delay(exc, retry_number=retries + 1, max_wait_seconds=settings.embedding_max_retry_wait_seconds)
                if delay is None:
                    raise IndexingError(
                        f"Embedding stopped after {len(cached)}/{len(nodes)} chunks; completed batches are saved. "
                        f"Re-run the same --rebuild command to resume. No partial index was published; any previous ready index is unchanged. {provider_error_message(exc)}"
                    ) from None
                retries += 1
                if progress:
                    progress(f"Gemini 429: progress saved at {len(cached)}/{len(nodes)}; waiting {delay:.1f}s before retry {retries}/{settings.embedding_max_retries}.")
                time.sleep(delay)
        if len(embeddings) != len(batch):
            raise IndexingError("Embedding response count is invalid; previous checkpoint and ready index were preserved.")
        updates = {}
        for node, vector in zip(batch, embeddings):
            if len(vector) != dimension or not all(math.isfinite(value) for value in vector) or not any(vector):
                raise IndexingError("Embedding dimension or numeric values are invalid; previous checkpoint was preserved.")
            updates[node.id_] = vector
        # Persist the entire validated batch before starting the next request.
        # A crash or quota failure can then lose at most the in-flight batch.
        checkpoint.save({**cached, **updates})
        cached.update(updates)
        for node in batch:
            node.embedding = cached[node.id_]
        if progress:
            progress(f"Embedding progress saved: {len(cached)}/{len(nodes)} chunks.")
    return reused


def ingest(
    settings: Settings,
    *,
    rebuild: bool = False,
    offline: bool = False,
    embed_model: BaseEmbedding | None = None,
    fresh_embeddings: bool = False,
    progress: Callable[[str], None] | None = None,
) -> IngestionResult:
    _validate_output_paths(settings, offline)
    if fresh_embeddings and not rebuild:
        raise IndexingError("--fresh-embeddings requires --rebuild.")
    if embed_model is not None and not offline:
        raise IndexingError("Injected embeddings are allowed only in the isolated offline test index.")
    manifest_path = get_manifest_path(settings, offline=offline)
    store = _storage_settings(settings, offline)
    if not rebuild:
        manifest = validate_index(settings, offline=offline)
        return IngestionResult(manifest["documents"], manifest["chunks"], False, offline, manifest_path, store.chroma_collection)
    documents = load_corpus(settings.knowledge_base_path)
    nodes = build_nodes(documents, settings)
    # Credentials are checked before creating storage or a lock. The offline
    # model is used only by explicit request and never in the production path.
    model = embed_model if embed_model is not None else (create_offline_embedding() if offline else create_embedding_model(settings))
    signature = _signature(settings, documents, offline)
    with _build_lock(manifest_path):
        checkpoint = EmbeddingCheckpoint(manifest_path.parent / "embedding_checkpoints", signature, nodes, signature["embedding"]["dimension"])
        try:
            reused = _embed_nodes(nodes, model, signature["embedding"]["dimension"], checkpoint=checkpoint, settings=settings, offline=offline, fresh_embeddings=fresh_embeddings, progress=progress)
        except CheckpointError as exc:
            raise IndexingError(f"Embedding progress could not be read or saved. No partial index was published; any previous ready index is unchanged. {exc}") from None
        client = open_client(store.chroma_path)
        generation_id = uuid4().hex
        staging = create_staging_collection(client, generation_id)
        try:
            as_vector_store(staging).add(nodes)
            _validate_collection(staging, nodes, signature["embedding"]["dimension"], generation_id)
            manifest = {
                **signature,
                "generation_id": generation_id,
                "documents": len(documents),
                "chunks": len(nodes),
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            promote_collection(client, staging.name, store.chroma_collection, lambda: _write_manifest_atomic(manifest_path, manifest))
        except BaseException:
            discard_staging_collection(client, "kingdomai_staging_" + generation_id)
            raise
    return IngestionResult(len(documents), len(nodes), True, offline, manifest_path, store.chroma_collection, reused)
