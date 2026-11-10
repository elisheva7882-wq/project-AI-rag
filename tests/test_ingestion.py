"""Real persistent Chroma with explicit offline embeddings, never Gemini calls."""

from __future__ import annotations

import json
import socket
from dataclasses import replace
from uuid import uuid4

import pytest
from llama_index.core.embeddings import MockEmbedding

from config import load_settings
from src.gemini_models import ProviderError
from src.indexer import IndexingError, get_manifest_path, ingest, validate_index
from src.vector_store import (
    GENERATION_METADATA_KEY,
    VectorStoreError,
    as_vector_store,
    create_staging_collection,
    get_collection,
    open_client,
    promote_collection,
)


OFFLINE_MODEL = "kingdomai-offline-hash-v1"
OFFLINE_COLLECTION = "kingdomai_altheria_docs_offline"


@pytest.fixture
def index_settings(tmp_path, corpus_directory):
    root = tmp_path / "project"
    root.mkdir()
    return load_settings(project_root=root, environ={"KNOWLEDGE_BASE_PATH": str(corpus_directory)})


def _collection(settings):
    return get_collection(open_client(settings.project_root / "storage" / "offline" / "chroma"), OFFLINE_COLLECTION)


class FailingEmbedding(MockEmbedding):
    def _get_text_embedding(self, text):
        raise RuntimeError("injected embedding error; fake-secret-must-not-be-printed")


def test_offline_build_is_persistent_and_clearly_separated_from_gemini(index_settings):
    result = ingest(index_settings, rebuild=True, offline=True)
    assert result.documents == 19
    assert result.chunks >= 19
    assert result.rebuilt is True
    assert result.offline is True
    assert result.collection_name == OFFLINE_COLLECTION
    assert result.manifest_path == get_manifest_path(index_settings, offline=True)
    assert result.manifest_path == index_settings.project_root / "storage" / "offline" / "index_manifest.json"
    assert not get_manifest_path(index_settings).exists()
    assert not index_settings.chroma_path.exists()
    manifest = validate_index(index_settings, offline=True)
    assert isinstance(manifest, dict)
    assert OFFLINE_MODEL in result.manifest_path.read_text(encoding="utf-8")

    reopened = _collection(index_settings)
    assert reopened.count() == result.chunks
    stored = reopened.get(include=["metadatas", "documents", "embeddings"])
    assert len(set(stored["ids"])) == result.chunks
    assert {item["document_id"] for item in stored["metadatas"]} == {f"ALT-{number:02d}" for number in range(1, 20)}
    assert all(len(vector) == 64 for vector in stored["embeddings"])
    assert all(item["chunk_id"] == node_id for item, node_id in zip(stored["metadatas"], stored["ids"]))
    assert reopened.configuration["hnsw"]["space"] == "cosine"


def test_rebuild_replaces_target_without_duplicates_or_changes_to_other_collections(index_settings):
    first = ingest(index_settings, rebuild=True, offline=True)
    client = open_client(index_settings.project_root / "storage" / "offline" / "chroma")
    unrelated = client.create_collection("unrelated_collection", embedding_function=None)
    unrelated.add(ids=["keep-me"], embeddings=[[1.0, 2.0, 3.0]], documents=["Unrelated technical data"])
    before_target = _collection(index_settings)
    before_ids = set(before_target.get()["ids"])
    second = ingest(index_settings, rebuild=True, offline=True)
    after_target = _collection(index_settings)
    assert first.chunks == second.chunks == after_target.count()
    assert set(after_target.get()["ids"]) == before_ids
    assert before_target.id != after_target.id
    assert get_collection(client, "unrelated_collection").get()["ids"] == ["keep-me"]
    assert {collection.name for collection in client.list_collections()} == {OFFLINE_COLLECTION, "unrelated_collection"}


def test_reuse_performs_no_embedding_and_does_not_rewrite_manifest(index_settings):
    first = ingest(index_settings, rebuild=True, offline=True)
    before = (first.manifest_path.read_bytes(), first.manifest_path.stat().st_mtime_ns, _collection(index_settings).id)
    failing = FailingEmbedding(embed_dim=64, model_name=OFFLINE_MODEL)
    reused = ingest(index_settings, offline=True, embed_model=failing)
    assert reused.rebuilt is False
    assert reused.chunks == first.chunks
    assert (reused.manifest_path.read_bytes(), reused.manifest_path.stat().st_mtime_ns, _collection(index_settings).id) == before


def test_failed_embeddings_leave_existing_collection_and_manifest_unchanged(index_settings):
    first = ingest(index_settings, rebuild=True, offline=True)
    before = (first.manifest_path.read_bytes(), _collection(index_settings).id, _collection(index_settings).get())
    with pytest.raises((IndexingError, ProviderError)) as failure:
        ingest(index_settings, rebuild=True, offline=True, embed_model=FailingEmbedding(embed_dim=64, model_name=OFFLINE_MODEL), fresh_embeddings=True)
    assert "fake-secret-must-not-be-printed" not in str(failure.value)
    assert (first.manifest_path.read_bytes(), _collection(index_settings).id, _collection(index_settings).get()) == before


def test_wrong_vector_dimension_cannot_replace_valid_index(index_settings):
    first = ingest(index_settings, rebuild=True, offline=True)
    before = (first.manifest_path.read_bytes(), _collection(index_settings).id)
    with pytest.raises(IndexingError):
        ingest(index_settings, rebuild=True, offline=True, embed_model=MockEmbedding(embed_dim=63, model_name=OFFLINE_MODEL), fresh_embeddings=True)
    assert (first.manifest_path.read_bytes(), _collection(index_settings).id) == before


def test_manifest_commit_failure_restores_original_collection(index_settings):
    from llama_index.core.schema import TextNode

    first = ingest(index_settings, rebuild=True, offline=True)
    client = open_client(index_settings.project_root / "storage" / "offline" / "chroma")
    before = (first.manifest_path.read_bytes(), _collection(index_settings).id, _collection(index_settings).get())
    staging = create_staging_collection(client, uuid4().hex)
    as_vector_store(staging).add([TextNode(id_="new-test-only-node", text="Uncommitted technical fixture", embedding=[0.5] * 64)])

    def failed_atomic_commit():
        raise OSError("Injected failure before replacing the manifest")

    with pytest.raises(VectorStoreError):
        promote_collection(client, staging.name, OFFLINE_COLLECTION, failed_atomic_commit)
    assert (first.manifest_path.read_bytes(), _collection(index_settings).id, _collection(index_settings).get()) == before
    assert {collection.name for collection in client.list_collections()} == {OFFLINE_COLLECTION}


def test_ingestion_atomic_manifest_failure_preserves_previous_valid_index(index_settings, monkeypatch):
    import src.indexer as indexer

    first = ingest(index_settings, rebuild=True, offline=True)
    before = (first.manifest_path.read_bytes(), _collection(index_settings).id, _collection(index_settings).get())

    def failing_writer(path, manifest):
        raise OSError("Injected atomic manifest failure")

    monkeypatch.setattr(indexer, "_write_manifest_atomic", failing_writer)
    with pytest.raises((IndexingError, VectorStoreError)):
        ingest(index_settings, rebuild=True, offline=True)
    assert (first.manifest_path.read_bytes(), _collection(index_settings).id, _collection(index_settings).get()) == before
    assert validate_index(index_settings, offline=True)


def test_missing_gemini_key_never_falls_back_or_writes_production_storage(index_settings):
    assert not index_settings.has_api_key
    with pytest.raises((IndexingError, ProviderError)):
        ingest(index_settings, rebuild=True)
    assert not index_settings.chroma_path.exists()
    assert not get_manifest_path(index_settings).exists()


def test_missing_index_cannot_be_reused_silently(index_settings):
    with pytest.raises(IndexingError):
        ingest(index_settings, offline=True)
    assert not get_manifest_path(index_settings, offline=True).exists()


def test_changed_source_bytes_require_rebuild(index_settings):
    ingest(index_settings, rebuild=True, offline=True)
    source = index_settings.knowledge_base_path / "ALT-01_בדיקה.md"
    source.write_bytes(source.read_bytes() + b"\nAdditional technical source data.\n")
    with pytest.raises(IndexingError):
        validate_index(index_settings, offline=True)


def test_changed_chunk_configuration_requires_rebuild(index_settings):
    ingest(index_settings, rebuild=True, offline=True)
    with pytest.raises(IndexingError):
        validate_index(replace(index_settings, chunk_size=256, chunk_overlap=32), offline=True)


@pytest.mark.parametrize("change", [{"embedding_model": "different-embedding-model"}, {"embedding_dimension": 1536}])
def test_changed_requested_embedding_model_or_dimension_requires_rebuild(index_settings, change):
    ingest(index_settings, rebuild=True, offline=True)
    with pytest.raises(IndexingError):
        validate_index(replace(index_settings, **change), offline=True)


@pytest.mark.parametrize("field,value", [("model", "incorrect-offline-model"), ("dimension", 63)])
def test_manifest_embedding_model_or_dimension_corruption_is_rejected(index_settings, field, value):
    result = ingest(index_settings, rebuild=True, offline=True)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["embedding"][field] = value
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(IndexingError):
        validate_index(index_settings, offline=True)


def test_corrupt_manifest_is_rejected(index_settings):
    result = ingest(index_settings, rebuild=True, offline=True)
    result.manifest_path.write_text("{ not valid JSON", encoding="utf-8")
    with pytest.raises(IndexingError):
        validate_index(index_settings, offline=True)


def test_generation_mismatch_is_detected(index_settings):
    ingest(index_settings, rebuild=True, offline=True)
    collection = _collection(index_settings)
    metadata = dict(collection.metadata or {})
    metadata[GENERATION_METADATA_KEY] = uuid4().hex
    collection.modify(metadata=metadata)
    with pytest.raises(IndexingError):
        validate_index(index_settings, offline=True)


def test_deleted_chunk_is_detected(index_settings):
    ingest(index_settings, rebuild=True, offline=True)
    collection = _collection(index_settings)
    collection.delete(ids=collection.get()["ids"][:1])
    with pytest.raises(IndexingError):
        validate_index(index_settings, offline=True)


def test_corrupted_source_metadata_is_detected(index_settings):
    ingest(index_settings, rebuild=True, offline=True)
    collection = _collection(index_settings)
    stored = collection.get(limit=1, include=["metadatas"])
    metadata = dict(stored["metadatas"][0])
    metadata["document_id"] = "ALT-99"
    collection.update(ids=stored["ids"], metadatas=[metadata])
    with pytest.raises(IndexingError):
        validate_index(index_settings, offline=True)


def test_offline_build_does_not_read_evaluation_material(index_settings, monkeypatch):
    from pathlib import Path

    index_settings.eval_path.mkdir()
    (index_settings.eval_path / "ALT-01_forbidden.md").write_text("FORBIDDEN_EVALUATION_SENTINEL", encoding="utf-8")
    original_open = Path.open

    def protected_open(path, *args, **kwargs):
        if path.resolve().is_relative_to(index_settings.eval_path):
            raise AssertionError("Indexer must never open evaluation material")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", protected_open)
    result = ingest(index_settings, rebuild=True, offline=True)
    assert "FORBIDDEN_EVALUATION_SENTINEL" not in result.manifest_path.read_text(encoding="utf-8")
    assert all("FORBIDDEN_EVALUATION_SENTINEL" not in text for text in _collection(index_settings).get(include=["documents"])["documents"])


def test_offline_build_requires_no_network(index_settings, monkeypatch):
    def forbidden_network(*args, **kwargs):
        raise AssertionError("Offline index build attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden_network)
    monkeypatch.setattr(socket.socket, "connect", forbidden_network)
    result = ingest(index_settings, rebuild=True, offline=True)
    assert result.offline
    assert validate_index(index_settings, offline=True)


def test_offline_cli_labels_test_vectors_and_cannot_create_production_index(index_settings, monkeypatch, capsys):
    import scripts.ingest as command

    monkeypatch.setattr(command, "load_settings", lambda: index_settings)
    assert command.main(["--offline", "--rebuild"]) == 0
    output = capsys.readouterr().out
    assert "OFFLINE TEST ONLY" in output
    assert "19 documents" in output
    assert not get_manifest_path(index_settings).exists()
    assert not index_settings.chroma_path.exists()


def test_production_cli_reports_missing_key_without_offline_fallback(index_settings, monkeypatch, capsys):
    import scripts.ingest as command

    monkeypatch.setattr(command, "load_settings", lambda: index_settings)
    assert command.main(["--rebuild"]) == 2
    output = capsys.readouterr()
    assert "GEMINI_API_KEY" in output.err
    assert "Built" not in output.out
    assert not get_manifest_path(index_settings).exists()
    assert not get_manifest_path(index_settings, offline=True).exists()
    assert not index_settings.chroma_path.exists()


@pytest.mark.parametrize("relative", ["storage", "storage/offline", "storage/offline/chroma", "storage/offline/chroma/nested"])
def test_offline_build_cannot_overlap_configured_production_storage(index_settings, relative):
    settings = load_settings(
        project_root=index_settings.project_root,
        environ={
            "KNOWLEDGE_BASE_PATH": str(index_settings.knowledge_base_path),
            "CHROMA_PATH": str(index_settings.project_root / relative),
            "CHROMA_COLLECTION": OFFLINE_COLLECTION,
        },
    )
    before = {path.relative_to(settings.project_root) for path in settings.project_root.rglob("*")}
    with pytest.raises(IndexingError):
        ingest(settings, rebuild=True, offline=True)
    assert {path.relative_to(settings.project_root) for path in settings.project_root.rglob("*")} == before


@pytest.mark.parametrize("directory", ["KNOWLEDGE_BASE_PATH", "EVAL_PATH"])
@pytest.mark.parametrize("relative", ["storage", "storage/offline", "storage/offline/data"])
def test_offline_workspace_cannot_overlap_sources_or_evaluation(index_settings, directory, relative):
    overrides = {
        "KNOWLEDGE_BASE_PATH": str(index_settings.knowledge_base_path),
        "CHROMA_PATH": str(index_settings.project_root / "other-db"),
    }
    overrides[directory] = str(index_settings.project_root / relative)
    settings = load_settings(project_root=index_settings.project_root, environ=overrides)
    before = {path.relative_to(settings.project_root) for path in settings.project_root.rglob("*")}
    with pytest.raises(IndexingError):
        ingest(settings, rebuild=True, offline=True)
    assert {path.relative_to(settings.project_root) for path in settings.project_root.rglob("*")} == before


@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("directory", ["KNOWLEDGE_BASE_PATH", "EVAL_PATH"])
def test_manifest_storage_must_not_be_inside_source_or_evaluation_directory(index_settings, directory, offline):
    overrides = {
        "KNOWLEDGE_BASE_PATH": str(index_settings.knowledge_base_path),
        "CHROMA_PATH": str(index_settings.project_root / "other-db"),
    }
    overrides[directory] = str(index_settings.project_root / "storage")
    settings = load_settings(project_root=index_settings.project_root, environ=overrides)
    before = {path.relative_to(settings.project_root) for path in settings.project_root.rglob("*")}
    with pytest.raises(IndexingError):
        ingest(settings, rebuild=True, offline=offline)
    assert {path.relative_to(settings.project_root) for path in settings.project_root.rglob("*")} == before
