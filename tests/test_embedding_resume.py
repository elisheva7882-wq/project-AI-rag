"""Offline recovery of paid-for embedding progress after a quota failure."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from llama_index.core.bridge.pydantic import Field
from llama_index.core.schema import MetadataMode, TextNode

from config import load_settings
from src.embedding_checkpoint import CheckpointError, EmbeddingCheckpoint
from src.file_loader import load_corpus
from src.gemini_models import OfflineEmbedding
from src.indexer import IndexingError, build_nodes, get_manifest_path, ingest, validate_index
from src.vector_store import get_collection, open_client


class ExplicitDailyQuotaError(RuntimeError):
    code = 429
    details = {
        "error": {
            "details": [{
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{"quotaId": "EmbedContentRequestsPerDay", "quotaMetric": "embed_content_requests"}],
            }],
        },
    }

    def __init__(self):
        super().__init__("SDK payload with fake-secret-must-not-be-printed")


class BatchRecordingEmbedding(OfflineEmbedding):
    batches: list[list[str]] = Field(default_factory=list, exclude=True)
    fail_after_batches: int | None = None

    def _get_text_embeddings(self, texts):
        if self.fail_after_batches is not None and len(self.batches) >= self.fail_after_batches:
            raise ExplicitDailyQuotaError()
        self.batches.append(list(texts))
        return [OfflineEmbedding._get_text_embedding(self, text) for text in texts]


@pytest.fixture
def resume_settings(tmp_path, corpus_directory):
    root = tmp_path / "resume-project"
    root.mkdir()
    return load_settings(
        project_root=root,
        environ={
            "KNOWLEDGE_BASE_PATH": str(corpus_directory),
            "EMBEDDING_MAX_RETRIES": "0",
            "EMBEDDING_REQUEST_INTERVAL_SECONDS": "0",
        },
    )


def _checkpoint_files(settings):
    return list((get_manifest_path(settings, offline=True).parent / "embedding_checkpoints").glob("*.json"))


def _offline_collection(settings):
    return get_collection(
        open_client(settings.project_root / "storage" / "offline" / "chroma"),
        "kingdomai_altheria_docs_offline",
    )


def test_quota_failure_saves_each_completed_batch_and_resume_only_embeds_remaining_nodes(resume_settings):
    nodes = build_nodes(load_corpus(resume_settings.knowledge_base_path), resume_settings)
    assert len(nodes) > 20
    failed = BatchRecordingEmbedding(fail_after_batches=2, embed_batch_size=10)
    with pytest.raises(IndexingError) as failure:
        ingest(resume_settings, rebuild=True, offline=True, embed_model=failed)
    assert "fake-secret-must-not-be-printed" not in str(failure.value)
    files = _checkpoint_files(resume_settings)
    assert len(files) == 1
    snapshot = json.loads(files[0].read_text(encoding="utf-8"))
    assert set(snapshot["embeddings"]) == {node.node_id for node in nodes[:20]}
    assert len(failed.batches) == 2
    assert not get_manifest_path(resume_settings, offline=True).exists()
    assert not get_manifest_path(resume_settings, offline=True).with_suffix(".lock").exists()
    with pytest.raises(IndexingError):
        validate_index(resume_settings, offline=True)

    resumed = BatchRecordingEmbedding(embed_batch_size=10)
    result = ingest(resume_settings, rebuild=True, offline=True, embed_model=resumed)
    expected_remaining = [node.get_content(MetadataMode.EMBED) for node in nodes[20:]]
    assert [text for batch in resumed.batches for text in batch] == expected_remaining
    assert result.chunks == len(nodes)
    assert validate_index(resume_settings, offline=True)["chunks"] == len(nodes)


def test_rebuild_reuses_complete_checkpoint_but_fresh_embeddings_explicitly_regenerates(resume_settings):
    first_model = BatchRecordingEmbedding(embed_batch_size=10)
    first = ingest(resume_settings, rebuild=True, offline=True, embed_model=first_model)
    assert first_model.batches
    no_calls = BatchRecordingEmbedding(fail_after_batches=0, embed_batch_size=10)
    second = ingest(resume_settings, rebuild=True, offline=True, embed_model=no_calls)
    assert second.chunks == first.chunks
    assert not no_calls.batches
    fresh = BatchRecordingEmbedding(embed_batch_size=10)
    ingest(resume_settings, rebuild=True, offline=True, embed_model=fresh, fresh_embeddings=True)
    assert sum(map(len, fresh.batches)) == first.chunks


def test_partial_rebuild_does_not_replace_preexisting_ready_index(resume_settings):
    first = ingest(resume_settings, rebuild=True, offline=True)
    before = (first.manifest_path.read_bytes(), _offline_collection(resume_settings).id)
    failed = BatchRecordingEmbedding(fail_after_batches=2, embed_batch_size=10)
    with pytest.raises(IndexingError):
        ingest(resume_settings, rebuild=True, offline=True, embed_model=failed, fresh_embeddings=True)
    assert before == (first.manifest_path.read_bytes(), _offline_collection(resume_settings).id)
    assert validate_index(resume_settings, offline=True)["chunks"] == first.chunks
    resumed = BatchRecordingEmbedding(embed_batch_size=10)
    result = ingest(resume_settings, rebuild=True, offline=True, embed_model=resumed)
    assert sum(map(len, resumed.batches)) == first.chunks - 20
    assert result.chunks == first.chunks
    assert _offline_collection(resume_settings).id != before[1]


def test_fresh_rebuild_rejects_checkpoint_file_before_embedding_and_preserves_ready_index(resume_settings):
    first = ingest(resume_settings, rebuild=True, offline=True)
    before = (first.manifest_path.read_bytes(), _offline_collection(resume_settings).id)
    checkpoints = _checkpoint_files(resume_settings)
    assert len(checkpoints) == 1
    checkpoint_directory = checkpoints[0].parent
    checkpoints[0].unlink()
    checkpoint_directory.rmdir()
    blocked_contents = b"A regular file cannot store embedding checkpoints."
    checkpoint_directory.write_bytes(blocked_contents)
    model = BatchRecordingEmbedding(embed_batch_size=10)

    with pytest.raises(IndexingError):
        ingest(resume_settings, rebuild=True, offline=True, embed_model=model, fresh_embeddings=True)

    assert model.batches == []
    assert checkpoint_directory.read_bytes() == blocked_contents
    assert before == (first.manifest_path.read_bytes(), _offline_collection(resume_settings).id)
    assert validate_index(resume_settings, offline=True)["chunks"] == first.chunks
    assert not first.manifest_path.with_suffix(".lock").exists()


@pytest.fixture
def checkpoint(tmp_path):
    nodes = [
        TextNode(id_="technical-a", text="Technical software fixture A; never kingdom knowledge."),
        TextNode(id_="technical-b", text="Technical software fixture B; never kingdom knowledge."),
    ]
    signature = {"embedding": {"model": "unit-test-model", "dimension": 3}, "private-test-marker": "must-not-be-stored"}
    return EmbeddingCheckpoint(tmp_path / "checkpoints", signature, nodes, 3)


def test_checkpoint_missing_is_empty_and_does_not_create_files(checkpoint):
    assert checkpoint.load() == {}
    assert not checkpoint.directory.exists()


def test_checkpoint_stores_only_opaque_identity_and_vectors(checkpoint):
    checkpoint.save({"technical-a": [1.0, 2.0, 3.0]})
    assert checkpoint.load() == {"technical-a": [1.0, 2.0, 3.0]}
    raw = checkpoint.path.read_text(encoding="utf-8")
    assert "Technical software fixture" not in raw
    assert "must-not-be-stored" not in raw
    assert "unit-test-model" not in raw
    assert not list(checkpoint.directory.glob("*.tmp"))


@pytest.mark.parametrize("change", ["model", "dimension", "payload", "order"])
def test_model_dimension_payload_or_node_order_gets_a_distinct_checkpoint(tmp_path, change):
    nodes = [TextNode(id_="a", text="Source fixture A"), TextNode(id_="b", text="Source fixture B")]
    signature = {"embedding": {"model": "unit-model", "dimension": 3}}
    original = EmbeddingCheckpoint(tmp_path, signature, nodes, 3)
    original.save({"a": [1.0, 2.0, 3.0]})
    dimension = 3
    if change == "model":
        signature = {"embedding": {"model": "different-model", "dimension": 3}}
    elif change == "dimension":
        dimension = 4
    elif change == "payload":
        nodes = [TextNode(id_="a", text="Changed source fixture A"), nodes[1]]
    else:
        nodes = list(reversed(nodes))
    changed = EmbeddingCheckpoint(tmp_path, signature, nodes, dimension)
    assert changed.path != original.path
    assert changed.load() == {}
    assert original.load() == {"a": [1.0, 2.0, 3.0]}


@pytest.mark.parametrize("bad_vector", [[1.0, 2.0], [True, 2.0, 3.0], [float("nan"), 2.0, 3.0], [float("inf"), 2.0, 3.0], [0.0, 0.0, 0.0], ["1", 2.0, 3.0]])
def test_invalid_save_preserves_last_good_checkpoint(checkpoint, bad_vector):
    checkpoint.save({"technical-a": [1.0, 2.0, 3.0]})
    before = checkpoint.path.read_bytes()
    with pytest.raises(CheckpointError):
        checkpoint.save({"technical-a": bad_vector})
    assert checkpoint.path.read_bytes() == before


@pytest.mark.parametrize("bad_vector", [[1.0, 2.0], [True, 2.0, 3.0], [float("nan"), 2.0, 3.0], [float("inf"), 2.0, 3.0], [0.0, 0.0, 0.0], ["1", 2.0, 3.0]])
def test_corrupt_persisted_vectors_are_not_silently_reused(checkpoint, bad_vector):
    checkpoint.save({"technical-a": [1.0, 2.0, 3.0]})
    snapshot = json.loads(checkpoint.path.read_text(encoding="utf-8"))
    snapshot["embeddings"]["technical-a"] = bad_vector
    checkpoint.path.write_text(json.dumps(snapshot), encoding="utf-8")
    with pytest.raises(CheckpointError):
        checkpoint.load()


@pytest.mark.parametrize("corruption", ["unknown_id", "wrong_fingerprint", "wrong_dimension", "duplicate_key", "malformed_json"])
def test_invalid_checkpoint_identity_or_json_is_rejected(checkpoint, corruption):
    checkpoint.save({"technical-a": [1.0, 2.0, 3.0]})
    snapshot = json.loads(checkpoint.path.read_text(encoding="utf-8"))
    if corruption == "unknown_id":
        snapshot["embeddings"] = {"unknown-source": [1.0, 2.0, 3.0]}
    elif corruption == "wrong_fingerprint":
        snapshot["fingerprint"] = "0" * 64
    elif corruption == "wrong_dimension":
        snapshot["dimension"] = 4
    raw = json.dumps(snapshot)
    if corruption == "duplicate_key":
        raw = raw.replace('"dimension": 3', '"dimension": 3, "dimension": 3', 1)
    elif corruption == "malformed_json":
        raw = "{ incomplete JSON"
    checkpoint.path.write_text(raw, encoding="utf-8")
    with pytest.raises(CheckpointError):
        checkpoint.load()


def test_atomic_checkpoint_failure_preserves_previous_progress(checkpoint, monkeypatch):
    import src.embedding_checkpoint as module

    checkpoint.save({"technical-a": [1.0, 2.0, 3.0]})
    before = checkpoint.path.read_bytes()

    def replace_failure(*args, **kwargs):
        raise OSError("Injected atomic replacement failure")

    monkeypatch.setattr(module.os, "replace", replace_failure)
    with pytest.raises(CheckpointError):
        checkpoint.save({"technical-a": [1.0, 2.0, 3.0], "technical-b": [3.0, 2.0, 1.0]})
    assert checkpoint.path.read_bytes() == before
    assert not list(checkpoint.directory.glob("*.tmp"))


def test_missing_index_with_key_has_specific_rebuild_guidance_without_api_calls(tmp_path, monkeypatch):
    import src.qa_service as module

    settings = load_settings(project_root=tmp_path, environ={"GEMINI_API_KEY": "test-only-not-real-api-key"})

    def forbidden(*args, **kwargs):
        raise AssertionError("A missing index must not trigger a model call")

    monkeypatch.setattr(module, "create_llm", forbidden)
    answer = module.QAService(settings).ask("שאלת בדיקת תוכנה למאגר שטרם נבנה")
    assert answer.status == "error"
    assert answer.error_code == "index_not_ready"
    assert "ingest" in answer.text and "--rebuild" in answer.text
    assert not answer.sources


@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("source_name", ["KNOWLEDGE_BASE_PATH", "EVAL_PATH"])
def test_checkpoint_directory_cannot_overlap_source_or_evaluation(tmp_path, offline, source_name):
    root = tmp_path / "project"
    root.mkdir()
    cache = root / "storage" / ("offline" if offline else "") / "embedding_checkpoints"
    overrides = {"CHROMA_PATH": str(root / "separate-db"), source_name: str(cache)}
    settings = load_settings(project_root=root, environ=overrides)
    before = set(root.rglob("*"))
    with pytest.raises(IndexingError):
        ingest(settings, rebuild=True, offline=offline)
    assert set(root.rglob("*")) == before
