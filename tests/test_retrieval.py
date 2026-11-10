"""Real local Chroma retrieval with explicit hash vectors, not semantic scoring."""

from __future__ import annotations

import json
import math
import socket
from dataclasses import replace

import pytest
from llama_index.core.bridge.pydantic import Field
from llama_index.core.schema import NodeRelationship, NodeWithScore, RelatedNodeInfo, TextNode

from config import load_settings
from src.gemini_models import OfflineEmbedding, create_offline_embedding
from src.indexer import get_manifest_path, ingest
from src.retriever import ArchiveRetriever, RetrievalError, _cosine_similarity


class CountingEmbedding(OfflineEmbedding):
    queries: list[str] = Field(default_factory=list, exclude=True)
    document_calls: int = 0

    def _get_query_embedding(self, query):
        self.queries.append(query)
        return OfflineEmbedding._get_text_embedding(self, query)

    def _get_text_embedding(self, text):
        self.document_calls += 1
        return OfflineEmbedding._get_text_embedding(self, text)


@pytest.fixture
def retrieval_settings(tmp_path, corpus_directory):
    root = tmp_path / "retrieval-project"
    root.mkdir()
    settings = load_settings(project_root=root, environ={"KNOWLEDGE_BASE_PATH": str(corpus_directory)})
    ingest(settings, rebuild=True, offline=True)
    return settings


@pytest.mark.parametrize("top_k", [1, 3, 5])
def test_retrieval_uses_original_question_and_only_queries_existing_vectors(retrieval_settings, top_k):
    settings = replace(retrieval_settings, top_k=top_k)
    model = CountingEmbedding()
    path = get_manifest_path(settings, offline=True)
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    retriever = ArchiveRetriever(settings, offline=True, embed_model=model)
    question = "  שאלת בדיקת תוכנה מקורית ללא הרחבה  "
    original_id = retriever._collection.id
    chunks = retriever.retrieve(question)
    assert len(chunks) == top_k
    assert len({chunk.chunk_id for chunk in chunks}) == top_k
    assert model.queries == [question]
    assert model.document_calls == 0
    assert retriever._collection.id == original_id
    assert retriever._collection.count() == retriever.chunk_count
    assert before == (path.read_bytes(), path.stat().st_mtime_ns)
    for chunk in chunks:
        assert chunk.document_id in {f"ALT-{number:02d}" for number in range(1, 20)}
        assert chunk.source_file.startswith(chunk.document_id)
        assert chunk.title and chunk.section and chunk.text
        assert chunk.document_type and chunk.period and chunk.topics
        assert -1 <= chunk.similarity <= 1


def test_returned_similarity_matches_raw_chroma_cosine_distance(retrieval_settings):
    retriever = ArchiveRetriever(retrieval_settings, offline=True)
    question = "שאלה טכנית להשוואת ציונים בלבד"
    chunks = retriever.retrieve(question)
    vector = create_offline_embedding().get_query_embedding(question)
    raw = retriever._collection.query(query_embeddings=[vector], n_results=5, include=["distances"])
    expected = {chunk_id: 1.0 - distance for chunk_id, distance in zip(raw["ids"][0], raw["distances"][0])}
    assert {chunk.chunk_id for chunk in chunks} == set(expected)
    for chunk in chunks:
        assert chunk.similarity == pytest.approx(expected[chunk.chunk_id], abs=1e-5)


def test_optional_threshold_uses_cosine_similarity_not_adapter_exponential(retrieval_settings):
    question = "שאלה טכנית לבדיקת סף בלבד"
    original = ArchiveRetriever(retrieval_settings, offline=True).retrieve(question)
    maximum = max(chunk.similarity for chunk in original)
    assert maximum < 1
    cutoff = (maximum + 1.0) / 2.0
    strict = ArchiveRetriever(replace(retrieval_settings, similarity_threshold=cutoff), offline=True)
    assert strict.retrieve(question) == ()


@pytest.mark.parametrize("score", [None, 0.0, -1.0, float("nan"), float("inf"), math.exp(-3.0)])
def test_invalid_or_out_of_range_adapter_scores_are_rejected(score):
    with pytest.raises(RetrievalError):
        _cosine_similarity(score)


def test_rebuilt_manifest_prevents_old_handle_from_serving_a_question(retrieval_settings):
    model = CountingEmbedding()
    retriever = ArchiveRetriever(retrieval_settings, offline=True, embed_model=model)
    ingest(retrieval_settings, rebuild=True, offline=True)
    with pytest.raises(RetrievalError):
        retriever.retrieve("שאלה אחרי שינוי דור האינדקס")
    assert not model.queries


def test_changed_source_prevents_query_without_automatic_rebuild(retrieval_settings):
    model = CountingEmbedding()
    retriever = ArchiveRetriever(retrieval_settings, offline=True, embed_model=model)
    source = retrieval_settings.knowledge_base_path / "ALT-01_בדיקה.md"
    source.write_bytes(source.read_bytes() + b"\nChanged technical source data.\n")
    with pytest.raises(RetrievalError):
        retriever.retrieve("שאלה אחרי שינוי מקור")
    assert not model.queries


def test_serialized_node_metadata_cannot_smuggle_an_unknown_source(retrieval_settings):
    retriever = ArchiveRetriever(retrieval_settings, offline=True)
    collection = retriever._collection
    stored = collection.get(include=["metadatas"])
    modified = []
    for metadata in stored["metadatas"]:
        metadata = dict(metadata)
        serialized = json.loads(metadata["_node_content"])
        serialized["metadata"]["document_id"] = "ALT-99"
        metadata["_node_content"] = json.dumps(serialized, ensure_ascii=False)
        modified.append(metadata)
    collection.update(ids=stored["ids"], metadatas=modified)
    with pytest.raises(RetrievalError):
        retriever.retrieve("בדיקת מטאדאטה מסולפת")


def test_multiple_chunks_from_same_document_remain_available_without_duplicates(retrieval_settings, monkeypatch):
    retriever = ArchiveRetriever(retrieval_settings, offline=True)
    selected = []
    target_doc = next(iter(retriever._chunks.values()))["metadata"]["document_id"]
    for chunk_id, cached in retriever._chunks.items():
        if cached["metadata"]["document_id"] != target_doc:
            continue
        node = TextNode(
            id_=chunk_id, text=cached["text"], metadata=dict(cached["metadata"]),
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=target_doc)},
        )
        selected.append(NodeWithScore(node=node, score=math.exp(-0.2)))
    assert len(selected) >= 2
    monkeypatch.setattr(retriever._retriever, "retrieve", lambda question: [selected[0], selected[0], selected[1]])
    chunks = retriever.retrieve("שאלה המצריכה שני מקטעים מאותו מקור")
    assert len(chunks) == 2
    assert {chunk.document_id for chunk in chunks} == {target_doc}
    assert len({chunk.chunk_id for chunk in chunks}) == 2


def test_retrieval_never_opens_evaluation_files(retrieval_settings, monkeypatch):
    from pathlib import Path

    evaluation = retrieval_settings.eval_path
    evaluation.mkdir()
    (evaluation / "expected_answers.jsonl").write_text("FORBIDDEN_EVALUATION_SENTINEL", encoding="utf-8")
    original_open = Path.open

    def protected_open(path, *args, **kwargs):
        if path.resolve().is_relative_to(evaluation):
            raise AssertionError("Retriever must not read evaluation material")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", protected_open)
    chunks = ArchiveRetriever(retrieval_settings, offline=True).retrieve("שאלת בדיקה טכנית")
    assert chunks
    assert all("FORBIDDEN_EVALUATION_SENTINEL" not in chunk.text for chunk in chunks)


def test_offline_retrieval_requires_no_network(retrieval_settings, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline retrieval must not access the network")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    assert ArchiveRetriever(retrieval_settings, offline=True).retrieve("שאלת בדיקה ללא רשת")


def test_injected_embeddings_are_not_accepted_in_production(retrieval_settings):
    with pytest.raises(RetrievalError):
        ArchiveRetriever(retrieval_settings, embed_model=create_offline_embedding())
