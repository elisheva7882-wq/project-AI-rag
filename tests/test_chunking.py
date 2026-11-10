"""Token-aware LlamaIndex chunking; all text here is a technical fixture."""

from __future__ import annotations

from dataclasses import replace

import pytest
from llama_index.core.schema import MetadataMode
from llama_index.core.utils import get_tokenizer

from config import load_settings
from src.file_loader import load_corpus, validate_corpus
from src.indexer import IndexingError, build_nodes


def _settings(tmp_path, corpus_directory):
    return load_settings(
        project_root=tmp_path,
        environ={"KNOWLEDGE_BASE_PATH": str(corpus_directory)},
    )


def test_nodes_preserve_all_source_metadata_and_archive_relationship(corpus_directory, tmp_path):
    documents = load_corpus(corpus_directory)
    nodes = build_nodes(documents, _settings(tmp_path, corpus_directory))
    assert {node.ref_doc_id for node in nodes} == {doc.document_id for doc in documents}
    originals = {document.document_id: document for document in documents}
    for node in nodes:
        original = originals[node.ref_doc_id]
        assert node.metadata["document_id"] == original.document_id
        assert node.metadata["source_file"] == original.source_file
        assert node.metadata["title"] == original.title
        assert node.metadata["document_type"] == original.document_type
        assert node.metadata["period"] == original.period
        assert node.metadata["topics"] == original.topics
        assert node.metadata["content_hash"] == original.content_hash
        assert node.metadata["chunk_id"] == node.node_id
        assert node.metadata["section"]
        assert node.text in original.text


def test_chunk_ids_are_stable_unique_and_sensitive_to_chunk_configuration(corpus_directory, tmp_path):
    documents = load_corpus(corpus_directory)
    settings = _settings(tmp_path, corpus_directory)
    first = build_nodes(documents, settings)
    second = build_nodes(documents, settings)
    assert [(node.node_id, node.text, node.metadata) for node in first] == [
        (node.node_id, node.text, node.metadata) for node in second
    ]
    assert len({node.node_id for node in first}) == len(first)
    changed = build_nodes(documents, replace(settings, chunk_size=256, chunk_overlap=32))
    assert {node.node_id for node in first}.isdisjoint(node.node_id for node in changed)


def test_long_sections_respect_real_token_budget_and_preserve_source_sentences(corpus_entries, corpus_directory, tmp_path):
    sentences = [f"Technical fixture sentence {number:04d} preserves distinct source data for verification. " for number in range(240)]
    filename = "ALT-01_בדיקה.md"
    addition = "\n\n## Long technical section\n\n" + "".join(sentences)
    corpus_entries[filename] += addition.encode("utf-8")
    nodes = build_nodes(validate_corpus(corpus_entries), _settings(tmp_path, corpus_directory))
    selected = [node for node in nodes if node.ref_doc_id == "ALT-01" and node.metadata["section"] == "Long technical section"]
    assert len(selected) > 2
    tokenizer = get_tokenizer()
    for node in nodes:
        assert len(tokenizer(node.get_content(metadata_mode=MetadataMode.EMBED))) <= 512
        assert len(tokenizer(node.get_content(metadata_mode=MetadataMode.LLM))) <= 512
    combined = "\n".join(node.text for node in selected)
    assert all(sentence.strip() in combined for sentence in sentences)

    # Distinct sentence numbers keep adjacent chunks from appearing to overlap
    # merely because the fixture repeats identical paragraphs. Count tokens,
    # rather than words, to exercise the splitter's actual overlap behavior.
    overlaps = []
    for previous, current in zip(selected, selected[1:]):
        # A leading space aligns BPE tokens at sentence starts after the
        # splitter strips boundary whitespace from returned chunks.
        left, right = tokenizer(" " + previous.text), tokenizer(" " + current.text)
        overlaps.append(max((size for size in range(1, min(len(left), len(right)) + 1) if left[-size:] == right[:size]), default=0))
    assert any(overlaps)
    assert max(overlaps) <= 64


def test_oversized_metadata_fails_before_embedding(corpus_entries, corpus_directory, tmp_path):
    filename = "ALT-01_בדיקה.md"
    text = corpus_entries[filename].decode("utf-8")
    text = text.replace("כותרת: מסמך טכני לבדיקת טעינה 01", "כותרת: " + "extremely long technical metadata " * 400)
    corpus_entries[filename] = text.encode("utf-8")
    with pytest.raises(IndexingError):
        build_nodes(validate_corpus(corpus_entries), _settings(tmp_path, corpus_directory))
