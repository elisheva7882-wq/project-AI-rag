"""Read-only semantic retrieval from a validated, explicitly selected index."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import MetadataMode

from config import Settings
from src.file_loader import read_bytes, reject_links
from src.gemini_models import create_embedding_model, create_offline_embedding, provider_error_message
from src.indexer import get_manifest_path, validate_index
from src.models import RetrievedChunk, validate_question
from src.service_errors import classify_service_error, provider_status_code
from src.vector_store import GENERATION_METADATA_KEY, as_vector_store, get_collection, open_client


_SOURCE_FIELDS = (
    "document_id", "source_file", "title", "document_type", "period",
    "topics", "section", "content_hash", "chunk_id",
)


class RetrievalError(ValueError):
    """An invalid/stale index or failed query, without SDK request payloads."""

    def __init__(self, message: str, *, code: int | None = None, kind: str | None = None) -> None:
        super().__init__(message)
        self.code = code if isinstance(code, int) and not isinstance(code, bool) else None
        self.kind = kind


def _cosine_similarity(adapter_score: float | None) -> float:
    """Undo the installed adapter's exp(-cosine_distance) transformation.

    ChromaVectorStore 0.6.0 does not return raw cosine similarity. Clamping
    tolerates normal floating-point drift at the [-1, 1] endpoints only.
    """
    if adapter_score is None or not math.isfinite(adapter_score) or adapter_score <= 0:
        raise RetrievalError("The vector store returned an invalid similarity score.")
    similarity = 1.0 + math.log(adapter_score)
    if not math.isfinite(similarity) or not -1.00001 <= similarity <= 1.00001:
        raise RetrievalError("The vector store score is outside the cosine range.")
    return max(-1.0, min(1.0, similarity))


class ArchiveRetriever:
    """Reuse persisted document vectors and embed only the original question.

    Offline mode is explicit and stays in its separately labelled test index.
    It exercises retrieval mechanics, not semantic relevance of the hash model.
    A changed index/corpus requires a new service instance rather than silent
    reuse of Chroma handles that still point at an older collection UUID.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        offline: bool = False,
        embed_model: BaseEmbedding | None = None,
    ) -> None:
        if not 1 <= settings.top_k <= 5:
            raise RetrievalError("Retrieval TOP_K must stay between 1 and 5.")
        if embed_model is not None and not offline:
            raise RetrievalError("Injected query embeddings are allowed only in the isolated offline test index.")
        self._settings = settings
        self.offline = offline
        self._manifest = validate_index(settings, offline=offline)
        self._manifest_path = get_manifest_path(settings, offline=offline)
        self._manifest_bytes = self._manifest_path.read_bytes()
        self.generation_id = self._manifest["generation_id"]
        self.collection_name = self._manifest["collection_name"]
        self.document_count = self._manifest["documents"]
        self.chunk_count = self._manifest["chunks"]
        self._client = open_client(Path(self._manifest["chroma_path"]))
        self._collection = get_collection(self._client, self.collection_name)
        self._collection_id = self._collection.id
        stored = self._collection.get(include=["metadatas", "documents"])
        self._chunks: dict[str, dict[str, Any]] = {}
        for index, chunk_id in enumerate(stored["ids"]):
            metadata = stored["metadatas"][index]
            self._chunks[chunk_id] = {
                "text": stored["documents"][index],
                "metadata": {key: metadata[key] for key in _SOURCE_FIELDS},
            }
        self._source_hashes = {
            source["source_file"]: source["content_hash"]
            for source in self._manifest["sources"]
        }
        self._check_current_index()
        self._embed_model = embed_model if embed_model is not None else (
            create_offline_embedding() if offline else create_embedding_model(settings)
        )
        # A supplied model prevents LlamaIndex from resolving an OpenAI default.
        # from_vector_store receives no documents and does not rebuild vectors.
        index = VectorStoreIndex.from_vector_store(
            as_vector_store(self._collection),
            embed_model=self._embed_model,
        )
        self._retriever = index.as_retriever(similarity_top_k=settings.top_k)

    def _check_current_index(self) -> None:
        try:
            reject_links(self._manifest_path)
            if self._manifest_path.read_bytes() != self._manifest_bytes:
                raise RetrievalError("The index manifest changed; reload the question service before querying.")
            knowledge = self._settings.knowledge_base_path
            reject_links(knowledge)
            actual = {child.name for child in knowledge.iterdir() if child.name != ".gitkeep"}
            if actual != set(self._source_hashes):
                raise RetrievalError("The knowledge corpus changed; rebuild the index and reload the question service.")
            for filename, expected_hash in self._source_hashes.items():
                if hashlib.sha256(read_bytes(knowledge / filename)).hexdigest() != expected_hash:
                    raise RetrievalError("A source document changed; rebuild the index and reload the question service.")
            current = get_collection(self._client, self.collection_name)
            if (
                current.id != self._collection_id
                or (current.metadata or {}).get(GENERATION_METADATA_KEY) != self.generation_id
                or current.count() != self.chunk_count
            ):
                raise RetrievalError("The index collection changed; reload the question service before querying.")
        except RetrievalError:
            raise
        except Exception:
            raise RetrievalError("Could not verify the current index. Check local storage and reload the question service.") from None

    def retrieve(self, question: str) -> tuple[RetrievedChunk, ...]:
        question = validate_question(question)
        self._check_current_index()
        try:
            results = self._retriever.retrieve(question)
        except Exception as exc:
            raise RetrievalError(
                f"Question embedding or retrieval failed. {provider_error_message(exc)}",
                code=provider_status_code(exc),
                kind=classify_service_error(exc),
            ) from None
        self._check_current_index()
        chunks: list[RetrievedChunk] = []
        seen: set[str] = set()
        for result in results:
            node = result.node
            expected = self._chunks.get(node.node_id)
            if (
                expected is None
                or node.get_content(MetadataMode.NONE) != expected["text"]
                or any(node.metadata.get(key) != value for key, value in expected["metadata"].items())
                or node.ref_doc_id != expected["metadata"]["document_id"]
            ):
                raise RetrievalError("Retrieved chunk identity, content or source metadata is not part of the validated corpus.")
            similarity = _cosine_similarity(result.score)
            if node.node_id in seen:
                continue
            seen.add(node.node_id)
            threshold = self._settings.similarity_threshold
            if threshold is not None and similarity < threshold:
                continue
            metadata = expected["metadata"]
            chunks.append(RetrievedChunk(
                chunk_id=node.node_id,
                document_id=metadata["document_id"],
                source_file=metadata["source_file"],
                title=metadata["title"],
                section=metadata["section"],
                text=expected["text"],
                similarity=similarity,
                document_type=metadata["document_type"],
                period=metadata["period"],
                topics=metadata["topics"],
            ))
            if len(chunks) == self._settings.top_k:
                break
        return tuple(chunks)
