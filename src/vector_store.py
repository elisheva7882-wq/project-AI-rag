"""Explicit local Chroma storage and recoverable collection replacement.

No default embedding function is installed: callers must supply vectors created
by the configured Gemini adapter (or explicit test doubles in offline tests).
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from uuid import uuid4

from src.file_loader import reject_links

if TYPE_CHECKING:
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection
    from llama_index.vector_stores.chroma import ChromaVectorStore


STAGING_PREFIX = "kingdomai_staging_"
BACKUP_PREFIX = "kingdomai_backup_"
GENERATION_METADATA_KEY = "kingdomai_generation"


class VectorStoreError(ValueError):
    """A safe local-storage error suitable for a command-line or UI message."""


def open_client(path: Path) -> ClientAPI:
    """Open persistent storage with telemetry and destructive reset disabled."""
    from chromadb import PersistentClient
    from chromadb.config import Settings as ChromaSettings

    path = Path(path)
    reject_links(path)
    if path.exists() and not path.is_dir():
        raise VectorStoreError("CHROMA_PATH must point to a directory.")
    try:
        return PersistentClient(
            path=str(path.resolve()),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
        )
    except Exception:
        raise VectorStoreError("Could not open the local Chroma database.") from None


def _generation_id(staging_name: str) -> str:
    if not staging_name.startswith(STAGING_PREFIX):
        raise VectorStoreError("Only a newly owned staging collection may be promoted or discarded.")
    generation_id = staging_name[len(STAGING_PREFIX):]
    if re.fullmatch(r"[0-9a-f]{32}", generation_id) is None:
        raise VectorStoreError("Staging generation must be a UUID hex string.")
    return generation_id


def _get_optional(client: ClientAPI, name: str) -> Collection | None:
    from chromadb.errors import NotFoundError

    try:
        return client.get_collection(name=name, embedding_function=None)
    except NotFoundError:
        return None
    except Exception:
        raise VectorStoreError("Could not read the requested Chroma collection.") from None


def get_collection(client: ClientAPI, name: str) -> Collection:
    """Fetch an existing collection without creating it or choosing embeddings."""
    collection = _get_optional(client, name)
    if collection is None:
        raise VectorStoreError("Index collection is missing; build the index first.")
    return collection


def create_staging_collection(client: ClientAPI, generation_id: str) -> Collection:
    """Create a uniquely named, generation-marked cosine collection."""
    staging_name = STAGING_PREFIX + generation_id
    _generation_id(staging_name)
    try:
        return client.create_collection(
            name=staging_name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={GENERATION_METADATA_KEY: generation_id},
        )
    except Exception:
        raise VectorStoreError("Could not create the staging Chroma collection.") from None


def as_vector_store(collection: Collection) -> ChromaVectorStore:
    """Wrap an explicit collection with LlamaIndex; no implicit client/defaults."""
    from llama_index.vector_stores.chroma import ChromaVectorStore

    return ChromaVectorStore(chroma_collection=collection)


def _owned_staging(client: ClientAPI, staging_name: str) -> Collection | None:
    generation_id = _generation_id(staging_name)
    collection = _get_optional(client, staging_name)
    if collection is not None and (collection.metadata or {}).get(GENERATION_METADATA_KEY) != generation_id:
        raise VectorStoreError("Staging ownership metadata does not match; collection was not changed.")
    return collection


def _delete_known_collection(client: ClientAPI, collection: Collection) -> None:
    """Delete only the same collection ID tracked by this operation."""
    current = _get_optional(client, collection.name)
    if current is None:
        return
    if current.id != collection.id:
        raise VectorStoreError("Collection identity changed; automatic cleanup was stopped.")
    client.delete_collection(name=collection.name)


def discard_staging_collection(client: ClientAPI, staging_name: str) -> None:
    """Remove one explicitly owned temporary collection after a failed build."""
    collection = _owned_staging(client, staging_name)
    if collection is None:
        return
    try:
        _delete_known_collection(client, collection)
    except Exception:
        raise VectorStoreError("Could not remove the owned staging collection.") from None


def promote_collection(
    client: ClientAPI,
    staging_name: str,
    target_name: str,
    commit: Callable[[], None],
) -> Collection:
    """Replace one target after a staged build, preserving it on ordinary failure.

    ``commit`` must atomically replace the index manifest, leaving the old
    manifest intact if it raises. Chroma renames and filesystem writes cannot
    form one transaction; generation metadata lets the caller detect crashes
    between operations. Only this target and collections owned by this call are
    changed. Independent writers must be excluded by the caller's build lock.
    """
    if target_name.startswith((STAGING_PREFIX, BACKUP_PREFIX)):
        raise VectorStoreError("The target collection must not use an internal staging or backup name.")
    staging = _owned_staging(client, staging_name)
    if staging is None:
        raise VectorStoreError("Staging collection is missing; nothing was promoted.")
    original = _get_optional(client, target_name)
    original_renamed = False
    backup_name = BACKUP_PREFIX + uuid4().hex
    try:
        if original is not None:
            original.modify(name=backup_name)
            original_renamed = True
        staging.modify(name=target_name)
        commit()
    except BaseException as exc:
        rollback_failed = False
        try:
            _delete_known_collection(client, staging)
        except Exception:
            rollback_failed = True
        if original is not None and original_renamed:
            try:
                original.modify(name=target_name)
            except Exception:
                rollback_failed = True
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if rollback_failed:
            raise VectorStoreError(
                "Promotion failed and rollback was incomplete; inspect the target and owned staging/backup collections."
            ) from None
        raise VectorStoreError("Promotion failed; existing index data was preserved.") from None

    # The collection and manifest are already committed. Failure to remove the
    # original backup is a cleanup warning, not a failed or partial new index.
    if original is not None and original_renamed:
        try:
            _delete_known_collection(client, original)
        except Exception:
            warnings.warn(
                "Index committed, but its owned old backup could not be removed.",
                RuntimeWarning,
                stacklevel=2,
            )
    return staging
