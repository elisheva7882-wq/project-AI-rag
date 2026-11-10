"""Atomic, resumable embedding snapshots tied to exact model and chunk inputs.

Snapshots contain vectors and opaque chunk IDs, never API keys or source text.
The ingestion build lock must serialize calls that write the same snapshot.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import warnings
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence

from llama_index.core.schema import MetadataMode, TextNode

from src.file_loader import CorpusError, reject_links


SCHEMA_VERSION = 1
MAX_CHECKPOINT_BYTES = 100 * 1024 * 1024


class CheckpointError(ValueError):
    """Invalid or inaccessible checkpoint, without exposing stored payloads."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError("Embedding checkpoint contains duplicate JSON keys.")
        result[key] = value
    return result


def _reject_constant(_value):
    raise CheckpointError("Embedding checkpoint contains non-finite JSON numbers.")


class EmbeddingCheckpoint:
    def __init__(
        self,
        directory: Path,
        signature: dict,
        nodes: Sequence[TextNode],
        dimension: int,
    ) -> None:
        if type(dimension) is not int or dimension <= 0:
            raise CheckpointError("Embedding checkpoint dimension must be a positive integer.")
        self.directory = Path(directory)
        self.dimension = dimension
        inputs = []
        self._node_ids: set[str] = set()
        for node in nodes:
            chunk_id = node.node_id
            if not isinstance(chunk_id, str) or not chunk_id or chunk_id in self._node_ids:
                raise CheckpointError("Embedding checkpoint requires unique, nonempty chunk IDs.")
            self._node_ids.add(chunk_id)
            inputs.append({
                "chunk_id": chunk_id,
                "payload_hash": hashlib.sha256(
                    node.get_content(MetadataMode.EMBED).encode("utf-8")
                ).hexdigest(),
            })
        if not inputs or not isinstance(signature, dict):
            raise CheckpointError("Embedding checkpoint requires a signature and chunk inputs.")
        try:
            identity = json.dumps(
                {"signature": signature, "nodes": inputs, "dimension": dimension},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, RecursionError):
            raise CheckpointError("Embedding checkpoint signature cannot be serialized.") from None
        self.fingerprint = hashlib.sha256(identity).hexdigest()
        self.path = self.directory / f"{self.fingerprint}.json"

    def _check_paths(self) -> None:
        try:
            reject_links(self.directory)
            reject_links(self.path)
            if self.directory.exists() and not self.directory.is_dir():
                raise CheckpointError("Embedding checkpoint location must be a directory.")
            if self.path.exists() and not self.path.is_file():
                raise CheckpointError("Embedding checkpoint must be a regular file.")
        except (CorpusError, OSError):
            raise CheckpointError("Embedding checkpoint paths are inaccessible or contain a link.") from None

    def validate_location(self) -> None:
        """Preflight file types and links without reading an old snapshot."""
        self._check_paths()

    def _vectors(self, vectors: Mapping) -> dict[str, list[float]]:
        if not isinstance(vectors, Mapping) or len(vectors) > len(self._node_ids):
            raise CheckpointError("Embedding checkpoint vector mapping is invalid.")
        normalized = {}
        for chunk_id, vector in vectors.items():
            if not isinstance(chunk_id, str) or chunk_id not in self._node_ids:
                raise CheckpointError("Embedding checkpoint references an unknown chunk.")
            if not isinstance(vector, (list, tuple)) or len(vector) != self.dimension:
                raise CheckpointError("Embedding checkpoint vector dimension is invalid.")
            if any(isinstance(value, bool) or not isinstance(value, Real) for value in vector):
                raise CheckpointError("Embedding checkpoint vector values must be finite numbers.")
            try:
                values = [float(value) for value in vector]
            except (OverflowError, ValueError):
                raise CheckpointError("Embedding checkpoint vector values must be finite numbers.") from None
            if not all(math.isfinite(value) for value in values) or not any(values):
                raise CheckpointError("Embedding checkpoint vector is non-finite or zero.")
            normalized[chunk_id] = values
        return normalized

    def load(self) -> dict[str, list[float]]:
        self._check_paths()
        if not self.path.exists():
            return {}
        try:
            with self.path.open("rb") as stream:
                data = stream.read(MAX_CHECKPOINT_BYTES + 1)
            if len(data) > MAX_CHECKPOINT_BYTES:
                raise CheckpointError("Embedding checkpoint exceeds the size limit.")
            snapshot = json.loads(
                data.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (OSError, UnicodeError, ValueError, RecursionError):
            raise CheckpointError("Embedding checkpoint could not be read or contains invalid JSON.") from None
        if not isinstance(snapshot, dict) or set(snapshot) != {
            "schema_version", "fingerprint", "dimension", "embeddings"
        }:
            raise CheckpointError("Embedding checkpoint schema is invalid.")
        if (
            type(snapshot["schema_version"]) is not int
            or snapshot["schema_version"] != SCHEMA_VERSION
            or snapshot["fingerprint"] != self.fingerprint
            or type(snapshot["dimension"]) is not int
            or snapshot["dimension"] != self.dimension
        ):
            raise CheckpointError("Embedding checkpoint does not match the current model and chunk inputs.")
        return self._vectors(snapshot["embeddings"])

    def save(self, vectors: Mapping[str, list[float]]) -> None:
        """Save the caller's full cumulative snapshot, preserving it atomically."""
        normalized = self._vectors(vectors)
        data = json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "fingerprint": self.fingerprint,
                "dimension": self.dimension,
                "embeddings": normalized,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if len(data) > MAX_CHECKPOINT_BYTES:
            raise CheckpointError("Embedding checkpoint exceeds the size limit.")
        self._check_paths()
        temporary = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._check_paths()
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self.directory, prefix=".checkpoint_", suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            self._check_paths()
            os.replace(temporary, self.path)
            temporary = None
        except OSError:
            raise CheckpointError("Embedding checkpoint could not be saved; any previous snapshot was preserved.") from None
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    warnings.warn("An uncommitted embedding checkpoint temporary file could not be removed.", RuntimeWarning)
