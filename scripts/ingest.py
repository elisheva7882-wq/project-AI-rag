"""Build an explicit persistent index, or verify/reuse an existing one."""

from __future__ import annotations

import argparse
import sys

from config import ConfigurationError, load_settings
from src.file_loader import CorpusError
from src.gemini_models import ProviderError
from src.indexer import IndexingError, ingest
from src.vector_store import VectorStoreError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Altheria Chroma index using explicit LlamaIndex/Gemini embeddings.")
    parser.add_argument("--rebuild", action="store_true", help="Explicitly rebuild and replace this project's configured collection.")
    parser.add_argument("--offline", action="store_true", help="Use test-only hash vectors in storage/offline; never build the production Gemini index.")
    parser.add_argument("--fresh-embeddings", action="store_true", help="With --rebuild, explicitly regenerate even compatible cached vectors.")
    args = parser.parse_args(argv)
    try:
        result = ingest(load_settings(), rebuild=args.rebuild, offline=args.offline, fresh_embeddings=args.fresh_embeddings, progress=lambda message: print(message, flush=True))
    except (ConfigurationError, CorpusError, ProviderError, IndexingError, VectorStoreError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception:
        print("ERROR: Indexing failed; inspect local storage and configuration. No successful build is claimed.", file=sys.stderr)
        return 2
    mode = "OFFLINE TEST ONLY: deterministic hash vectors, no semantic-quality validation" if result.offline else "GEMINI PRODUCTION INDEX"
    print(mode)
    print(f"{'Built' if result.rebuilt else 'Verified and reused'}: {result.documents} documents, {result.chunks} chunks.")
    print(f"Collection: {result.collection_name}")
    if result.rebuilt:
        print(f"Embeddings reused: {result.reused_embeddings}; newly created: {result.chunks - result.reused_embeddings}.")
    print(f"Manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
