"""Check installed adapters and configuration without making a Gemini request."""

from __future__ import annotations

import importlib
import argparse
import sys
from importlib.metadata import PackageNotFoundError, version

DEPENDENCIES = (
    ("streamlit", "streamlit", None),
    ("python-dotenv", "dotenv", "dotenv_values"),
    ("chromadb", "chromadb", "PersistentClient"),
    ("llama-index-core", "llama_index.core.node_parser", "SentenceSplitter"),
    ("llama-index-vector-stores-chroma", "llama_index.vector_stores.chroma", "ChromaVectorStore"),
    ("llama-index-llms-google-genai", "llama_index.llms.google_genai", "GoogleGenAI"),
    ("llama-index-embeddings-google-genai", "llama_index.embeddings.google_genai", "GoogleGenAIEmbedding"),
    ("google-genai", "google.genai", "Client"),
    ("pytest", "pytest", "main"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check dependencies and the production index, or an explicit offline test index.")
    parser.add_argument("--offline", action="store_true", help="Verify the isolated test index; does not establish Gemini readiness.")
    args = parser.parse_args(argv)
    print(f"Python {sys.version.split()[0]}")
    if sys.version_info < (3, 11):
        print("ERROR: Python 3.11 or newer is required.")
        return 2

    failed = False
    for distribution, module_name, symbol in DEPENDENCIES:
        try:
            installed_version = version(distribution)
            module = importlib.import_module(module_name)
            if symbol is not None:
                getattr(module, symbol)
        except (PackageNotFoundError, ImportError, AttributeError):
            print(f"ERROR: {distribution} is missing or its required import failed.")
            failed = True
        else:
            print(f"OK: {distribution}=={installed_version}")
    if failed:
        return 2

    from config import ConfigurationError, load_settings
    from src.file_loader import CorpusError, load_corpus

    try:
        settings = load_settings()
    except ConfigurationError as exc:
        print(f"ERROR: {exc}")
        return 2

    print(f"LLM_MODEL={settings.llm_model}")
    print(f"EMBEDDING_MODEL={settings.embedding_model}")
    print(f"EMBEDDING_DIMENSION={settings.embedding_dimension}")
    print(f"CHUNK_SIZE={settings.chunk_size}; CHUNK_OVERLAP={settings.chunk_overlap}; TOP_K={settings.top_k}")
    print(f"API key configured: {'yes' if settings.has_api_key else 'no'}")
    if settings.knowledge_base_path.is_dir() and any(
        entry.name != ".gitkeep" for entry in settings.knowledge_base_path.iterdir()
    ):
        try:
            documents = load_corpus(settings.knowledge_base_path)
        except (CorpusError, OSError) as exc:
            print(f"ERROR: Invalid knowledge corpus: {exc}")
            return 2
        print(f"Validated source documents: {len(documents)} (ALT-01 through ALT-19).")
    from src.indexer import IndexingError, get_manifest_path, validate_index
    from src.vector_store import VectorStoreError

    manifest_path = get_manifest_path(settings, offline=args.offline)
    if manifest_path.exists():
        try:
            manifest = validate_index(settings, offline=args.offline)
        except (IndexingError, CorpusError, VectorStoreError, OSError) as exc:
            print(f"ERROR: Invalid index: {exc}")
            return 2
        print(f"Validated {'OFFLINE TEST' if args.offline else 'GEMINI'} index: {manifest['documents']} documents, {manifest['chunks']} chunks; cosine.")
    elif args.offline:
        print("NOT READY: Offline test index is missing. Run python -m scripts.ingest --offline --rebuild.")
        return 1
    if args.offline:
        print("Offline test index only; Gemini access and semantic answer quality were not tested.")
        return 0
    issues = settings.setup_issues()
    for issue in issues:
        print(f"NOT READY: {issue}")
    print("Offline check only; account model access, quotas and Gemini calls were not tested.")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
