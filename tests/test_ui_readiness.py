"""Local readiness checks use temporary synthetic data and never Gemini."""

from __future__ import annotations

import socket
from dataclasses import replace

import pytest

from config import load_settings
from src import gemini_models, indexer, ui
from src.file_loader import CorpusError
from src.indexer import IndexingError, get_manifest_path, ingest, validate_index
from src.vector_store import VectorStoreError


@pytest.fixture(autouse=True)
def forbid_external_services(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Readiness must not construct an embedding/QA service or use the network")

    monkeypatch.setattr(gemini_models, "create_embedding_model", forbidden)
    monkeypatch.setattr(indexer, "create_embedding_model", forbidden)
    monkeypatch.setattr(ui, "QAService", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture
def settings(tmp_path, corpus_directory):
    project = tmp_path / "isolated_project"
    project.mkdir()
    return load_settings(
        project_root=project,
        environ={
            "GEMINI_API_KEY": "synthetic-readiness-key-never-sent",
            "KNOWLEDGE_BASE_PATH": str(corpus_directory),
        },
    )


@pytest.mark.parametrize("create_directory", [False, True])
def test_missing_documents_are_reported_before_key_or_index(settings, tmp_path, create_directory):
    missing = tmp_path / "missing_knowledge"
    if create_directory:
        missing.mkdir()
        (missing / ".gitkeep").touch()
    status = ui.check_readiness(replace(settings, knowledge_base_path=missing, gemini_api_key=""))
    assert not status.ready
    assert status.code == "missing_documents"
    assert status.documents == status.chunks == 0
    assert status.message
    assert not (settings.project_root / "storage").exists()


def test_incomplete_corpus_is_invalid_not_ready(settings):
    next(settings.knowledge_base_path.glob("*.md")).unlink()
    status = ui.check_readiness(settings)
    assert not status.ready
    assert status.code == "invalid_documents"
    assert status.documents == status.chunks == 0
    assert status.message
    assert not (settings.project_root / "storage").exists()


def test_missing_key_is_reported_before_attempting_validation(settings, monkeypatch):
    def unexpected_validation(*args, **kwargs):
        pytest.fail("Index validation should wait until the local key is configured")

    monkeypatch.setattr(ui, "validate_index", unexpected_validation)
    status = ui.check_readiness(replace(settings, gemini_api_key=""))
    assert not status.ready
    assert status.code == "missing_api_key"
    assert status.documents == 19
    assert status.chunks == 0
    assert not (settings.project_root / "storage").exists()


def test_missing_production_index_is_reported_without_creating_storage(settings):
    status = ui.check_readiness(settings)
    assert not status.ready
    assert status.code == "index_not_ready"
    assert status.documents == 19
    assert status.chunks == 0
    assert not (settings.project_root / "storage").exists()


def test_validated_manifest_controls_displayed_counts_and_generation(settings, monkeypatch):
    calls = []

    def local_validation(actual_settings):
        calls.append(actual_settings)
        return {"documents": 19, "chunks": 73, "generation_id": "b" * 32}

    monkeypatch.setattr(ui, "validate_index", local_validation)
    status = ui.check_readiness(settings)
    assert status == ui.SetupStatus(True, documents=19, chunks=73, generation_id="b" * 32)
    assert calls == [settings]
    assert not (settings.project_root / "storage").exists()


@pytest.mark.parametrize("error_type", [IndexingError, CorpusError, VectorStoreError, OSError])
def test_invalid_index_failures_are_sanitized(settings, monkeypatch, error_type):
    def broken_validation(actual_settings):
        raise error_type("sensitive-test-payload-must-not-appear")

    monkeypatch.setattr(ui, "validate_index", broken_validation)
    status = ui.check_readiness(settings)
    assert not status.ready
    assert status.code == "index_invalid"
    assert status.documents == 19
    assert status.chunks == 0
    assert "sensitive-test-payload" not in status.message
    assert settings.gemini_api_key not in status.message


def test_offline_manifest_cannot_masquerade_as_production_ready(settings):
    result = ingest(settings, rebuild=True, offline=True)
    # Only a temporary synthetic offline manifest is placed at the temporary
    # production location. Real validation must reject its provider/storage tag.
    get_manifest_path(settings).write_bytes(result.manifest_path.read_bytes())
    status = ui.check_readiness(settings)
    assert not status.ready
    assert status.code == "index_invalid"
    assert status.documents == 19
    assert not settings.chroma_path.exists()


def test_changed_source_invalidates_previous_ready_status_without_manifest_change(settings, monkeypatch):
    result = ingest(settings, rebuild=True, offline=True)
    manifest_before = (result.manifest_path.read_bytes(), result.manifest_path.stat().st_mtime_ns)
    validations = []

    def validate_isolated_index(actual_settings):
        # Preserve real corpus/vector validation while substituting the explicit
        # offline store, which remains wholly below pytest's temporary directory.
        validations.append(actual_settings)
        return validate_index(actual_settings, offline=True)

    monkeypatch.setattr(ui, "validate_index", validate_isolated_index)
    first = ui.check_readiness(settings)
    assert first.ready
    assert first.documents == 19
    assert first.chunks == result.chunks
    assert len(first.generation_id) == 32

    source = next(settings.knowledge_base_path.glob("*.md"))
    source.write_bytes(source.read_bytes() + b"\nAdditional synthetic source bytes.\n")
    second = ui.check_readiness(settings)
    assert not second.ready
    assert second.code == "index_invalid"
    assert validations == [settings, settings]
    assert (result.manifest_path.read_bytes(), result.manifest_path.stat().st_mtime_ns) == manifest_before
    assert not settings.chroma_path.exists()
