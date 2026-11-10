"""Explicit Gemini adapter configuration and isolated deterministic test vectors."""

from __future__ import annotations

import socket
import pytest

from config import ConfigurationError, load_settings
from src.gemini_models import ProviderError, create_embedding_model, create_offline_embedding


@pytest.mark.parametrize("value", ["1", "768", "1536", "3072"])
def test_supported_embedding_dimensions_can_be_configured(tmp_path, value):
    settings = load_settings(project_root=tmp_path, environ={"EMBEDDING_DIMENSION": value})
    assert settings.embedding_dimension == int(value)


@pytest.mark.parametrize("value", ["0", "-1", "3073", "not-a-number", "", "1.5"])
def test_invalid_embedding_dimensions_are_rejected(tmp_path, value):
    with pytest.raises(ConfigurationError):
        load_settings(project_root=tmp_path, environ={"EMBEDDING_DIMENSION": value})


def test_missing_project_key_cannot_use_google_environment_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "environment-fake-key-not-authorized-for-project")
    settings = load_settings(project_root=tmp_path, environ={})
    with pytest.raises(ProviderError):
        create_embedding_model(settings)


@pytest.mark.parametrize("value", ["true", "1", "false"])
def test_implicit_vertex_mode_is_rejected(tmp_path, monkeypatch, value):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", value)
    settings = load_settings(project_root=tmp_path, environ={"GEMINI_API_KEY": "unit-test-fake-key"})
    with pytest.raises(ProviderError):
        create_embedding_model(settings)


def test_actual_adapter_receives_explicit_model_dimension_and_bounded_http_retries(tmp_path, monkeypatch):
    import google.genai
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    captured = {}

    def constructor_only(**kwargs):
        captured.update(kwargs)
        return object()

    def forbidden_network(*args, **kwargs):
        raise AssertionError("Constructor verification must not make an API request")

    monkeypatch.setattr(google.genai, "Client", constructor_only)
    monkeypatch.setattr(socket, "create_connection", forbidden_network)
    monkeypatch.setattr(socket.socket, "connect", forbidden_network)
    settings = load_settings(project_root=tmp_path, environ={"GEMINI_API_KEY": "unit-test-fake-key"})
    model = create_embedding_model(settings)
    assert isinstance(model, GoogleGenAIEmbedding)
    assert model.model_name == "gemini-embedding-001"
    assert model.embedding_config["output_dimensionality"] == 3072
    assert captured["api_key"] == settings.gemini_api_key
    assert not captured.get("vertexai")
    assert 0 < captured["http_options"].timeout <= 120000
    # Ingestion owns bounded quota retries; SDK/adapter attempts must not multiply them.
    assert captured["http_options"].retry_options.attempts == 1
    assert model.retries == 1


def test_offline_vectors_are_deterministic_finite_and_clearly_named():
    import math

    model = create_offline_embedding()
    first = model.get_text_embedding("Technical source fixture A")
    repeated = create_offline_embedding().get_text_embedding("Technical source fixture A")
    different = model.get_text_embedding("Technical source fixture B")
    assert model.model_name == "kingdomai-offline-hash-v1"
    assert len(first) == len(different) == 64
    assert first == repeated
    assert first != different
    assert all(math.isfinite(value) for value in first)
    assert any(value != 0 for value in first)


def test_actual_adapter_distinguishes_document_and_query_tasks_without_network(tmp_path, monkeypatch):
    import google.genai
    from google.genai import types

    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    requests = []

    class ConstructorStub:
        def __init__(self, **kwargs):
            self.models = self

        def embed_content(self, *, model, contents, config):
            requests.append({"model": model, "contents": contents, "config": dict(config)})
            return types.EmbedContentResponse(
                embeddings=[types.ContentEmbedding(values=[0.25] * config["output_dimensionality"]) for _ in contents]
            )

    def forbidden_network(*args, **kwargs):
        raise AssertionError("Adapter task verification must remain offline")

    monkeypatch.setattr(google.genai, "Client", ConstructorStub)
    monkeypatch.setattr(socket, "create_connection", forbidden_network)
    monkeypatch.setattr(socket.socket, "connect", forbidden_network)
    settings = load_settings(project_root=tmp_path, environ={"GEMINI_API_KEY": "unit-test-fake-key"})
    model = create_embedding_model(settings)
    assert len(model.get_text_embedding("Original technical source")) == 3072
    assert len(model.get_query_embedding("Original technical question")) == 3072
    assert len(model.get_text_embedding_batch(["First technical source", "Second technical source"])) == 2
    assert [request["config"]["task_type"] for request in requests] == [
        "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT"
    ]
    assert requests[1]["contents"] == [["Original technical question"]]
    assert all(request["model"] == "gemini-embedding-001" for request in requests)
    assert all(request["config"]["output_dimensionality"] == 3072 for request in requests)
    assert "task_type" not in model.embedding_config
