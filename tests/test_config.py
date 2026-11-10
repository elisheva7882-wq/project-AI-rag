"""Offline tests for settings boundaries, error states and secret handling."""

from dataclasses import FrozenInstanceError
import pytest

from config import ConfigurationError, load_settings


def test_defaults_and_absent_prerequisites(tmp_path):
    settings = load_settings(project_root=tmp_path, environ={})
    assert settings.llm_model == "gemini-3.8-flash"
    assert settings.embedding_model == "gemini-embedding-001"
    assert settings.llm_request_timeout_seconds == 120.0
    assert settings.llm_thinking_level == "medium"
    assert (settings.chunk_size, settings.chunk_overlap, settings.top_k) == (512, 64, 5)
    assert settings.similarity_threshold is None
    assert not settings.has_api_key
    assert len(settings.setup_issues()) == 3
    assert not list(tmp_path.iterdir())  # Loading configuration writes nothing.


def test_dotenv_utf8_bom_and_environment_precedence(tmp_path):
    (tmp_path / ".env").write_text(
        "GEMINI_API_KEY=dotenv-test-value\nTOP_K=4\nEVAL_PATH=בדיקות\n",
        encoding="utf-8-sig",
    )
    settings = load_settings(project_root=tmp_path, environ={"TOP_K": "5"})
    assert settings.gemini_api_key == "dotenv-test-value"
    assert settings.top_k == 5
    assert settings.eval_path == tmp_path / "בדיקות"


def test_paths_resolve_against_project_when_working_directory_changes(tmp_path, monkeypatch):
    other = tmp_path / "unrelated"
    other.mkdir()
    monkeypatch.chdir(other)
    settings = load_settings(project_root=tmp_path, environ={})
    assert settings.knowledge_base_path == tmp_path / "knowledge_base"
    assert settings.eval_path == tmp_path / "eval"
    assert settings.chroma_path == tmp_path / "storage" / "chroma"


@pytest.mark.parametrize("key", ["", "your_gemini_api_key_here", " CHANGEME "])
def test_placeholder_keys_are_not_configured(tmp_path, key):
    settings = load_settings(project_root=tmp_path, environ={"GEMINI_API_KEY": key})
    assert not settings.has_api_key


def test_api_key_is_not_in_settings_representation(tmp_path):
    fake_key = "unit-test-secret-never-a-real-api-key"
    settings = load_settings(project_root=tmp_path, environ={"GEMINI_API_KEY": fake_key})
    assert settings.has_api_key
    assert fake_key not in repr(settings)
    assert fake_key not in str(settings.setup_issues())


@pytest.mark.parametrize(
    "environment",
    [
        {"CHUNK_SIZE": "0"},
        {"CHUNK_SIZE": "abc"},
        {"CHUNK_OVERLAP": "-1"},
        {"CHUNK_OVERLAP": "512"},
        {"TOP_K": "0"},
        {"TOP_K": "6"},
        {"SIMILARITY_THRESHOLD": "NaN"},
        {"SIMILARITY_THRESHOLD": "inf"},
        {"SIMILARITY_THRESHOLD": "1.1"},
        {"SIMILARITY_THRESHOLD": "unknown"},
        {"LLM_MODEL": " "},
        {"KNOWLEDGE_BASE_PATH": ""},
        {"CHROMA_COLLECTION": "ארכיון"},
        {"CHROMA_COLLECTION": "ab"},
        {"CHROMA_COLLECTION": "bad..name"},
        {"CHROMA_COLLECTION": "127.0.0.1"},
        {"CHROMA_COLLECTION": "-bad-name"},
        {"CHROMA_COLLECTION": "a" * 513},
    ],
)
def test_invalid_configuration_is_rejected(tmp_path, environment):
    with pytest.raises(ConfigurationError):
        load_settings(project_root=tmp_path, environ=environment)


@pytest.mark.parametrize(
    "environment",
    [
        {"EVAL_PATH": "knowledge_base"},
        {"EVAL_PATH": "knowledge_base/eval"},
        {"KNOWLEDGE_BASE_PATH": "eval/docs"},
        {"CHROMA_PATH": "knowledge_base/index"},
        {"CHROMA_PATH": "eval/index"},
        {"KNOWLEDGE_BASE_PATH": "."},
    ],
)
def test_knowledge_evaluation_and_storage_cannot_overlap(tmp_path, environment):
    with pytest.raises(ConfigurationError):
        load_settings(project_root=tmp_path, environ=environment)


def test_data_path_cannot_point_to_a_file(tmp_path):
    (tmp_path / "knowledge_base").write_text("not a directory", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="KNOWLEDGE_BASE_PATH"):
        load_settings(project_root=tmp_path, environ={})


def test_threshold_is_optional_and_validated_as_similarity(tmp_path):
    settings = load_settings(project_root=tmp_path, environ={"SIMILARITY_THRESHOLD": "-0.25"})
    assert settings.similarity_threshold == -0.25


def test_settings_are_immutable(tmp_path):
    settings = load_settings(project_root=tmp_path, environ={})
    with pytest.raises(FrozenInstanceError):
        settings.top_k = 3


def test_existing_empty_knowledge_directory_is_reported(tmp_path):
    (tmp_path / "knowledge_base").mkdir()
    settings = load_settings(project_root=tmp_path, environ={})
    assert any("Markdown" in issue for issue in settings.setup_issues())


def test_explicit_env_file_and_environment_key_override(tmp_path):
    alternative = tmp_path / "local.env"
    alternative.write_text("GEMINI_API_KEY=file-value\n", encoding="utf-8")
    settings = load_settings(
        project_root=tmp_path,
        env_file=alternative,
        environ={"GEMINI_API_KEY": "environment-value"},
    )
    assert settings.gemini_api_key == "environment-value"


@pytest.mark.parametrize("value", ["10", "12.345", "120", "300"])
def test_llm_request_timeout_accepts_finite_seconds_within_bounds(tmp_path, value):
    settings = load_settings(project_root=tmp_path, environ={"LLM_REQUEST_TIMEOUT_SECONDS": value})
    assert settings.llm_request_timeout_seconds == float(value)


@pytest.mark.parametrize("value", ["", "unknown", "NaN", "inf", "-inf", "-1", "0", "9.99", "300.01"])
def test_llm_request_timeout_rejects_invalid_or_unbounded_deadlines(tmp_path, value):
    with pytest.raises(ConfigurationError, match="LLM_REQUEST_TIMEOUT_SECONDS"):
        load_settings(project_root=tmp_path, environ={"LLM_REQUEST_TIMEOUT_SECONDS": value})


def test_environment_overrides_llm_deadline_in_local_dotenv(tmp_path):
    (tmp_path / ".env").write_text("LLM_REQUEST_TIMEOUT_SECONDS=240\n", encoding="utf-8")
    settings = load_settings(project_root=tmp_path, environ={"LLM_REQUEST_TIMEOUT_SECONDS": "180"})
    assert settings.llm_request_timeout_seconds == 180


@pytest.mark.parametrize("value", ["low", "medium", "high"])
def test_llm_thinking_level_accepts_only_supported_choices(tmp_path, value):
    settings = load_settings(project_root=tmp_path, environ={"LLM_THINKING_LEVEL": value})
    assert settings.llm_thinking_level == value


@pytest.mark.parametrize("value", ["", " ", "LOW", "minimal", "dynamic", "none", "0"])
def test_llm_thinking_level_rejects_unsupported_choices(tmp_path, value):
    with pytest.raises(ConfigurationError, match="LLM_THINKING_LEVEL"):
        load_settings(project_root=tmp_path, environ={"LLM_THINKING_LEVEL": value})


def test_environment_overrides_llm_thinking_level_in_local_dotenv(tmp_path):
    (tmp_path / ".env").write_text("LLM_THINKING_LEVEL=high\n", encoding="utf-8")
    settings = load_settings(project_root=tmp_path, environ={"LLM_THINKING_LEVEL": "low"})
    assert settings.llm_thinking_level == "low"
