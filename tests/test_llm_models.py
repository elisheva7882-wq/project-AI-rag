"""Real chat adapter with a fake SDK transport; no external network calls."""

from __future__ import annotations

import socket

import pytest

from config import load_settings
from src.gemini_models import ProviderError, create_llm


def test_llm_factory_requires_the_project_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "environment-only-test-key")
    settings = load_settings(project_root=tmp_path, environ={})
    with pytest.raises(ProviderError):
        create_llm(settings)


@pytest.mark.parametrize("value", ["true", "false"])
def test_llm_factory_rejects_implicit_vertex_mode(tmp_path, monkeypatch, value):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", value)
    settings = load_settings(project_root=tmp_path, environ={"GEMINI_API_KEY": "unit-test-fake-key"})
    with pytest.raises(ProviderError):
        create_llm(settings)


@pytest.mark.parametrize("timeout_seconds", [120, 12.345, 300])
@pytest.mark.parametrize("thinking_level", ["low", "medium", "high"])
def test_actual_chat_adapter_uses_explicit_policy_json_output_and_no_tools(tmp_path, monkeypatch, timeout_seconds, thinking_level):
    import google.genai
    from google.genai import types
    from llama_index.core.llms import ChatMessage, MessageRole

    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    captured = {}

    class ForbiddenMetadata:
        def get(self, *args, **kwargs):
            raise AssertionError("Constructor and chat must not request model metadata")

    class FakeChat:
        def send_message(self, parts):
            captured["question_parts"] = parts
            return types.GenerateContentResponse(candidates=[types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text='{"status":"no_answer","claims":[]}')]),
                finish_reason=types.FinishReason.STOP,
            )])

    class FakeChats:
        def create(self, **kwargs):
            captured["chat"] = kwargs
            return FakeChat()

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.models = ForbiddenMetadata()
            self.chats = FakeChats()

    def forbidden_external(*args, **kwargs):
        raise AssertionError("LLM adapter test must not access an external destination")

    original_connect = socket.socket.connect

    def loopback_only(stream, address):
        # asyncio.run creates a local socketpair on Windows. It is unrelated to
        # an SDK request, so allow only this loopback plumbing.
        if isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1"):
            return original_connect(stream, address)
        return forbidden_external(stream, address)

    monkeypatch.setattr(google.genai, "Client", FakeClient)
    monkeypatch.setattr(socket, "create_connection", forbidden_external)
    monkeypatch.setattr(socket.socket, "connect", loopback_only)
    settings = load_settings(project_root=tmp_path, environ={
        "GEMINI_API_KEY": "unit-test-fake-key",
        "LLM_REQUEST_TIMEOUT_SECONDS": str(timeout_seconds),
        "LLM_THINKING_LEVEL": thinking_level,
    })
    model = create_llm(settings)
    policy = "מדיניות בדיקת תוכנה: מקורות מצורפים בלבד וללא כלי חיפוש."
    question = '{"question":"שאלת בדיקה טכנית","sources":[]}'
    response = model.chat([
        ChatMessage(role=MessageRole.SYSTEM, content=policy),
        ChatMessage(role=MessageRole.USER, content=question),
    ])
    assert response.message.content == '{"status":"no_answer","claims":[]}'
    assert captured["client"]["api_key"] == settings.gemini_api_key
    assert captured["chat"]["model"] == settings.llm_model
    configuration = captured["chat"]["config"]
    assert configuration.system_instruction == policy
    assert configuration.response_mime_type == "application/json"
    assert configuration.response_schema is not None
    assert configuration.max_output_tokens == 4096
    assert configuration.tools == []
    assert configuration.automatic_continuation is False
    assert configuration.automatic_function_calling.disable is True
    assert configuration.automatic_function_calling.maximum_remote_calls is None
    assert configuration.temperature is None
    assert configuration.thinking_config.thinking_level == types.ThinkingLevel[thinking_level.upper()]
    assert configuration.thinking_config.include_thoughts is False
    assert configuration.thinking_config.thinking_budget is None
    assert captured["client"]["http_options"].timeout == round(timeout_seconds * 1000)
    assert captured["client"]["http_options"].retry_options.attempts == 1
    assert all(getattr(part, "text", None) == question for part in captured["question_parts"])
