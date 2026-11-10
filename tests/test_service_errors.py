"""Public failure categories never depend on raw provider error messages."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
import requests

from src.service_errors import classify_service_error, provider_status_code


class StructuredFailure(RuntimeError):
    def __init__(self, *, code=None, response_code=None, kind=None):
        super().__init__("private provider payload: api-key-and-request-content")
        self.code = code
        self.response = SimpleNamespace(status_code=response_code)
        self.kind = kind

    def __str__(self):
        raise AssertionError("Classification must never stringify a provider error")


@pytest.mark.parametrize("code", [401, 403, 404, 408, 429, 500, 502, 503, 504, 0, 501])
def test_provider_integer_codes_are_read_without_messages(code):
    assert provider_status_code(StructuredFailure(code=code)) == code


@pytest.mark.parametrize("invalid", [True, False, "503", "private-payload", 503.0, None, [], {}])
def test_provider_codes_are_not_coerced(invalid):
    assert provider_status_code(StructuredFailure(code=invalid)) is None


@pytest.mark.parametrize("invalid", [True, False, "503", 503.0, None, [], {}])
def test_http_response_is_used_when_provider_code_is_not_an_integer(invalid):
    error = StructuredFailure(code=invalid, response_code=503)
    assert provider_status_code(error) == 503
    assert classify_service_error(error) == "service_unavailable"


@pytest.mark.parametrize("invalid", [True, False, "503", 503.0, None, [], {}])
def test_http_response_codes_are_not_coerced(invalid):
    assert provider_status_code(StructuredFailure(response_code=invalid)) is None


def test_provider_code_takes_precedence_over_http_response_code():
    error = StructuredFailure(code=429, response_code=503)
    assert provider_status_code(error) == 429
    assert classify_service_error(error) == "rate_limit"


@pytest.mark.parametrize("code,kind", [
    (401, "access_denied"), (403, "access_denied"),
    (404, "model_unavailable"), (429, "rate_limit"),
    (408, "service_timeout"), (504, "service_timeout"),
    (500, "service_unavailable"), (502, "service_unavailable"), (503, "service_unavailable"),
])
@pytest.mark.parametrize("field", ["code", "response_code"])
def test_supported_status_codes_have_safe_public_categories(code, kind, field):
    assert classify_service_error(StructuredFailure(**{field: code})) == kind


@pytest.mark.parametrize("kind", [
    "access_denied", "model_unavailable", "rate_limit", "service_timeout",
    "connection_error", "service_unavailable",
])
def test_sanitized_wrapper_kind_is_retained(kind):
    assert classify_service_error(StructuredFailure(kind=kind)) == kind


def test_explicit_sanitized_wrapper_kind_takes_precedence_over_status_code():
    assert classify_service_error(StructuredFailure(kind="service_timeout", code=503)) == "service_timeout"


@pytest.mark.parametrize("kind", ["private-payload", "SERVICE_TIMEOUT", "index_invalid", "", True, 503, [], {}])
def test_arbitrary_wrapper_kinds_are_ignored(kind):
    assert classify_service_error(StructuredFailure(kind=kind)) is None
    assert classify_service_error(StructuredFailure(kind=kind, code=503)) == "service_unavailable"


@pytest.mark.parametrize("failure_type", [
    TimeoutError, httpx.TimeoutException, httpx.ConnectTimeout, httpx.ReadTimeout,
    httpx.WriteTimeout, httpx.PoolTimeout,
    requests.exceptions.Timeout, requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout,
])
def test_transport_timeouts_have_the_timeout_category(failure_type):
    assert classify_service_error(failure_type("private transport message")) == "service_timeout"


@pytest.mark.parametrize("failure_type", [
    ConnectionError, httpx.NetworkError, httpx.ConnectError, httpx.ReadError,
    httpx.WriteError, httpx.CloseError, requests.exceptions.ConnectionError,
])
def test_transport_network_errors_have_the_connection_category(failure_type):
    assert classify_service_error(failure_type("private transport message")) == "connection_error"


def test_httpx_http_status_error_uses_only_response_status():
    request = httpx.Request("POST", "https://unit-test.invalid/private")
    response = httpx.Response(503, request=request)
    error = httpx.HTTPStatusError("private request data", request=request, response=response)
    assert provider_status_code(error) == 503
    assert classify_service_error(error) == "service_unavailable"


@pytest.mark.parametrize("error", [None, RuntimeError("503 timeout private api key"), StructuredFailure(code=400), StructuredFailure(code=501)])
def test_unknown_errors_are_not_guessed_from_messages(error):
    assert classify_service_error(error) is None


def test_exception_context_is_not_unwrapped_or_stringified():
    outer = StructuredFailure()
    outer.__cause__ = StructuredFailure(code=429)
    outer.__context__ = StructuredFailure(kind="service_timeout")
    assert provider_status_code(outer) is None
    assert classify_service_error(outer) is None
