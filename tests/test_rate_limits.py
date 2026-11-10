"""Structured quota advice and bounded ingestion timing, with no real waits."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace

import pytest
from llama_index.core.schema import TextNode

from config import ConfigurationError, load_settings
from src.embedding_checkpoint import EmbeddingCheckpoint
from src.indexer import IndexingError, _embed_nodes
from src.rate_limits import next_retry_delay, retry_advice


class StructuredFailure(RuntimeError):
    def __init__(self, *, code=429, details=None, headers=None):
        super().__init__("private-sdk-payload-and-fake-secret-must-not-be-printed")
        self.code = code
        self.details = details
        self.response = SimpleNamespace(headers={} if headers is None else headers)


def _quota(identifier, *, value=None):
    violation = {"quotaId": identifier, "quotaMetric": "embedding_requests"}
    if value is not None:
        violation["quotaValue"] = value
    return {"error": {"details": [{
        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
        "violations": [violation],
    }]}}


def _retry_info(seconds):
    return {"error": {"details": [{
        "@type": "type.googleapis.com/google.rpc.RetryInfo",
        "retryDelay": seconds,
    }]}}


@pytest.mark.parametrize("identifier", ["EmbedRequestsPerDay", "embed_requests_per_day", "DAILY_EMBED_REQUESTS"])
def test_explicit_daily_quota_stops_retry(identifier):
    error = StructuredFailure(details=_quota(identifier))
    assert retry_advice(error).kind == "daily"
    assert next_retry_delay(error, retry_number=1, max_wait_seconds=60) is None


@pytest.mark.parametrize("identifier", ["EmbedRequestsPerMinute", "embed_requests_per_minute"])
def test_explicit_minute_quota_is_classified_without_guessing_from_progress(identifier):
    assert retry_advice(StructuredFailure(details=_quota(identifier))).kind == "rate"
    assert retry_advice(StructuredFailure()).kind == "unknown"


def test_daily_quota_takes_precedence_when_minute_and_daily_limits_are_both_reported():
    details = {"error": {"details": [_quota("EmbedRequestsPerMinute"), _quota("EmbedRequestsPerDay")]}}
    assert retry_advice(StructuredFailure(details=details)).kind == "daily"


@pytest.mark.parametrize("value", [0, "0", 0.0])
def test_explicit_zero_entitlement_stops_retry(value):
    error = StructuredFailure(details=_quota("GenericEmbeddingRequests", value=value))
    assert retry_advice(error).quota_zero is True
    assert next_retry_delay(error, retry_number=1, max_wait_seconds=60) is None


@pytest.mark.parametrize("value", [False, True, "unknown", "zero", 1])
def test_boolean_or_nonzero_quota_value_is_not_a_zero_entitlement(value):
    assert retry_advice(StructuredFailure(details=_quota("GenericEmbeddingRequests", value=value))).quota_zero is False


def test_retry_info_and_retry_after_are_combined_using_the_longest_provider_delay(monkeypatch):
    import src.rate_limits as module

    monkeypatch.setattr(module.random, "uniform", lambda *args: 0.0)
    error = StructuredFailure(details=_retry_info("49.4s"), headers={"Retry-After": "55"})
    advice = retry_advice(error)
    assert advice.kind == "unknown"
    assert advice.retry_after == pytest.approx(55.0)
    assert next_retry_delay(error, retry_number=1, max_wait_seconds=60) == pytest.approx(55.0)


def test_retry_after_http_date_is_parsed_against_utc_without_real_wait(monkeypatch):
    import src.rate_limits as module

    now = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(module, "datetime", FixedDateTime)
    date = format_datetime(now + timedelta(seconds=35), usegmt=True)
    advice = retry_advice(StructuredFailure(headers={"retry-after": date}))
    assert advice.retry_after == pytest.approx(35.0)


@pytest.mark.parametrize("delay", ["-3s", "NaNs", "infs", "not-a-duration", ""])
def test_malformed_structured_duration_is_ignored(delay):
    assert retry_advice(StructuredFailure(details=_retry_info(delay))).retry_after is None


@pytest.mark.parametrize("header", ["-1", "nan", "inf", "not-a-date", ""])
def test_malformed_retry_after_header_is_ignored(header):
    assert retry_advice(StructuredFailure(headers={"Retry-After": header})).retry_after is None


def test_server_delay_beyond_budget_stops_instead_of_retrying_too_early(monkeypatch):
    import src.rate_limits as module

    monkeypatch.setattr(module.random, "uniform", lambda *args: 0.0)
    error = StructuredFailure(details=_retry_info("61s"))
    assert next_retry_delay(error, retry_number=1, max_wait_seconds=60) is None


def test_unknown_quota_uses_bounded_exponential_retry_with_jitter(monkeypatch):
    import src.rate_limits as module

    monkeypatch.setattr(module.random, "uniform", lambda *args: 0.25)
    error = StructuredFailure()
    assert next_retry_delay(error, retry_number=1, max_wait_seconds=60) == 5.25
    assert next_retry_delay(error, retry_number=2, max_wait_seconds=60) == 10.25
    assert next_retry_delay(error, retry_number=5, max_wait_seconds=60) is None


def test_diagnostics_do_not_return_or_print_payload_project_or_secret(capsys):
    details = _quota("GenericEmbeddingRequests")
    details["error"]["message"] = "private-sdk-payload-and-fake-secret-must-not-be-printed"
    details["error"]["subject"] = "projects/private-project-id"
    advice = retry_advice(StructuredFailure(details=details))
    assert "private" not in repr(advice)
    output = capsys.readouterr()
    assert not output.out
    assert not output.err


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.waits = []
        self.on_sleep = None

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        assert 0 <= seconds <= 60
        if self.on_sleep is not None:
            self.on_sleep(seconds)
        self.waits.append(seconds)
        self.now += seconds


class ScriptedEmbedding:
    def __init__(self, clock, *, failures=None, duration=0.0):
        self.clock = clock
        self.failures = {} if failures is None else failures
        self.duration = duration
        self.calls = []
        self.starts = []

    def get_text_embedding_batch(self, texts):
        self.calls.append(list(texts))
        self.starts.append(self.clock.now)
        self.clock.now += self.duration
        failure = self.failures.get(len(self.calls))
        if failure is not None:
            raise failure
        return [[0.5] * 64 for _ in texts]


@pytest.fixture
def embedding_run(tmp_path, monkeypatch):
    import src.indexer as indexer
    import src.rate_limits as limits

    nodes = [TextNode(id_=f"technical-{number:02d}", text=f"Software-only source fixture {number:02d}.") for number in range(25)]
    checkpoint = EmbeddingCheckpoint(tmp_path / "snapshots", {"model": "technical-only", "dimension": 64}, nodes, 64)
    settings = load_settings(project_root=tmp_path, environ={})
    settings = replace(settings, embedding_request_interval_seconds=0.0, embedding_max_retries=2, embedding_max_retry_wait_seconds=60.0)
    clock = FakeClock()
    monkeypatch.setattr(indexer.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(indexer.time, "sleep", clock.sleep)
    monkeypatch.setattr(limits.random, "uniform", lambda *args: 0.0)
    return nodes, checkpoint, settings, clock


def _run(run, model, *, offline=True, progress=None):
    nodes, checkpoint, settings, _ = run
    return _embed_nodes(nodes, model, 64, checkpoint=checkpoint, settings=settings, offline=offline, fresh_embeddings=False, progress=progress)


def test_transient_429_preserves_completed_batch_before_wait_and_finishes_once(embedding_run):
    nodes, checkpoint, _, clock = embedding_run
    error = StructuredFailure(details=_retry_info("7s"))
    model = ScriptedEmbedding(clock, failures={2: error})
    saved_at_wait = []
    clock.on_sleep = lambda seconds: saved_at_wait.append(set(checkpoint.load()))
    progress = []
    assert _run(embedding_run, model, progress=progress.append) == 0
    assert len(model.calls) == 4  # 10 success, 10 fail, same10 retry, final5.
    assert model.calls[1] == model.calls[2]
    assert clock.waits == [7.0]
    assert saved_at_wait == [{node.node_id for node in nodes[:10]}]
    assert set(checkpoint.load()) == {node.node_id for node in nodes}
    assert "private-sdk-payload" not in "\n".join(progress)


def test_repeated_429_stops_after_configured_retry_budget(embedding_run):
    _, checkpoint, _, clock = embedding_run
    model = ScriptedEmbedding(clock, failures={1: StructuredFailure(), 2: StructuredFailure(), 3: StructuredFailure()})
    with pytest.raises(IndexingError) as failure:
        _run(embedding_run, model)
    assert len(model.calls) == 3
    assert clock.waits == [5.0, 10.0]
    assert checkpoint.load() == {}
    assert "private-sdk-payload" not in str(failure.value)


@pytest.mark.parametrize("details", [_quota("EmbeddingPerDay"), _quota("GenericEmbeddingRequests", value=0)])
def test_daily_or_zero_quota_stops_immediately_with_durable_prior_progress(embedding_run, details):
    nodes, checkpoint, _, clock = embedding_run
    model = ScriptedEmbedding(clock, failures={2: StructuredFailure(details=details)})
    with pytest.raises(IndexingError):
        _run(embedding_run, model)
    assert len(model.calls) == 2
    assert clock.waits == []
    assert set(checkpoint.load()) == {node.node_id for node in nodes[:10]}


@pytest.mark.parametrize("code", [401, 403, 404, 500, 503])
def test_non_429_errors_are_not_retried_and_do_not_lose_saved_vectors(embedding_run, code):
    nodes, checkpoint, _, clock = embedding_run
    model = ScriptedEmbedding(clock, failures={2: StructuredFailure(code=code)})
    with pytest.raises(IndexingError):
        _run(embedding_run, model)
    assert len(model.calls) == 2
    assert clock.waits == []
    assert set(checkpoint.load()) == {node.node_id for node in nodes[:10]}


def test_provider_delay_above_wait_budget_stops_and_can_resume_without_repeating_successes(embedding_run):
    nodes, checkpoint, _, clock = embedding_run
    model = ScriptedEmbedding(clock, failures={2: StructuredFailure(headers={"Retry-After": "120"})})
    with pytest.raises(IndexingError):
        _run(embedding_run, model)
    assert len(model.calls) == 2
    assert not clock.waits
    resumed = ScriptedEmbedding(clock)
    assert _run(embedding_run, resumed) == 10
    assert sum(map(len, resumed.calls)) == 15
    assert set(checkpoint.load()) == {node.node_id for node in nodes}


@pytest.mark.parametrize("offline", [False, True])
def test_request_pacing_uses_mock_clock_and_is_disabled_for_explicit_offline_mode(embedding_run, offline):
    nodes, checkpoint, settings, clock = embedding_run
    settings = replace(settings, embedding_request_interval_seconds=6.0)
    run = nodes, checkpoint, settings, clock
    model = ScriptedEmbedding(clock, duration=2.0)
    _run(run, model, offline=offline)
    assert model.starts == ([0.0, 2.0, 4.0] if offline else [0.0, 6.0, 12.0])
    assert clock.waits == ([] if offline else [4.0, 4.0])


def test_new_retry_settings_defaults_and_finite_boundary_values(tmp_path):
    defaults = load_settings(project_root=tmp_path, environ={})
    assert defaults.embedding_request_interval_seconds == 6.0
    assert defaults.embedding_max_retries == 2
    assert defaults.embedding_max_retry_wait_seconds == 60.0
    settings = load_settings(project_root=tmp_path, environ={
        "EMBEDDING_REQUEST_INTERVAL_SECONDS": "0",
        "EMBEDDING_MAX_RETRIES": "0",
        "EMBEDDING_MAX_RETRY_WAIT_SECONDS": "0",
    })
    assert settings.embedding_request_interval_seconds == settings.embedding_max_retries == settings.embedding_max_retry_wait_seconds == 0
    settings = load_settings(project_root=tmp_path, environ={
        "EMBEDDING_REQUEST_INTERVAL_SECONDS": "60",
        "EMBEDDING_MAX_RETRIES": "5",
        "EMBEDDING_MAX_RETRY_WAIT_SECONDS": "60",
    })
    assert settings.embedding_request_interval_seconds == settings.embedding_max_retry_wait_seconds == 60
    assert settings.embedding_max_retries == 5


@pytest.mark.parametrize("name,value", [
    ("EMBEDDING_REQUEST_INTERVAL_SECONDS", "-1"),
    ("EMBEDDING_REQUEST_INTERVAL_SECONDS", "61"),
    ("EMBEDDING_REQUEST_INTERVAL_SECONDS", "nan"),
    ("EMBEDDING_REQUEST_INTERVAL_SECONDS", "inf"),
    ("EMBEDDING_REQUEST_INTERVAL_SECONDS", ""),
    ("EMBEDDING_MAX_RETRY_WAIT_SECONDS", "-1"),
    ("EMBEDDING_MAX_RETRY_WAIT_SECONDS", "61"),
    ("EMBEDDING_MAX_RETRY_WAIT_SECONDS", "NaN"),
    ("EMBEDDING_MAX_RETRY_WAIT_SECONDS", "infinity"),
    ("EMBEDDING_MAX_RETRY_WAIT_SECONDS", "bad"),
    ("EMBEDDING_MAX_RETRIES", "-1"),
    ("EMBEDDING_MAX_RETRIES", "6"),
    ("EMBEDDING_MAX_RETRIES", "1.5"),
    ("EMBEDDING_MAX_RETRIES", ""),
])
def test_new_retry_settings_reject_unbounded_or_malformed_values(tmp_path, name, value):
    with pytest.raises(ConfigurationError):
        load_settings(project_root=tmp_path, environ={name: value})
