"""Tests for the federated job queue abstraction (federated_queue.py).

Covers:
  - BaseJobQueue protocol (structural conformance of existing JobQueue)
  - LocalJobQueue (wrapper around JobQueue)
  - FederatedJobQueue (fail-closed fallback to local when no federation)
  - make_job_queue factory
"""

import asyncio
import os
import tempfile

import pytest

from federated_queue import (
    BaseJobQueue,
    LocalJobQueue,
    FederatedJobQueue,
    make_job_queue,
    _serialize_result,
)
from rfsn_job_queue import JobQueue, JobStatus


class FakeOrchestrator:
    """Minimal orchestrator for queue tests — returns a fake result."""

    async def run(self, query, force_route=None):
        class FakeResult:
            final_answer = f"answer to: {query}"
            route_taken = "fake"
            clr_score = 1.0
            routing_confidence = 1.0
            specialist_used = "fake"
            raw_traces = {}
        return FakeResult()


@pytest.fixture
def audit_log_path():
    path = tempfile.mktemp(suffix=".jsonl")
    yield path
    if os.path.exists(path):
        os.unlink(path)


class TestBaseJobQueueProtocol:
    """Tests that the existing JobQueue satisfies the protocol."""

    def test_job_queue_satisfies_protocol(self):
        q = JobQueue(FakeOrchestrator(), audit_log=None)
        assert isinstance(q, BaseJobQueue)

    def test_local_job_queue_satisfies_protocol(self):
        q = LocalJobQueue(FakeOrchestrator(), audit_log=None)
        assert isinstance(q, BaseJobQueue)

    def test_federated_job_queue_satisfies_protocol(self):
        q = FederatedJobQueue(FakeOrchestrator(), audit_log=None)
        assert isinstance(q, BaseJobQueue)


class TestLocalJobQueue:
    """Tests for the local wrapper."""

    @pytest.mark.asyncio
    async def test_submit_and_wait(self, audit_log_path):
        q = LocalJobQueue(
            FakeOrchestrator(), max_concurrent=2, audit_log=audit_log_path
        )
        await q.start()
        try:
            job = q.submit("test query", priority=5)
            assert job.status == JobStatus.PENDING
            result = await q.wait_for(job.job_id, timeout=10.0)
            assert "answer to" in result.final_answer
            assert q.status(job.job_id) == JobStatus.COMPLETED
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_list_jobs(self, audit_log_path):
        q = LocalJobQueue(
            FakeOrchestrator(), max_concurrent=2, audit_log=audit_log_path
        )
        await q.start()
        try:
            q.submit("query 1")
            q.submit("query 2")
            jobs = q.list_jobs()
            assert len(jobs) == 2
        finally:
            await q.stop()

    def test_inner_property(self):
        q = LocalJobQueue(FakeOrchestrator(), audit_log=None)
        assert isinstance(q.inner, JobQueue)


class TestFederatedJobQueue:
    """Tests for the federated queue (fail-closed fallback)."""

    def test_no_federation_url_is_local_only(self):
        q = FederatedJobQueue(FakeOrchestrator(), federation_url="", audit_log=None)
        assert not q._federation_configured()
        assert not q.is_federated

    def test_url_without_certs_is_local_only(self):
        q = FederatedJobQueue(
            FakeOrchestrator(),
            federation_url="https://swarm:7443",
            audit_log=None,
        )
        assert not q._federation_configured()

    def test_url_with_nonexistent_certs_is_local_only(self):
        q = FederatedJobQueue(
            FakeOrchestrator(),
            federation_url="https://swarm:7443",
            mtls_cert="/nonexistent/cert.pem",
            mtls_key="/nonexistent/key.pem",
            mtls_ca="/nonexistent/ca.pem",
            audit_log=None,
        )
        assert not q._federation_configured()

    @pytest.mark.asyncio
    async def test_submit_runs_locally_when_no_federation(self, audit_log_path):
        """Without a federation, jobs still run locally (fail-closed)."""
        q = FederatedJobQueue(
            FakeOrchestrator(),
            federation_url="",
            max_concurrent=2,
            audit_log=audit_log_path,
        )
        await q.start()
        try:
            job = q.submit("federated test query", priority=5)
            result = await q.wait_for(job.job_id, timeout=10.0)
            assert "answer to" in result.final_answer
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_cancel_works(self, audit_log_path):
        q = FederatedJobQueue(
            FakeOrchestrator(),
            federation_url="",
            max_concurrent=2,
            audit_log=audit_log_path,
        )
        await q.start()
        try:
            job = q.submit("to be cancelled", priority=0)
            # Cancel before it runs (priority 0, low)
            cancelled = q.cancel(job.job_id)
            assert cancelled is True
        finally:
            await q.stop()

    def test_inner_property(self):
        q = FederatedJobQueue(FakeOrchestrator(), audit_log=None)
        assert isinstance(q.inner, JobQueue)


class TestAttributeDelegation:
    """Tests that the wrappers delegate REPL-accessed attributes to the
    inner JobQueue. Without this, the CLI crashes when using
    make_job_queue() because the REPL accesses queue.bitemporal,
    queue.job_history, and queue.state_as_of directly."""

    @pytest.fixture
    def audit_log_path(self):
        path = tempfile.mktemp(suffix=".jsonl")
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_local_delegates_bitemporal(self, audit_log_path):
        q = LocalJobQueue(FakeOrchestrator(), audit_log=audit_log_path)
        assert hasattr(q, "bitemporal")
        assert q.bitemporal is not None
        assert q.bitemporal is q.inner.bitemporal

    def test_local_delegates_job_history(self, audit_log_path):
        q = LocalJobQueue(FakeOrchestrator(), audit_log=audit_log_path)
        assert hasattr(q, "job_history")
        assert callable(q.job_history)

    def test_local_delegates_state_as_of(self, audit_log_path):
        q = LocalJobQueue(FakeOrchestrator(), audit_log=audit_log_path)
        assert hasattr(q, "state_as_of")
        assert callable(q.state_as_of)

    def test_federated_delegates_bitemporal(self, audit_log_path):
        q = FederatedJobQueue(
            FakeOrchestrator(), federation_url="", audit_log=audit_log_path
        )
        assert hasattr(q, "bitemporal")
        assert q.bitemporal is not None
        assert q.bitemporal is q.inner.bitemporal

    def test_federated_delegates_job_history(self, audit_log_path):
        q = FederatedJobQueue(
            FakeOrchestrator(), federation_url="", audit_log=audit_log_path
        )
        assert hasattr(q, "job_history")
        assert callable(q.job_history)

    def test_federated_delegates_state_as_of(self, audit_log_path):
        q = FederatedJobQueue(
            FakeOrchestrator(), federation_url="", audit_log=audit_log_path
        )
        assert hasattr(q, "state_as_of")
        assert callable(q.state_as_of)

    def test_make_job_queue_delegates_bitemporal(self, audit_log_path):
        """End-to-end: make_job_queue returns a wrapper that delegates
        bitemporal — this is what the CLI path uses."""
        from federated_queue import make_job_queue
        q = make_job_queue(FakeOrchestrator(), audit_log=audit_log_path)
        assert q.bitemporal is not None
        assert q.bitemporal is q.inner.bitemporal


class TestMakeJobQueueFactory:
    """Tests for the make_job_queue factory."""

    def test_no_federation_url_returns_local(self):
        q = make_job_queue(FakeOrchestrator(), audit_log=None)
        assert isinstance(q, LocalJobQueue)

    def test_federation_url_returns_federated(self):
        q = make_job_queue(
            FakeOrchestrator(),
            federation_url="https://swarm:7443",
            audit_log=None,
        )
        assert isinstance(q, FederatedJobQueue)

    def test_empty_federation_url_returns_local(self):
        q = make_job_queue(
            FakeOrchestrator(),
            federation_url="",
            audit_log=None,
        )
        assert isinstance(q, LocalJobQueue)


# ---------------------------------------------------------------------- #
# _serialize_result (v1.0)
# ---------------------------------------------------------------------- #
class TestSerializeResult:
    """Tests for the _serialize_result helper that converts OrchestratorResult
    to a JSON-safe dict for federation POST-back."""

    def test_none_returns_empty_dict(self):
        assert _serialize_result(None) == {}

    def test_dict_passthrough(self):
        d = {"answer": "42", "score": 0.95}
        assert _serialize_result(d) == d

    def test_dataclass_serialized(self):
        """A simple dataclass is serialized via __dict__."""
        from dataclasses import dataclass

        @dataclass
        class FakeResult:
            final_answer: str = "42"
            score: float = 0.95
            verified: bool = True

        result = _serialize_result(FakeResult())
        assert result["final_answer"] == "42"
        assert result["score"] == 0.95
        assert result["verified"] is True

    def test_non_serializable_raw_traces_handled(self):
        """raw_traces with non-serializable nested objects doesn't crash."""
        from dataclasses import dataclass, field
        from typing import Any, Dict

        @dataclass
        class FakeResult:
            final_answer: str = "42"
            raw_traces: Dict[str, Any] = field(default_factory=dict)

        # raw_traces contains a non-serializable object (a custom class).
        class NonSerializable:
            def __repr__(self):
                return "<NonSerializable>"

        result = FakeResult(raw_traces={"nested": NonSerializable()})
        serialized = _serialize_result(result)
        # Should not crash — the non-serializable value becomes a string.
        assert serialized["final_answer"] == "42"
        assert "raw_traces" in serialized

    def test_orchestrator_result_shape(self):
        """Test with the actual OrchestratorResult dataclass."""
        from hybrid_orchestrator import OrchestratorResult

        result = OrchestratorResult(
            final_answer="4",
            route_taken="specialist_clr",
            specialist_used="VibeThinker-3B",
            clr_score=0.95,
            raw_traces={"clr_result": {"best_answer": "4", "best_score": 0.95}},
        )
        serialized = _serialize_result(result)
        assert serialized["final_answer"] == "4"
        assert serialized["route_taken"] == "specialist_clr"
        assert serialized["clr_score"] == 0.95
        assert serialized["raw_traces"]["clr_result"]["best_answer"] == "4"

    def test_non_dict_non_dataclass_returns_string(self):
        """A bare string is wrapped in a dict."""
        result = _serialize_result("just a string")
        assert result == {"result": "just a string"}
