"""Tests for the Python-native federation server (v0.4.1).

Tests the FederationState (in-memory job store with asyncio.Lock) and
the FastAPI endpoints (submit, claim, complete, jobs, health).
"""

import asyncio
import json
import pytest

from federation_server import FederationState, create_federation_app


# ---------------------------------------------------------------------- #
# FederationState (unit tests, no HTTP)
# ---------------------------------------------------------------------- #
class TestFederationState:
    @pytest.mark.asyncio
    async def test_submit_creates_pending_job(self):
        state = FederationState()
        job = await state.submit("job-1", "What is 2+2?")
        assert job.job_id == "job-1"
        assert job.query == "What is 2+2?"
        assert job.status == "pending"
        assert job.priority == 0

    @pytest.mark.asyncio
    async def test_submit_generates_job_id_when_empty(self):
        state = FederationState()
        job = await state.submit("", "query")
        assert len(job.job_id) == 12  # uuid hex[:12]

    @pytest.mark.asyncio
    async def test_claim_returns_pending_job(self):
        state = FederationState()
        await state.submit("j1", "query 1")
        job = await state.claim("worker-1")
        assert job is not None
        assert job.job_id == "j1"
        assert job.status == "claimed"
        assert job.claimed_by == "worker-1"
        assert job.claimed_at is not None

    @pytest.mark.asyncio
    async def test_claim_returns_none_when_no_jobs(self):
        state = FederationState()
        job = await state.claim("worker-1")
        assert job is None

    @pytest.mark.asyncio
    async def test_claim_returns_none_when_all_claimed(self):
        state = FederationState()
        await state.submit("j1", "query 1")
        await state.claim("worker-1")
        job = await state.claim("worker-2")
        assert job is None

    @pytest.mark.asyncio
    async def test_claim_priority_ordering(self):
        """Higher priority jobs are claimed first."""
        state = FederationState()
        await state.submit("low", "q1", priority=1)
        await state.submit("high", "q2", priority=10)
        await state.submit("mid", "q3", priority=5)
        first = await state.claim("w1")
        assert first.job_id == "high"
        second = await state.claim("w2")
        assert second.job_id == "mid"
        third = await state.claim("w3")
        assert third.job_id == "low"

    @pytest.mark.asyncio
    async def test_claim_fifo_for_same_priority(self):
        """Same-priority jobs are claimed in FIFO order."""
        state = FederationState()
        await state.submit("j1", "q1", priority=5)
        await state.submit("j2", "q2", priority=5)
        first = await state.claim("w1")
        assert first.job_id == "j1"
        second = await state.claim("w2")
        assert second.job_id == "j2"

    @pytest.mark.asyncio
    async def test_complete_sets_done(self):
        state = FederationState()
        await state.submit("j1", "q1")
        await state.claim("w1")
        ok = await state.complete("j1", result={"answer": "4"})
        assert ok is True
        job = await state.get_job("j1")
        assert job.status == "done"
        assert job.result == {"answer": "4"}
        assert job.completed_at is not None

    @pytest.mark.asyncio
    async def test_complete_with_error_sets_error(self):
        state = FederationState()
        await state.submit("j1", "q1")
        await state.claim("w1")
        ok = await state.complete("j1", error="timeout")
        assert ok is True
        job = await state.get_job("j1")
        assert job.status == "error"
        assert job.error == "timeout"

    @pytest.mark.asyncio
    async def test_complete_unknown_job_returns_false(self):
        state = FederationState()
        ok = await state.complete("nonexistent")
        assert ok is False

    @pytest.mark.asyncio
    async def test_list_jobs(self):
        state = FederationState()
        await state.submit("j1", "q1")
        await state.submit("j2", "q2")
        jobs = await state.list_jobs()
        assert len(jobs) == 2
        assert all("job_id" in j for j in jobs)
        assert all("status" in j for j in jobs)

    @pytest.mark.asyncio
    async def test_get_job_returns_none_for_unknown(self):
        state = FederationState()
        job = await state.get_job("nope")
        assert job is None

    @pytest.mark.asyncio
    async def test_concurrent_claim_race_safety(self):
        """Two workers claiming simultaneously get different jobs (asyncio.Lock)."""
        state = FederationState()
        await state.submit("j1", "q1")
        await state.submit("j2", "q2")
        # Claim both concurrently — the lock should serialize them.
        results = await asyncio.gather(
            state.claim("w1"),
            state.claim("w2"),
        )
        job_ids = {r.job_id for r in results if r is not None}
        assert job_ids == {"j1", "j2"}


# ---------------------------------------------------------------------- #
# FastAPI endpoints (integration via TestClient)
# ---------------------------------------------------------------------- #
class TestFederationApp:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        app = create_federation_app()
        return TestClient(app)

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_submit(self, client):
        resp = client.post("/submit", json={"query": "What is 2+2?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_submit_with_explicit_job_id(self, client):
        resp = client.post("/submit", json={"job_id": "my-job", "query": "q"})
        assert resp.json()["job_id"] == "my-job"

    def test_claim_returns_job(self, client):
        client.post("/submit", json={"query": "What is 2+2?"})
        resp = client.post("/claim", json={"worker_id": "laptop-1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] is not None
        assert data["status"] == "claimed"
        assert data["query"] == "What is 2+2?"

    def test_claim_no_jobs_returns_null(self, client):
        resp = client.post("/claim", json={"worker_id": "laptop-1"})
        assert resp.json()["job_id"] is None
        assert resp.json()["status"] == "no_jobs"

    def test_complete_with_result(self, client):
        client.post("/submit", json={"job_id": "j1", "query": "q"})
        client.post("/claim", json={"worker_id": "w1"})
        resp = client.post("/complete", json={
            "job_id": "j1",
            "result": {"answer": "4", "verified": True},
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_complete_with_error(self, client):
        client.post("/submit", json={"job_id": "j1", "query": "q"})
        client.post("/claim", json={"worker_id": "w1"})
        resp = client.post("/complete", json={
            "job_id": "j1",
            "error": "model server down",
        })
        assert resp.status_code == 200

    def test_complete_unknown_job_404(self, client):
        resp = client.post("/complete", json={"job_id": "nope"})
        assert resp.status_code == 404

    def test_list_jobs(self, client):
        client.post("/submit", json={"job_id": "j1", "query": "q1"})
        client.post("/submit", json={"job_id": "j2", "query": "q2"})
        resp = client.get("/jobs")
        assert resp.status_code == 200
        jobs = resp.json()
        assert len(jobs) == 2

    def test_get_job(self, client):
        client.post("/submit", json={"job_id": "j1", "query": "q"})
        resp = client.get("/jobs/j1")
        assert resp.status_code == 200
        assert resp.json()["job_id"] == "j1"

    def test_get_job_not_found(self, client):
        resp = client.get("/jobs/nope")
        assert resp.status_code == 404

    def test_full_lifecycle(self, client):
        """Submit → claim → complete → verify done."""
        # Submit
        resp = client.post("/submit", json={"job_id": "lifecycle", "query": "q"})
        assert resp.status_code == 200
        # Claim
        resp = client.post("/claim", json={"worker_id": "w1"})
        assert resp.json()["job_id"] == "lifecycle"
        # Complete
        resp = client.post("/complete", json={
            "job_id": "lifecycle",
            "result": {"answer": "42"},
        })
        assert resp.status_code == 200
        # Verify
        resp = client.get("/jobs/lifecycle")
        job = resp.json()
        assert job["status"] == "done"
        assert job["result"] == {"answer": "42"}
