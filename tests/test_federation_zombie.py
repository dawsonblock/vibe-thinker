"""Tests for federation zombie claim detection (Phase 4.2).

Tests the heartbeat mechanism and stale-claim reaping in
InMemoryFederationState and the federation server endpoints.
"""

import asyncio
import importlib.util
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _has_module(name: str) -> bool:
    """Check if a module is importable without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ImportError):
        return False


_FASTAPI_AVAILABLE = _has_module("fastapi")

pytestmark = [
    pytest.mark.web,
    pytest.mark.federation,
    pytest.mark.skipif(
        not _FASTAPI_AVAILABLE,
        reason="requires fastapi web extra (pip install -e '.[web]')",
    ),
]

# Guard the fastapi/federation_server imports: federation_server imports
# fastapi at module load time. When deps are absent the names are set to
# None; they are never referenced at runtime because all tests are
# skipped via skipif above.
if _FASTAPI_AVAILABLE:
    from fastapi.testclient import TestClient  # noqa: E402
    from federation_server import (
        InMemoryFederationState,
        FederatedJob,
        create_federation_app,
    )
else:
    TestClient = None  # type: ignore[assignment]
    InMemoryFederationState = None  # type: ignore[assignment]
    FederatedJob = None  # type: ignore[assignment]
    create_federation_app = None  # type: ignore[assignment]


class TestHeartbeat:
    """Tests for the heartbeat method on InMemoryFederationState."""

    @pytest.mark.asyncio
    async def test_heartbeat_updates_timestamp(self):
        state = InMemoryFederationState()
        await state.submit("job1", "test query")
        job = await state.claim("worker-1")
        assert job is not None
        assert job.heartbeat_at is not None  # set on claim

        # Wait a tiny bit, then heartbeat.
        await asyncio.sleep(0.01)
        old_hb = job.heartbeat_at
        ok = await state.heartbeat("job1", "worker-1")
        assert ok is True

        job = await state.get_job("job1")
        assert job.heartbeat_at > old_hb

    @pytest.mark.asyncio
    async def test_heartbeat_wrong_worker_fails(self):
        state = InMemoryFederationState()
        await state.submit("job1", "test query")
        await state.claim("worker-1")
        # A different worker can't heartbeat this job.
        ok = await state.heartbeat("job1", "worker-2")
        assert ok is False

    @pytest.mark.asyncio
    async def test_heartbeat_nonexistent_job_fails(self):
        state = InMemoryFederationState()
        ok = await state.heartbeat("nonexistent", "worker-1")
        assert ok is False

    @pytest.mark.asyncio
    async def test_heartbeat_on_completed_job_fails(self):
        state = InMemoryFederationState()
        await state.submit("job1", "test query")
        await state.claim("worker-1")
        await state.complete("job1", result={"answer": "42"})
        # Can't heartbeat a completed job.
        ok = await state.heartbeat("job1", "worker-1")
        assert ok is False

    @pytest.mark.asyncio
    async def test_heartbeat_on_pending_job_fails(self):
        state = InMemoryFederationState()
        await state.submit("job1", "test query")
        # Job is pending, not claimed — heartbeat should fail.
        ok = await state.heartbeat("job1", "worker-1")
        assert ok is False


class TestReapStaleClaims:
    """Tests for the reap_stale_claims method."""

    @pytest.mark.asyncio
    async def test_reap_stale_claim_requeues_job(self):
        state = InMemoryFederationState()
        await state.submit("job1", "test query")
        job = await state.claim("worker-1")
        assert job is not None
        # Manually backdate the heartbeat to simulate a stale claim.
        job.heartbeat_at = time.time() - 400  # 400s ago (timeout=300)
        reaped = await state.reap_stale_claims(timeout=300.0)
        assert reaped == ["job1"]
        # Job should be back to pending.
        job = await state.get_job("job1")
        assert job.status == "pending"
        assert job.claimed_by is None
        assert job.claimed_at is None
        assert job.heartbeat_at is None

    @pytest.mark.asyncio
    async def test_reap_skips_fresh_claims(self):
        state = InMemoryFederationState()
        await state.submit("job1", "test query")
        await state.claim("worker-1")
        # Heartbeat is fresh — should not be reaped.
        reaped = await state.reap_stale_claims(timeout=300.0)
        assert reaped == []
        job = await state.get_job("job1")
        assert job.status == "claimed"

    @pytest.mark.asyncio
    async def test_reap_uses_claimed_at_when_no_heartbeat(self):
        state = InMemoryFederationState()
        await state.submit("job1", "test query")
        job = await state.claim("worker-1")
        # Clear heartbeat_at — should fall back to claimed_at.
        job.heartbeat_at = None
        job.claimed_at = time.time() - 400  # 400s ago
        reaped = await state.reap_stale_claims(timeout=300.0)
        assert reaped == ["job1"]

    @pytest.mark.asyncio
    async def test_reap_skips_completed_jobs(self):
        state = InMemoryFederationState()
        await state.submit("job1", "test query")
        await state.claim("worker-1")
        await state.complete("job1", result={"answer": "42"})
        reaped = await state.reap_stale_claims(timeout=300.0)
        assert reaped == []

    @pytest.mark.asyncio
    async def test_reap_skips_pending_jobs(self):
        state = InMemoryFederationState()
        await state.submit("job1", "test query")
        # Job is pending — not a stale claim.
        reaped = await state.reap_stale_claims(timeout=300.0)
        assert reaped == []

    @pytest.mark.asyncio
    async def test_reap_multiple_stale_claims(self):
        state = InMemoryFederationState()
        await state.submit("job1", "query1")
        await state.submit("job2", "query2")
        job1 = await state.claim("worker-1")
        job2 = await state.claim("worker-2")
        # Backdate both.
        job1.heartbeat_at = time.time() - 400
        job2.heartbeat_at = time.time() - 400
        reaped = await state.reap_stale_claims(timeout=300.0)
        assert len(reaped) == 2
        assert set(reaped) == {"job1", "job2"}

    @pytest.mark.asyncio
    async def test_reap_requeued_job_can_be_claimed_again(self):
        state = InMemoryFederationState()
        await state.submit("job1", "test query")
        await state.claim("worker-1")
        job = await state.get_job("job1")
        job.heartbeat_at = time.time() - 400
        await state.reap_stale_claims(timeout=300.0)
        # A different worker should be able to claim the re-queued job.
        job = await state.claim("worker-2")
        assert job is not None
        assert job.job_id == "job1"
        assert job.claimed_by == "worker-2"
        assert job.status == "claimed"

    @pytest.mark.asyncio
    async def test_reap_with_custom_timeout(self):
        state = InMemoryFederationState()
        await state.submit("job1", "test query")
        job = await state.claim("worker-1")
        # Only 10s stale, but timeout is 5s.
        job.heartbeat_at = time.time() - 10
        reaped = await state.reap_stale_claims(timeout=5.0)
        assert reaped == ["job1"]


class TestHeartbeatEndpoint:
    """Tests for the /heartbeat HTTP endpoint."""

    def test_heartbeat_endpoint_updates(self):
        state = InMemoryFederationState()
        app = create_federation_app(state=state)
        client = TestClient(app)
        # Submit and claim a job via the API.
        client.post("/submit", json={"job_id": "job1", "query": "test"})
        client.post("/claim", json={"worker_id": "worker-1"})
        # Send a heartbeat.
        resp = client.post("/heartbeat", json={
            "job_id": "job1", "worker_id": "worker-1",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_heartbeat_endpoint_wrong_worker_404(self):
        state = InMemoryFederationState()
        app = create_federation_app(state=state)
        client = TestClient(app)
        client.post("/submit", json={"job_id": "job1", "query": "test"})
        client.post("/claim", json={"worker_id": "worker-1"})
        resp = client.post("/heartbeat", json={
            "job_id": "job1", "worker_id": "worker-2",
        })
        assert resp.status_code == 404

    def test_heartbeat_endpoint_nonexistent_job_404(self):
        state = InMemoryFederationState()
        app = create_federation_app(state=state)
        client = TestClient(app)
        resp = client.post("/heartbeat", json={
            "job_id": "nonexistent", "worker_id": "worker-1",
        })
        assert resp.status_code == 404


class TestWebHeartbeatEndpoint:
    """Tests for the web/app.py /api/jobs/heartbeat endpoint."""

    def test_web_heartbeat_updates(self):
        from web.app import create_app, AppState
        mock_orch = MagicMock()
        mock_orch.start = AsyncMock()
        mock_orch.cleanup = AsyncMock()

        async def slow_route(query, **kwargs):
            await asyncio.sleep(10.0)

        mock_orch.route = slow_route
        app = create_app(mock_orch)
        client = TestClient(app)
        # Submit via the query API (same as existing web federation tests).
        resp = client.post("/api/query", json={"query": "test query"})
        job_id = resp.json()["job_id"]
        # Claim it.
        resp = client.post("/api/jobs/claim", json={"worker_id": "w1"})
        # The claim may return null if the background task already picked
        # up the job. In that case, skip (the job is already running).
        if resp.json().get("job_id") is None:
            pytest.skip("Background task picked up job before claim")
        assert resp.json()["job_id"] == job_id
        # Heartbeat.
        resp = client.post("/api/jobs/heartbeat", json={
            "job_id": job_id, "worker_id": "w1",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_web_heartbeat_wrong_worker_404(self):
        from web.app import create_app, AppState
        mock_orch = MagicMock()
        mock_orch.start = AsyncMock()
        mock_orch.cleanup = AsyncMock()

        async def slow_route(query, **kwargs):
            await asyncio.sleep(10.0)

        mock_orch.route = slow_route
        app = create_app(mock_orch)
        client = TestClient(app)
        resp = client.post("/api/query", json={"query": "test query"})
        job_id = resp.json()["job_id"]
        resp = client.post("/api/jobs/claim", json={"worker_id": "w1"})
        if resp.json().get("job_id") is None:
            pytest.skip("Background task picked up job before claim")
        resp = client.post("/api/jobs/heartbeat", json={
            "job_id": job_id, "worker_id": "w2",
        })
        assert resp.status_code == 404
