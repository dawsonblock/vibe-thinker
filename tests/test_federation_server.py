"""Tests for the Python-native federation server (v0.4.1).

Tests the FederationState (in-memory job store with asyncio.Lock) and
the FastAPI endpoints (submit, claim, complete, jobs, health).
"""

import asyncio
import importlib.util
import json

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

# Guard the federation_server import: it imports fastapi at module load
# time. When deps are absent the names are set to None; they are never
# referenced at runtime because all tests are skipped via skipif above.
if _FASTAPI_AVAILABLE:
    from federation_server import FederationState, create_federation_app
else:
    FederationState = None  # type: ignore[assignment]
    create_federation_app = None  # type: ignore[assignment]


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
@pytest.fixture
def client():
    """Module-level TestClient fixture shared by all endpoint test classes."""
    from fastapi.testclient import TestClient
    app = create_federation_app()
    return TestClient(app)


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


# ---------------------------------------------------------------------- #
# v3.2: Sybil-resistant node reputation
# ---------------------------------------------------------------------- #
class TestReputationStore:
    """Unit tests for the ReputationStore EMA + downweight logic."""

    @pytest.mark.asyncio
    async def test_verified_true_raises_score(self):
        from federation_server import ReputationStore
        r = ReputationStore()
        s0 = await r.get_score("node-A")
        assert s0 == 0.7  # default
        s1 = await r.record_outcome("node-A", +1.0)
        assert s1 > s0
        # Clamped to [0,1].
        for _ in range(20):
            s = await r.record_outcome("node-A", +1.0)
        assert s == 1.0

    @pytest.mark.asyncio
    async def test_verified_false_drops_below_floor(self):
        """A node that repeatedly fails verification drops below the
        gossip floor and its downweight factor becomes 0."""
        from federation_server import ReputationStore
        r = ReputationStore()
        # ~3 consecutive -2 outcomes should drop a 0.7 default below 0.3.
        for _ in range(4):
            await r.record_outcome("bad-node", -2.0)
        dw = await r.downweight_factor("bad-node")
        assert dw == 0.0  # below floor -> discarded

    @pytest.mark.asyncio
    async def test_single_failure_does_not_zero_reputation(self):
        """One fluke failure should NOT drop a good node below the floor
        — the EMA is resistant to a single bad outcome."""
        from federation_server import ReputationStore
        r = ReputationStore()
        # Build up trust.
        for _ in range(5):
            await r.record_outcome("good-node", +1.0)
        # One failure.
        await r.record_outcome("good-node", -2.0)
        dw = await r.downweight_factor("good-node")
        assert dw > 0.0  # still trusted

    @pytest.mark.asyncio
    async def test_separate_identities_separate_reputations(self):
        """Sybil resistance: two different cert identities have
        independent reputations. A bad identity cannot poison a good one."""
        from federation_server import ReputationStore
        r = ReputationStore()
        await r.record_outcome("cn:good", +1.0)
        await r.record_outcome("cn:bad", -2.0)
        assert await r.get_score("cn:good") > await r.get_score("cn:bad")


class TestReputationEndpoints:
    """Integration tests for the /complete -> reputation -> SONA flow."""

    def test_complete_verified_true_updates_reputation(self, client):
        client.post("/submit", json={"job_id": "r1", "query": "q"})
        client.post("/claim", json={"worker_id": "w1"})
        client.post("/complete", json={
            "job_id": "r1",
            "worker_id": "w1",
            "result": {"answer": "4", "verified": True},
        })
        resp = client.get("/api/reputation")
        assert resp.status_code == 200
        ids = resp.json()["identities"]
        # In dev mode (no mTLS), identity is wid:w1.
        assert any(k.startswith("wid:w1") for k in ids)
        score = next(v["score"] for k, v in ids.items() if k.startswith("wid:w1"))
        assert score > 0.7  # verified=True nudged it up

    def test_complete_verified_false_drops_reputation(self, client):
        client.post("/submit", json={"job_id": "r2", "query": "q"})
        client.post("/claim", json={"worker_id": "w-bad"})
        for _ in range(4):  # repeat to push below floor
            client.post("/submit", json={"job_id": f"r2-{_}", "query": "q"})
            client.post("/claim", json={"worker_id": "w-bad"})
            client.post("/complete", json={
                "job_id": f"r2-{_}",
                "worker_id": "w-bad",
                "result": {"answer": "wrong", "verified": False},
            })
        # Also complete the original r2.
        client.post("/complete", json={
            "job_id": "r2",
            "worker_id": "w-bad",
            "result": {"answer": "wrong", "verified": False},
        })
        resp = client.get("/api/reputation")
        ids = resp.json()["identities"]
        score = next(v["score"] for k, v in ids.items() if k.startswith("wid:w-bad"))
        assert score < 0.3  # below gossip floor

    def test_sona_sync_downweights_low_reputation_node(self, client):
        """A node below the gossip floor has its SONA patterns discarded."""
        # Drive w-bad below the floor.
        for i in range(5):
            client.post("/submit", json={"job_id": f"s{i}", "query": "q"})
            client.post("/claim", json={"worker_id": "w-bad"})
            client.post("/complete", json={
                "job_id": f"s{i}",
                "worker_id": "w-bad",
                "result": {"answer": "wrong", "verified": False},
            })
        # w-bad tries to export SONA patterns.
        resp = client.post("/api/sona/sync", json={
            "worker_id": "w-bad",
            "patterns": [{"id": "p1", "centroid": [0.1, 0.2]}],
            "stats": {"total": 1},
        })
        assert resp.status_code == 200
        body = resp.json()
        # All patterns skipped because reputation is below floor.
        assert body["skipped"] == 1
        assert body["imported"] == 0
        assert body["reputation_downweight"] == 0.0

    def test_sona_sync_imports_high_reputation_node(self, client):
        """A good node (verified=True completions) exports patterns normally."""
        for i in range(3):
            client.post("/submit", json={"job_id": f"g{i}", "query": "q"})
            client.post("/claim", json={"worker_id": "w-good"})
            client.post("/complete", json={
                "job_id": f"g{i}",
                "worker_id": "w-good",
                "result": {"answer": "right", "verified": True},
            })
        resp = client.post("/api/sona/sync", json={
            "worker_id": "w-good",
            "patterns": [{"id": "p-good", "centroid": [0.3, 0.4]}],
            "stats": {"total": 1},
        })
        body = resp.json()
        assert body["imported"] == 1
        assert body["skipped"] == 0
        assert body["reputation_downweight"] > 0.0


class TestExtractIdentity:
    """Tests for the mTLS cert identity extraction (Sybil resistance)."""

    def test_falls_back_to_worker_id_when_no_cert(self):
        from federation_server import _extract_identity
        from starlette.requests import Request
        # A request with no ssl scope -> wid: fallback.
        req = Request(scope={"type": "http", "ssl": None})
        ident = _extract_identity(req, "w1")
        assert ident == "wid:w1"

    def test_uses_cert_common_name_when_present(self):
        from federation_server import _extract_identity
        from starlette.requests import Request
        # uvicorn-style ssl scope with a subject CN.
        scope = {
            "type": "http",
            "ssl": {
                "subject": ((("commonName", "node-alpha"),),),
            },
        }
        req = Request(scope=scope)
        ident = _extract_identity(req, "w1")
        assert ident == "cn:node-alpha"

    def test_uses_cert_fingerprint_when_no_cn(self):
        from federation_server import _extract_identity
        from starlette.requests import Request
        scope = {
            "type": "http",
            "ssl": {"subject": None, "cert": b"der-bytes-here"},
        }
        req = Request(scope=scope)
        ident = _extract_identity(req, "w1")
        assert ident.startswith("fp:")
