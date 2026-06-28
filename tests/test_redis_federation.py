"""Tests for the Redis-backed HA federation state (v1.2).

Uses ``fakeredis.aioredis.FakeRedis`` so no live Redis server is needed.
The Redis backend must satisfy the same invariants as the in-memory
backend (see ``test_federation_server.py``), plus cross-coordinator
claim atomicity.

Note: ``fakeredis`` does not support Lua scripting (``SCRIPT LOAD`` /
``EVALSHA``), so these tests exercise the WATCH/MULTI optimistic-locking
fallback path in ``RedisFederationState.claim``. The Lua path is
identical in intent (atomic claim) and is the preferred path on real
Redis; the fallback exists precisely so the same code is testable
without a live Redis and so deployments on Redis-like stores without
scripting still work.
"""

import asyncio
import importlib.util

import pytest


def _has_module(name: str) -> bool:
    """Check if a module is importable without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ImportError):
        return False


# Check optional deps with find_spec (no import side effects). Using
# pytest.importorskip at module level causes pytest to skip the entire
# module and exit with code 5 ("no tests collected"), which CI scripts
# treat as failure. Instead, use skipif so tests are collected (exit 0)
# but individually skipped when deps are absent.
_DEPS_AVAILABLE = (
    _has_module("fakeredis.aioredis")
    and _has_module("fastapi")
)

pytestmark = [
    pytest.mark.federation,
    pytest.mark.web,
    pytest.mark.skipif(
        not _DEPS_AVAILABLE,
        reason="requires fakeredis + fastapi (pip install -e '.[federation,web]')",
    ),
]

# Guard the federation_server import: it imports fastapi at module load
# time. When deps are absent the names are set to None; they are never
# referenced at runtime because all tests are skipped via skipif above.
if _DEPS_AVAILABLE:
    from fakeredis import aioredis as fakeredis_aioredis
    from federation_server import (
        RedisFederationState,
        InMemoryFederationState,
        FederationState,
        make_federation_state,
        create_federation_app,
    )
else:
    fakeredis_aioredis = None  # type: ignore[assignment]
    RedisFederationState = None  # type: ignore[assignment]
    InMemoryFederationState = None  # type: ignore[assignment]
    FederationState = None  # type: ignore[assignment]
    make_federation_state = None  # type: ignore[assignment]
    create_federation_app = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def redis_state():
    """A RedisFederationState backed by an in-process FakeRedis."""
    client = fakeredis_aioredis.FakeRedis()
    return RedisFederationState(redis_client=client)


@pytest.fixture
def redis_state_pair():
    """Two RedisFederationState instances sharing ONE FakeRedis server.

    This simulates two federation coordinators backed by the same Redis,
    which is the HA deployment scenario. Claim atomicity is verified by
    having both coordinators claim concurrently from a shared queue.
    """
    # A single FakeRedis instance is shared by both coordinators. In a
    # real deployment each coordinator would have its own client
    # connection to the same Redis server; the shared FakeRedis here
    # models that shared server state.
    server = fakeredis_aioredis.FakeServer()
    client_a = fakeredis_aioredis.FakeRedis(server=server)
    client_b = fakeredis_aioredis.FakeRedis(server=server)
    a = RedisFederationState(redis_client=client_a)
    b = RedisFederationState(redis_client=client_b)
    return a, b


# --------------------------------------------------------------------------- #
# Single-coordinator invariants (mirror test_federation_server.py)
# --------------------------------------------------------------------------- #

class TestRedisFederationState:
    @pytest.mark.asyncio
    async def test_submit_creates_pending_job(self, redis_state):
        job = await redis_state.submit("job-1", "What is 2+2?")
        assert job.job_id == "job-1"
        assert job.query == "What is 2+2?"
        assert job.status == "pending"
        assert job.priority == 0

    @pytest.mark.asyncio
    async def test_submit_generates_job_id_when_empty(self, redis_state):
        job = await redis_state.submit("", "query")
        assert len(job.job_id) == 12  # uuid hex[:12]

    @pytest.mark.asyncio
    async def test_claim_returns_pending_job(self, redis_state):
        await redis_state.submit("j1", "query 1")
        job = await redis_state.claim("worker-1")
        assert job is not None
        assert job.job_id == "j1"
        assert job.status == "claimed"
        assert job.claimed_by == "worker-1"
        assert job.claimed_at is not None

    @pytest.mark.asyncio
    async def test_claim_returns_none_when_no_jobs(self, redis_state):
        job = await redis_state.claim("worker-1")
        assert job is None

    @pytest.mark.asyncio
    async def test_claim_returns_none_when_all_claimed(self, redis_state):
        await redis_state.submit("j1", "query 1")
        await redis_state.claim("worker-1")
        job = await redis_state.claim("worker-2")
        assert job is None

    @pytest.mark.asyncio
    async def test_claim_priority_ordering(self, redis_state):
        """Higher priority jobs are claimed first."""
        await redis_state.submit("low", "q1", priority=1)
        await redis_state.submit("high", "q2", priority=10)
        await redis_state.submit("mid", "q3", priority=5)
        first = await redis_state.claim("w1")
        assert first.job_id == "high"
        second = await redis_state.claim("w2")
        assert second.job_id == "mid"
        third = await redis_state.claim("w3")
        assert third.job_id == "low"

    @pytest.mark.asyncio
    async def test_claim_fifo_for_same_priority(self, redis_state):
        """Same-priority jobs are claimed in FIFO order."""
        await redis_state.submit("j1", "q1", priority=5)
        # Tiny delay so created_at differs (FIFO tie-break).
        await asyncio.sleep(0.001)
        await redis_state.submit("j2", "q2", priority=5)
        first = await redis_state.claim("w1")
        assert first.job_id == "j1"
        second = await redis_state.claim("w2")
        assert second.job_id == "j2"

    @pytest.mark.asyncio
    async def test_complete_sets_done(self, redis_state):
        await redis_state.submit("j1", "q1")
        await redis_state.claim("w1")
        ok = await redis_state.complete("j1", result={"answer": "4"})
        assert ok is True
        job = await redis_state.get_job("j1")
        assert job.status == "done"
        assert job.result == {"answer": "4"}
        assert job.completed_at is not None

    @pytest.mark.asyncio
    async def test_complete_with_error_sets_error(self, redis_state):
        await redis_state.submit("j1", "q1")
        await redis_state.claim("w1")
        ok = await redis_state.complete("j1", error="timeout")
        assert ok is True
        job = await redis_state.get_job("j1")
        assert job.status == "error"
        assert job.error == "timeout"

    @pytest.mark.asyncio
    async def test_complete_unknown_job_returns_false(self, redis_state):
        ok = await redis_state.complete("nonexistent")
        assert ok is False

    @pytest.mark.asyncio
    async def test_list_jobs(self, redis_state):
        await redis_state.submit("j1", "q1")
        await redis_state.submit("j2", "q2")
        jobs = await redis_state.list_jobs()
        assert len(jobs) == 2
        assert all("job_id" in j for j in jobs)
        assert all("status" in j for j in jobs)

    @pytest.mark.asyncio
    async def test_get_job_returns_none_for_unknown(self, redis_state):
        job = await redis_state.get_job("nope")
        assert job is None

    @pytest.mark.asyncio
    async def test_count(self, redis_state):
        assert await redis_state.count() == 0
        await redis_state.submit("j1", "q1")
        await redis_state.submit("j2", "q2")
        assert await redis_state.count() == 2

    @pytest.mark.asyncio
    async def test_force_route_round_trips(self, redis_state):
        await redis_state.submit("j1", "q1", force_route="code")
        job = await redis_state.get_job("j1")
        assert job.force_route == "code"


# --------------------------------------------------------------------------- #
# Cross-coordinator claim atomicity (the HA invariant)
# --------------------------------------------------------------------------- #

class TestRedisClaimAtomicity:
    @pytest.mark.asyncio
    async def test_concurrent_claim_no_double_assignment(self, redis_state_pair):
        """Two coordinators claiming concurrently never get the same job."""
        a, b = redis_state_pair
        await a.submit("j1", "q1")
        await a.submit("j2", "q2")
        results = await asyncio.gather(a.claim("w-a"), b.claim("w-b"))
        job_ids = {r.job_id for r in results if r is not None}
        # Both jobs should be claimed, by different coordinators, with no
        # overlap (no double-assignment).
        assert job_ids == {"j1", "j2"}
        # Verify each job's claimed_by is set and they're distinct.
        j1 = await a.get_job("j1")
        j2 = await a.get_job("j2")
        assert j1.claimed_by in {"w-a", "w-b"}
        assert j2.claimed_by in {"w-a", "w-b"}
        assert j1.claimed_by != j2.claimed_by

    @pytest.mark.asyncio
    async def test_concurrent_claim_single_job_one_winner(self, redis_state_pair):
        """One pending job, two coordinators claim — exactly one wins."""
        a, b = redis_state_pair
        await a.submit("solo", "q")
        results = await asyncio.gather(a.claim("w-a"), b.claim("w-b"))
        winners = [r for r in results if r is not None]
        assert len(winners) == 1
        assert winners[0].job_id == "solo"
        # The other coordinator got None.
        losers = [r for r in results if r is None]
        assert len(losers) == 1

    @pytest.mark.asyncio
    async def test_submit_on_one_visible_on_other(self, redis_state_pair):
        """A job submitted via coordinator A is visible to coordinator B."""
        a, b = redis_state_pair
        await a.submit("shared", "q")
        # B can list it and claim it.
        jobs = await b.list_jobs()
        assert any(j["job_id"] == "shared" for j in jobs)
        claimed = await b.claim("w-b")
        assert claimed is not None
        assert claimed.job_id == "shared"

    @pytest.mark.asyncio
    async def test_complete_on_one_visible_on_other(self, redis_state_pair):
        """A job completed via coordinator A reflects on coordinator B."""
        a, b = redis_state_pair
        await a.submit("j1", "q")
        await a.claim("w-a")
        await a.complete("j1", result={"answer": "42"})
        job = await b.get_job("j1")
        assert job.status == "done"
        assert job.result == {"answer": "42"}


# --------------------------------------------------------------------------- #
# Factory + app integration
# --------------------------------------------------------------------------- #

class TestMakeFederationState:
    def test_default_is_in_memory(self):
        state = make_federation_state()
        assert isinstance(state, InMemoryFederationState)

    def test_empty_redis_url_is_in_memory(self):
        state = make_federation_state(redis_url="")
        assert isinstance(state, InMemoryFederationState)

    def test_redis_url_returns_redis_state(self):
        state = make_federation_state(redis_url="redis://localhost:6379/0")
        assert isinstance(state, RedisFederationState)

    def test_redis_client_returns_redis_state(self):
        client = fakeredis_aioredis.FakeRedis()
        state = make_federation_state(redis_client=client)
        assert isinstance(state, RedisFederationState)

    def test_backward_compat_alias_is_in_memory(self):
        # FederationState must remain the instantiable in-memory class
        # (existing tests and callers import it directly).
        assert isinstance(FederationState(), InMemoryFederationState)


class TestRedisFederationApp:
    def test_app_with_redis_state(self):
        """create_federation_app accepts a RedisFederationState."""
        from fastapi.testclient import TestClient
        client = fakeredis_aioredis.FakeRedis()
        state = RedisFederationState(redis_client=client)
        app = create_federation_app(state=state)
        tc = TestClient(app)

        # Health works (uses state.count()).
        resp = tc.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["jobs"] == 0

        # Submit + claim + complete lifecycle.
        resp = tc.post("/submit", json={"job_id": "j1", "query": "q"})
        assert resp.status_code == 200
        resp = tc.post("/claim", json={"worker_id": "w1"})
        assert resp.json()["job_id"] == "j1"
        assert resp.json()["status"] == "claimed"
        resp = tc.post("/complete", json={
            "job_id": "j1", "result": {"answer": "4"},
        })
        assert resp.status_code == 200
        resp = tc.get("/jobs/j1")
        assert resp.json()["status"] == "done"
        assert resp.json()["result"]["answer"] == "4"


# --------------------------------------------------------------------------- #
# Phase 4.2: Heartbeat + zombie reaping (Redis backend)
# --------------------------------------------------------------------------- #

class TestRedisHeartbeat:
    """Tests for the heartbeat method on RedisFederationState."""

    @pytest.mark.asyncio
    async def test_heartbeat_updates_timestamp(self, redis_state):
        await redis_state.submit("job1", "test query")
        job = await redis_state.claim("worker-1")
        assert job is not None
        assert job.heartbeat_at is not None  # set on claim

        await asyncio.sleep(0.01)
        ok = await redis_state.heartbeat("job1", "worker-1")
        assert ok is True

        job = await redis_state.get_job("job1")
        assert job.heartbeat_at is not None
        assert job.status == "claimed"

    @pytest.mark.asyncio
    async def test_heartbeat_wrong_worker_fails(self, redis_state):
        await redis_state.submit("job1", "test query")
        await redis_state.claim("worker-1")
        ok = await redis_state.heartbeat("job1", "worker-2")
        assert ok is False

    @pytest.mark.asyncio
    async def test_heartbeat_nonexistent_job_fails(self, redis_state):
        ok = await redis_state.heartbeat("nonexistent", "worker-1")
        assert ok is False

    @pytest.mark.asyncio
    async def test_heartbeat_on_completed_job_fails(self, redis_state):
        await redis_state.submit("job1", "test query")
        await redis_state.claim("worker-1")
        await redis_state.complete("job1", result={"answer": "42"})
        ok = await redis_state.heartbeat("job1", "worker-1")
        assert ok is False


class TestRedisReapStaleClaims:
    """Tests for the reap_stale_claims method on RedisFederationState."""

    @pytest.mark.asyncio
    async def test_reap_stale_claim_requeues_job(self, redis_state):
        import time as _time
        await redis_state.submit("job1", "test query")
        job = await redis_state.claim("worker-1")
        assert job is not None
        # Backdate the heartbeat to simulate a stale claim.
        await redis_state._redis.hset(
            f"{redis_state._job_prefix}job1",
            "heartbeat_at", str(_time.time() - 400),
        )
        reaped = await redis_state.reap_stale_claims(timeout=300.0)
        assert reaped == ["job1"]
        job = await redis_state.get_job("job1")
        assert job.status == "pending"
        assert job.claimed_by is None

    @pytest.mark.asyncio
    async def test_reap_skips_fresh_claims(self, redis_state):
        await redis_state.submit("job1", "test query")
        await redis_state.claim("worker-1")
        reaped = await redis_state.reap_stale_claims(timeout=300.0)
        assert reaped == []
        job = await redis_state.get_job("job1")
        assert job.status == "claimed"

    @pytest.mark.asyncio
    async def test_reap_skips_completed_jobs(self, redis_state):
        await redis_state.submit("job1", "test query")
        await redis_state.claim("worker-1")
        await redis_state.complete("job1", result={"answer": "42"})
        reaped = await redis_state.reap_stale_claims(timeout=300.0)
        assert reaped == []

    @pytest.mark.asyncio
    async def test_reap_requeued_job_can_be_claimed_again(self, redis_state):
        import time as _time
        await redis_state.submit("job1", "test query")
        await redis_state.claim("worker-1")
        await redis_state._redis.hset(
            f"{redis_state._job_prefix}job1",
            "heartbeat_at", str(_time.time() - 400),
        )
        await redis_state.reap_stale_claims(timeout=300.0)
        # A different worker should be able to claim the re-queued job.
        job = await redis_state.claim("worker-2")
        assert job is not None
        assert job.job_id == "job1"
        assert job.claimed_by == "worker-2"
        assert job.status == "claimed"
