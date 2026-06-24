"""Pytest tests for the job queue (no model servers needed)."""

import asyncio
import os
import tempfile

import pytest

from rfsn_job_queue import JobQueue, JobStatus


class MockResult:
    def __init__(self, answer="42", route="specialist_clr", clr_score=1.0):
        self.final_answer = answer
        self.route_taken = route
        self.clr_score = clr_score
        self.raw_traces = {}
        self.timestamp = "t"
        self.routing_confidence = 0.9


class MockOrchestrator:
    def __init__(self, delay=0.01, fail_on=None):
        self.delay = delay
        self.fail_on = fail_on or set()
        self.calls = []

    async def run(self, query, force_route=None):
        self.calls.append(query)
        await asyncio.sleep(self.delay)
        if any(q in query for q in self.fail_on):
            raise RuntimeError(f"failed on: {query}")
        return MockResult()


@pytest.fixture
def log_path():
    path = tempfile.mktemp(suffix=".jsonl")
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def queue(log_path):
    orch = MockOrchestrator()
    return JobQueue(orch, max_concurrent=2, audit_log=log_path)


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_submit_and_complete(self, queue):
        await queue.start()
        j = queue.submit("test query", priority=5)
        assert j.status == JobStatus.PENDING
        r = await queue.wait_for(j.job_id, timeout=5)
        assert r.final_answer == "42"
        assert queue.status(j.job_id) == JobStatus.COMPLETED
        await queue.stop()

    @pytest.mark.asyncio
    async def test_failed_job(self, log_path):
        orch = MockOrchestrator(fail_on={"BOOM"})
        q = JobQueue(orch, max_concurrent=2, audit_log=log_path)
        await q.start()
        j = q.submit("this will BOOM", priority=1)
        with pytest.raises(RuntimeError, match="BOOM"):
            await q.wait_for(j.job_id, timeout=5)
        assert q.status(j.job_id) == JobStatus.FAILED
        assert "BOOM" in j.error
        await q.stop()

    @pytest.mark.asyncio
    async def test_cancel_pending(self, log_path):
        orch = MockOrchestrator(delay=0.5)
        q = JobQueue(orch, max_concurrent=1, audit_log=log_path)
        await q.start()
        j1 = q.submit("running", priority=10)
        j2 = q.submit("pending", priority=1)
        await asyncio.sleep(0.05)
        assert q.cancel(j2.job_id) is True
        assert q.status(j2.job_id) == JobStatus.CANCELLED
        # Can't cancel a running job
        assert q.cancel(j1.job_id) is False
        # Can't cancel unknown
        assert q.cancel("nonexistent") is False
        await q.wait_for(j1.job_id, timeout=5)
        await q.stop()

    @pytest.mark.asyncio
    async def test_wait_timeout(self, log_path):
        orch = MockOrchestrator(delay=1.0)
        q = JobQueue(orch, max_concurrent=1, audit_log=log_path)
        await q.start()
        j = q.submit("slow", priority=1)
        with pytest.raises(TimeoutError):
            await q.wait_for(j.job_id, timeout=0.1)
        await q.stop()

    @pytest.mark.asyncio
    async def test_wait_unknown_job(self, queue):
        await queue.start()
        with pytest.raises(KeyError):
            await queue.wait_for("nonexistent", timeout=0.1)
        await queue.stop()


class TestPriority:
    @pytest.mark.asyncio
    async def test_priority_order(self, log_path):
        orch = MockOrchestrator(delay=0.1)
        q = JobQueue(orch, max_concurrent=1, audit_log=log_path)
        await q.start()
        order = []
        orig = orch.run
        async def tracking(query, force_route=None):
            order.append(query)
            return await orig(query, force_route)
        orch.run = tracking
        q.submit("low", priority=1)
        q.submit("high", priority=10)
        q.submit("mid", priority=5)
        await asyncio.sleep(0.5)
        await q.stop()
        assert order == ["high", "mid", "low"]


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_never_exceeds_max(self, log_path):
        orch = MockOrchestrator(delay=0.15)
        q = JobQueue(orch, max_concurrent=2, audit_log=log_path)
        await q.start()
        concurrent = 0
        max_seen = 0
        orig = orch.run
        async def counting(query, force_route=None):
            nonlocal concurrent, max_seen
            concurrent += 1
            max_seen = max(max_seen, concurrent)
            try:
                return await orig(query, force_route)
            finally:
                concurrent -= 1
        orch.run = counting
        jobs = [q.submit(f"q{i}", priority=1) for i in range(5)]
        await asyncio.gather(*[q.wait_for(j.job_id, timeout=5) for j in jobs])
        await q.stop()
        assert max_seen <= 2
        assert max_seen == 2


class TestShutdown:
    @pytest.mark.asyncio
    async def test_inflight_job_completes(self, log_path):
        orch = MockOrchestrator(delay=0.2)
        q = JobQueue(orch, max_concurrent=2, audit_log=log_path)
        await q.start()
        j = q.submit("in-flight at shutdown", priority=1)
        await q.stop()
        assert q.status(j.job_id) == JobStatus.COMPLETED
        assert j.result is not None


class TestBitemporalIntegration:
    @pytest.mark.asyncio
    async def test_log_records_lifecycle(self, queue):
        await queue.start()
        j = queue.submit("test", priority=5)
        await queue.wait_for(j.job_id, timeout=5)
        await queue.stop()
        hist = queue.job_history(j.job_id)
        events = [e["event"] for e in hist]
        assert events == ["submitted", "started", "completed"]
        completed = [e for e in hist if e["event"] == "completed"][0]
        assert "route" in completed["extra"]
        assert "clr_score" in completed["extra"]

    @pytest.mark.asyncio
    async def test_log_records_failure(self, log_path):
        orch = MockOrchestrator(fail_on={"BOOM"})
        q = JobQueue(orch, max_concurrent=2, audit_log=log_path)
        await q.start()
        j = q.submit("BOOM", priority=1)
        try:
            await q.wait_for(j.job_id, timeout=5)
        except RuntimeError:
            pass
        await q.stop()
        hist = q.job_history(j.job_id)
        assert any(e["event"] == "failed" for e in hist)
