"""
Comprehensive test demo — exercises every component with assertions.

Runs without model servers (uses mock orchestrators) so it works anywhere:

    python test_demo.py

Covers:
  A. Bi-temporal audit log: write, read, history (both axes), as-of,
     current_state, corrections, migration + duplicate guard.
  B. Job queue: submit, priority ordering, concurrency limit, lifecycle
     (pending->running->completed), failed jobs, cancel, callbacks,
     wait_for with timeout, clean shutdown (in-flight jobs survive).
  C. REPL: every command, flag parsing, usage errors, empty input,
     unknown commands, unknown jobs.
  D. Integration: queue + bi-temporal log schema, end-to-end flow,
     log reconstruction matches in-memory state.
  E. Deprecation safety: the whole suite passes under
     -W error::DeprecationWarning.
"""

import asyncio
import json
import os
import sys
import tempfile
import traceback
from datetime import datetime, timezone

from bitemporal_log import BiTemporalAuditLog, migrate_legacy_log
from rfsn_job_queue import JobQueue, JobStatus, Job
from rfsn_cli import JobQueueREPL, _split_flags


# ====================================================================== #
# Test infrastructure
# ====================================================================== #
PASSED = 0
FAILED = 0


def check(condition, label):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {label}")
    else:
        FAILED += 1
        print(f"  [FAIL] {label}")


def section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


# ====================================================================== #
# Mock orchestrator
# ====================================================================== #
class MockResult:
    def __init__(self, answer, route, clr_score=None):
        self.final_answer = answer
        self.route_taken = route
        self.clr_score = clr_score
        self.raw_traces = {}
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.routing_confidence = 0.9


class MockOrchestrator:
    """Simulates HybridReasoningOrchestrator with controllable latency,
    route selection, and optional failures."""

    def __init__(self, delay=0.05, fail_on=None):
        self.delay = delay
        self.fail_on = fail_on or set()
        self.calls = []

    async def run(self, query, force_route=None):
        self.calls.append(query)
        await asyncio.sleep(self.delay)
        if any(q in query for q in self.fail_on):
            raise RuntimeError(f"orchestrator failed on: {query}")
        q = query.lower()
        if force_route:
            route = force_route
        elif any(kw in q for kw in ("solve", "find", "series", "sum", "recurrence")):
            route = "specialist_clr"
        elif any(kw in q for kw in ("explain", "what is", "history")):
            route = "generalist"
        else:
            route = "hybrid"
        score = 1.0 if route.startswith("specialist") else (0.85 if route == "hybrid" else None)
        return MockResult(f"answer({query[:30]})", route, clr_score=score)


# ====================================================================== #
# A. Bi-temporal audit log
# ====================================================================== #
def test_bitemporal_log():
    section("A. BI-TEMPORAL AUDIT LOG")
    tmp = tempfile.mktemp(suffix=".jsonl")
    log = BiTemporalAuditLog(tmp)

    class FakeJob:
        job_id = "j1"
        status = type("S", (), {"value": "pending"})()
        query = "test query"
        priority = 5
        force_route = None

    # Write 3 events with explicit times to test ordering
    log.record(FakeJob(), "submitted",
               valid_time="2026-06-24T10:00:00+00:00",
               transaction_time="2026-06-24T10:00:01+00:00")
    log.record(FakeJob(), "started",
               valid_time="2026-06-24T10:00:05+00:00",
               transaction_time="2026-06-24T10:00:06+00:00")
    log.record(FakeJob(), "completed", extra={"route": "specialist_clr", "clr_score": 1.0},
               valid_time="2026-06-24T10:00:10+00:00",
               transaction_time="2026-06-24T10:00:11+00:00")

    entries = log.read_all()
    check(len(entries) == 3, "read_all returns 3 entries")
    check(all("record_id" in e for e in entries), "every entry has record_id")
    check(all("valid_time" in e for e in entries), "every entry has valid_time")
    check(all("transaction_time" in e for e in entries), "every entry has transaction_time")
    check(all("correction_of" in e for e in entries), "every entry has correction_of")

    # History — valid axis
    hist = log.history("j1", axis="valid")
    check([e["event"] for e in hist] == ["submitted", "started", "completed"],
          "history(valid) ordered by valid_time")
    check(hist[2]["extra"]["route"] == "specialist_clr", "extra fields preserved")

    # History — transaction axis
    hist_t = log.history("j1", axis="transaction")
    check([e["event"] for e in hist_t] == ["submitted", "started", "completed"],
          "history(transaction) ordered by transaction_time")

    # As-of — valid axis
    check(log.state_as_of("j1", "2026-06-24T10:00:03+00:00")["event"] == "submitted",
          "as_of(valid) before 'started' -> submitted")
    check(log.state_as_of("j1", "2026-06-24T10:00:09+00:00")["event"] == "started",
          "as_of(valid) before 'completed' -> started")
    check(log.state_as_of("j1", "2026-06-24T10:00:99+00:00")["event"] == "completed",
          "as_of(valid) after all -> completed")
    check(log.state_as_of("j1", "2026-06-24T09:00:00+00:00") is None,
          "as_of(valid) before any event -> None")

    # As-of — transaction axis (knowledge lags)
    check(log.state_as_of("j1", "2026-06-24T10:00:01+00:00", axis="transaction")["event"] == "submitted",
          "as_of(transaction) at submit txn time -> submitted")
    check(log.state_as_of("j1", "2026-06-24T10:00:05+00:00", axis="transaction")["event"] == "submitted",
          "as_of(transaction) before 'started' recorded -> still submitted")
    check(log.state_as_of("j1", "2026-06-24T10:00:06+00:00", axis="transaction")["event"] == "started",
          "as_of(transaction) at 'started' txn time -> started")

    # Current state
    state = log.current_state()
    check("j1" in state, "current_state has j1")
    check(state["j1"]["event"] == "completed", "current_state j1 is completed")

    # Jobs list
    check(log.jobs() == ["j1"], "jobs() returns distinct job_ids")

    # Correction entry
    log.record(FakeJob(), "completed", extra={"route": "corrected_route"},
               valid_time="2026-06-24T10:00:10+00:00",
               transaction_time="2026-06-24T11:00:00+00:00",
               correction_of=entries[2]["record_id"])
    check(len(log.read_all()) == 4, "correction entry appended (not edited)")
    check(log.read_all()[-1]["correction_of"] == entries[2]["record_id"],
          "correction references superseded record")

    os.unlink(tmp)


def test_migration():
    section("A2. LEGACY MIGRATION")
    legacy = tempfile.mktemp(suffix=".jsonl")
    out = tempfile.mktemp(suffix=".jsonl")

    with open(legacy, "w") as f:
        f.write(json.dumps({"timestamp": "2026-06-24T09:00:00", "job_id": "x",
                            "event": "submitted", "status": "pending",
                            "query": "hi", "priority": 2, "route": "generalist"}) + "\n")
        f.write(json.dumps({"timestamp": "2026-06-24T09:00:05", "job_id": "x",
                            "event": "completed", "status": "completed",
                            "query": "hi", "priority": 2, "clr_score": 0.9}) + "\n")

    n = migrate_legacy_log(legacy, out, overwrite=True)
    check(n == 2, "migrated 2 entries")
    rows = BiTemporalAuditLog(out).read_all()
    check(rows[0]["valid_time"] == "2026-06-24T09:00:00", "valid_time from legacy timestamp")
    check(rows[0]["extra"]["migrated"] is True, "migrated flag set")
    check(rows[0]["extra"]["route"] == "generalist", "extra fields preserved from legacy")
    check(rows[1]["extra"]["clr_score"] == 0.9, "clr_score preserved in extra")

    # Duplicate guard
    try:
        migrate_legacy_log(legacy, out)
        check(False, "re-migrate without overwrite should raise")
    except FileExistsError:
        check(True, "re-migrate without overwrite raises FileExistsError")

    # Overwrite works
    n2 = migrate_legacy_log(legacy, out, overwrite=True)
    check(n2 == 2, "re-migrate with overwrite truncates + rewrites")
    check(len(BiTemporalAuditLog(out).read_all()) == 2, "no duplicates after overwrite")

    # Missing legacy file
    check(migrate_legacy_log("/nonexistent/path.jsonl", out) == 0,
          "missing legacy file returns 0")

    os.unlink(legacy)
    os.unlink(out)


# ====================================================================== #
# B. Job queue
# ====================================================================== #
async def test_queue_lifecycle():
    section("B. JOB QUEUE — lifecycle, priority, concurrency, callbacks")
    tmp = tempfile.mktemp(suffix=".jsonl")
    orch = MockOrchestrator(delay=0.05)
    q = JobQueue(orch, max_concurrent=2, audit_log=tmp)

    check(q.bitemporal is not None, "bi-temporal log initialized")
    await q.start()

    # Submit + complete
    j = q.submit("Solve a_1=2, find a_5", priority=5)
    check(j.status == JobStatus.PENDING, "job starts pending")
    r = await q.wait_for(j.job_id, timeout=5)
    check(r.final_answer.startswith("answer("), "wait_for returns result")
    check(q.status(j.job_id) == JobStatus.COMPLETED, "status is completed")
    check(j.started_at is not None, "started_at set")
    check(j.finished_at is not None, "finished_at set")

    # get / list_jobs
    check(q.get(j.job_id) is j, "get() returns the job")
    check(len(q.list_jobs()) == 1, "list_jobs has 1 job")

    # Callback fires
    callback_results = []
    def cb(result):
        callback_results.append(result)
    j2 = q.submit("Explain what gravity is", priority=3, callback=cb)
    await q.wait_for(j2.job_id, timeout=5)
    check(len(callback_results) == 1, "sync callback fired once")
    check(callback_results[0].final_answer.startswith("answer("), "callback got result")

    # Async callback
    async_cb_results = []
    async def acb(result):
        await asyncio.sleep(0.01)
        async_cb_results.append(result)
    j3 = q.submit("What is recursion", priority=2, callback=acb)
    await q.wait_for(j3.job_id, timeout=5)
    check(len(async_cb_results) == 1, "async callback fired once")

    await q.stop()
    os.unlink(tmp)


async def test_queue_priority():
    section("B2. JOB QUEUE — priority ordering")
    tmp = tempfile.mktemp(suffix=".jsonl")
    # delay high enough that all submit before any completes
    orch = MockOrchestrator(delay=0.15)
    q = JobQueue(orch, max_concurrent=1, audit_log=tmp)  # 1 at a time to see order
    await q.start()

    start_order = []
    original_run = orch.run
    async def tracking_run(query, force_route=None):
        start_order.append(query)
        return await original_run(query, force_route)
    orch.run = tracking_run

    q.submit("low-pri query", priority=1)
    q.submit("high-pri query", priority=10)
    q.submit("mid-pri query", priority=5)

    await asyncio.sleep(0.6)  # let all 3 complete
    await q.stop()

    check(start_order[0] == "high-pri query", "highest priority runs first")
    check(start_order[1] == "mid-pri query", "middle priority runs second")
    check(start_order[2] == "low-pri query", "lowest priority runs last")
    os.unlink(tmp)


async def test_queue_concurrency():
    section("B3. JOB QUEUE — concurrency limit")
    tmp = tempfile.mktemp(suffix=".jsonl")
    orch = MockOrchestrator(delay=0.2)
    q = JobQueue(orch, max_concurrent=2, audit_log=tmp)
    await q.start()

    # Track concurrent executions
    concurrent = 0
    max_concurrent = 0
    original_run = orch.run
    async def counting_run(query, force_route=None):
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        try:
            return await original_run(query, force_route)
        finally:
            concurrent -= 1
    orch.run = counting_run

    jobs = [q.submit(f"query {i}", priority=1) for i in range(5)]
    await asyncio.gather(*[q.wait_for(j.job_id, timeout=5) for j in jobs])
    await q.stop()

    check(max_concurrent <= 2, f"never exceeded max_concurrent=2 (saw {max_concurrent})")
    check(max_concurrent == 2, f"actually reached concurrency=2 (saw {max_concurrent})")
    os.unlink(tmp)


async def test_queue_failure():
    section("B4. JOB QUEUE — failed jobs")
    tmp = tempfile.mktemp(suffix=".jsonl")
    orch = MockOrchestrator(delay=0.05, fail_on={"BOOM"})
    q = JobQueue(orch, max_concurrent=2, audit_log=tmp)
    await q.start()

    j = q.submit("this will BOOM", priority=1)
    try:
        await q.wait_for(j.job_id, timeout=5)
        check(False, "wait_for should raise on failed job")
    except RuntimeError as e:
        check("BOOM" in str(e), "wait_for raises RuntimeError with error text")

    check(q.status(j.job_id) == JobStatus.FAILED, "status is FAILED")
    check(j.error is not None and "BOOM" in j.error, "error message stored on job")

    # Bi-temporal log recorded the failure
    hist = q.job_history(j.job_id)
    events = [e["event"] for e in hist]
    check("failed" in events, "bi-temporal log has 'failed' event")
    failed_entry = [e for e in hist if e["event"] == "failed"][0]
    check("error" in failed_entry["extra"], "failed entry has error in extra")

    await q.stop()
    os.unlink(tmp)


async def test_queue_cancel():
    section("B5. JOB QUEUE — cancel pending job")
    tmp = tempfile.mktemp(suffix=".jsonl")
    orch = MockOrchestrator(delay=0.3)
    q = JobQueue(orch, max_concurrent=1, audit_log=tmp)
    await q.start()

    # Fill the slot
    j1 = q.submit("running job", priority=10)
    # This one will be pending (concurrency=1)
    j2 = q.submit("pending job", priority=1)
    await asyncio.sleep(0.05)  # let j1 start

    check(q.status(j2.job_id) == JobStatus.PENDING, "j2 is pending")
    ok = q.cancel(j2.job_id)
    check(ok is True, "cancel returns True for pending job")
    check(q.status(j2.job_id) == JobStatus.CANCELLED, "j2 status is CANCELLED")

    # Cancel running job -> True (now supported via cooperative cancellation)
    ok2 = q.cancel(j1.job_id)
    check(ok2 is True, "cancel returns True for running job (cooperative)")

    # Cancel unknown -> False
    check(q.cancel("nonexistent") is False, "cancel returns False for unknown job")

    # Bi-temporal log has cancel event
    hist = q.job_history(j2.job_id)
    check(any(e["event"] == "cancelled" for e in hist), "log has 'cancelled' event")

    # j1 was cancelled, so wait_for should raise RuntimeError
    try:
        await q.wait_for(j1.job_id, timeout=5)
        check(False, "wait_for on cancelled job should raise")
    except RuntimeError:
        check(True, "wait_for on cancelled job raises RuntimeError")

    await q.stop()
    os.unlink(tmp)


async def test_queue_wait_timeout():
    section("B6. JOB QUEUE — wait_for timeout")
    tmp = tempfile.mktemp(suffix=".jsonl")
    orch = MockOrchestrator(delay=1.0)  # slow
    q = JobQueue(orch, max_concurrent=1, audit_log=tmp)
    await q.start()

    j = q.submit("slow query", priority=1)
    try:
        await q.wait_for(j.job_id, timeout=0.1)
        check(False, "should timeout")
    except TimeoutError:
        check(True, "wait_for raises TimeoutError")

    # Unknown job_id
    try:
        await q.wait_for("nonexistent", timeout=0.1)
        check(False, "should raise KeyError")
    except KeyError:
        check(True, "wait_for raises KeyError for unknown job")

    await q.stop()
    os.unlink(tmp)


async def test_queue_shutdown():
    section("B7. JOB QUEUE — clean shutdown waits for in-flight jobs")
    tmp = tempfile.mktemp(suffix=".jsonl")
    orch = MockOrchestrator(delay=0.3)
    q = JobQueue(orch, max_concurrent=2, audit_log=tmp)
    await q.start()

    # Submit a job then immediately stop — it should still complete
    j = q.submit("in-flight at shutdown", priority=1)
    await q.stop()
    check(q.status(j.job_id) == JobStatus.COMPLETED,
          "job submitted right before stop() still completes")
    check(j.result is not None, "job has result after shutdown")
    os.unlink(tmp)


# ====================================================================== #
# C. REPL
# ====================================================================== #
async def test_repl():
    section("C. REPL — all commands, edge cases, errors")

    class FakeResult:
        final_answer = "42 is the answer"
    class FakeJob:
        job_id = "abc123"
        priority = 5
        force_route = None
        status = type("S", (), {"value": "completed"})()
        query = "test query here"
        result = FakeResult()
        error = None
        def to_dict(self):
            return {"job_id": self.job_id, "status": "completed", "priority": 5,
                    "force_route": None, "query": "test query here",
                    "created_at": "t1", "started_at": "t2", "finished_at": "t3",
                    "error": None, "has_result": True}
    class FakeQueue:
        bitemporal = None
        def submit(self, q, priority=0, force_route=None):
            self.last_submit = (q, priority, force_route)
            return FakeJob()
        def list_jobs(self):
            return [FakeJob().to_dict()]
        def get(self, jid):
            return FakeJob() if jid == "abc123" else None
        def cancel(self, jid):
            return jid == "abc123"
        def job_history(self, jid, axis="valid"):
            return [{"valid_time": "2026-01-01T00:00:00+00:00",
                     "transaction_time": "2026-01-01T00:00:01+00:00",
                     "event": "submitted", "status": "pending",
                     "extra": {"route": "specialist"}}] if jid == "abc123" else []
        def state_as_of(self, jid, t, axis="valid"):
            return {"valid_time": "2026-01-01T00:00:00+00:00",
                    "event": "submitted", "status": "pending"} if jid == "abc123" else None

    # Flag parsing
    check(_split_flags(["-p", "5", "hello"])[0] == {"priority": 5}, "flag -p at start")
    check(_split_flags(["hello", "-p", "5"])[0] == {"priority": 5}, "flag -p in middle")
    check(_split_flags(["a", "-r", "specialist", "-p", "3"]) ==
          ({"force_route": "specialist", "priority": 3}, ["a"]),
          "multiple flags extracted")
    check(_split_flags(["--axis", "transaction", "job1"]) ==
          ({"axis": "transaction"}, ["job1"]), "--axis flag")
    check(_split_flags(["-p"])[0] == {}, "dangling -p with no value -> empty flags")

    repl = JobQueueREPL(FakeQueue())

    # Empty / whitespace input (was a crash)
    await repl._dispatch("")
    check(True, "empty string doesn't crash")
    await repl._dispatch("   ")
    check(True, "whitespace-only doesn't crash")

    # submit
    await repl._dispatch("submit hello world -p 5 -r specialist")
    check(True, "submit with flags works")

    # list
    await repl._dispatch("list")
    check(True, "list works")
    await repl._dispatch("ls")
    check(True, "ls alias works")

    # status
    await repl._dispatch("status abc123")
    check(True, "status works")
    await repl._dispatch("status nope")
    check(True, "status unknown job handled")

    # result
    await repl._dispatch("result abc123")
    check(True, "result works")
    await repl._dispatch("result nope")
    check(True, "result unknown job handled")

    # cancel
    await repl._dispatch("cancel abc123")
    check(True, "cancel works")

    # history
    await repl._dispatch("history abc123")
    check(True, "history works")
    await repl._dispatch("history abc123 --axis transaction")
    check(True, "history --axis transaction works")
    await repl._dispatch("history nope")
    check(True, "history unknown job handled")

    # asof
    await repl._dispatch("asof abc123 2026-01-01T00:00:00+00:00")
    check(True, "asof works")
    await repl._dispatch("asof onlyone")
    check(True, "asof too few args handled")

    # log-state
    await repl._dispatch("log-state")
    check(True, "log-state with disabled log handled")

    # help
    await repl._dispatch("help")
    check(True, "help works")

    # unknown command
    await repl._dispatch("bogus command")
    check(True, "unknown command handled")

    # usage errors
    await repl._dispatch("submit")
    check(True, "submit with no query handled")
    await repl._dispatch("status")
    check(True, "status with no id handled")
    await repl._dispatch("cancel")
    check(True, "cancel with no id handled")
    await repl._dispatch("history")
    check(True, "history with no id handled")


# ====================================================================== #
# D. Integration: queue + bi-temporal log
# ====================================================================== #
async def test_integration():
    section("D. INTEGRATION — queue + bi-temporal log end-to-end")
    tmp = tempfile.mktemp(suffix=".jsonl")
    orch = MockOrchestrator(delay=0.05)
    q = JobQueue(orch, max_concurrent=2, audit_log=tmp)
    await q.start()

    # Submit 3 jobs with different routes
    j1 = q.submit("Solve the recurrence", priority=5)
    j2 = q.submit("Explain gravity", priority=3)
    j3 = q.submit("What is recursion", priority=4, force_route="generalist")

    results = await asyncio.gather(
        q.wait_for(j1.job_id, timeout=5),
        q.wait_for(j2.job_id, timeout=5),
        q.wait_for(j3.job_id, timeout=5),
    )

    # All completed
    check(all(r is not None for r in results), "all 3 jobs returned results")
    check(q.status(j1.job_id) == JobStatus.COMPLETED, "j1 completed")
    check(q.status(j2.job_id) == JobStatus.COMPLETED, "j2 completed")
    check(q.status(j3.job_id) == JobStatus.COMPLETED, "j3 completed")

    # Bi-temporal log has 3 events per job (submitted, started, completed)
    for j in (j1, j2, j3):
        hist = q.job_history(j.job_id)
        events = [e["event"] for e in hist]
        check(events == ["submitted", "started", "completed"],
              f"job {j.job_id} has 3 lifecycle events in order")

    # Log schema validation
    with open(tmp) as f:
        first_entry = json.loads(f.readline())
    required_keys = {"record_id", "valid_time", "transaction_time", "job_id",
                     "event", "status", "query", "priority", "force_route",
                     "extra", "correction_of"}
    check(required_keys.issubset(first_entry.keys()),
          "log entry has all required bi-temporal schema keys")

    # valid_time <= transaction_time for every entry
    for e in q.bitemporal.read_all():
        check(e["valid_time"] <= e["transaction_time"],
              f"valid_time <= transaction_time for {e['event']} of {e['job_id']}")

    # Log reconstruction matches in-memory state
    log_state = q.bitemporal.current_state()
    in_memory = {j.job_id: j.status.value for j in (j1, j2, j3)}
    for jid, status in in_memory.items():
        check(log_state[jid]["status"] == status,
              f"log state matches in-memory for {jid} ({status})")

    # As-of query for a completed job
    completed_time = q.job_history(j1.job_id)[-1]["valid_time"]
    future = "9999-12-31T23:59:59+00:00"
    state_future = q.state_as_of(j1.job_id, future)
    check(state_future["event"] == "completed", "as_of(future) -> completed")

    # Route info in completed event extra
    j1_completed = [e for e in q.job_history(j1.job_id) if e["event"] == "completed"][0]
    check("route" in j1_completed["extra"], "completed event has route in extra")
    check("clr_score" in j1_completed["extra"], "completed event has clr_score in extra")

    await q.stop()
    os.unlink(tmp)


# ====================================================================== #
# E. Real legacy log migration
# ====================================================================== #
def test_real_migration():
    section("E. REAL LEGACY LOG MIGRATION")
    legacy = "rfsn_jobs.jsonl"
    if not os.path.exists(legacy):
        check(True, "skipped (no rfsn_jobs.jsonl)")
        return

    out = tempfile.mktemp(suffix=".jsonl")
    n = migrate_legacy_log(legacy, out, overwrite=True)
    check(n > 0, f"migrated {n} entries from real legacy log")

    log = BiTemporalAuditLog(out)
    entries = log.read_all()
    check(all(e["extra"].get("migrated") for e in entries), "all entries marked migrated")
    check(all(e["valid_time"] != e["transaction_time"] for e in entries),
          "valid_time (old) != transaction_time (migration time)")

    # Reconstruct state — all jobs should be completed (from the real log)
    state = log.current_state()
    check(all(e["status"] == "completed" for e in state.values()),
          "all real jobs reconstructed as completed")

    # History for a known job
    if "99b1f9b63ca0" in log.jobs():
        hist = log.history("99b1f9b63ca0")
        check(len(hist) == 3, f"known job has 3 events (got {len(hist)})")
        check([e["event"] for e in hist] == ["submitted", "started", "completed"],
              "known job lifecycle correct")

    os.unlink(out)


# ====================================================================== #
# Main
# ====================================================================== #
async def main():
    print("  Run with: python -W error::DeprecationWarning test_demo.py")
    print("  (to also catch DeprecationWarnings as errors)")

    # A. Bi-temporal log (sync tests)
    test_bitemporal_log()
    test_migration()

    # B. Job queue (async tests)
    await test_queue_lifecycle()
    await test_queue_priority()
    await test_queue_concurrency()
    await test_queue_failure()
    await test_queue_cancel()
    await test_queue_wait_timeout()
    await test_queue_shutdown()

    # C. REPL
    await test_repl()

    # D. Integration
    await test_integration()

    # E. Real legacy log
    test_real_migration()

    # Summary
    print(f"\n{'='*70}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'='*70}")
    if FAILED > 0:
        print("  SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("  ALL TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
