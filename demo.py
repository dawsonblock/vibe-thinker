"""
End-to-end demo of the RFSN job queue + bi-temporal audit log + REPL.

Runs entirely with a mock orchestrator (no model servers required) so it
can be executed anywhere:

    python demo.py

What it shows:
  1. Migrating the legacy flat audit log into bi-temporal format.
  2. Starting the job queue with the bi-temporal log.
  3. Submitting jobs with different priorities and force-routes.
  4. Watching priority-ordered dispatch + concurrent execution.
  5. Querying the bi-temporal log: history, as-of (both time axes).
  6. Driving the REPL programmatically (same commands a human would type).
  7. Clean shutdown that waits for in-flight jobs.
"""

import asyncio
import json
import os
import tempfile

from rfsn_job_queue import JobQueue, JobStatus
from bitemporal_log import BiTemporalAuditLog, migrate_legacy_log
from rfsn_cli import JobQueueREPL


# ====================================================================== #
# Mock orchestrator — simulates routing + latency without model servers
# ====================================================================== #
class MockResult:
    def __init__(self, answer, route, clr_score=None):
        self.final_answer = answer
        self.route_taken = route
        self.clr_score = clr_score
        self.raw_traces = {}
        self.timestamp = "mock"
        self.routing_confidence = 0.9


class MockOrchestrator:
    """Pretends to be HybridReasoningOrchestrator with realistic delays."""

    def __init__(self):
        self.call_count = 0

    async def run(self, query, force_route=None):
        self.call_count += 1
        # Simulate different latencies per route
        await asyncio.sleep(0.3)

        q = query.lower()
        if force_route:
            route = force_route
        elif any(kw in q for kw in ("solve", "find", "series", "sum", "recurrence")):
            route = "specialist_clr"
        elif any(kw in q for kw in ("explain", "what is", "history")):
            route = "generalist"
        else:
            route = "hybrid"

        if route.startswith("specialist"):
            return MockResult(f"[specialist] The answer to '{query[:40]}...' is 1807.",
                              route, clr_score=1.0)
        elif route == "generalist":
            return MockResult(f"[generalist] Here's a brief explanation of '{query[:40]}...'",
                              route, clr_score=None)
        else:
            return MockResult(f"[hybrid] Synthesized answer for '{query[:40]}...'",
                              route, clr_score=0.85)


# ====================================================================== #
# Helpers
# ====================================================================== #
def header(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


def pretty(obj):
    if isinstance(obj, str):
        return obj
    return json.dumps(obj, indent=2, default=str)


# ====================================================================== #
# Demo
# ====================================================================== #
async def run_demo():
    workdir = tempfile.mkdtemp(prefix="rfsn_demo_")
    bitemporal_path = os.path.join(workdir, "jobs_bitemporal.jsonl")
    legacy_path = os.path.join(workdir, "jobs_legacy.jsonl")

    # -- 1. Create a small legacy log, then migrate it ----------------- #
    header("1. MIGRATE legacy flat log -> bi-temporal")
    with open(legacy_path, "w") as f:
        for evt in [
            {"timestamp": "2026-06-24T09:00:00", "job_id": "oldjob1",
             "event": "submitted", "status": "pending", "query": "old math problem",
             "priority": 5},
            {"timestamp": "2026-06-24T09:00:10", "job_id": "oldjob1",
             "event": "completed", "status": "completed", "query": "old math problem",
             "priority": 5, "route": "specialist_clr", "clr_score": 1.0},
        ]:
            f.write(json.dumps(evt) + "\n")

    n = migrate_legacy_log(legacy_path, bitemporal_path, overwrite=True)
    print(f"  Migrated {n} legacy entries -> {bitemporal_path}")
    log = BiTemporalAuditLog(bitemporal_path)
    for e in log.read_all():
        print(f"  valid={e['valid_time']}  txn={e['transaction_time']}  "
              f"job={e['job_id']}  event={e['event']}  status={e['status']}")
    print("  -> Note: valid_time from legacy timestamp, transaction_time = migration time")

    # -- 2. Start the queue with the SAME bi-temporal log (appends) ---- #
    header("2. START job queue (max_concurrent=2, bi-temporal logging ON)")
    orch = MockOrchestrator()
    queue = JobQueue(orch, max_concurrent=2, audit_log=bitemporal_path)
    await queue.start()

    # -- 3. Submit jobs with different priorities ---------------------- #
    header("3. SUBMIT jobs (priority-ordered dispatch)")
    jobs = []
    queries = [
        ("Explain what the Riemann Hypothesis is.", 3, None),
        ("Solve the recurrence a_1=2, a_{n+1}=a_n^2-a_n+1, find a_5.", 5, None),
        ("Sum the geometric series 1 + 1/3 + 1/9 + ...", 4, None),
        ("What is the capital of France?", 1, "generalist"),
    ]
    for q, pri, route in queries:
        j = queue.submit(q, priority=pri, force_route=route)
        jobs.append(j)
        print(f"  submitted {j.job_id}  pri={pri}  route={route or 'auto'}  q={q[:50]}")

    print("\n  Waiting for all jobs to complete...")
    results = await asyncio.gather(
        *[queue.wait_for(j.job_id, timeout=10) for j in jobs],
        return_exceptions=True,
    )

    # -- 4. Show results ----------------------------------------------- #
    header("4. RESULTS")
    for j, r in zip(jobs, results):
        if isinstance(r, Exception):
            print(f"  {j.job_id}  ERROR: {r}")
        else:
            print(f"  {j.job_id}  status={j.status.value:10}  route={r.route_taken:16}  "
                  f"clr={r.clr_score}  answer={r.final_answer[:60]}")

    # -- 5. Bi-temporal queries ---------------------------------------- #
    header("5. BI-TEMPORAL QUERIES")

    print("\n  5a. Full history of one job (valid_time axis = true-world order):")
    target = jobs[1].job_id  # the math one
    for e in queue.job_history(target, axis="valid"):
        extra = e.get("extra", {})
        tail = f"  extra={extra}" if extra else ""
        print(f"    valid={e['valid_time']}  event={e['event']:10}  "
              f"status={e['status']}{tail}")

    print(f"\n  5b. State of job {target} as of a mid-point timestamp:")
    # Pick a timestamp between 'started' and 'completed'
    hist = queue.job_history(target, axis="valid")
    mid_ts = hist[1]["valid_time"]  # the 'started' event's valid_time
    state = queue.state_as_of(target, mid_ts, axis="valid")
    print(f"    as_of={mid_ts}  ->  event={state['event']}  status={state['status']}")

    print(f"\n  5c. Same job, transaction_time axis (what we KNEW, and when):")
    for e in queue.job_history(target, axis="transaction"):
        print(f"    txn={e['transaction_time']}  valid={e['valid_time']}  "
              f"event={e['event']:10}")
    print("    -> transaction_time >= valid_time (recording lags the event)")

    print("\n  5d. Current state of ALL jobs reconstructed from the log:")
    for jid, e in queue.bitemporal.current_state(axis="valid").items():
        print(f"    {jid}  status={e['status']:10}  event={e['event']:10}  "
              f"query={e['query'][:40]}")

    # -- 6. REPL demo (programmatic) ----------------------------------- #
    header("6. REPL demo (driving the same commands a user would type)")
    repl = JobQueueREPL(queue)

    repl_cmds = [
        "list",
        f"status {jobs[0].job_id}",
        f"result {jobs[1].job_id}",
        f"history {jobs[2].job_id}",
        f"history {jobs[2].job_id} --axis transaction",
        "log-state",
        "help",
    ]
    for cmd in repl_cmds:
        print(f"\n  rfsn> {cmd}")
        await repl._dispatch(cmd)

    # -- 7. Clean shutdown --------------------------------------------- #
    header("7. SHUTDOWN (waits for in-flight jobs)")
    # Submit one more job right before shutdown to prove stop() waits
    extra = queue.submit("Solve x^2 + 2x - 3 = 0", priority=2)
    print(f"  Submitted {extra.job_id} right before stop() — should still complete")
    await queue.stop()
    print(f"  {extra.job_id} final status: {queue.status(extra.job_id).value}")

    # -- Final log summary --------------------------------------------- #
    header("SUMMARY")
    total = len(log.read_all())
    n_jobs = len(queue.bitemporal.jobs())
    print(f"  Bi-temporal log: {total} entries across {n_jobs} jobs")
    print(f"  Log file: {bitemporal_path}")
    print(f"  Mock orchestrator handled {orch.call_count} calls")
    print("\n  Demo complete. All components working correctly.")


if __name__ == "__main__":
    asyncio.run(run_demo())
