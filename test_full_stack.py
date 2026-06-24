"""
Full-stack integration test.

Exercises:
  - RFSN job queue (priorities, concurrency, status tracking)
  - Hybrid orchestrator (routing: specialist + generalist + hybrid)
  - Both local models (VibeThinker-3B on :8080, Llama 3.2 3B on :8081)
  - Persistent route cache + CLR result cache (second run should hit caches)

Requires embedding deps for the full experience:
  pip install sentence-transformers scikit-learn numpy aiohttp

If embedding deps are missing, it falls back to keyword routing and skips
the CLR result cache (still tests the queue + both models).

Usage:
    python test_full_stack.py

Exit code is nonzero if any assertion fails, so this is CI-safe.
"""

import asyncio
import json
import os
import sys
import tempfile

from hybrid_orchestrator import HybridReasoningOrchestrator
from rfsn_job_queue import JobQueue, JobStatus


# Use a temp directory for cache files so we never destroy project artifacts.
# The old version deleted route_cache.json / clr_result_cache.json / rfsn_jobs.jsonl
# at import time — that was a destructive side effect that made the module
# unsafe to import. Now cleanup is explicit and scoped to a temp dir.
_CACHE_DIR = None


def _cache_dir() -> str:
    global _CACHE_DIR
    if _CACHE_DIR is None:
        _CACHE_DIR = tempfile.mkdtemp(prefix="vibe_test_")
    return _CACHE_DIR


QUERIES = [
    # specialist (math, verifiable) — should route to VibeThinker + CLR
    {
        "q": "Solve this step by step: a_1=2, a_{n+1}=a_n^2 - a_n + 1. Find a_5.",
        "priority": 5,
        "expect_route_prefix": "specialist",
        "expect_answer_contains": "1807",
    },
    # generalist (knowledge/explanation) — should route to Llama 3.2
    {
        "q": "In two sentences, explain what the Riemann Hypothesis is.",
        "priority": 3,
        "expect_route_prefix": "generalist",
        "expect_answer_contains": None,  # free-form
    },
    # hybrid (mixed) — generalist plans, specialist solves, generalist synthesizes
    {
        "q": "Explain the math behind geometric series and compute the sum of "
             "1 + 1/3 + 1/9 + 1/27 + ...",
        "priority": 4,
        "expect_route_prefix": None,  # could be hybrid or specialist
        "expect_answer_contains": None,
    },
]


async def run_once(label: str, orchestrator: HybridReasoningOrchestrator, audit_log: str):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    queue = JobQueue(orchestrator, max_concurrent=2, audit_log=audit_log)
    await queue.start()

    jobs = []
    for spec in QUERIES:
        job = queue.submit(spec["q"], priority=spec["priority"])
        jobs.append((job, spec))
        print(f"  Submitted job {job.job_id} (pri={spec['priority']}): {spec['q'][:60]}...")

    results = []
    for job, spec in jobs:
        try:
            result = await queue.wait_for(job.job_id, timeout=600)
            results.append((job, spec, result, None))
        except Exception as e:
            results.append((job, spec, None, str(e)))

    await queue.stop()

    # Report
    print(f"\n--- {label} Results ---")
    all_ok = True
    for job, spec, result, err in results:
        if err:
            print(f"  [{job.job_id}] ERROR: {err}")
            all_ok = False
            continue
        route = result.route_taken
        ans = (result.final_answer or "").replace("\n", " ")
        score = result.clr_score
        ok = True
        if spec["expect_route_prefix"] and not route.startswith(spec["expect_route_prefix"]):
            ok = False
            print(f"  [{job.job_id}] ROUTE MISMATCH: got '{route}', "
                  f"expected prefix '{spec['expect_route_prefix']}'")
        if spec["expect_answer_contains"] and spec["expect_answer_contains"] not in ans:
            ok = False
            print(f"  [{job.job_id}] ANSWER MISMATCH: expected to contain "
                  f"'{spec['expect_answer_contains']}', got: {ans[:120]}")
        if ok:
            tag = "OK"
            print(f"  [{job.job_id}] {tag} route={route} score={score} "
                  f"answer={ans[:100]}")
        else:
            all_ok = False
    return all_ok


async def main():
    print("Checking embedding deps...")
    try:
        import sentence_transformers  # noqa: F401
        import sklearn  # noqa: F401
        have_embeddings = True
        print("  Embedding deps present — full routing + CLR cache enabled.")
    except ImportError:
        have_embeddings = False
        print("  Embedding deps MISSING — using keyword routing, no CLR cache.")

    cdir = _cache_dir()
    route_cache = os.path.join(cdir, "route_cache.json")
    clr_cache = os.path.join(cdir, "clr_result_cache.json")
    audit_log = os.path.join(cdir, "rfsn_jobs_bitemporal.jsonl")

    orchestrator = HybridReasoningOrchestrator(
        vibe_endpoint="http://127.0.0.1:8080",
        generalist_endpoint="http://127.0.0.1:8081",
        use_clr=True,
        clr_k=4,                 # small k to keep the test fast
        max_concurrent_clr=4,
        use_embedding_router=have_embeddings,
        use_clr_cache=have_embeddings,
        router_cache_size=512,
        clr_cache_path=clr_cache,
    )
    # Override the route cache path (the orchestrator constructor doesn't expose it)
    if hasattr(orchestrator, "router") and orchestrator.router is not None:
        if orchestrator.router.persistent is not None:
            orchestrator.router.persistent.path = route_cache

    # Run 1: cold caches
    ok1 = await run_once("RUN 1 (cold caches)", orchestrator, audit_log)

    # Run 2: warm caches — the specialist query should hit the CLR result cache
    # and the route decisions should hit the route cache.
    ok2 = await run_once("RUN 2 (warm caches — expect cache hits)", orchestrator, audit_log)

    print("\n" + "=" * 70)
    print(f"Run 1 OK: {ok1}")
    print(f"Run 2 OK: {ok2}")
    print("=" * 70)

    # Show cache files
    for p in (route_cache, clr_cache):
        if os.path.exists(p):
            sz = os.path.getsize(p)
            print(f"  {os.path.basename(p)}: {sz} bytes")
        else:
            print(f"  {os.path.basename(p)}: (not created — embedding deps likely missing)")

    # Show job audit log tail
    if os.path.exists(audit_log):
        with open(audit_log) as f:
            lines = f.readlines()
        print(f"\n  {os.path.basename(audit_log)}: {len(lines)} events")
        for line in lines[-4:]:
            e = json.loads(line)
            print(f"    {e['event']:10s} job={e['job_id']} status={e['status']}")

    if have_embeddings and orchestrator.clr_cache is not None:
        print(f"\n  CLR cache entries: {len(orchestrator.clr_cache)}")

    return ok1 and ok2


if __name__ == "__main__":
    ok = asyncio.run(main())
    print("\nFULL STACK TEST:", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)
