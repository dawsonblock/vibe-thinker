# vibe-thinker

Hybrid reasoning orchestrator for local LLMs. Routes queries between a
high-precision reasoning specialist (VibeThinker-3B with Claim-Level
Reliability) and a generalist model, with a priority async job queue,
bi-temporal audit logging, and an interactive CLI.

## Architecture

```
Query -> EmbeddingRouter -> specialist | generalist | hybrid
                                |
                    VibeThinkerCLRAsync
                    (k parallel trajectories)
                                |
                    extract claims -> verify claims -> score
                                |
                    best trajectory -> OrchestratorResult
                                |
                    JobQueue (priority, concurrency-limited)
                                |
                    BiTemporalAuditLog (valid_time + transaction_time)
```

## Requirements

- Python 3.10+
- A running llama-server for the specialist (default: `http://127.0.0.1:8080`)
- A running llama-server for the generalist (default: `http://127.0.0.1:8081`)
- `pip install -r requirements.txt` (aiohttp is required; sentence-transformers
  is optional but enables semantic routing + CLR result caching)

## Quick start

```bash
pip install -r requirements.txt

# Start your model servers (e.g.):
# llama-server --model vibethinker-3b.gguf --port 8080
# llama-server --model llama-3.2-3b.gguf --port 8081

# Run the interactive REPL:
python rfsn_cli.py

# Or run the full-stack integration test (needs live servers):
python test_full_stack.py
```

## Components

| File | Purpose |
|---|---|
| `vibe_clr_async.py` | Async parallel CLR wrapper (k trajectories, claim extraction, verification, scoring) |
| `hybrid_orchestrator.py` | Routes between specialist, generalist, or hybrid path |
| `persistent_cache.py` | Disk-backed route cache + semantic CLR result cache |
| `rfsn_job_queue.py` | In-process async priority job queue with concurrency limit |
| `bitemporal_log.py` | Bi-temporal JSONL audit log (valid_time + transaction_time) |
| `rfsn_cli.py` | Interactive CLI/REPL over the job queue |
| `test_demo.py` | Comprehensive test suite (no model servers needed) |
| `test_full_stack.py` | Full-stack integration test (needs live servers) |

## CLR scoring

The reliability score is fail-closed:

- No final answer -> score = 0
- Fewer than 3 meaningful claims -> score = 0
- Garbage/prompt-fragment claims are rejected before scoring
- Any unverified claim -> score capped at 0.3
- Only all-verified, meaningful trajectories can reach 1.0

Bad answers are never cached. The cache rejects empty answers, sentinel
strings ("No clear answer found"), zero-score results, and results with
no trajectories.

## Testing

```bash
# Unit + integration tests (no model servers needed):
python test_demo.py

# With pytest:
pytest tests/

# Full-stack test (needs live model servers):
python test_full_stack.py
```

## Known limitations

- Self-verification: the same model generates, extracts, and verifies claims.
  This is a weak heuristic, not independent verification. For math/code tasks,
  deterministic verification (execution, unit tests, symbolic checks) should
  be added.
- The job queue is in-process only. For multi-process/multi-node, swap the
  dispatcher for RQ/Celery/Dramatiq.
- The bi-temporal log is append-only JSONL without hash-chain integrity. For
  audit-grade use, add record hashes, sequence numbers, and signatures.
