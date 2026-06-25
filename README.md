# vibe-thinker

Hybrid reasoning orchestrator for local LLMs. Routes queries between a
high-precision reasoning specialist (VibeThinker-3B with Claim-Level
Reliability) and a generalist model, with a priority async job queue,
bi-temporal audit logging, deterministic verifiers, and an interactive CLI.

## Status: ALPHA SOFTWARE

**This is alpha software.** It is a local reasoning control plane prototype,
not a production reasoning engine. The following limitations are real and
intentional:

- **Self-claim verification is not proof of correctness.** The model checking
  its own claims is a weak heuristic, not independent verification. Confidence
  based solely on self-claims is hard-capped at 0.65.
- **High-confidence answers require deterministic verifier support.** Without
  a math verifier, code verifier, or retrieval source, confidence cannot
  exceed the self-claims cap. Deterministic verification blends 70% verifier
  weight + 30% claim consistency.
- **Without verifier support, confidence is capped.** The system will not
  report high confidence for answers it cannot independently check.
- **Model endpoint failures fail closed.** A dead model server raises a
  `RuntimeError` and marks the job FAILED — it does not return a fake
  "No clear answer found" completed result.
- **Weak self-verification is never cached by default.** The cache rejects
  self-claims-only results, transport failures, low claim counts (<5),
  and scores below 0.75. Caching uncertainty as truth is the core epistemic
  hazard this system guards against.

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
| `vibe_clr_async.py` | Async parallel CLR wrapper (k trajectories, claim extraction, verification, scoring, fail-closed) |
| `vibe_clr.py` | Synchronous facade over the async CLR (single reliability engine) |
| `hybrid_orchestrator.py` | Routes between specialist, generalist, or hybrid path |
| `persistent_cache.py` | Disk-backed route cache + semantic CLR result cache with strict promotion rules |
| `rfsn_job_queue.py` | In-process async priority job queue with concurrency limit |
| `bitemporal_log.py` | Bi-temporal JSONL audit log (valid_time + transaction_time, hash-chain, strict verification) |
| `rfsn_cli.py` | Interactive CLI/REPL over the job queue (env var + CLI flag support) |
| `scoring.py` | Separated confidence fields (model_confidence, claim_consistency, deterministic_verification) |
| `verifiers/` | Deterministic verifier adapters (math, code, factual) |
| `test_demo.py` | Comprehensive test suite (no model servers needed) |
| `test_full_stack.py` | Full-stack integration test (needs live servers) |

## CLR scoring

The reliability score is fail-closed:

- No final answer -> score = 0
- Fewer than 5 meaningful claims -> score = 0
- Garbage/prompt-fragment claims are rejected before scoring
- Any unverified claim -> score capped at 0.3
- Only all-verified, meaningful trajectories can reach 1.0
- **All trajectories fail (dead endpoint) -> RuntimeError, job FAILED**
- **Partial trajectory failure -> continues with warning metadata**

## Confidence scoring

Confidence is split into separate fields to avoid conflating model
self-agreement with deterministic verification:

```json
{
  "model_confidence": 0.82,
  "claim_consistency": 0.74,
  "deterministic_verification": 1.0,
  "final_score": 0.92,
  "verified": true,
  "verification_method": "python_eval"
}
```

- With deterministic verification: `final_score = det * 0.7 + consistency * 0.3`
- Without deterministic verification: `final_score = min(consistency, 0.65)`
- Self-claims-only confidence is **hard-capped at 0.65**

## Cache promotion rules

Bad answers are never cached. The strict `should_cache()` policy rejects:

- No answer or sentinel failure strings ("No clear answer found", etc.)
- `self_claims_only` verification (unless `allow_weak_cache=True`)
- Transport failures (>0 failed trajectories)
- Low claim count (<5 meaningful claims)
- Low score (<0.75)
- Explicit failure results

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
  This is a weak heuristic, not independent verification. Deterministic
  verifiers (math, code, factual) are provided but must be wired into the
  orchestrator pipeline for full effect.
- The factual verifier is an honest placeholder: without retrieval sources,
  it returns `verified=False` rather than pretending to verify.
- The job queue is in-process only. For multi-process/multi-node, swap the
  dispatcher for RQ/Celery/Dramatiq.
- The bi-temporal log has hash-chain integrity with strict verification
  (`verify_chain(strict=True)`), but does not use cryptographic signatures.
  For audit-grade use, add signing keys.
