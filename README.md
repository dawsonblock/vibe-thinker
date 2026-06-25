# vibe-thinker

Hybrid reasoning orchestrator for local LLMs. Routes queries between a
high-precision reasoning specialist (VibeThinker-3B with Claim-Level
Reliability) and a generalist model, with a priority async job queue,
bi-temporal audit logging, deterministic verifiers, and an interactive CLI.

## Status: ALPHA SOFTWARE (v0.3.1)

**This is alpha software.** It is a local reasoning control plane prototype,
not a production reasoning engine. The following limitations are real and
intentional:

- **Self-claim verification is capped at 0.65.** The model checking its own
  claims is a weak heuristic, not independent verification. The runtime
  enforces this cap in the active scoring path — `compute_confidence()` is
  called by `_calculate_reliability()`, not just defined in a module nobody
  imports. No code path returns >0.65 unless `verification_method` is not
  `self_claims_only`.
- **High confidence requires an independent verifier.** MathVerifier,
  CodeVerifier, or FactualVerifier must pass for the score to exceed 0.65.
  Verifiers are wired into the orchestrator path: `select_verifier()` picks
  the verifier based on `task_type`, and the CLR `run()` calls it against
  the best answer. A passing verifier blends 70% verifier weight + 30% claim
  consistency. A refuting verifier sets the score to 0.0.
- **Cache lookup rejects weak or legacy entries by default.** The cache
  enforces the same trust rules on lookup as on insertion.
  `is_cache_entry_trustworthy()` rejects entries with `schema_version < 3`,
  `self_claims_only` verification (unless `allow_weak_cache=True`), transport
  failures, low claim counts (<5), and sentinel failure answers. Old v0.1 /
  v0.2 / v0.3 cache entries are silently invalidated by the schema version
  bump.
- **Factual answers without retrieval are marked unsupported.**
  FactualVerifier returns `verified=False` with `method="unsupported_factual"`
  when no retrieval sources are provided. A factual answer without citations
  cannot enter the high-confidence cache.
- **Code answers require executable tests to be verified.** CodeVerifier
  refuses to verify code when no `unit_tests` or `expected_output` are
  provided. Running code without checking its output is not verification.
- **Model endpoint failures fail closed.** A dead model server raises a
  `RuntimeError` and marks the job FAILED — it does not return a fake
  "No clear answer found" completed result.
- **Routing does not send generalist tasks to specialist CLR.** The route
  is determined by `task_type`, not just the embedding router. "code of
  conduct" routes to generalist because its task type is conversation, not
  code. "sum of human knowledge" routes to generalist or hybrid, not
  specialist.

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
| `sandbox/` | Isolated code execution layer (Docker container, sbx microVM, local subprocess) |
| `math_solver.py` | Deterministic solver for simple math (arithmetic, sums, recurrences) — derives expected_answer for MathVerifier |
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
  This is a weak heuristic, not independent verification. The runtime caps
  self-claims-only confidence at 0.65. Deterministic verifiers (math, code,
  factual) are wired into the orchestrator path and are the only way to
  exceed the cap.
- The factual verifier is an honest placeholder: without retrieval sources,
  it returns `verified=False` with `method="unsupported_factual"` rather than
  pretending to verify. A factual answer without citations cannot reach
  high confidence.
- The code verifier refuses to verify code without executable test
  specifications (`unit_tests` or `expected_output`). Running code without
  checking output is not verification.
- The code verifier runs in a sandbox executor, not raw subprocess. By
  default it uses Docker containers with `--network=none`, `--read-only`,
  `--memory=128m`, and `--cap-drop=ALL`. If Docker is not available, it
  tries sbx microVMs. If neither is available, it **refuses to verify**
  rather than running untrusted code on the host. See `sandbox/README.md`
  for the full isolation model and sbx setup instructions.
- The job queue is in-process only. For multi-process/multi-node, swap the
  dispatcher for RQ/Celery/Dramatiq.
- The bi-temporal log has hash-chain integrity with strict verification
  (`verify_chain(strict=True)`), but does not use cryptographic signatures.
  For audit-grade use, add signing keys.
- Cache entries from v0.1/v0.2/v0.3 (schema_version < 3) are silently
  rejected on lookup. Delete `clr_result_cache.json` to start fresh.
