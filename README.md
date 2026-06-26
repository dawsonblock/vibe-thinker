# vibe-thinker

Hybrid reasoning orchestrator for local LLMs. Routes queries between a
high-precision reasoning specialist (VibeThinker-3B with Claim-Level
Reliability) and a generalist model, with a priority async job queue,
bi-temporal audit logging, deterministic verifiers, and an interactive CLI.

## Status: ALPHA SOFTWARE (v0.3.7)

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
- *Optional:* a dedicated code-specialist server (default: `http://127.0.0.1:8082`).
  When configured via `CODE_SPECIALIST_URL` / `--code-specialist`, code tasks
  route here for fast plain generation (verified downstream by CodeVerifier)
  instead of the VibeThinker CLR path. Math/reasoning still uses the specialist.
- `pip install -r requirements.txt` (aiohttp is required; sentence-transformers
  is optional but enables semantic routing + CLR result caching)

## Quick start

```bash
pip install -r requirements.txt

# Start your model servers (e.g.):
# llama-server --model vibethinker-3b.gguf --port 8080
# llama-server --model llama-3.2-3b.gguf --port 8081
# Optional code specialist (e.g. ruvltra-claude-code-0.5b):
# llama-server --model ruvltra-claude-code-0.5b-q4_k_m.gguf --port 8082

# Run the interactive REPL (auto-loads .env, incl. CODE_SPECIALIST_URL):
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
| `persistent_cache.py` | Disk-backed route cache + semantic CLR result cache + verified trajectory store |
| `rfsn_job_queue.py` | In-process async priority job queue with concurrency limit |
| `bitemporal_log.py` | Bi-temporal JSONL audit log (valid_time + transaction_time, hash-chain, strict verification) |
| `rfsn_cli.py` | Interactive CLI/REPL over the job queue (env var + CLI flag support) |
| `scoring.py` | Separated confidence fields (model_confidence, claim_consistency, deterministic_verification) |
| `verifiers/` | Deterministic verifier adapters (math, code, factual) |
| `sandbox/` | Isolated code execution layer (Docker container, warm pool, sbx microVM, local subprocess) |
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
- **Cross-trajectory consistency is NOT deterministic verification.**
  The model agreeing with itself is not proof of correctness.
  Consensus gives a small boost within the 0.65 cap (+0.05) but can
  NEVER exceed it. Only external verifiers can exceed 0.65.

## Adaptive compute (v0.3.2)

The CLR runtime uses adaptive compute (dynamic sampling / early exiting)
instead of brute-force trajectory generation:

**Phase 1 — Fast Path (System 1):**
- If a verifier exists: generate k=1 lightweight trajectory (answer
  extraction only, no claim verification — saves 6 LLM calls), verify
  immediately. If `verified=True`, exit early.
- If no verifier: generate k=2 full trajectories.

**Phase 2 — Consensus Check:**
- If no verifier confirmed, check if trajectories agree.
- If all `\boxed{}` answers match, exit early — more trajectories won't
  raise the score above 0.65 without a verifier.
- **Score is capped at 0.65.** Consensus saves compute, not trust.
- High-risk tasks (code, file modification, security, medical, legal,
  financial) cannot early-exit from self-consensus alone.

**Phase 3 — Branching (System 2):**
- If trajectories disagree, verifier refuted, or high-risk task without
  verifier: scale up to `max_k=6` trajectories.
- Aggregate all trajectories to find the most consistent answer.
- Self-only results still capped at 0.65.

**Decision table:**

| Condition | Action | Max score |
|---|---|---|
| Deterministic verifier passes | Early exit at k=1 | >0.65 allowed |
| Deterministic verifier refutes | Branch to max_k | 0 unless later verified |
| Verifier unsupported | Continue self-check | ≤0.65 |
| 2 trajectories agree, no verifier | Early exit at k=2 | ≤0.65 |
| 2 trajectories disagree | Branch to max_k | ≤0.65 unless verifier passes |
| No final answer | Branch; if still none, fail | 0 |
| Model/server error | Fail job if all attempts fail | 0 |
| High-risk task, no verifier | Branch to max_k (no consensus exit) | ≤0.65 |

**Queue-load-aware max_k:** When the job queue is under pressure,
max_k is automatically reduced to improve throughput:
- Queue load < 50% → max_k = 6 (full budget)
- Queue load 50–80% → max_k = 4
- Queue load > 80% → max_k = 2

**CLRResult metadata:** Every result includes adaptive compute metadata:
```json
{
  "best_answer": "42",
  "best_score": 0.65,
  "verified": false,
  "verification_method": "self_claims_only",
  "verification_status": "self_only",
  "adaptive": true,
  "trajectories_used": 2,
  "max_trajectories": 6,
  "early_exit_reason": "self_consensus_cap_reached",
  "agreement": true
}
```

## Multi-candidate code generation (v0.3.3)

When a dedicated code-specialist endpoint is configured (`CODE_SPECIALIST_URL`)
and a code verifier is available (default: `CodeVerifier` with Docker sandbox),
code tasks run through a sandbox-verified multi-candidate loop:

1. **Generalist writes tests** — the generalist model produces unit-test
   `assert` statements for the query (the "Software Architect" step).
2. **Code specialist generates N candidates** — ruvltra (or any configured
   code model) generates `CODE_CANDIDATES` (default 6) solutions in parallel
   with diverse temperatures. Higher is better for fast 0.5B models.
3. **Sandbox verification** — `CodeVerifier` runs each candidate against the
   test spec in a hardened Docker container (`--network=none`, `--read-only`,
   `--cap-drop=ALL`, `--security-opt=no-new-privileges`). Uses the warm Docker
   pool by default (2.5x faster than cold `docker run`).
4. **First passing candidate wins** — the first candidate that prints
   `ALL_TESTS_PASSED` is returned with `clr_score=1.0`, `verified=True`.
5. **Fail-closed** — if no candidate passes, the first candidate is returned
   with `clr_score=0.0`, `verified=False`. The system never fakes verification.

Requires Docker running + `python:3.12-slim` image. Without Docker, the
verifier fail-closes and the loop returns unverified best-effort (score 0.0).

## Verified trajectory store (v0.3.4)

Self-improving few-shot memory. When a deterministic verifier (MathVerifier,
CodeVerifier) confirms an answer, the `(query, verified_answer, method, score,
task_type)` tuple is stored semantically. On future similar queries, verified
trajectories are retrieved as few-shot context and prepended to the model
prompt — improving first-attempt success rate without weight updates.

This is RAG over verified solutions, not fake "self-learning" weight updates.
The system genuinely improves over time: each verified solution becomes a
retrievable example for similar future problems.

**Trust model (fail-closed):**
- Only `verified=True` with deterministic verifiers are stored
- `self_claims_only` and unverified results are **never** stored
- Entries re-checked for trustworthiness on retrieval
- `task_type` filtering prevents math examples polluting code queries
- The store provides **context, not answers** — the verifier still must confirm

Config: `RFSN_USE_TRAJECTORY_STORE` (default true), `TRAJECTORY_STORE_PATH`,
`--no-trajectory-store` CLI flag.

## Warm Docker pool (v0.3.4)

`WarmDockerPool` keeps N containers running and uses `docker exec` instead of
`docker run --rm` for each verification. Measured speedup: **2.5x** (0.494s →
0.197s per verification).

Same security hardening as cold runs: `--network=none`, `--read-only`,
`--security-opt=no-new-privileges`, `--cap-drop=ALL`, `--pids-limit=64`,
tmpfs `/tmp`. The `/tmp` directory is cleaned between executions via
`find -delete` to prevent state leakage. Containers are recycled after 50
uses to prevent resource accumulation.

`select_executor` prefers the warm pool by default. Disable with
`prefer_warm_pool=False`. The orchestrator manages the pool lifecycle via
`start()` / `cleanup()`.

## Grammar enforcement (v0.3.4)

GBNF grammar passed to llama-server's `/completion` endpoint for CLR claim
extraction. Forces valid JSON output: `{"claims": [...], "final_answer": "..."}`.
This prevents small models from producing malformed JSON that causes trajectory
scoring to fail — the main weakness of sub-3B models.

Applied to the VibeThinker-3B extraction path (where JSON parsing happens).
The existing regex fallback parser is retained as defense-in-depth. When the
in-process backend (below) is active, the same grammar is enforced natively via
`LlamaGrammar`.

## In-process specialist backend (v0.3.5, pool in v0.3.6)

For ultra-tiny specialists (e.g. ruvltra-claude-code-0.5b, ~398MB, ~100+ tok/s),
HTTP overhead to a separate llama-server dominates inference time. The
in-process backend loads the GGUF directly into the orchestrator's Python
process via `llama-cpp-python` and calls it through a thread executor — zero
network latency, zero JSON-serialization overhead.

Auto-preferred over HTTP when configured; falls back to HTTP if
`llama-cpp-python` is missing or the load fails.

**Pool mode (v0.3.6)**: `--local-specialist-pool-size N` loads N separate
`Llama` instances into a `queue.Queue` for true parallel inference. Each call
checks out one instance, runs it in a thread executor, and returns it. For a
0.5B model, 4 instances cost ~1.6GB and enable 4 concurrent trajectories —
fixing the serialization bottleneck of single-instance mode. Default is 1
(single instance + Lock, lowest memory).

```bash
# Install (Apple Silicon, Metal acceleration):
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python

# Run with 4 in-process instances + fast-specialist policy:
python rfsn_cli.py \
  --local-specialist-model ~/models/ruvltra-claude-code-0.5b-q4_k_m.gguf \
  --local-specialist-pool-size 4 \
  --fast-specialist \
  --generalist http://127.0.0.1:8081
```

| Flag / env | Default | Purpose |
|---|---|---|
| `--local-specialist-model` / `VIBE_THINKER_LOCAL_MODEL` | empty | `.gguf` path or `repo_id/filename.gguf` |
| `--local-specialist-n-ctx` / `VIBE_THINKER_LOCAL_N_CTX` | 4096 | context window |
| `--local-specialist-n-threads` / `VIBE_THINKER_LOCAL_N_THREADS` | 8 | CPU threads (divided across pool) |
| `--local-specialist-pool-size` / `VIBE_THINKER_LOCAL_POOL_SIZE` | 1 | N parallel Llama instances |

## Test-feedback loop (v0.3.6)

When ALL code candidates fail with `TEST_ERROR` (the test harness itself
crashed — not an assertion failure), the generalist gets one retry to rewrite
the tests with the error fed back as context. This distinguishes "bad tests"
from "bad code":

- `ASSERTION_FAILED` / `IMPORT_ERROR` → candidate is wrong, no retry
- `TEST_ERROR` → test spec is broken, retry once with error feedback

Max 2 attempts. If the retry also fails, returns best-effort unverified
(score 0.0, fail-closed).

## Fast-specialist adaptive profile (v0.3.5)

`make_fast_specialist_policy()` returns an `AdaptivePolicy` with
`initial_k_with_verifier=3`, `initial_k_without_verifier=5`, `max_k=15` (all
capped at `--clr-k`) — shotgun-sampling tuned for a 0.5B model where 15 parallel trajectories cost
roughly what 2 cost on a 3B model. The `self_claim_cap` stays 0.65: a fast
model agreeing with itself more often is NOT independent verification.

Gated behind `--fast-specialist` / `RFSN_FAST_SPECIALIST` (default off). **Do
not enable with a 3B+ specialist on 16GB RAM** — 15 parallel trajectories will
thrash or OOM. The default 1/2/6 policy is unchanged.

## TurboQuant+ (optional llama-server backend)

[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)
is a fork of llama.cpp that adds Walsh-Hadamard rotated polar quantization for
both KV cache (`turbo2`/`turbo3`/`turbo4`) and weights (`TQ3_1S`/`TQ4_1S`).
vibe-thinker uses it as a drop-in replacement for the `llama-server` binary —
no Python changes needed.

**What it helps with on 16GB**: long-context KV cache compression for models
that already fit in RAM (VibeThinker-3B with 32k context, Qwen 7B with large
codebases). It does NOT make Command R 35B fit — weights alone are ~19.7GB
even at `TQ4_1S` (~4.5 bits/param).

The core finding: **V tolerates aggressive compression, K does not**. Always
keep K at higher precision than V. Recommended default: `--cache-type-k q8_0
--cache-type-v turbo3` (~4.6× V compression, <1.5% PPL loss).

```bash
# Build (Apple Silicon / Metal)
git clone -b feature/turboquant-kv-cache https://github.com/TheTom/llama-cpp-turboquant.git
cd llama-cpp-turboquant
cmake -B build -DGGML_METAL=ON && cmake --build build -j

# VibeThinker-3B with 32k context (fits on 16GB with TurboQuant+ KV compression)
llama-server -m ~/models/vibethinker-3b-q4_k_m.gguf \
  --host 127.0.0.1 --port 8080 -c 32768 -t 6 --jinja \
  --cache-type-k q8_0 --cache-type-v turbo3
```

See AGENTS.md for the full config ladder and the asymmetric K/V compression
details.

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
# Full suite (345 tests, no model servers needed):
python3 -m pytest -q

# Specific subsystems:
python3 -m pytest tests/test_warm_pool.py -q          # warm Docker pool
python3 -m pytest tests/test_trajectory_store.py -q    # verified trajectory store
python3 -m pytest tests/test_clr_scoring.py -q         # CLR scoring + grammar
python3 -m pytest tests/test_routing.py -q             # routing + code specialist

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
  default it uses the warm Docker pool (pre-started containers with `docker exec`
  for 2.5x speedup). Falls back to cold `docker run --rm`, then sbx microVMs.
  If none are available, it **refuses to verify** rather than running untrusted
  code on the host. See `sandbox/README.md` for the full isolation model.
- The job queue is in-process only. For multi-process/multi-node, swap the
  dispatcher for RQ/Celery/Dramatiq.
- The bi-temporal log has hash-chain integrity with strict verification
  (`verify_chain(strict=True)`), but does not use cryptographic signatures.
  For audit-grade use, add signing keys.
- Cache entries from v0.1/v0.2/v0.3 (schema_version < 3) are silently
  rejected on lookup. Delete `clr_result_cache.json` to start fresh.
