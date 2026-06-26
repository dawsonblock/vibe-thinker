# vibe-thinker — project notes for agents

## Verify / test
- Full suite: `python3 -m pytest -q` (502 tests, ~90s, no live servers needed)
- Routing + REPL only: `python3 -m pytest tests/test_routing.py tests/test_repl.py -q`
- Warm pool + code verifier: `python3 -m pytest tests/test_warm_pool.py tests/test_code_verifier.py -q`
- Trajectory store: `python3 -m pytest tests/test_trajectory_store.py -q`
- Grammar enforcement: `python3 -m pytest tests/test_clr_scoring.py::TestGrammarEnforcement -q`
- In-process backend + pool + fast-specialist: `python3 -m pytest tests/test_clr_scoring.py::TestInProcessBackend tests/test_clr_scoring.py::TestInProcessPool tests/test_clr_scoring.py::TestFastSpecialistPolicy -q`
- Test-feedback loop: `python3 -m pytest tests/test_routing.py::TestCodeSpecialistRouting -q`
- Sandbox nonce anti-spoofing: `python3 -m pytest tests/test_code_verifier.py::TestNonceAntiSpoofing -q`
- Bi-temporal log HMAC signatures: `python3 -m pytest tests/test_bitemporal_log.py::TestHmacSignatures -q`
- Ed25519 + signer abstraction: `python3 -m pytest tests/test_signers.py -q`
- Vector store + AgentDB abstraction: `python3 -m pytest tests/test_vector_store.py -q`
- RuvLLM adapter + CLI flags: `python3 -m pytest tests/test_ruvllm_adapter.py -q`
- Federated job queue: `python3 -m pytest tests/test_federated_queue.py -q`
- Factual verifier NLI + negation: `python3 -m pytest tests/test_factual_verifier.py -q`
- Job queue disk persistence: `python3 -m pytest tests/test_job_queue.py::TestDiskPersistence -q`
- Full-stack integration (needs live model servers): `python test_full_stack.py`
- A benign `ResourceTracker.__del__` AttributeError prints after pytest exits on
  macOS Python 3.12 — it is multiprocessing teardown noise, NOT a test failure.

## RuFlo integration abstractions (v0.3.9)
Four pluggable abstractions from the Vibe-Thinker + RuFlo Integration Plan.
All are opt-in with backward-compatible defaults — no behavior change unless
the new parameters/flags are used. External ruvnet repos (ruflo/AgentDB,
ruvllm, exo-federation) are NOT required; the abstractions fail-closed to
the existing local behavior when the sidecars/bindings are absent.

### Ed25519 audit-log signatures (signers.py)
Upgrades the bi-temporal log from HMAC-SHA256 (symmetric, stdlib) to
Ed25519 (asymmetric, SLSA L2 compliant) — the plan's Phase 1.1. The
`Signer` protocol abstracts the signature scheme:
- `HmacSigner` — the v0.3.8 default (stdlib, unchanged behavior).
- `Ed25519Signer` — asymmetric via the optional `cryptography` package.
  Each node has a private key; the public key can verify but NOT forge.
  `Ed25519Signer.generate()` creates a keypair; `.from_private_key_hex()`
  loads a persisted key; `.from_public_key_hex()` is verify-only.
- `make_signer()` factory: precedence Ed25519 private > Ed25519 public >
  HMAC key > none.
`BiTemporalAuditLog` accepts `signer=`, `ed25519_private_key_hex=`, or
`ed25519_public_key_hex=`. The old `signing_key=` parameter still works
(builds an HmacSigner). The signature field uses a scheme prefix
(`ed25519:` or `hmac-sha256:`) so `verify_chain` dispatches correctly
even during a migration. The old `_sign_entry()` function is retained
for backward compat (tests import it directly).

### Vector store abstraction (vector_store.py)
Abstracts the semantic similarity search behind a `VectorStore` protocol
— the plan's Phase 1.2. Moves the embedding matrix out of the
orchestrator process into a purpose-built vector index (AgentDB):
- `LocalVectorStore` — in-memory numpy + sklearn cosine similarity.
  The default; reproduces the exact existing behavior.
- `AgentDBVectorStore` — HTTP sidecar (RuFlo/AgentDB,
  `POST /v1/vector/search`). Fail-closed: returns `[]` when the sidecar
  is down (caller falls back). Moves lookups from ms to <25µs with zero
  RAM bloat on the Python side.
- `ShadowVectorStore` — dual-write, primary-read-with-fallback. The
  plan's "Shadow Mode" migration step: write to both local JSON and
  AgentDB, read from local first; once AgentDB recall is verified, cut
  over to AgentDB-only.
`CLRResultCache` and `VerifiedTrajectoryStore` accept `vector_store=`
or `agentdb_url=` (convenience: builds a ShadowVectorStore wrapping the
local matrix). Default is None = unchanged in-memory behavior.

### RuvLLM inference backend (ruvllm_adapter.py)
Documents and wires the RuvLLM Rust inference engine with TurboQuant KV
cache compression — the plan's Phase 2.1. Two integration modes:
- **HTTP sidecar** (zero-code, recommended start): RuvLLM exposes the
  same OpenAI-compatible API as llama-server. The `--ruvllm-url` CLI
  flag (or `RUVLLM_URL` env) overrides `--vibe` to point at the RuvLLM
  port. The existing HTTP path handles it unchanged.
  `RuvLLMHTTPBackend.recommended_start_command()` prints the start
  command with TurboQuant flags (`--cache-type-k q8_0 --cache-type-v
  turbo3`). TurboQuantConfig presets: `TURBOQUANT_SAFE` (f16/turbo4),
  `TURBOQUANT_CONSERVATIVE` (q8_0/turbo4), `TURBOQUANT_DEFAULT`
  (q8_0/turbo3), `TURBOQUANT_AGGRESSIVE_V` (q8_0/turbo2).
- **In-process PyO3 binding** (zero-HTTP-overhead, optional):
  `RuvLLMBinding` wraps a hypothetical `ruvllm_py` extension built from
  the Rust crate via PyO3/maturin. When installed, `vibe_clr_async.
  _init_local_backend` prefers it over llama-cpp-python (same pool
  mode). When not installed (current state), it raises ImportError and
  the llama-cpp-python path is used unchanged.
`is_ruvllm_binding_available()` checks for the extension.

### Fast code-specialist preset (CLI flag)
`--fast-code-specialist` (or `RFSN_FAST_CODE_SPECIALIST` env) — the
plan's Phase 2.2. Bumps `CODE_CANDIDATES` to 15 for ultra-fast 0.5B
code models (ruvltra). At 0.5B speed (~100+ tok/s), 15 parallel
candidates cost roughly what 2 cost on a 3B model. Does NOT hardcode a
model path — pair with `--code-specialist <ruvltra-url>` or
`--local-specialist-model <ruvltra.gguf>`. Warns if no code specialist
is configured.

### Federated job queue (federated_queue.py)
Abstracts the job queue behind a `BaseJobQueue` protocol — the plan's
Phase 4.1. Enables multi-node reasoning swarms via exo-federation:
- `LocalJobQueue` — thin wrapper around the existing `JobQueue`. The
  default; zero behavior change.
- `FederatedJobQueue` — pushes jobs to an exo-federation network
  (ruvnet/exo-federation, Rust crate) over mTLS. Any idle node on the
  network can claim and run pending jobs. Fail-closed-fallback: when
  the federation is unreachable (sidecar down, mTLS certs missing),
  jobs still run locally via the wrapped `LocalJobQueue`. mTLS config
  via `mtls_cert`, `mtls_key`, `mtls_ca` params.
- `make_job_queue()` factory: `federation_url` non-empty →
  FederatedJobQueue; empty/None → LocalJobQueue.
The existing `JobQueue` satisfies `BaseJobQueue` structurally
(runtime_checkable Protocol) — no changes needed to use it as a
`BaseJobQueue`.

## Security hardening (v0.3.8)
Ten fixes from a full codebase audit. All stdlib-only, no new dependencies.

## Security hardening (v0.3.8)
Ten fixes from a full codebase audit. All stdlib-only, no new dependencies.

### Sandbox nonce anti-spoofing (CRITICAL)
The old test harness printed a hardcoded `ALL_TESTS_PASSED` string that
candidate code could trivially forge (just `print("ALL_TESTS_PASSED")`).
Worse, a candidate that overrode `sys.exit = lambda x: None` could force
a 0 exit code while tests failed, and the verifier checked only stdout.
Fix: `sandbox/base.py:build_test_harness()` generates a per-execution
nonce (`secrets.token_hex(8)`) passed via the `VT_TEST_NONCE` env var.
The harness prints `VT_PASS_<nonce>` only if the test block completes
without raising. The verifier (`code_verifier.py:_interpret_test_result`)
requires BOTH the exact nonce marker in stdout AND exit_code 0. The nonce
is not in the script source, so candidate code cannot guess it. All four
executors (docker, warm_pool, local, sbx) use the centralized harness.

### Safe AST math evaluator
`math_solver.py` no longer uses `eval()`. Replaced with `_safe_eval()`,
a whitelisting AST evaluator that permits only `BinOp`, `UnaryOp`,
`Constant`, and `Name` nodes. The plan's suggestion of `ast.literal_eval()`
was incorrect — it rejects arithmetic expressions.

### Test-spec vacuity rejection
`_generate_test_spec` now validates the test spec via `_validate_test_spec`
(AST check): every `assert` must reference a `Name` or `Call` node.
`assert True`, `assert 1 == 1`, etc. are rejected. If the first attempt
produces a vacuous spec, the generalist gets one retry with feedback.

### Bi-temporal log HMAC signatures (optional)
`BiTemporalAuditLog(signing_key=...)` adds an `hmac-sha256:` signature
field to each entry. `verify_chain(strict=True)` with a key configured
mathematically proves the log was written by a process holding the key
(tamper-proof, not just tamper-evident). No key = unchanged behavior.

### Factual verifier NLI judge + negation detection
`FactualVerifier(llm_judge=...)` uses the orchestrator's generalist as an
NLI judge (ENTAILMENT/CONTRADICTION/NEUTRAL). The lexical fallback now
detects negation polarity mismatches ("Paris is NOT the capital" vs a
source saying "Paris is the capital") and rejects them. The orchestrator
wires `self._call_generalist` as the judge automatically.

### Session init race fix
`_get_session()` is now async with a double-checked `asyncio.Lock`. The
shared `aiohttp.ClientSession` is eagerly created in `start()` to avoid
concurrent first-requests creating overlapping sessions.

### Warm pool zombie reaping
Before each execution, the warm pool kills all processes except PID 1
and the reaping shell itself, using `/proc` + shell builtins (no
`ps`/`pkill` needed in `python:3.12-slim`). Prevents leftover background
processes from consuming the `--pids-limit=64` budget.

### Hardened Python block extraction
`_extract_python_block` now handles `py` (abbreviated) and no-tag fences
in addition to `python`. Test-spec generation retries once with feedback
if extraction/validation fails.

### LRU eviction for trajectory store + CLR cache
Both `VerifiedTrajectoryStore` and `CLRResultCache` now evict by
`(access_count, score)` instead of score alone. Access counts are
persisted to disk, so rarely-retrieved entries are evicted across
restarts. Both stores already had disk persistence + bounded size
(since v0.3.4) — this adds access-frequency awareness.

### Job queue disk persistence (crash recovery)
`JobQueue(persist_path=...)` persists pending/running jobs to a JSONL
file. On `start()`, non-terminal jobs are recovered and re-queued. No
Redis needed — same durability pattern as the bi-temporal log.

## Model servers (OpenAI-compatible HTTP)
The orchestrator talks to llama-server / mlx_lm.server over HTTP. Three endpoints:
- `VIBE_THINKER_URL` (default 8080): specialist — VibeThinker-3B CLR (math/reasoning)
- `GENERALIST_URL` (default 8081): generalist — any OpenAI-compatible server
- `CODE_SPECIALIST_URL` (default empty): OPTIONAL dedicated code model (e.g.
  ruvltra-claude-code-0.5b). When set, `task_type=="code"` routes here for plain
  generation (no CLR); math still uses VibeThinker CLR. CLI flag: `--code-specialist`.
- `CODE_CANDIDATES` (default 6): number of parallel candidates in the multi-
  candidate code loop. Higher is better for fast 0.5B models. CLI flag:
  `--code-candidates`.

## Multi-candidate code loop (v0.3.3)
When `CODE_SPECIALIST_URL` is set AND a code verifier is available (default:
CodeVerifier with Docker sandbox), code tasks run through a verified loop:
1. Generalist writes unit-test asserts for the query (test spec).
2. Code specialist (ruvltra) generates N candidates in parallel (diverse temps).
3. CodeVerifier runs each candidate against the test spec in the Docker sandbox.
4. First candidate that passes (ALL_TESTS_PASSED) wins → score 1.0, verified=True.
5. If none pass → returns first candidate, score 0.0, verified=False (fail-closed).
Requires Docker running + `python:3.12-slim` image. Without Docker, the verifier
fail-closes (verified=False) and the loop returns unverified best-effort.

## Verified trajectory store (v0.3.4)
Self-improving few-shot memory. When a deterministic verifier confirms an answer,
the (query, verified_answer, method, score, task_type) tuple is stored
semantically. On future similar queries, verified trajectories are retrieved as
few-shot context and prepended to the model prompt — improving first-attempt
success rate without weight updates. This is RAG over verified solutions, not
fake SONA/MicroLoRA weight updates.
- Only `verified=True` with deterministic verifiers are stored (never self_claims_only)
- Entries re-checked for trustworthiness on retrieval (defensive against corruption)
- `task_type` filtering prevents math examples polluting code queries
- Config: `RFSN_USE_TRAJECTORY_STORE` (default true), `TRAJECTORY_STORE_PATH`,
  `--no-trajectory-store` CLI flag

## Warm Docker pool (v0.3.4)
`WarmDockerPool` in `sandbox/warm_pool_executor.py` keeps N containers running
and uses `docker exec` instead of `docker run --rm` for each verification.
Measured speedup: 2.5x (0.494s → 0.197s per verification). Same security
hardening as cold runs. `/tmp` cleaned between executions. `select_executor`
prefers warm pool by default. Orchestrator manages lifecycle via
`start()` / `cleanup()`.

## Grammar enforcement (v0.3.4)
GBNF grammar passed to llama-server `/completion` endpoint for CLR claim
extraction. Forces valid JSON output (`{"claims": [...], "final_answer": "..."}`
). Prevents small models from producing malformed JSON that causes trajectory
scoring to fail. Applied to VibeThinker-3B extraction path. Regex fallback
parser retained as defense-in-depth. When the in-process backend is active
(see below), the same grammar is enforced natively via `LlamaGrammar`.

Health check: `curl http://127.0.0.1:<port>/health` → `{"status":"ok"}`

## In-process specialist backend (v0.3.5, pool in v0.3.6)
Eliminates HTTP overhead for tiny specialists. When `--local-specialist-model`
(or `VIBE_THINKER_LOCAL_MODEL`) is set, the GGUF is loaded directly into the
orchestrator's Python process via `llama-cpp-python` and called through a
thread executor (`loop.run_in_executor`). Auto-preferred over HTTP: if the
load succeeds, `_call_model` bypasses aiohttp entirely. If `llama-cpp-python`
is missing or the load fails, it warns and falls back to HTTP at `--vibe`.
- **Pool mode (v0.3.6)**: `--local-specialist-pool-size N` loads N separate
  `Llama` instances into a `queue.Queue`. Each call checks out one instance,
  runs it in a thread executor, and returns it to the pool. This enables true
  parallel inference — N trajectories run simultaneously on N instances. For a
  0.5B model (~398MB each), 4 instances cost ~1.6GB. Threads are divided:
  `n_threads // pool_size` per instance. Default is 1 (single instance + Lock).
- **Single mode** (pool_size=1): one `Llama` instance, serialized with
  `self._local_lock` (`threading.Lock`). No parallelism but lowest memory.
- Grammar enforcement is wired in: `_CLAIMS_JSON_GRAMMAR` is pre-compiled once
  at init into a `LlamaGrammar` and reused on every extraction call.
- Accepts a local `.gguf` path OR `repo_id/filename.gguf` (HF Hub download).
- Optional dep: `llama-cpp-python`. On Apple Silicon build with Metal:
  `CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python`
- CLI: `--local-specialist-model`, `--local-specialist-n-ctx` (default 4096),
  `--local-specialist-n-threads` (default 8), `--local-specialist-pool-size`
  (default 1). Env equivalents: `VIBE_THINKER_LOCAL_MODEL`,
  `VIBE_THINKER_LOCAL_N_CTX`, `VIBE_THINKER_LOCAL_N_THREADS`,
  `VIBE_THINKER_LOCAL_POOL_SIZE`.

## Test-feedback loop (v0.3.6)
In `_run_code_specialist_verified`, if ALL candidates fail with `TEST_ERROR`
(the test harness itself crashed — e.g. the test references a function the
solution doesn't define, or the test has a syntax error), the generalist gets
ONE retry to rewrite the tests with the error fed back as context. This
distinguishes "bad tests" from "bad code":
- `ASSERTION_FAILED` = candidate is wrong (tests ran, code gave wrong answer) → no retry
- `IMPORT_ERROR` = candidate code failed to define/import → no retry
- `TEST_ERROR` = the test spec itself is broken → retry once with feedback
The retry prompt includes the error message so the generalist can fix the
specific problem. Max 2 attempts (initial + 1 retry). If the retry also fails,
returns best-effort unverified (score 0.0, fail-closed).

## Performance fixes (v0.3.7)
Seven fixes from a full codebase audit. The most impactful:
- **Parallel claim verification**: `_verify_claims` was sequential (5 serial
  LLM calls per trajectory, 40 total with k=8). Now uses `asyncio.gather`.
  Single claim failures return verdict 0 instead of crashing the trajectory.
- **Parallel candidate verification**: the code loop verified 6 candidates
  one-by-one while 3 warm containers sat idle. Now uses `asyncio.gather` —
  all candidates verified concurrently (~3x speedup).
- **Shared HTTP session**: `_call_generalist` / `_call_code_specialist` /
  `_call_specialist_plain` each created a new `aiohttp.ClientSession` per
  call. Now one shared session via `_get_session()`, closed in `cleanup()`.
- **Warm pool `cleanup()` fix**: was passing `["rm", "-f", name]` (missing
  `"docker"`) to `create_subprocess_exec` — containers leaked on every
  shutdown since v0.3.4.
- **Timeout recycling**: a timed-out container is now recycled to a fresh
  state (killed process may leave zombies/held file handles).
- **Recycle failure handling**: if `docker run` fails twice during recycle,
  the container is removed from the pool rather than leaving an invalid entry.
- **In-process session skip**: `_run_adaptive` / `_run_static` use
  `contextlib.nullcontext()` instead of creating an unused `aiohttp.ClientSession`
  when the in-process backend is active.

## Fast-specialist adaptive profile (v0.3.5)
`make_fast_specialist_policy(k=15)` returns an `AdaptivePolicy(3, 5, max_k=15)`
(all capped at `k`) tuned for ultra-tiny fast specialists (e.g. ruvltra 0.5B, ~100+ tok/s). At
that speed, shotgun-sampling 15 trajectories costs roughly what 2 cost on a
3B model. The `self_claim_cap` stays 0.65 — a fast model agreeing with itself
more often is NOT independent verification; only a deterministic verifier
exceeds the cap. Gated behind `--fast-specialist` / `RFSN_FAST_SPECIALIST`
(default off). Do NOT enable with a 3B+ specialist on 16GB RAM — 15 parallel
trajectories will thrash/OOM. The default 1/2/6 policy is unchanged. An
explicit `policy=` to `VibeThinkerCLRAsync` overrides the flag.

### Local setup used on this machine (Apple M2 Pro, 16GB RAM)
GGUF models stored in `~/models`:
- `vibethinker-3b-q4_k_m.gguf` (1.8G) — from `oussaber/VibeThinker-3B-Q4_K_M-GGUF`
- `ruvltra-claude-code-0.5b-q4_k_m.gguf` (379M) — from `ruv/ruvltra`

Start commands:
```
llama-server -m ~/models/vibethinker-3b-q4_k_m.gguf --host 127.0.0.1 --port 8080 -c 8192 -t 6 --jinja
mlx_lm.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --host 127.0.0.1 --port 8081
llama-server -m ~/models/ruvltra-claude-code-0.5b-q4_k_m.gguf --host 127.0.0.1 --port 8082 -c 4096 -t 4 --jinja
```
Use `hf download <repo> <file> --local-dir ~/models` (NOT `huggingface-cli download`,
which prints help on the installed version).

## TurboQuant+ (optional llama-server backend)
[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) is a
fork of llama.cpp that adds Walsh-Hadamard rotated polar quantization for both
KV cache (`turbo2`/`turbo3`/`turbo4`) and weights (`TQ3_1S`/`TQ4_1S`). It is
additive — every existing llama.cpp quant/model/backend still works. vibe-thinker
uses it as a drop-in replacement for the `llama-server` binary; no Python changes
needed.

### What it helps with on 16GB M2 Pro
TurboQuant+ does NOT make Command R 35B fit (weights alone are ~19.7GB at
`TQ4_1S` ~4.5 bits). What it DOES help with is **long-context KV cache
compression** for models that already fit in RAM:
- VibeThinker-3B with `-c 32768` (32k context) instead of the default 8192
- Qwen 2.5 Coder 7B with `-c 32768` for large codebase ingestion
- The asymmetric finding: **V tolerates aggressive compression, K does not**.
  Always keep K at higher precision than V.

### Build (Apple Silicon / Metal)
```bash
git clone -b feature/turboquant-kv-cache https://github.com/TheTom/llama-cpp-turboquant.git
cd llama-cpp-turboquant
cmake -B build -DGGML_METAL=ON && cmake --build build -j
# Use build/bin/llama-server instead of the stock binary
```

### Recommended KV cache configs (start light, then compress)
| Config | `--cache-type-k` | `--cache-type-v` | When |
|---|---|---|---|
| Safest start | `f16` | `turbo4` | First contact with any new model |
| Conservative | `q8_0` | `turbo4` | Verified safe, want a memory win |
| **Recommended default** | `q8_0` | `turbo3` | Most models — ~4.6× V compression, <1.5% PPL loss |
| Aggressive V | `q8_0` | `turbo2` | Memory-bound long context, after validating step 3 |

Never start with symmetric turbo K compression (both sides `turbo*`) — that's
where models break. See the fork's
[asymmetric-kv-compression paper](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md).

### Example: VibeThinker-3B with 32k context (TurboQuant+)
```bash
# Stock llama-server: 32k context would blow up RAM on 16GB
# TurboQuant+ with asymmetric KV: fits comfortably
llama-server -m ~/models/vibethinker-3b-q4_k_m.gguf \
  --host 127.0.0.1 --port 8080 -c 32768 -t 6 --jinja \
  --cache-type-k q8_0 --cache-type-v turbo3
```

## Hardware constraints learned
- 16GB unified memory is too small for Command R 35B. Even with TurboQuant+
  weight quantization (`TQ4_1S` ~4.5 bits/param), 35B weights alone are
  ~19.7GB — still over budget. KV cache compression doesn't help when the
  weights don't fit. See the TurboQuant+ section below for what DOES help
  on 16GB (long-context KV compression for models that already fit).
- ruvLLM's "SONA self-learning" is stubs and mocks (training loop body is
  `// would go here`, embeddings are zero vectors, inference is mock templates).
  The verified trajectory store is the honest version of this concept.
- The "private o1" framing is aspirational; this is an alpha control-plane prototype
  with hard caps (self-claim confidence ≤0.65; high confidence requires an
  independent verifier). See README "Status: ALPHA SOFTWARE".
