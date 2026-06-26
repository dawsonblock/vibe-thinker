# vibe-thinker — project notes for agents

## Verify / test
- Full suite: `python3 -m pytest -q` (366 tests, ~60s, no live servers needed)
- Routing + REPL only: `python3 -m pytest tests/test_routing.py tests/test_repl.py -q`
- Warm pool + code verifier: `python3 -m pytest tests/test_warm_pool.py tests/test_code_verifier.py -q`
- Trajectory store: `python3 -m pytest tests/test_trajectory_store.py -q`
- Grammar enforcement: `python3 -m pytest tests/test_clr_scoring.py::TestGrammarEnforcement -q`
- In-process backend + fast-specialist: `python3 -m pytest tests/test_clr_scoring.py::TestInProcessBackend tests/test_clr_scoring.py::TestFastSpecialistPolicy -q`
- Full-stack integration (needs live model servers): `python test_full_stack.py`
- A benign `ResourceTracker.__del__` AttributeError prints after pytest exits on
  macOS Python 3.12 — it is multiprocessing teardown noise, NOT a test failure.

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

## In-process specialist backend (v0.3.5)
Eliminates HTTP overhead for tiny specialists. When `--local-specialist-model`
(or `VIBE_THINKER_LOCAL_MODEL`) is set, the GGUF is loaded directly into the
orchestrator's Python process via `llama-cpp-python` and called through a
thread executor (`loop.run_in_executor`). Auto-preferred over HTTP: if the
load succeeds, `_call_model` bypasses aiohttp entirely. If `llama-cpp-python`
is missing or the load fails, it warns and falls back to HTTP at `--vibe`.
- A single `Llama` instance is NOT thread-safe, so calls are serialized with
  `self._local_lock` (a `threading.Lock`) inside the executor. The asyncio
  semaphore is not the gate for the in-process path.
- Grammar enforcement is wired in: `_CLAIMS_JSON_GRAMMAR` is pre-compiled once
  at init into a `LlamaGrammar` and reused on every extraction call.
- Accepts a local `.gguf` path OR `repo_id/filename.gguf` (HF Hub download).
- Optional dep: `llama-cpp-python`. On Apple Silicon build with Metal:
  `CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python`
- CLI: `--local-specialist-model`, `--local-specialist-n-ctx` (default 4096),
  `--local-specialist-n-threads` (default 8). Env equivalents:
  `VIBE_THINKER_LOCAL_MODEL`, `VIBE_THINKER_LOCAL_N_CTX`,
  `VIBE_THINKER_LOCAL_N_THREADS`.

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

## Hardware constraints learned
- 16GB unified memory is too small for Command R 35B (Q4 weights alone ~20GB).
  TurboQuant compresses the KV cache, not the weights, so it does not rescue this.
- ruvLLM's "SONA self-learning" is stubs and mocks (training loop body is
  `// would go here`, embeddings are zero vectors, inference is mock templates).
  The verified trajectory store is the honest version of this concept.
- The "private o1" framing is aspirational; this is an alpha control-plane prototype
  with hard caps (self-claim confidence ≤0.65; high confidence requires an
  independent verifier). See README "Status: ALPHA SOFTWARE".
