# vibe-thinker — project notes for agents

> **Version-tag convention:** Tags like `v3.2`, `v3.2.1`, `v0.3.9`,
> `v1.1`, `v1.2` etc. refer to **historical internal phase numbering**
> from earlier development cycles. They do NOT correspond to the
> package version (currently `v0.4.6a9`, set in `pyproject.toml`).
> Treat them as historical context labels, not current release markers.

## Verify / test
- Fast core gate: `./scripts/test_core.sh` (~280 curated tests, ~45s — the iteration gate, zero skips)
- Broad local gate: `./scripts/test_local.sh` (~1000+ core-marker tests, ~70s — the pre-release gate)
- Full suite (all markers, needs all optional deps): `python3 -m pytest -q`
- **Release gate profiles** (self-contained venvs, fresh-clone safe):
  - Fast core gate: `./scripts/test_core.sh` (env-aware; standalone creates `.venv-core` + installs `-e ".[dev,test]"`; runs compile + doctor + smoke + a curated fast subset — spine, anti-regression static checks, routing/REPL/cache/scoring/signers/deterministic/math-verifier/format-enforcer/trajectory — ~280 tests, ~45s, zero skips)
  - Broad local gate: `./scripts/test_local.sh` (env-aware; the full ~1000+ core-marker selection, ~70s — the pre-release confidence gate)
  - Compose sidecar gate: `./scripts/test_compose.sh` (starts `redis` + hardened `sni-proxy`, validates HTTP allow, HTTPS allow, blocked domain)
  - Docker sandbox gate: `./scripts/test_docker.sh` (`.venv-docker`, `sandbox`/`requires_docker_gateway` markers)
  - Embeddings gate: `./scripts/test_embeddings.sh` (`.venv-embeddings`, `embeddings` marker)
  - Federation/web gate: `./scripts/test_federation.sh` (`.venv-federation`, `federation`/`web` markers)
  - RuvLLM gate: `./scripts/check_ruvllm.sh` (cargo check + maturin build + import; needs Rust)
  - Full release: `./scripts/release_gate.sh` (build wheel + clean-install smoke + core gate)
- **ZIP release build + self-test**:
  - Build: `python scripts/build_clean_zip.py` (compile + core pytest gate + ZIP; verifies every `.sh` in the ZIP has the +x bit set in `external_attr`)
  - Self-test: `./scripts/test_zip_release.sh dist/vibe-thinker-v<version>.zip` (extracts to a temp dir, checks +x bits, no junk, installs in a fresh venv, runs `bash scripts/test_core.sh`)
- **Anti-regression static checks** (AST-based, no execution):
  - Missing private methods: `python3 -m pytest tests/test_static_missing_self_methods.py -q` (flags `self.<name>()` calls with no defining method on the class)
  - Unreachable dead code: `python3 -m pytest tests/test_static_unreachable_code.py -q` (flags statements after return/raise/break/continue in the same block — the orphaned-method-body bug class)
- **Orchestrator runtime spine**: `python3 -m pytest tests/test_orchestrator_runtime_spine.py -q` (verifies `_run_clr_with_cache` exists on the class and `run()` flows through the real method)
- **Lint**: `ruff check .` (green baseline: B + C4 + W rules; B905 ignored)
- **Smoke + doctor**: `python rfsn_cli.py smoke` (incl. orchestrator spine check) / `python rfsn_cli.py doctor` (Python >=3.11)
- v3.2 verifier golden-set regression suite: `python3 -m pytest tests/test_verifier_golden_set.py -q`
- v3.2 tool-callback scaffold: `python3 -m pytest tests/test_tool_callbacks.py -q`
- v3.2 train-lora diversity stats + gate: `python3 -m pytest tests/test_train_lora.py -q`
- v3.2 federation reputation (Sybil-resistant): `python3 -m pytest tests/test_federation_server.py::TestReputationStore tests/test_federation_server.py::TestReputationEndpoints tests/test_federation_server.py::TestExtractIdentity -q`
- v3.2 wasmtime fuel + wall-clock: `python3 -m pytest tests/test_static_analysis.py::TestWasmtimeFuel -q`
- v3.2 static-fallback gate: `python3 -m pytest tests/test_static_analysis.py::TestStaticFallbackGate -q`
- v3.2 CommonMark fence matcher: `python3 -m pytest tests/test_routing.py::TestExtractPythonBlock -q`
- v3.2 structured-output telemetry: `python3 -m pytest tests/test_clr_scoring.py::TestStructuredOutputTelemetry -q`
- v3.2.1 vector-store precedence + ShadowVectorStore removal: `python3 -m pytest tests/test_vector_store.py -q`
- Routing + REPL only: `python3 -m pytest tests/test_routing.py tests/test_repl.py -q`
- Format enforcer + chat transports + repair loop: `python3 -m pytest tests/test_format_enforcer.py -q`
- Citation-backed NLI factual verifier: `python3 -m pytest tests/test_factual_verifier.py -q`
- Encoder NLI judge (optional, default ON since Phase 3.3): `python3 -m pytest tests/test_nli_encoder.py -q`
- Warm pool + code verifier: `python3 -m pytest tests/test_warm_pool.py tests/test_code_verifier.py -q`
- Trajectory store: `python3 -m pytest tests/test_trajectory_store.py -q`
- Grammar enforcement: `python3 -m pytest tests/test_clr_scoring.py::TestGrammarEnforcement -q`
- In-process backend + pool + fast-specialist: `python3 -m pytest tests/test_clr_scoring.py::TestInProcessBackend tests/test_clr_scoring.py::TestInProcessPool tests/test_clr_scoring.py::TestFastSpecialistPolicy -q`
- Test-feedback loop: `python3 -m pytest tests/test_routing.py::TestCodeSpecialistRouting -q`
- Iterative code repair: `python3 -m pytest tests/test_routing.py::TestCodeSpecialistRouting -k repair -q`
- DPO/SFT exporter: `python3 -m pytest tests/test_export_dpo.py -q`
- Active retrieval: `python3 -m pytest tests/test_retrieval.py -q`
- Schema + Logic verifiers: `python3 -m pytest tests/test_schema_verifier.py -q`
- Dynamic compute limits: `python3 -m pytest tests/test_compute_limits.py -q`
- Trajectory synthesis: `python3 -m pytest tests/test_trajectory_synthesis.py -q`
- AgentDB migration: `python3 -m pytest tests/test_migration.py -q`
- Network allow-listing: `python3 -m pytest tests/test_network_allowlist.py -q`
- Static analysis fallback (v2.0): `python3 -m pytest tests/test_static_analysis.py -q`
- Envoy sidecar + DNS pinning (v2.0): `python3 -m pytest tests/test_envoy_sidecar.py -q`
- Sandbox nonce anti-spoofing: `python3 -m pytest tests/test_code_verifier.py::TestNonceAntiSpoofing -q`
- Bi-temporal log HMAC signatures: `python3 -m pytest tests/test_bitemporal_log.py::TestHmacSignatures -q`
- Ed25519 + signer abstraction: `python3 -m pytest tests/test_signers.py -q`
- Vector store + AgentDB abstraction: `python3 -m pytest tests/test_vector_store.py -q`
- RuvLLM adapter + CLI flags: `python3 -m pytest tests/test_ruvllm_adapter.py -q`
- Federated job queue: `python3 -m pytest tests/test_federated_queue.py -q`
- Factual verifier NLI + negation + offline RAG: `python3 -m pytest tests/test_factual_verifier.py -q`
- Job queue disk persistence: `python3 -m pytest tests/test_job_queue.py::TestDiskPersistence -q`
- Mutation testing (vacuous test detection, Phase 3.1): `python3 -m pytest tests/test_mutation.py -q`
- Hardware guardrails (OOM prevention, Phase 4.1): `python3 -m pytest tests/test_hardware_guardrail.py -q`
- Federation zombie claim detection (Phase 4.2): `python3 -m pytest tests/test_federation_zombie.py -q`
- Redis federation heartbeat + reaping (Phase 4.2): `python3 -m pytest tests/test_redis_federation.py::TestRedisHeartbeat tests/test_redis_federation.py::TestRedisReapStaleClaims -q`
- Full-stack integration (needs live model servers): `python test_full_stack.py`
- A benign `ResourceTracker.__del__` AttributeError prints after pytest exits on
  macOS Python 3.12 — it is multiprocessing teardown noise, NOT a test failure.
- **Compose `sni-proxy` hardening**: the Docker Compose stack uses the Python
  SNI proxy (`sandbox/sni_proxy.py`) as the HTTP/HTTPS CONNECT proxy for the
  sandbox. The `sni-proxy` service is hardened with read-only root filesystem,
  `cap_drop: ALL`, `no-new-privileges`, non-root user `1000:1000`, a 64-process
  PID limit, 128 MB memory limit, and a 10 MB tmpfs `/tmp`. The Envoy sidecar
  (`--envoy-sidecar` / `sandbox/envoy_sidecar.py`) is for standalone
  transparent-routing experiments on the host and is **not** used as the
  HTTP proxy in compose. Validate with `./scripts/test_compose.sh`.
- **Wheel packaging**: the wheel now ships `web/static/index.html`,
  `sandbox/entrypoint.sh`, all shell scripts, Docker files, and compose files
  as package/data files. `vibe-thinker-ui` is a console entry point for the web
  UI. `finalize-migration` imports from the `agentdb_migration` module, so it
  works from wheel installs. Validate with
  `python3 -m pytest tests/test_wheel_install.py -q`.
- **IP/CIDR enforcement**: `sni_proxy.py` now enforces allow-list entries for
  plain IP addresses and CIDR ranges, not just domains/wildcards.
- **AgentDB-only mode**: when `agentdb_only=True` and `--agentdb-url` is set,
  `CLRResultCache.lookup()` and `VerifiedTrajectoryStore.retrieve()` search the
  AgentDB vector store directly instead of the local embeddings matrix, even
  when the local JSON file is empty or archived.
- **Compose sandbox networking**: `RFSN_DOCKER_NETWORK` / `--docker-network` lets
  executor-spawned containers join an existing Docker network (e.g.
  `vibe-thinker_default`), so they can reach the compose `sni-proxy` service.

## RuFlo integration abstractions (v0.3.9)
Four pluggable abstractions from the Vibe-Thinker + RuFlo Integration Plan.
All are opt-in with backward-compatible defaults — no behavior change unless
the new parameters/flags are used. External ruvnet repos (ruflo/AgentDB,
ruvllm) are NOT required; the abstractions fail-closed to
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
- `ShadowVectorStore` — REMOVED in v3.2.1. The dual-write behavior moved
  into `CLRResultCache` / `VerifiedTrajectoryStore`: when `agentdb_url`
  is set, inserts are mirrored to AgentDB while the local JSON file
  remains the primary read index. Use `--agentdb-only` after running
  `finalize-migration` to make AgentDB the authoritative read index.
`CLRResultCache` and `VerifiedTrajectoryStore` accept `vector_store=`
or `agentdb_url=` (convenience: builds an AgentDBVectorStore directly).
Default is None = unchanged in-memory behavior.

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
Phase 4.1. Enables multi-node reasoning swarms via a Python-native
federation coordinator:
- `LocalJobQueue` — thin wrapper around the existing `JobQueue`. The
  default; zero behavior change.
- `FederatedJobQueue` — pushes jobs to a Python-native federation
  coordinator (`federation_server.py`, a FastAPI app) over mTLS. Any
  idle node on the network can claim and run pending jobs.
  Fail-closed-fallback: when the coordinator is unreachable (server
  down, mTLS certs missing), jobs still run locally via the wrapped
  `LocalJobQueue`. mTLS config via `mtls_cert`, `mtls_key`, `mtls_ca`
  params.
- `federation_server.py` — standalone FastAPI coordinator that any
  node can run: `python3 -m federation_server --mtls-cert node.crt
  --mtls-key node.key --mtls-ca ca.crt`. Implements POST /submit,
  /claim, /complete, GET /jobs, /health. Replaces the stubbed
  exo-federation Rust crate.
- `make_job_queue()` factory: `federation_url` non-empty →
  FederatedJobQueue; empty/None → LocalJobQueue.
The existing `JobQueue` satisfies `BaseJobQueue` structurally
(runtime_checkable Protocol) — no changes needed to use it as a
`BaseJobQueue`.

### Federation zombie claim detection (Phase 4.2)
Heartbeat-based zombie detection prevents jobs from being stuck in
"claimed" state forever when a worker crashes after claiming but before
completing.
- `FederatedJob.heartbeat_at`: updated by the `/heartbeat` endpoint
  while a worker is actively processing a job. Initialized on claim.
- `InMemoryFederationState.heartbeat(job_id, worker_id)`: updates the
  timestamp, validates worker_id matches (prevents stale workers from
  extending re-queued claims).
- `InMemoryFederationState.reap_stale_claims(timeout=300)`: scans
  claimed jobs, re-queues those whose heartbeat (or claimed_at
  fallback) is older than the timeout. Transitions claimed → pending,
  clears claim fields.
- `RedisFederationState`: implements the same `heartbeat()` and
  `reap_stale_claims()` methods using Redis hashes + sorted sets.
  `heartbeat_at` is stored in the job hash and read by
  `_job_from_hash`.
- `POST /heartbeat` endpoint (federation_server.py): workers call this
  periodically. `POST /api/jobs/heartbeat` (web/app.py): same for the
  web UI coordinator.
- Background reaper task: runs every `claim_timeout / 2` seconds
  (default timeout 300s), logs reaped job IDs.
- `federated_queue.py._claim_and_report`: sends heartbeats every 60s
  while a job is running. Stops after 3 consecutive failures
  (coordinator may have re-queued the job). Heartbeat loop is
  cancelled in a finally block.

### AgentDB-only mode (Phase 4.3)
`--agentdb-only` CLI flag (or `VIBE_THINKER_AGENTDB_ONLY` env) for
post-cut-over AgentDB-only mode. With `--agentdb-url` alone, inserts
are mirrored to AgentDB but reads still use the local JSON file. After
running `finalize-migration` (which archives local JSON files), restart
with `--agentdb-only` to make AgentDB the authoritative read index.
Fail-closed: searches return empty if AgentDB is down. Requires
`--agentdb-url` (warns and falls back to in-memory numpy if set without
it). `agentdb_only` is threaded through the orchestrator →
CLRResultCache and VerifiedTrajectoryStore.

### Hardware guardrails (Phase 4.1)
`hardware_guardrail.py` prevents OOM crashes by checking model size
against available RAM before loading. `check_model_fits_ram()` estimates
model RAM (file size + KV cache + safety multiplier × pool_size) and
compares against available RAM (via psutil, or a 4GB fallback when
psutil is absent). Wired into `VibeThinkerCLRAsync._init_local_backend`:
before loading a local .gguf model, the guardrail checks whether it
fits. If not, the load is refused and the CLR falls back to HTTP
(instead of crashing with OOM). Non-local paths (HuggingFace repo_ids)
skip the guardrail (can't estimate size without a network fetch). The
error message includes actionable remediation (smaller model, reduce
pool_size, reduce n_ctx, install psutil).

### Mutation testing for vacuous test detection (Phase 3.1)
`verifiers/mutation.py` with 5 syntactic mutation operators
(`flip_arithmetic`, `flip_comparison`, `return_none`, `swap_constants`,
`drop_statement`). `mutate_code()` tries each operator in order and
returns the first mutation that produces syntactically valid Python
that differs from the original. Wired into
`_run_code_specialist_verified` at both candidate PASSED return points:
before accepting a verified=1.0 score, the winning code is mutated and
re-run. If the mutated (broken) code still passes, the tests are
vacuous — the candidate is rejected and the test-feedback loop is
triggered. Fail-safe: if the mutation check itself errors, the
candidate is accepted (infrastructure error doesn't reject a passing
candidate).

### Z3/SMT translation retry loop (Phase 3.2)
`LogicVerifier.validate_constraints()` checks whether constraint
strings parse as valid Z3 expressions WITHOUT running the full
verification. Returns None if all parse, or an error message describing
the first failure. `_translate_logic_constraints_with_retry()` wraps
the translation with a parse-validation + retry loop (default 2
retries). When the generalist's constraints fail to parse as Z3, the
specific parse error is fed back for a corrected attempt. This
separates "bad translation" (retryable) from "bad answer" (not
retryable — that's a verification failure). Z3 boolean functions
(`And`, `Or`, `Not`, `Implies`, `If`, `Xor`) are exposed directly in
the eval namespace so constraint strings like `Implies(a, b)` work
without a `z3.` prefix.

### NLI encoder default (Phase 3.3)
The encoder-only NLI judge (`EncoderNLIJudge`) is now the DEFAULT for
factual verification when available, instead of requiring
`--prefer-encoder-nli` opt-in. `select_verifier()` and the orchestrator
constructor default `prefer_encoder_nli=True`. Added `--no-encoder-nli`
CLI flag (and `VIBE_THINKER_NO_ENCODER_NLI` env) to explicitly disable
it. Fallback chain: EncoderNLIJudge (if available) → LLM judge →
fail-closed.

### Explicit sandbox network mode (v0.4.6a1)
`--sandbox-network {auto,none,best-effort-proxy,enforced-gateway}` CLI
flag (env `VIBE_NETWORK_MODE`, default `auto`) for explicit opt-in to the
sandbox network mode. `auto` maps to `None` (preserves auto-detect:
BEST_EFFORT_PROXY when an allow-list is present, DISABLED otherwise) —
zero behavior change. An explicit choice overrides auto-detection so the
operator's intent is never silently inferred from the allow-list.
`DISABLED` ignores the allow-list entirely. `BEST_EFFORT_PROXY` is NOT a
security boundary (bypassable via raw sockets / direct IP). Only
`ENFORCED_GATEWAY` may be treated as egress enforcement, and only after
bypass tests pass. Threaded through the orchestrator constructor's
`network_mode=` param → `executor.set_network_mode()`. Doctor warns for
both networked modes.

### Trajectory store embedding-model injection (v0.4.6a1)
`VerifiedTrajectoryStore.__init__` accepts `embedding_model=` (any object
with an `.encode(texts)` method returning numpy arrays). When injected
(and no explicit mode/vector store is pinned), it forces EMBEDDINGS mode
and uses the injected model directly, so the real semantic store/retrieve
path runs without sentence-transformers. Unit tests inject a deterministic
bag-of-words fake model. Defaults preserve existing behavior
(`embedding_model=None` → shared sentence-transformers model).

## Robust answer extraction (v0.4.1)
The system relies on regex to parse `\boxed{...}` from model output. Two
fixes:
1. **Whitespace-tolerant `\boxed` regex.** Models sometimes output
   `\boxed {42}` (space before the brace). The regex now uses
   `\\boxed\s*\{` to handle this. Fixed in `verifiers/math_verifier.py`
   (`_extract_boxed`) and `vibe_clr_async.py` (`_extract_boxed_answer`
   and the fallback in `_extract_claims_and_answer`).
2. **Structured JSON output schema.** A GBNF grammar
   (`STRUCTURED_OUTPUT_GRAMMAR`, now defined in `format_enforcer.py` and
   re-imported into `vibe_clr_async.py` as `_STRUCTURED_OUTPUT_GRAMMAR`)
   forces the specialist to output JSON with distinct keys:
   `reasoning_steps` (array), `boxed_answer` (string|null),
   `code_solution` (string|null). The `parse_structured_output()` static
   method delegates to `format_enforcer.parse_structured_output` and
   extracts these keys directly — no regex scraping needed. When the
   specialist uses this grammar, the orchestrator reads the answer from
   the `boxed_answer` or `code_solution` key. Falls back to regex
   extraction for unstructured outputs (backward-compatible).

## API-agnostic structured outputs (v1.1)
Decouples the structured-output contract from llama.cpp GBNF so the
specialist can target any upstream API (OpenAI, Anthropic, vLLM) without
silently dropping the JSON constraint. Three pieces:

### FormatEnforcer abstraction (`format_enforcer.py`)
A `FormatEnforcer` protocol maps one structured-output contract to each
provider's native enforcement mechanism. Two contracts (`SchemaKind`):
- `STRUCTURED_OUTPUT` — `reasoning_steps`, `boxed_answer`, `code_solution`.
- `CLAIMS` — `claims`, `final_answer` (the claim-extraction grammar).
Three enforcers:
- `LlamaCppEnforcer` — returns the canonical GBNF strings
  (`CLAIMS_JSON_GRAMMAR`, `STRUCTURED_OUTPUT_GRAMMAR`). These are the
  single source of truth; `vibe_clr_async.py` imports them back as
  `_CLAIMS_JSON_GRAMMAR` / `_STRUCTURED_OUTPUT_GRAMMAR` so the
  in-process and `/completion` paths keep byte-identical behavior
  (identity is preserved — `grammar == _CLAIMS_JSON_GRAMMAR` still works).
- `OpenAIEnforcer` — `to_openai_response_format()` returns
  `{"type":"json_schema","json_schema":{"strict":true,"schema":...}}`.
  `strict=False` falls back to `{"type":"json_object"}`.
- `AnthropicEnforcer` — `to_anthropic_tool()` returns a tool definition;
  the transport sets `tool_choice={"type":"tool","name":...}` to force
  the payload as a `tool_use` block. The transport extracts the tool
  `input` and returns it as a JSON string for the shared parser.
All enforcers share `parse_structured_output()` (the JSON shape is
provider-independent) and `repair_prompt()` for the parse-repair loop.
`make_enforcer(kind, transport)` factory.

### Specialist chat-completions transport
The specialist HTTP path was hardcoded to llama-server's `/completion`
endpoint, which OpenAI/Anthropic don't have. New `specialist_transport`
constructor param (CLI `--specialist-transport {completion,openai_chat,anthropic}`,
env `VIBE_THINKER_SPECIALIST_TRANSPORT`):
- `completion` (default) — unchanged llama-server/RuvLLM `/completion`.
- `openai_chat` — POSTs to `/v1/chat/completions` with
  `response_format` from the enforcer. Auth via `Authorization: Bearer`.
- `anthropic` — POSTs to `/v1/messages` with `tools` + `tool_choice`
  from the enforcer. Auth via `x-api-key` + `anthropic-version`.
ChatML is stripped from the raw prompt into a `messages` array (with
optional assistant prefill). `--specialist-api-key` /
`VIBE_THINKER_SPECIALIST_API_KEY` and `--specialist-model-name` /
`VIBE_THINKER_SPECIALIST_MODEL_NAME` configure auth and model name.
Ignored when `--local-specialist-model` is set (in-process backend has
its own path). The key is never logged.

### Parse-repair loop
On transports without native grammar enforcement, the model can emit
malformed JSON. `_call_model_with_repair` wraps `_call_model`: after the
response, it parses with the enforcer; on failure, it feeds the bad text
+ a concrete parse error (`parse_error_detail`) back to the model at
`temperature=0.0` for a corrected attempt. Bounded by `max_parse_repairs`
(default 2, 0 disables; CLI `--max-parse-repairs` / `MAX_PARSE_REPAIRS`).
Fail-closed: if no repair parses, returns `(raw, None)` and the caller
falls back to regex extraction — never fakes a successful parse. Wired
into `_generate_one_trajectory` and `_generate_lightweight_trajectory`
for structured-output calls. Mirrors the code-specialist
`max_repair_attempts` pattern but is distinct (this repairs JSON
parsing, not code bugs).

### Tests
`tests/test_format_enforcer.py` (32 tests): enforcer rendering for all
three transports, byte-identity of the GBNF strings, shared parser
(valid/invalid/markdown/claims/nullish), repair-prompt construction,
the repair loop (recover / cap-respected / disabled / grammar-None),
and the OpenAI + Anthropic transports (native enforcement applied,
ChatML stripped, auth headers, tool_use extraction, text fallback).
Run: `python3 -m pytest tests/test_format_enforcer.py -q`.

## Iterative code repair (v0.4.0)
Extends the multi-candidate code loop (`_run_code_specialist_verified`) with
targeted repair. Previously, candidates that failed an assertion were simply
discarded. Now, when the best candidate fails with a code bug
(`ASSERTION_FAILED` or `IMPORT_ERROR` — a real defect in the candidate, NOT
a broken test), the failing code + the verifier's error message are fed back
to the code specialist via `_CODE_REPAIR_PROMPT` for a corrected attempt.
- Bounded by `max_repair_attempts` (default 2, 0 disables). CLI:
  `--max-repair-attempts` / `MAX_REPAIR_ATTEMPTS` env.
- Distinct from the v0.3.6 test-feedback loop: that fires only when ALL
  candidates fail with `TEST_ERROR` (the test spec itself is broken) and
  retries test generation. Repair fires on candidate code bugs and retries
  code generation. The two are mutually exclusive — `TEST_ERROR` never
  triggers repair, and `ASSERTION_FAILED`/`IMPORT_ERROR` never trigger a
  test-spec retry.
- Fail-closed: if no repair passes, the best-effort unverified result is
  returned with `score 0.0` (never fake verification). The verified result's
  `raw_traces` carries `repair_attempts`; the unverified result carries
  `repair_attempts` = rounds actually attempted.
- Repair candidates are generated in parallel (same `code_candidates` count
  and diverse temperatures as the initial round) and verified concurrently.
  Each repair round's traces are appended to `all_verification_traces` with a
  `repair_attempt` index. The next repair round feeds back the latest
  failing candidate.

## DPO / SFT exporter (v0.4.0)
`scripts/export_dpo.py` turns the system into an automated data-flywheel for
fine-tuning. Verified solutions become the "chosen" column; failed/unverified
completions become the "rejected" column, in standard HuggingFace formats.
- Chosen: drawn ONLY from `verified_trajectories.json` entries that are
  `verified=True` with a deterministic `verification_method` (not
  `self_claims_only`). The trajectory store already enforces this on load;
  the exporter re-checks defensively. Never learns from self-claims.
- Rejected: drawn ONLY from clearly-worse completions in
  `orchestrator_memory.jsonl`:
  - CLR trajectories whose score < `--reject-threshold` (default 0.5) AND
    whose answer differs from the verified chosen (near-ties are NOT labeled
    rejected — that would teach a wrong preference).
  - CLR runs whose `best_score` < `--min-score` (default 0.75, matching the
    cache trust threshold) — the low-confidence best answer is rejected.
  - Code tasks where `raw_traces.verified` is False — the sandbox explicitly
    rejected the candidate.
- Formats: `--format dpo` (`{"prompt","chosen","rejected"}` JSONL),
  `--format sft` (`{"messages":[user,assistant]}` JSONL), or `--format both`
  (writes `<out>.dpo.jsonl` + `<out>.sft.jsonl`).
- `--max-pairs-per-query` (default 3) caps pairs per query so one popular
  query can't dominate. Chosen entries with no matching rejected completion
  are still emitted as SFT (verified data is always safe to learn from).
- Note: the plan referenced a `clr_trace.json` file that does not exist; the
  real trace data lives in `orchestrator_memory.jsonl` (via `log_to_memory`).

## Active retrieval (v0.4.0)
`retrieval.py` provides pluggable search-API backends that fetch real source
text for factual verification. This closes the "honesty gap" where
FactualVerifier always returned `unsupported_factual` because no sources
existed. Now, when a retrieval backend is configured, factual queries fetch
real Google search snippets and feed them to the NLI judge for
ENTAILMENT / CONTRADICTION / NEUTRAL classification.

### Trust model (fail-closed, no epistemic contamination)
- Every backend returns `[]` on ANY failure: missing API key, network error,
  timeout, non-2xx response, malformed JSON, empty results. The orchestrator
  treats `[]` as "no sources" — the FactualVerifier returns
  `unsupported_factual`, which is the honest, unchanged behavior. No backend
  ever fabricates sources or returns hardcoded text.
- Sources are real text snippets from search-engine organic results (title +
  snippet + link). These are genuine web text, not model-generated — the NLI
  judge classifies the model's answer against them, same as a human checking
  a citation.
- Results without a snippet are skipped — a title alone is too thin for the
  NLI judge to classify entailment against.
- API keys are read from constructor args or env vars. They are NEVER
  hardcoded, logged, or committed to the repo.

### Backends
- `SerperBackend` — google.serper.dev (POST, `X-API-KEY` header). Key from
  `api_key` arg or `SERPER_API_KEY` env. Extracts `organic[].snippet` +
  `knowledgeGraph.description`.
- `SearchApiBackend` — www.searchapi.io (GET, `api_key` query param). Key
  from `api_key` arg or `SEARCHAPI_API_KEY` env. Extracts
  `organic_results[].snippet` + `knowledge_graph.description`.
- `None` (default) — no retrieval. Unchanged fail-closed behavior.

### Factory
`make_retrieval_backend(serper_key, searchapi_key, timeout)` — precedence:
explicit serper_key > explicit searchapi_key > `SERPER_API_KEY` env >
`SEARCHAPI_API_KEY` env > None.

### Orchestrator integration
`_build_verifier_context` is now async (it was sync; the only caller,
`_run_clr_with_cache`, is already async). For `task_type == "factual"`, it
calls `await self._retrieval_backend.search(query)` and puts the results in
`context["sources"]`. If the backend is None, returns [], or raises, no
sources are set — the verifier returns `unsupported_factual`. Math and code
tasks never trigger retrieval. The orchestrator's `__init__` accepts
`retrieval_backend=` (default `_UNSET` → auto-detect from env vars).

### CLI flags
`--serper-key` / `SERPER_API_KEY` env, `--searchapi-key` /
`SEARCHAPI_API_KEY` env. When set, the CLI prints which backend is active.
Keys are never logged.

## Schema + Logic verifiers (v0.4.0)
Two new deterministic verifiers expand the set of tasks that can be
independently verified beyond math, code, and factual.

### SchemaVerifier (`verifiers/schema_verifier.py`)
Validates structural conformance of an answer against a JSON-schema subset
or regex pattern. Deterministic — no model calls.
- Supported context keys: `schema` (JSON-schema dict with type, properties,
  required, items, enum, minimum, maximum, minLength, maxLength, pattern,
  additionalProperties), `pattern` (regex fullmatch), `expected_keys`
  (shortcut for required keys), `format` (`"json"` default, `"yaml"`,
  `"text"`).
- All provided checks must pass (AND semantics). No schema/pattern/keys ->
  `verified=False` (honest — no criteria). Parse error -> `verified=False`
  with the error. Match -> `verified=True`, score 1.0.
- JSON uses stdlib `json`; YAML requires optional PyYAML (fail-closed when
  absent). `bool` is correctly excluded from `integer`/`number` checks.
- Wired into `select_verifier(task_type="schema")`.

### LogicVerifier (`verifiers/logic_verifier.py`)
Validates logical constraints via Z3/SMT. Z3 is a proof tool, not a model —
satisfaction is mathematical, not approximate.
- Context keys: `constraints` (list of Z3 assertion strings, e.g.
  `["x > 0", "x + y == 10"]`), `variables` (name -> sort: `"Int"`,
  `"Real"`, `"Bool"`), `values` (name -> numeric, the answer's values).
- Trust model: Z3 not installed -> `verified=False`, method
  `"smt_unavailable"` (never falls back to a weaker check). No constraints
  -> `verified=False`. UNSAT -> `verified=False` (problem is infeasible).
  SAT + values satisfy all constraints -> `verified=True`, score 1.0, with
  the Z3 model as evidence.
- Requires optional `z3-solver` package: `pip install z3-solver`.
- Wired into `select_verifier(task_type="logic")`.
- **v0.4.1: Z3 constraint translation.** `_build_verifier_context` now
  has a `task_type == "logic"` branch that prompts the generalist to
  translate the natural-language query into a JSON block with
  `constraints`, `variables`, and `values` (Z3-compatible). The
  `_translate_logic_constraints` method parses the JSON and feeds it
  into the verifier context. Fail-closed: if the generalist is
  unreachable or returns malformed JSON, no constraints are set — the
  verifier returns `verified=False` (honest — we can't verify what we
  can't parse).

## Dynamic compute limits (v0.4.0)
The sandbox's memory and timeout were previously hardcoded (128m / 5.0s for
code, 10s for the verifier). Now `route_structured` outputs a
`compute_limits` dict that the CodeVerifier uses to size the sandbox
dynamically — a data-processing script gets 512m / 30s, a one-liner gets
64m / 5s.
- `_suggest_compute_limits(query, task_type)` starts from per-task-type
  defaults, then bumps memory (128m -> 256m -> 512m) and timeout (+10s per
  keyword, capped at 60s) when heavy-computation keywords are detected
  (dataframe, pandas, numpy, torch, matrix, recursive, simulation, etc.).
- The CodeVerifier reads `compute_limits` from the verifier context and
  passes `timeout`/`memory_limit` to the executor. Absent compute_limits ->
  uses the verifier's own defaults (backward-compatible).
- `_build_verifier_context` passes `compute_limits` into the context for
  code tasks only (math/factual/schema/logic don't need sandbox sizing).

## Trajectory synthesis / memory pruning (v0.4.0)
The verified trajectory store accumulates overlapping solutions over time.
`synthesize_trajectories` finds clusters of highly-similar verified
trajectories (cosine similarity >= 0.85), asks the generalist to distill
each cluster into a single "master trajectory" (a general rule capturing
the common pattern), stores the master, and removes the raw entries —
pruning memory without losing the distilled knowledge.
- `VerifiedTrajectoryStore.find_clusters()` — greedy agglomerative
  clustering on the embedding matrix. Returns clusters of >= min_cluster_size
  (default 3) entries above the similarity threshold.
  **v0.4.1: chunked computation for large N.** For N <= 512 (the default
  max_entries), the full N×N similarity matrix is computed in one shot.
  For N > 512, chunked computation keeps memory bounded at
  O(chunk_size × N) instead of O(N²), using normalized dot-product
  similarity. This prevents the event loop from locking up when the
  memory vault grows to 10,000+ entries.
- `VerifiedTrajectoryStore.store_synthesized()` — stores the master with
  `verification_method="synthesized"`, `verified=False`, `synthesized=True`,
  and `source_queries` for provenance.
- `VerifiedTrajectoryStore.remove_entries()` — removes the raw entries and
  rebuilds the embedding matrix.
- `HybridReasoningOrchestrator.synthesize_trajectories()` — the full loop:
  find clusters, call generalist with `_SYNTHESIS_PROMPT`, store masters,
  remove raws. Bounded by `max_clusters` (default 5). Fail-closed: if the
  generalist fails for a cluster, the raw entries are kept (no data loss).
- **Trust model (no epistemic contamination):** synthesized masters are
  `verified=False` with `verification_method="synthesized"`. The existing
  `is_cache_entry_trustworthy` check in `retrieve()` filters them out —
  they are NEVER served as few-shot "verified examples." They exist for
  provenance/audit and potential future re-verification, not as substitute
  evidence. Only independently-verified entries are eligible for clustering;
  synthesized entries are never re-synthesized.

## AgentDB migration script + finalize-migration (v0.4.0)
One-shot backfill and zero-downtime migration from local JSON cache files
to AgentDB. With `--agentdb-url` the caches mirror inserts to AgentDB
while keeping the local JSON file as the primary read index.
`finalize-migration` verifies AgentDB recall, archives the local JSON
files, and then `--agentdb-only` switches reads to AgentDB.

### `scripts/migrate_to_agentdb.py`
Reads all entries from the local CLR cache and trajectory store JSON files
and pushes their embeddings + metadata into AgentDB. After backfill, runs
a recall check comparing local vs AgentDB search results.
- `--dry-run`: report what would be migrated without writing to AgentDB
- `--verify-only`: skip backfill, just check recall
- `--recall-threshold`: minimum recall to pass (default 0.95)
- Fail-closed: if AgentDB is unreachable, exits with code 1 (no data loss).
  If recall fails below threshold, exits with code 2 (refuses to finalize).
  Does NOT modify the local JSON files — read-only.

### `rfsn_cli.py finalize-migration`
Verifies AgentDB recall, then archives the local JSON files (renamed to
`.bak`) to complete the cut-over to AgentDB-only.
- Fail-closed: if recall is below threshold, refuses to finalize (exit 2)
  and does NOT archive local files (no data loss).
- `--force`: override the recall check (DANGEROUS — data loss risk).
- To roll back: rename `.bak` files back and restart without `--agentdb-only`.

## Network allow-listing (v0.4.0, hardened)
Granular egress filtering for the Docker sandbox. Replaces the binary
`--network=none` / `--network=default` choice with iptables-based
allow-listing: only specified domains, IPs, and CIDRs are reachable from
the sandbox.

**IMPORTANT — current status is "best-effort egress restriction":**
The iptables-based approach provides defense-in-depth but is NOT a
complete network security boundary. Known limitations:
- Domain-to-IP resolution is done at rule-generation time. CDN IPs
  (pypi.org, etc.) change, so rules may break or over-allow shared
  CDN infrastructure. For reliable domain allow-listing, use an
  HTTP(S) proxy that enforces SNI/Host headers instead of IP-based
  rules.
- Wildcard domains (`*.pypi.org`) rely on the DNS allow rule and have
  no IP-level filtering.
- DNS is allowed (to a configurable resolver) — this is an exfil
  channel. Data can be encoded in DNS query names. Restricting DNS
  to a controlled resolver with query logging mitigates but does not
  eliminate this.
- The candidate code runs as non-root with no NET_ADMIN, but the
  entrypoint phase has NET_ADMIN to apply iptables rules. If the
  entrypoint script has a vulnerability, it could be exploited before
  privilege dropping.
- For production-grade network isolation, use host-side enforcement
  (e.g., Docker network plugins, eBPF, or a separate proxy container)
  rather than in-container iptables.

### `sandbox/network_allowlist.py`
`NetworkAllowList` parses a list of allowed destinations and generates
iptables rules that DROP all egress except the allow-listed destinations.
- Supported formats: `pypi.org`, `*.pypi.org` (wildcard), `10.0.0.1` (IP),
  `10.0.0.0/24` (CIDR), `pypi.org:443` (port-specific).
- `from_string("pypi.org:443,10.0.0.0/24")` or `from_file("allowlist.txt")`.
- `generate_iptables_rules(dns_resolver=...)`: returns shell commands that
  (1) allow loopback, (2) allow established/related, (3) allow DNS
  (port 53) — to a specific resolver if `dns_resolver` is set, otherwise
  to any resolver, (4) allow each allow-listed destination, (5) DROP
  everything else (IPv4), (6) deny all IPv6 (ip6tables DROP policy).
- Fail-closed: empty allow-list = deny all (same as `--network=none`).
  Unresolvable domains are skipped (fail-closed for that domain, not the
  whole execution). Wildcard domains rely on the DNS allow rule (no
  IP-level filtering).

### Purpose-built sandbox image (`sandbox/Dockerfile`)
The sandbox uses a purpose-built Docker image (`vibe-thinker-sandbox`)
that includes iptables, ip6tables, and curl baked in at build time —
no `apt-get` at runtime. This closes the TOCTOU window where the
container has open network before firewall lockdown.
- Build: `docker build -f sandbox/Dockerfile -t vibe-thinker-sandbox:latest .`
- Creates a non-root `sandbox` user (uid 1000) for candidate code.
- Uses an entrypoint script that applies firewall rules as root, then
  drops to the sandbox user before exec'ing the candidate command.

### Entrypoint script (`sandbox/entrypoint.sh`)
The entrypoint runs as root (container starts as root), applies firewall
rules from the `VT_IPTABLES_RULES` env var (base64-encoded), then drops
privileges to the `sandbox` user and exec's the candidate command.
Security properties:
1. Firewall rules are applied BEFORE candidate code runs (no TOCTOU).
2. Candidate code runs as uid 1000 (sandbox user), NOT root.
3. Candidate code has no NET_ADMIN capability (dropped by Docker after
   the entrypoint's `runuser` call).
4. IPv6 is denied via ip6tables DROP policy (prevents IPv6 bypass).
5. DNS can be restricted to a specific resolver via `VT_DNS_RESOLVER`.
6. If an iptables rule fails, the entrypoint exits (fail-closed) —
   candidate code never runs without firewall protection.

### DockerSandboxExecutor integration
When an allow-list is set, the executor uses `--network=default` +
`--cap-add=NET_ADMIN` (for the entrypoint's iptables phase only) and
passes firewall rules via the `VT_IPTABLES_RULES` env var. The
entrypoint applies rules, drops to the sandbox user, then exec's the
candidate code. The candidate code has no NET_ADMIN and cannot modify
the firewall. Without an allow-list, the executor uses `--network=none`
(unchanged behavior). Audit evidence (rules hash, network mode, DNS
restriction) is attached to the `ExecutionResult.evidence` dict.

### CLI flags
`--network-allowlist "pypi.org:443,10.0.0.0/24"` (comma-separated) or
`--network-allowlist-file allowlist.txt` (one per line, `#` comments).
`--dns-resolver 8.8.8.8` (restrict DNS to a specific resolver).
`--sandbox-image vibe-thinker-sandbox:latest` (override the Docker image).
Env: `RFSN_NETWORK_ALLOWLIST`, `RFSN_NETWORK_ALLOWLIST_FILE`,
`RFSN_DNS_RESOLVER`, `RFSN_SANDBOX_IMAGE`.

## SNI-aware egress proxy (v0.4.1)
Solves the CDN IP rotation problem with iptables-based allow-listing.
Modern CDNs (Fastly, Cloudflare) rotate IPs constantly — iptables rules
resolved at generation time break. The SNI proxy inspects the domain in
the TLS ClientHello (SNI) or HTTP Host header and allows/denies based on
the domain, not the IP.

### `sandbox/sni_proxy.py`
A lightweight async CONNECT proxy that inspects SNI without TLS
interception (no MITM). The proxy reads the cleartext SNI from the
TLS ClientHello and checks it against the allow-list. Allowed
connections are tunneled to the destination; denied connections are
closed.
- Run: `python3 -m sandbox.sni_proxy --port 8888
  --allowlist "pypi.org:443,*.pypi.org:443,files.pythonhosted.org:443"`
- Wildcard support: `*.pypi.org` matches `foo.pypi.org` (single-level).
- No TLS interception — the proxy only reads the SNI (cleartext) and
  tunnels the connection. The actual TLS traffic passes through
  untouched.

### `--proxy-egress` CLI flag
`--proxy-egress 127.0.0.1:8888` (or `RFSN_PROXY_EGRESS` env). When set,
the sandbox container routes traffic through the proxy via
HTTP_PROXY/HTTPS_PROXY env vars instead of using iptables. No
NET_ADMIN capability needed. Audit evidence includes `network_mode:
"proxy"` and the proxy address.

## Python-native federation server (v0.4.1)
Replaces the stubbed `exo-federation` Rust crate with a simple,
maintained Python implementation. The Rust crate had no working
networking layer and relied on archived post-quantum crypto deps
(pqcrypto-kyber, pqcrypto-internals from PQClean).

### `federation_server.py`
A FastAPI app that any node can run to become a federation coordinator.
Implements: POST /submit, /claim, /complete; GET /jobs, /health.
- Run: `python3 -m federation_server --mtls-cert node.crt
  --mtls-key node.key --mtls-ca ca.crt`
- In-memory state (single-coordinator design). For multi-coordinator
  deployments, back with Redis or a shared database.
- mTLS: server presents cert and verifies client certs against CA.
- `--no-tls` flag for dev mode.

### FederatedJobQueue worker loop (v0.4.1)
`FederatedJobQueue.start()` now launches a background claim loop that
polls the coordinator for pending jobs every 2 seconds. When the local
node has spare capacity, it claims a job and submits it locally for
execution. This is the "pull" side of the federation — the "push" side
(submit publishing) was already implemented.

## Rust dependency probe (v0.4.0)
Probed `ruvllm 2.3.0` and `exo-federation 0.1.1` from crates.io.
**exo-federation has been replaced** by a Python-native federation
server (`federation_server.py`). The Rust crate was stubbed (no working
networking, archived PQ crypto deps). The probe findings are kept below
for historical reference.
Both compile and link cleanly. Full report at `docs/rust_dependency_report.md`.

### ruvllm 2.3.0 — viable for sidecar
- Real LLM inference via `LlmBackend` trait (`load_model`, `generate`,
  `generate_stream`, `get_embeddings`) backed by candle-core.
- `ServingEngine` wraps a backend with scheduling, batching, KV cache,
  speculative decoding.
- `unsafe` is only in SIMD quantization kernels (NEON/AVX2) — expected.
- **3 cargo audit vulnerabilities** in transitive `rustls-webpki 0.101.7`
  (CRL panic, cert name constraint bypass). Mitigated by loading models
  from local disk (no TLS needed).
- **1 unsound crate** `lru 0.12.5` (Stacked Borrows violation in IterMut).
- Dependency tree pinned and vendored at `rust/probes/ruvllm_exo_probe/vendor/`.
- Next: build `rfsn-ruvllm-sidecar` HTTP binary wrapping `CandleBackend`.

### exo-federation 0.1.1 — NOT viable (replaced by Python-native server)
- `FederatedMesh` networking is **stubbed** — `federated_query` returns
  placeholder data, `SubstrateInstance` is an empty struct.
- Post-quantum crypto stack (`pqcrypto-kyber`, `pqcrypto-internals`) is
  **unmaintained** — PQClean project archived, replaced by `pqcrypto-mlkem`.
- Zero `unsafe` blocks, but the crate is scaffolding, not a working
  federation.
- Next: implement federation in Python with well-maintained crypto.
- **DONE**: `federation_server.py` is a Python-native FastAPI coordinator
  that replaces exo-federation. Standard mTLS, no post-quantum deps.

## Local web UI (v0.4.0)
A FastAPI backend + single-page HTML frontend for running queries,
viewing results, inspecting traces, and browsing stored data.

### Running
```
python3 run_ui.py [--vibe URL] [--generalist URL] [--port 8000]
```
All orchestrator CLI flags are accepted (`--ruvllm-url`, `--code-specialist`,
`--network-allowlist`, `--local-specialist-model`, etc.). The UI server
binds to `127.0.0.1:8000` by default.

### Architecture
- `web/app.py` — FastAPI app wrapping `HybridReasoningOrchestrator`.
  REST endpoints for status/query/jobs/memory/trajectories/audit-log.
  WebSocket at `/ws` for live job updates (pending → running → done/error).
- `web/static/index.html` — single-page UI (no build step, no JS framework).
  Dark theme, tabbed interface: Query, Jobs, Memory, Trajectories, Audit Log.
- `run_ui.py` — entry point that parses CLI flags, builds the orchestrator,
  and launches uvicorn.

### API endpoints
- `GET /api/status` — system config + job counts
- `POST /api/query` — submit a query (`{"query": "...", "force_route": null}`)
- `GET /api/jobs` — list all jobs
- `GET /api/jobs/{id}` — single job detail
- `GET /api/memory?limit=50` — orchestrator memory vault (JSONL)
- `GET /api/trajectories?limit=50` — verified trajectory store
- `GET /api/audit-log?limit=100` — bi-temporal audit log
- `POST /api/synthesize` — synthesize trajectories from memory
- `WS /ws` — live job updates (init message + job_update events)
- `GET /api/docs` — Swagger/OpenAPI docs

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

### Citation-backed NLI (v1.1)
The LLM-judge NLI path was hardened to prevent a hallucinating judge from
fabricating support. The judge prompt now requires JSON:
`{"verdict": "...", "supporting_quote": "..."}`. For an ENTAILMENT
verdict, the verifier performs a **normalized substring check** — the
judge's `supporting_quote` must actually appear in the source text (after
casefolding, whitespace collapse, and quote/punctuation stripping). If
the quote is absent, the verdict is voided — fail-closed.
- Citation verified → `nli_citation_backed`, score **0.8** (above the
  0.75 cache trust threshold — safe because the quote is real).
- Citation mismatch (fabricated quote) → `nli_citation_mismatch`,
  score **0.0**.
- Old-style single-word verdict (no JSON/citation, backward compat) →
  `nli_llm_judge`, score **0.7** (BELOW the 0.75 cache threshold, so
  un-cited entailment can no longer poison the CLR cache — this was the
  v0.4.0 risk: a hallucinating judge at 0.85 could get cached).
- CONTRADICTION (with or without a quote) → `nli_llm_judge`, score 0.0.
  The citation is NOT verified for CONTRADICTION (only ENTAILMENT gets
  the normalized substring check), so the method tag is `nli_llm_judge`
  even when a quote is present. The quote is included in evidence for
  debugging.
- NEUTRAL / judge error / no judge → unchanged fail-closed paths.
The normalization is deliberately conservative (no lemmatization, no
synonym mapping) so a real match is meaningful. Run:
`python3 -m pytest tests/test_factual_verifier.py -q` (26 tests).

### Encoder-only NLI judge (v1.1, optional)
`verifiers/nli_encoder.py` provides an alternative to the LLM-judge NLI
path: a dedicated encoder-only model fine-tuned for Natural Language
Inference (default `cross-encoder/nli-deberta-v3-base`, override via
`VIBE_THINKER_NLI_MODEL`). Encoder-only models output fixed probabilities
for [entailment, neutral, contradiction] — they are not generative, so
they cannot hallucinate a verdict.
- **Optional extra**: `pip install "vibe-thinker[nli]"` (transformers +
  torch). Not a core dependency — follows the project's pattern for
  `cryptography`, `z3-solver`, `llama-cpp-python`.
- **Default ON (Phase 3.3)**: the encoder NLI judge is preferred
  whenever available. `--no-encoder-nli` CLI flag /
  `VIBE_THINKER_NO_ENCODER_NLI` env var disables it. The model
  downloads from HuggingFace on first use (lazy loading).
- **Fail-closed**: when transformers/torch are absent, or the model can't
  load, `select_verifier` falls back to the LLM judge (or
  `nli_unavailable`). `is_available()` checks deps without importing them
  eagerly. The model loads lazily on first `__call__` (construction is
  cheap — no network, no RAM).
- **Honest scoring**: the encoder returns the same JSON shape as the LLM
  judge but with an empty `supporting_quote` (encoder models classify but
  don't extract spans). The FactualVerifier treats this as an un-cited
  verdict — score 0.7, below the 0.75 cache threshold. An encoder NLI
  verdict is stronger than self-claims but does not carry a verifiable
  citation, so it does not get the 0.8 citation-backed score.
- **Determinism**: on CPU the encoder is deterministic. The constructor
  sets `torch.manual_seed(0)` and enables deterministic algorithms when
  `deterministic=True` (default). On GPU/MPS, set seeds explicitly for
  true reproducibility.
- **Low-confidence downgrade**: verdicts below `threshold` (default 0.6)
  are downgraded to NEUTRAL (honest uncertainty → fail-closed).
Run: `python3 -m pytest tests/test_nli_encoder.py -q` (17 tests, no model
download — inference is mocked).

### Process-pool mode for in-process specialist (v1.1, REMOVED in v2.0)
The in-process specialist pool historically used a `queue.Queue` of Llama
instances + `ThreadPoolExecutor` (shared GIL). A `ProcessPoolExecutor`
option was added in v1.1 where each worker loads its own Llama instance
(worker-local global), giving each worker its own GIL. This eliminated
Python-side lock contention under extreme concurrency.

**Removed in v2.0**: The ruvllm_py PyO3 binding releases the GIL
natively, making process-pool mode unnecessary. The
`--local-specialist-pool-kind process` flag, `LocalPoolKind.PROCESS`
enum, and all worker functions were removed. Thread mode is the only
pool kind now.
- **Opt-in**: `--local-specialist-pool-kind process` CLI flag /
  `VIBE_THINKER_LOCAL_POOL_KIND` env var. Default `thread` (unchanged).
- **RAM guardrail**: before starting the process pool, estimates the
  model size from the GGUF file (`_estimate_model_ram_mb`) and compares
  `model_mb * pool_size * 1.5` (KV cache + runtime overhead) against
  available RAM (`_available_ram_mb`, uses psutil if installed, else a
  conservative 4GB cap). If the estimate exceeds available RAM, falls
  back to thread mode with a warning. This prevents the OOM the original
  plan flagged. For HuggingFace repo IDs (not local files), the guardrail
  is skipped with a warning — the user monitors RAM manually.
- **Worker functions** (`_process_pool_worker_init`, `_process_pool_worker_call`):
  module-level functions (picklable). The init function loads one Llama
  + compiles one `LlamaGrammar` per worker (LlamaGrammar is not
  picklable, so it can't be shared across processes). The call function
  resolves the grammar inside the worker (pre-compiled claims grammar or
  on-demand compile) and runs inference.
- **Dispatch**: `_call_model` checks `_local_process_pool` in addition to
  `_local_llm` and `_local_llm_pool`. `_call_model_inprocess` has a
  process-pool branch that submits to the executor and awaits the future.
- **Backend tag**: `in-process-process-pool` (distinct from
  `in-process-pool` for thread mode and `in-process` for single-instance).
- **RuvLLM compatible**: the worker init function handles both
  `llama-cpp-python` and `RuvLLMBinding` (passes the raw GBNF string).
- **Fail-closed**: on any worker error, raises `RuntimeError` (caller
  handles). On pool construction failure, falls back to HTTP.
Run: `python3 -m pytest tests/test_clr_scoring.py::TestProcessPool -q`
(8 tests, mock-based — no real model load, no worker processes spawned).

### RuvLLM direction decision (v1.1)
The original integration plan conflated two RuvLLM integration directions:
(a) a sync `__call__`-compatible binding that drops into the existing
pool as a Llama replacement, and (b) an async batched engine that
bypasses the executor. **Decision: pursue (a), not (b).** The existing
pool infrastructure (thread queue, process pool with RAM guardrail,
per-instance grammar) already handles parallelism — a sync binding
reuses all of it with zero new code paths. An async batched engine would
require a new dispatch path, new tests, and a new concurrency model
bypassing the semaphore/executor guardrails, for negligible latency win.
The process-pool mode (above) already gives each worker its own GIL, so
Python-side lock contention is not a reason to go async. The binding
contract: `__call__(prompt, max_tokens, temperature, stop, grammar)`
returns `{"choices": [{"text": "..."}]}`, grammar is a raw GBNF string.
Documented in `ruvllm_adapter.py` module docstring.

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

## v1.2 remediation (HA federation + RuvLLM binding + enterprise egress)

Three-phase remediation completed. All changes are backward-compatible
(opt-in via flags); the defaults preserve v0.4.1 behavior unless the
new flags are used.

### Phase 1: HA federation (federation_server.py + web/app.py + federated_queue.py)
- **FederationState Protocol**: `InMemoryFederationState` (default) +
  `RedisFederationState` (atomic Lua claim via sorted-set). Factory
  `make_federation_state()`. CLI: `--redis-url` / `--no-redis` on
  `federation_server`. Redis is optional (pip install redis).
- **Broadcaster Protocol**: `LocalBroadcaster` (default) +
  `RedisBroadcaster` (Pub/Sub fan-out for multi-UI-server HA). The
  web UI's WebSocket broadcast now goes through the Broadcaster, so
  multiple UI servers behind a load balancer stay in sync.
- **FederatedJobQueue HA failover**: `--federation-url` accepts a
  comma-separated list of coordinator URLs. The client tries each
  sticky-first and falls over on failure. `make_job_queue()` factory
  passes the list through.
- Tests: `test_redis_federation.py`, `test_web_pubsub.py`, HA failover
  tests in `test_federated_queue.py`. Run:
  `python3 -m pytest tests/test_federation_server.py tests/test_redis_federation.py tests/test_federated_queue.py tests/test_web_federation.py tests/test_web_pubsub.py -q`

### Phase 2: RuvLLM PyO3 binding (ruvllm_py/)
- **ruvllm 2.3.0** crate is published on crates.io and wired into
  `ruvllm_py/Cargo.toml`. The binding compiles in three modes:
  - Stub (default, no inference): `cargo check --release`
  - CPU candle: `cargo check --release --features candle`
  - Apple Silicon Metal: `cargo check --release --features inference-metal`
- **`ruvllm_py/src/lib.rs`**: PyO3 0.22 binding wrapping
  `CandleBackend` + `LlmBackend::generate()`. GIL released during
  generation (`py.allow_threads`). Backend in `Arc<Mutex<>>` for
  thread-safe pool mode. Grammar param accepted (API compat) but
  candle backend doesn't enforce GBNF — the format enforcer handles
  structured output.
- **Process-pool mode deprecated** (v1.2) → **removed** (v2.0): The
  `--local-specialist-pool-kind process` flag was deprecated in v1.2
  and fully removed in v2.0. Superseded by the ruvllm_py binding
  (releases GIL natively, no RAM duplication).
- NOTE: `ruvllm 2.3.0` has a bug where `pub mod claude_flow`
  unconditionally uses `tokio` (gated by `async-runtime`). We work
  around this by enabling `async-runtime` in our Cargo.toml.

### Phase 3: Enterprise egress (sandbox/)
- **SNI-proxy is the default egress mode** when an allow-list is
  present. The sandbox routes traffic through a proxy at
  `DEFAULT_PROXY_EGRESS` (127.0.0.1:8888) via HTTP_PROXY/HTTPS_PROXY
  env vars. No NET_ADMIN cap needed. Solves CDN IP rotation.
- **`--legacy-iptables-egress`** flag (v1.2) → **removed** (v2.0):
  The v0.4.0 iptables path (in-container firewall rules, NET_ADMIN cap)
  was deprecated in v1.2 and fully removed in v2.0. SNI-proxy is now
  the only egress mode.
- **`sandbox/envoy_sidecar.py`** (new): Envoy config generator +
  launcher. Generates SNI-aware tcp_proxy config from a
  NetworkAllowList. Fail-closed if envoy binary not on PATH.
  `--envoy-sidecar` CLI flag auto-launches Envoy as a child process
  with cleanup on exit.
- Tests: `test_envoy_sidecar.py` (16 tests). Integration tests
  (`test_network_integration.py`) test the SNI-proxy egress path. Run:
  `python3 -m pytest tests/test_envoy_sidecar.py tests/test_network_allowlist.py -q`

## v2.0 remediation (deprecated code removal + PyO3 HNSW/SONA + verification hardening)

### Phase 2: Deprecated code removal
- **Process-pool mode removed** from `vibe_clr_async.py` and `rfsn_cli.py`.
  The `--local-specialist-pool-kind process` flag and `LocalPoolKind.PROCESS`
  enum value were removed. The ruvllm_py binding (Phase 3) releases the GIL
  natively, making process-pool mode unnecessary. Thread mode is the only
  pool kind now.
- **iptables egress path removed** from `sandbox/docker_executor.py`. The
  `--legacy-iptables-egress` flag, `set_legacy_iptables_egress()`,
  `_build_firewall_env()`, and `_compute_rules_hash()` methods were removed.
  SNI-proxy is now the ONLY egress mode when an allow-list is present.
  The `execute()` method uses `--entrypoint python3` to bypass the sandbox
  image's iptables entrypoint (no SETGID/SETUID caps needed). Both proxy
  and non-proxy paths use `--cap-drop ALL` + `--security-opt
  no-new-privileges` for hardening.
- `hybrid_orchestrator.py`: `legacy_iptables_egress` and `local_pool_kind`
  constructor params removed. The orchestrator no longer prints the
  "use --legacy-iptables-egress" deprecation message.

### Phase 4: HNSW + SONA PyO3 bindings (ruvllm_py/)
- **`HnswIndex`** class: wraps `ruvllm::ruvector_integration::UnifiedIndex`.
  Constructor: `HnswIndex(dim, m=16, ef_construction=200, ef_search=64)`.
  Methods: `add(id, vector, source, quality_score)`, `search(query, k)`,
  `stats()`. Enables in-process HNSW vector search without an HTTP sidecar.
- **`SonaRecorder`** class: wraps `ruvllm::sona::SonaIntegration`.
  Constructor: `SonaRecorder(hidden_dim=256, embedding_dim=384,
  quality_threshold=0.7)`. Methods: `record(request_id, session_id,
  query_embedding, response_embedding, quality_score, model_index)`,
  `search_patterns(query, limit)`, `trigger_background_loop()`,
  `trigger_deep_loop()`, `stats()`. Enables in-process SONA trajectory
  recording for the data flywheel.
- Both classes have stub implementations (no-op) when built without
  `--features candle`, and real implementations when built with it.
- `chrono` added to `ruvllm_py/Cargo.toml` dependencies (used by
  `Trajectory.timestamp`).

### Phase 5: Verification hardening
- **5.1: Wildcard DNS egress loophole fix** (`sandbox/envoy_sidecar.py`):
  `generate_envoy_config()` now accepts a `dns_resolver` parameter. When
  set, the `dynamic_upstream` cluster uses `STRICT_DNS` with the trusted
  resolver instead of `ORIGINAL_DST` (which connects to the client's
  resolved IP). This closes the wildcard DNS loophole: the proxy resolves
  the SNI hostname via the trusted resolver and connects to that IP —
  not the IP the client resolved. Access logging (JSON format with
  `upstream_host`, `server_name`) added to the tcp_proxy filter for DNS
  resolution auditing.
- **5.2: Code verification static analysis fallback**
  (`hybrid_orchestrator.py`): When the Generalist fails to generate unit
  tests, the code loop now runs a static analysis pass on the candidate
  code. `_static_analysis_fallback()` uses Python's `ast` module to check:
  (1) the code parses cleanly, (2) no restricted imports (os, subprocess,
  socket, etc.). If both pass, assigns a partial heuristic score of 0.4
  (capped — NOT full verification). Route: `code_specialist_static_analysis`.
  If the code has syntax errors or restricted imports, falls back to
  `code_specialist_unverified` with score 0.0.
- **5.3: Factual verification offline RAG fallback**
  (`verifiers/factual_verifier.py`): `FactualVerifier` now accepts an
  `offline_sources` parameter (list of local document strings). When no
  online retrieval sources are available (Serper/SearchApi keys missing)
  AND offline_sources exist AND an LLM judge is configured, the verifier
  uses the NLI judge against the offline documents. Only returns
  `unsupported_factual` if BOTH online and offline sources are empty.

## v3.0 (enterprise swarm hardening)

### Phase 1: Rust build pipeline hardening
- **`ruvllm_py/Cargo.toml`**: The `[patch.crates-io]` block was a
  comment-only no-op. Replaced with explicit documentation of the
  security posture: `rustls-webpki` and `lru` are NOT in the dep tree
  (verified via `cargo tree --features candle`). If a future ruvllm
  version pulls them in, add direct deps to force safe versions.
- `cargo update` run, `Cargo.lock` committed for deterministic builds.

### Phase 2: Rust-native embedding pipeline (SONA decoupling)
- **`RuvLLMBinding.get_embeddings(text, dim=384)`**: New method on
  `ruvllm_adapter.RuvLLMBinding`. Uses character n-gram hashing (not
  semantic — the ruvllm crate's `LlmBackend` trait doesn't expose
  embeddings). Deterministic, L2-normalized, configurable dimension.
  Fallback for when `sentence-transformers` is not installed.
- **Orchestrator embedding source priority** (`_store_if_verified`):
  1. `RuvLLMBinding.get_embeddings()` (Rust-native, no Python deps)
  2. `trajectory_store.model.encode()` (sentence-transformers)
  3. Skip (no embedding source available)
- This decouples SONA from the `sentence-transformers` Python extra.

### Phase 3: Static analysis evasion vector sealing
- **`_static_analysis_fallback`** now checks `ast.Call` nodes for:
  - `__import__('os')` — direct builtin call
  - `importlib.import_module('os')` — method call on importlib
  - `exec()` / `eval()` — dynamic code execution
  - Any `ast.Name` referencing `importlib` or `__builtins__`
- All evasion vectors result in score 0.0 (same as restricted imports).
- Tests: `tests/test_static_analysis.py::TestStaticAnalysisEvasionVectors`
  (8 tests covering each vector + clean code still passes).

### Phase 4: Zero-trust federation (encrypted payloads)
- **`--federation-secret` CLI flag** / `FEDERATION_SECRET` env var:
  shared secret for AEAD encryption of federation payloads.
- **`FederatedJobQueue._encrypt_payload()` / `_decrypt_payload()`**:
  Fernet (AES-128-CBC + HMAC-SHA256) encryption layer. When enabled,
  all job queries and results are encrypted before transmission.
  Nodes without the secret see only opaque ciphertext in Redis/HTTP.
- **`federation_server.py`**: `create_federation_app()` accepts
  `federation_secret=` and decrypts incoming payloads on /submit and
  /complete endpoints.
- Tests: `tests/test_federated_queue.py::TestFederationEncryption`
  (5 tests: roundtrip, passthrough, error on missing secret, ciphertext
  doesn't contain plaintext).

### Phase 5: SONA gossip protocol (Distributed Brain)
- **`/api/sona/sync` endpoints** on `federation_server.py`:
  - `POST /api/sona/sync`: Workers export learned patterns (centroid,
    cluster_size, avg_quality) + stats. Coordinator merges into a
    global set (dedup by pattern ID).
  - `GET /api/sona/sync`: Workers retrieve the global aggregated
    pattern set from all nodes.
- **Orchestrator methods**:
  - `_sona_export_patterns()`: Uses `search_patterns` with a zero
    vector to retrieve all local patterns.
  - `_sona_import_patterns(patterns)`: Records global patterns as
    trajectories in the local SONA engine.
  - `sona_sync_once()`: One sync cycle (export + import).
  - `_sona_sync_loop()`: Background loop for periodic sync.
- **CLI flags**: `--sona-sync-url` / `SONA_SYNC_URL`,
  `--sona-sync-interval` / `SONA_SYNC_INTERVAL` (default 3600s).
- Tests: `tests/test_sona_gossip.py` (8 tests: endpoint POST/GET,
  pattern dedup, orchestrator export/import, disabled state).

## Phase 5: Hardening (security, reliability, operational maturity)

### Track 1: Security hardening
- **Vulnerable dependency pins**: `aiohttp>=3.10.11` (CVE-2024-52304),
  `fastapi>=0.109.1` (CVE-2024-24762 ReDoS), `cryptography>=43.0.1`
  (CVE-2024-6119). Updated in `requirements.txt` and `pyproject.toml`.
- **Safe Z3 constraint parsing**: Replaced bypassable `eval()` in
  `verifiers/logic_verifier.py` with `_SafeZ3Evaluator` — an AST-based
  whitelist evaluator that only allows arithmetic, comparisons, boolean
  ops, and calls to predefined Z3 functions. Prevents code injection via
  Python introspection (e.g. `().__class__.__base__.__subclasses__()`).
- **API key authentication** (`web_security.py`): Shared middleware for
  both `federation_server.py` and `web/app.py`. Header-based (`X-API-Key`),
  constant-time comparison (`hmac.compare_digest`), env var fallback
  (`VIBE_THINKER_API_KEY`). CLI flag `--api-key` on federation server.
  Health endpoint exempted.
- **CORS configuration**: Explicit allowed origins (localhost by default).
  Configurable via `allowed_origins` parameter.
- **Rate limiting**: In-memory sliding-window per-IP limiter. Configurable
  via `--rate-limit` CLI flag (requests per minute, 0=disabled).
- **Request body size limits**: Configurable via `--max-body-bytes`.
  Returns 413 when exceeded.
- **Input validation on federation endpoints**: Query length (max 10000),
  job_id format (alphanumeric/underscore/hyphen), worker_id format,
  priority range, error message length. Returns 400 on invalid input.
- **Input validation on web UI**: Query length limit (10000 chars).
- **Error message sanitization**: Web UI returns generic "internal error"
  to clients; full exception logged server-side only.
- **Federation secret logging**: Removed mention of "federation_secret"
  from log messages; now says "encryption configured" generically.
- **Sandbox entrypoint.sh**: Replaced `eval "$rule"` with validated
  execution — rules must start with `iptables ` and contain no shell
  metacharacters (`;|&\`$(){}`). Prevents command injection via
  `VT_IPTABLES_RULES` env var.
- **TLS warnings**: Federation server now emits `warnings.warn()` when
  running without TLS or without API key auth.

### Track 2: Reliability
- **Reaper task shutdown handler**: Added `@app.on_event("shutdown")` to
  cancel the background reaper task. Previously the reaper leaked on
  server shutdown.
- **Timeout consistency**: Reaper timeout changed from 300s to 180s
  (3x heartbeat interval of 60s). This is well below the job execution
  timeout (600s), preventing the race where a slow worker is reaped
  while still legitimately processing.
- **State-transition guard in `complete()`**: Both `InMemoryFederationState`
  and `RedisFederationState` now reject completions for jobs already in
  "done" or "error" state. Prevents stale workers from overwriting
  results after a reaper re-queue.
- **CLR cache save() lock**: Added `threading.Lock` to `CLRResultCache`
  and `VerifiedTrajectoryStore` `save()` methods. Prevents concurrent
  autosave calls from racing on the atomic write.
- **ShadowVectorStore retry tracking**: Failed secondary writes are now
  tracked in `_failed_writes` set. `reconcile_failed_writes()` method
  for periodic reconciliation. `failed_write_count` property for
  monitoring.
- **Z3 value type validation**: `_translate_logic_constraints` now
  validates that values are numeric (int/float/bool) before returning.
  Non-numeric values (e.g. "seven") are dropped with a warning instead
  of causing a confusing Z3 substitution error.

### Track 3: Operational maturity
- **Structured logging module** (`vt_logging.py`): Component-based
  loggers with consistent format (timestamp, level, component, message).
  Level control via `VIBE_THINKER_LOG_LEVEL` env var (default INFO).
  Existing `print()` statements can be incrementally migrated.
- **Config module** (`vt_config.py`): Centralized hardcoded timeouts,
  intervals, and limits with env var overrides. Covers federation
  timing (claim poll, heartbeat, job timeout, reaper timeout), sandbox
  timeout, HTTP timeouts, hardware guardrail parameters, and input
  validation limits. All overridable via `VIBE_THINKER_*` env vars.
- **Shared embedding model singleton** (`get_shared_embedding_model`):
  `CLRResultCache`, `VerifiedTrajectoryStore`, and `EmbeddingRouter` now
  share a single `SentenceTransformer` instance per model name, reducing
  memory usage by ~200-600MB. Cache clearable via
  `_clear_embedding_model_cache()` (for tests).
- **Connection pooling**: `FederatedJobQueue` now creates a shared
  `aiohttp.ClientSession` in `start()` and reuses it in the claim loop,
  avoiding TCP handshake overhead on every poll. Closed in `stop()`.
- **Shared serialization utility** (`serialization.py`):
  `serialize_for_json()` and `serialize_result_dict()` extracted from
  duplicated code in `web/app.py` and `federated_queue.py`.
- **Mutation operator refactor**: `_swap_constants` in
  `verifiers/mutation.py` refactored from unreadable one-liners to
  clear multi-line logic with proper indentation and comments.

## v3.2 (production-hardening pass)

A revision of the multi-month improvement plan, grounded in the actual
codebase state. Several original-plan items were over-engineered or
rested on misreads of the current code; this pass implements the
critique-grounded versions. All changes are backward-compatible unless
noted.

### ResourceTracker teardown noise fix (`tests/conftest.py`)
Suppresses the benign `AttributeError` from `multiprocess.resource_tracker.
ResourceTracker.__del__` at interpreter shutdown on macOS Python 3.12.
The noise was masking real CI failures. The conftest wraps `__del__` to
swallow `AttributeError` only (any other exception type still propagates).
Patches both the `multiprocess` package (the actual source of the noise)
and stdlib `multiprocessing`. macOS-only; no-op elsewhere.

### Static-analysis fallback gate + score lowering (Step 3.1 revised)
`_static_analysis_fallback` partial score lowered 0.4 -> 0.2 (less
epistemic weight for an unexecuted signal). The fallback is now GATED
behind `--allow-static-fallback` (env `VIBE_THINKER_ALLOW_STATIC_FALLBACK`,
default OFF). When the gate is closed and no sandbox is available, the
code route returns `verified=False, score=0.0` instead of the heuristic.
When open (local dev), the route is renamed `code_specialist_unverified_static_only`
(so consumers can't confuse it with any verified path). The original
plan's "return hard 0.0 always" was MORE epistemically wrong (it claims
"we know this is bad" when the truth is "we know it parses + has no
restricted imports"). The gate preserves the signal for dev without
letting it influence production confidence.

### CommonMark-aware fence matcher (Step 2.2 revised)
`_extract_python_block` now parses fenced code blocks per the CommonMark
spec: 3+ backticks with <=3 leading spaces, closing fence must match the
opener's backtick count and have no info string. The old
`stripped.startswith("```")` toggle broke on (a) `>3`-backtick lines
closing a `3`-backtick block early, and (b) nested backticks in markdown
explanations flipping the open/close state mid-block and swallowing real
code. 5 new tests cover the CommonMark cases.

### Wasmtime fuel + wall-clock belt-and-suspenders (Step 3.2 revised)
`_wasmtime_sandbox_fallback` now enables Wasmtime fuel metering
(`Config.consume_fuel = True` + `store.set_fuel(VIBE_WASM_FUEL)`, default
1e9 instructions) for deterministic infinite-loop detection at the
CPU-instruction level. The wall-clock timeout (`VIBE_WASM_WALL_CLOCK_TIMEOUT`,
default 10s) is kept as the OUTER belt — the wasm call runs in a thread
executor so `asyncio.wait_for` can abandon a hang that fuel can't see
(e.g. a host syscall). Out-of-fuel traps return `score=0.0` with a
"fuel" message; wall-clock timeouts return `score=0.0` with a "timeout"
message. Fuel enabling is guarded for older wasmtime versions (falls
back to wall-clock only).

### Sybil-resistant federation reputation (Step 4.3 revised)
`federation_server.py` gains a `ReputationStore` (EMA in [0,1]) keyed on
the mTLS client cert's cryptographic identity (subject CN -> fingerprint
-> raw worker_id fallback), NOT the self-asserted `worker_id`. This
prevents Sybil attacks: one cert = one reputation; minting new
`worker_id`s doesn't inflate influence. `/complete` updates reputation
from the result's INDEPENDENT `verified` verdict (True: +1, False: -2
[hallucination is worst], error: -1). `/api/sona/sync` downweights a
low-reputation node's gossip imports (0 below floor 0.3, else the score
itself) so hallucinated patterns don't propagate. `/api/reputation`
diagnostic endpoint returns the per-identity snapshot. A single failure
does NOT zero a good node (EMA with alpha=0.3 needs ~3 consecutive bad
outcomes to drop from 1.0 to ~0.4).

### Vector-store precedence + ShadowVectorStore deprecation (Step 1.3 revised)
`make_vector_store` now prefers `RuvLLMVectorStore` (in-process Rust HNSW,
zero Python RAM bloat) when the binding is installed AND `dim` is
provided AND no `agentdb_url` is set. `LocalVectorStore` is KEPT as the
zero-build default (NOT deprecated) because it supports `delete` and
`cluster`, which the HNSW binding does NOT expose — deprecating it would
regress those features and force a hard Rust build to run the orchestrator.
`ShadowVectorStore` has been REMOVED. New deployments should use AgentDB
directly via `--agentdb-url` and cut over reads with `--agentdb-only` after
verifying recall.

### Z3-grammar constraint on logic translation (Step 2.3 revised)
New `SchemaKind.LOGIC_CONSTRAINTS` + `LOGIC_CONSTRAINTS_GRAMMAR` (GBNF)
in `format_enforcer.py` forces the generalist to emit valid JSON with
`constraints`/`variables`/`values` keys during the NL->Z3 translation
step. `_translate_logic_constraints` passes the grammar to
`_call_generalist` (mapped to `response_format` on the chat endpoint,
`grammar` on the /completion fallback). The prompt adds an explicit Z3-
syntax allowlist (And/Or/Not/Implies/If/Xor, no `z3.` prefix). The
existing `_translate_logic_constraints_with_retry` + `validate_constraints`
loop is kept — the grammar handles JSON shape, the validator handles Z3
semantics. Replaces the original plan's "fine-tuned Translator-Specialist
model" (10x the operational cost for a marginal win) with few-shot +
schema-validated retry.

### Structured-output + regex fallback telemetry (Step 2.1 revised)
`vibe_clr_async.py` gains `_structured_success_count` /
`_structured_fallback_count` counters and a `structured_fallback_rate()`
method. Both the full-trajectory and lightweight-trajectory fallback
points increment the fallback counter and log the rate when regex
`\boxed{}` scraping is used instead of structured JSON. This is the
metric-gated rollout: the regex fallback is KEPT (not removed "un-
bypassably" on day one) so a single upstream schema bug doesn't mark
everything `verified=False`. Once the fallback rate drops below a
threshold (e.g. 2%) over a sufficient sample, the regex path can be
removed. `structured_telemetry()` returns the raw counters for logging.

### ruvllm_py per-token logprobs (Step 1.1)
`ruvllm_py/src/lib.rs` exposes `Engine.complete_with_logprobs()` which
runs forward passes with `candle_transformers::models::quantized_llama`
and applies `log_softmax` over the vocab to return per-token logprobs.
`Engine.supports_logprobs()` and module-level `SUPPORTS_LOGPROBS`
enable capability detection. `scripts/turboquant_ppl_check.py` uses
this so `eval_inprocess()` returns a real `PplResult` when the binding
is built with the `candle` feature; raises `NotImplementedError`
(fail-closed) when the binding is absent or built without candle.
Backward-compat: `complete()` is unchanged. The fail-closed contract
for the absent-binding case is preserved in the tests.
End-to-end verified: built with `maturin build --release --features
inference-metal` on Apple Silicon, installed into pyenv 3.12, and ran
`eval-inprocess` against Llama-3.2-3B-Instruct-Q5_K_M.gguf (with a
tokenizer.json from HuggingFace alongside it) — produces a real
PplResult with source="ruvllm_py". Note: candle's `quantized_llama`
only supports Llama/Mistral GGUF architectures (NOT Qwen2/Phi/Gemma).
The test `test_inprocess_logprobs_returns_ppl_result` passes with
`RUVLLM_PPL_TEST_MODEL` set (timeout 300s for model load + inference).

### Unified Docker deployment stack (Step 1.2)
New files: `docker-compose.yml` (redis/envoy/agentdb sidecars always
start + 3 orchestrator profiles: `--profile cpu`/`gpu`/`ruvllm`),
`Dockerfile` (multi-stage python:3.12-slim), `docker/agentdb.Dockerfile`
(placeholder — no published AgentDB image exists), `docker/envoy/envoy.yaml`
(SNI-aware egress matching `sandbox/envoy_sidecar.py`), `.dockerignore`,
`docs/deployment.md`. All healthchecks include `start_period` + retries.
Honesty notes: AgentDB/RuvLLM have no published images (must be supplied
by the user); GPU profile requires a `.gguf` model volume mount. This is
GREENFIELD (no existing compose/Dockerfile to consolidate).
Verified: `docker compose config` passes for all 3 profiles; hadolint
clean (only version-pinning warnings); envoy.yaml matches
`envoy_sidecar.py` generate_envoy_config() structure; all runtime
.py modules confirmed in Dockerfile COPY via AST import scan; llama-server
uses a single port (8080) for both specialist and generalist (not separate
ports — llama-server doesn't support multi-port).

### Manual train-lora subcommand (Step 4.1 revised)
`scripts/train_lora.py` is the HUMAN-IN-THE-LOOP boundary of the data
flywheel. It reads an exported DPO/SFT dataset, computes and prints
DIVERSITY STATS (task-type distribution, unique-query count,
verification-method breakdown, selection-bias warning when one task type
is >70%), and requires an explicit `--yes` flag to proceed past the
stats review. It then shells out to `mlx_lm.lora` (Apple Silicon) or
`unsloth` (CUDA) and writes a LoRA adapter to `--output-dir` and STOPS.
NO auto-training (the original plan's "train when X patterns accumulate"
bakes in selection bias toward verifiable task types). NO hot-swap
(llama-cpp-python has no `set_lora()` — it's a model reload, and silently
swapping a learned adapter into production is an unreviewed deployment).
The operator reviews the adapter and loads it manually.
`--dry-run` validates the dataset format against the trainer's expectations
(SFT vs DPO, messages structure) and prints the exact command without
executing — exits 0 on valid, 1 on format error, 3 when no trainer is
detected. `validate_dataset_format()` catches mlx_lm/DPO mismatches and
malformed messages entries before shelling out.

### Inter-turn tool-callback scaffold (Step 4.2 revised)
`tool_callbacks.py` implements INTER-TURN tool calls (generate -> tool
call -> continue), NOT mid-token pausing (which is genuinely hard with
llama-cpp streaming). A `ToolCallback` protocol + `CallbackRegistry` map
tool names to async callables. The specialist signals a tool call via
`<tool_call name="...">query</tool_call>` markers; `run_with_callbacks`
drives the loop (bounded by `max_rounds`, default 3), executes tools
fail-safe (unknown tool / exception -> error note fed back, not
crashed), and strips markers from the final output. Without a registry,
the loop is a no-op (backward-compat). This is a SCAFFOLD — the
orchestrator wires the registry (e.g. "generalist_query" -> a
FactualVerifier RAG call) and passes it to the specialist invocation.

### Verifier golden-set regression suite
`tests/test_verifier_golden_set.py` — a curated set of (verifier, query,
answer, context, expected_verified) tuples for MathVerifier and
LogicVerifier. Guards against regressions: if a verifier starts
accepting a hallucinated answer or rejecting a correct one, this suite
catches it. Meta-test (`TestGoldenSetCompleteness`) ensures the golden
set covers BOTH True and False outcomes for each verifier (a one-
direction set is vacuous). Already documented a real parser limitation
(`_extract_numeric` doesn't parse scientific notation — `1e2` -> 2.0).
