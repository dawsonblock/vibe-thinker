# vibe-thinker — project notes for agents

## Verify / test
- Full suite: `python3 -m pytest -q` (970 tests, ~130s, no live servers needed)
- Routing + REPL only: `python3 -m pytest tests/test_routing.py tests/test_repl.py -q`
- Format enforcer + chat transports + repair loop: `python3 -m pytest tests/test_format_enforcer.py -q`
- Citation-backed NLI factual verifier: `python3 -m pytest tests/test_factual_verifier.py -q`
- Encoder NLI judge (optional): `python3 -m pytest tests/test_nli_encoder.py -q`
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
- Full-stack integration (needs live model servers): `python test_full_stack.py`
- A benign `ResourceTracker.__del__` AttributeError prints after pytest exits on
  macOS Python 3.12 — it is multiprocessing teardown noise, NOT a test failure.

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
to AgentDB. The `ShadowVectorStore` (dual-write, primary-read-with-fallback)
was already implemented in v0.3.9; this adds the operational tooling to
actually perform the migration.

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
- To roll back: rename `.bak` files back and restart with `--agentdb-url`
  (shadow mode).

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
- **Opt-in**: `--prefer-encoder-nli` CLI flag /
  `VIBE_THINKER_PREFER_ENCODER_NLI` env var. Default off — the model
  downloads from HuggingFace on first use, so it's not enabled by default.
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
