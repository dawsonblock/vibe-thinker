# Changelog

## v0.4.6a2

Post-audit cleanup + allowlisted gateway container. The ENFORCED_GATEWAY
mode now automatically starts and validates a real gateway container
running the SNI egress proxy — the "not yet validated" gap from v0.4.6a1
is closed.

### Added
- `sandbox/docker_executor.py`: the executor now automatically starts a
  gateway container running the SNI egress proxy (`sandbox.sni_proxy`)
  on `python:3.12-slim` with the `sandbox/` directory mounted as a
  read-only volume. The gateway is connected to both the default Docker
  bridge (for internet access) and the `--internal` network (for sandbox
  access). The sandbox container routes traffic through the gateway via
  `HTTP_PROXY`/`HTTPS_PROXY` env vars pointing at the gateway's IP on
  the internal network. Raw socket egress is blocked by the `--internal`
  network — the proxy is the only path out. The gateway is stopped and
  removed in `cleanup()`.
- `tests/test_sandbox_network_enforcement.py`: 3 new integration tests
  in `TestGatewayEgressEnforcement` that run real Docker containers:
  - `test_gateway_allows_allowlisted_domain`: verifies an allowlisted
    domain (example.com) returns HTTP 200 through the proxy.
  - `test_gateway_blocks_non_allowlisted_domain`: verifies a
    non-allowlisted domain (httpbin.org) is blocked by the proxy.
  - `test_gateway_blocks_raw_socket_bypass`: verifies raw socket egress
    (bypassing the proxy) is blocked by the `--internal` network.

### Fixed
- `sandbox/sni_proxy.py` `_handle_http()`: headers were not terminated
  with `\r\n`, causing the remote server to wait for more headers and
  never respond. Also removed a duplicate request line in the forwarded
  headers (the original first_line was included in the headers variable,
  resulting in two request lines being sent to the upstream).
- `sandbox/docker_executor.py`: fixed Go template hyphen issue — Docker
  network names with hyphens (e.g. `vibe-thinker-gateway-net`) need
  `index()` syntax in `docker inspect --format` instead of dot-notation
  (which interprets hyphens as subtraction).
- `scripts/test_optional.sh`: removed `sandbox` from the marker filter
  (sandbox tests now belong to `test_docker.sh`, which needs Docker +
  the sandbox extra). The marker filter is now `logic or embeddings or
  federation or web or nli` — consistent with `optional.yml`.
- `scripts/build_clean_zip.py`: stale example version `v0.4.6a0` →
  `v0.4.6a2`.
- Sandbox wording in `README.md`, `sandbox/README.md`,
  `sandbox/docker_executor.py`, and `sandbox/base.py`: updated to
  reflect that the gateway IS now started and the egress path IS
  validated by integration tests. No longer says "not yet validated".

### Verified end-to-end (macOS, cargo + Docker daemon)
- `test_core.sh`: 248 passed, 32 skipped (27s)
- `test_local.sh`: 1021 passed, 45 skipped (56s)
- `test_docker.sh`: 9 passed (12s) — 6 network isolation + 3 gateway
  egress enforcement
- `check_ruvllm.sh`: PASSED (inference-metal, SUPPORTS_INFERENCE=True)
- `release_gate.sh all`: PASS
- `ruff check .`: clean

### Still experimental
- RuvLLM (Rust inference engine — default build is stub).
- Distributed federation (Redis-backed HA).

## v0.4.6a1

Release-hygiene repair over v0.4.6a0. No new features. Splits the
test gate into fast vs. broad, fixes nested-venv waste, implements
real Docker network enforcement tests, and aligns CI with the local
gate model.

### Fixed
- `scripts/test_core.sh`: now a **fast curated gate** (~250 tests,
  ~30s) covering the historically-regressing failure classes —
  orchestrator runtime spine, anti-regression static AST checks
  (missing-self / unreachable-code), routing, REPL, cache, scoring,
  signers, deterministic check, math verifier, format enforcer,
  trajectory store. Env-aware: reuses an active `VIRTUAL_ENV` instead
  of creating a nested `.venv-core`, eliminating double venv/install
  overhead when called from `release_gate.sh core` or
  `test_zip_release.sh`.
- `scripts/test_local.sh` (new): the **broad local gate** (~1000+
  core-marker tests, ~70s) — the former `test_core.sh` behavior, now
  the pre-release confidence gate.
- `scripts/release_gate.sh`: `phase_core` now calls
  `bash scripts/test_core.sh` which detects the active venv and
  reuses it — no more nested venv creation.
- `tests/test_static_missing_self_methods.py` and
  `tests/test_static_unreachable_code.py`: fixed latent `.venv*`
  exclusion bug. Both static checks only excluded `.venv` and
  `.venv-core` by exact name, so per-profile venvs (`.venv-local`,
  `.venv-docker`, etc.) caused 551 false positives from third-party
  packages. Added a `.venv*` prefix guard (`_is_excluded_part`).
- `.gitignore`: added `.venv-local/` and `.venv-*/` glob to match
  the per-profile venv naming convention.
- `README.md`: version bump `v0.4.6a0` → `v0.4.6a1` (matches
  `pyproject.toml`). Stale "Full suite (509 tests)" testing section
  replaced with the current fast-core / broad-local / full-suite
  gate model.
- `AGENTS.md`: stale "Full suite: `python3 -m pytest -q` (~1173
  tests)" replaced with the current three-gate model.
- `scripts/build_clean_zip.py`: updated stale comments referencing
  `test_core.sh` marker filter (now `test_local.sh` owns the broad
  filter; `test_core.sh` is the fast curated subset).
- `.github/workflows/test.yml`: now calls `bash scripts/test_core.sh`
  instead of bare `pytest -q` — same gate as local dev and release.
- `.github/workflows/core.yml`: now calls `bash scripts/test_core.sh`
  instead of the broad marker-selected pytest directly.
- `.github/workflows/optional.yml`: added `sandbox` extra install
  and a dedicated `sandbox` job that builds the sandbox image and
  runs `bash scripts/test_docker.sh` (real Docker enforcement).
  The `optional` job no longer includes `sandbox` in its marker
  filter (those tests need Docker + the sandbox extra).

### Added
- `tests/test_sandbox_network_enforcement.py`: replaced 6 stub
  Docker enforcement tests (always skipped with "requires real
  enforced-gateway Docker fixture") with **real tests** that run
  hardened containers against the live Docker daemon:
  - `--network none` (DISABLED mode): verifies containers cannot
    connect to 1.1.1.1 (internet), 169.254.169.254 (cloud metadata),
    or 192.168.1.1 (host LAN).
  - `--internal` network (ENFORCED_GATEWAY mode): verifies
    containers cannot reach 1.1.1.1, metadata service, or 10.0.0.1.
  All 6 pass against a real Docker daemon.

### Verified end-to-end (macOS, cargo + Docker daemon)
- `release_gate.sh all`: PASS (build + install-smoke + fast core)
- `check_ruvllm.sh`: PASSED (inference-metal build,
  `SUPPORTS_INFERENCE=True`)
- `test_docker.sh`: 6 passed (real Docker network enforcement)
- `test_local.sh`: 1021 passed, 45 skipped in 68s
- `ruff check .`: clean

### Still experimental
- RuvLLM (Rust inference engine — default build is stub).
- Distributed federation (Redis-backed HA).

## v0.4.6a0

Stabilization-only release. No new features. Release-engineering
stabilization, documentation cleanup, and FastAPI deprecation fix
addressing the v30/v31 validation reports.

### Fixed
- CLI help text: removed remaining iptables references from
  user-visible `--help` output. Egress filtering help now says
  "egress filtering" (not "iptables filtering") and marks the feature
  as EXPERIMENTAL.
- `sandbox/entrypoint.sh`: "production path" → neutral wording.
  "NEVER set in production" → "NEVER set in normal operation".
- `sandbox/network_allowlist.py`: module docstring now explicitly
  marks the in-container iptables path as DEPRECATED. The SNI proxy /
  Envoy sidecar is described as a legacy/prototype egress path (not
  "recommended"), and ENFORCED egress is marked EXPERIMENTAL / not
  production-safe.
- `sandbox/Dockerfile`: removed "in favor of" SNI proxy recommendation;
  both iptables and SNI proxy paths are now described as
  experimental/deprecated.
- `README.md`: replaced the ad-hoc "Security status" section with the
  canonical "Sandbox network status" block describing all three network
  modes (`DISABLED`, `BEST_EFFORT_PROXY`, `ENFORCED_GATEWAY`) with
  honest enforcement claims.
- `sandbox/README.md`: added the same canonical "Sandbox network
  status" block.
- `README.md`: RuvLLM section now explicitly marked **experimental**
  (default Rust build may be stub; main wheel does not include
  `ruvllm_py`; "preferred backend" wording replaced with "used only
  when explicitly installed and enabled; experimental").
- `scripts/build_clean_zip.py`: added `--use-current-env` and
  `--self-contained` mode flags. `--self-contained` creates a temp
  venv, installs `.[dev,test]`, and runs the core gate there (best for
  release use). Default mode now fails if pytest is missing instead of
  silently skipping. Added `require_test_deps()` check with actionable
  remediation message.
- `pyproject.toml`: added `requires_docker_gateway` marker for
  integration tests that need a real enforced-gateway Docker fixture.
- `tests/test_sandbox_network_enforcement.py`: Docker bypass tests
  now marked `@pytest.mark.integration` +
  `@pytest.mark.requires_docker_gateway` with clearer skip reason
  ("requires real enforced-gateway Docker fixture"). Added docstring
  listing the future required bypass tests (raw sockets, DNS bypass,
  direct IP, metadata service, host LAN, RFC1918, Docker DNS bypass).
- FastAPI `on_event` deprecation warnings eliminated: `web/app.py`
  and `federation_server.py` now use the modern `lifespan` context
  manager pattern instead of the deprecated `@app.on_event`
  decorators. Web/federation tests now produce 0 FastAPI/Starlette
  deprecation warnings (down from 21).
- `scripts/check_ruvllm.sh`: fixed to build with `--features candle`
  (CPU) or `--features inference-metal` (Apple Silicon) instead of
  the default stub build. The old script was designed to fail —
  `SUPPORTS_INFERENCE` is only true with inference features enabled.
- `ruvllm_py/pyproject.toml`: fixed deprecated `license = { text = "MIT" }`
  → `license = "MIT"` (matches main pyproject.toml fix).
- `docker-compose.yml`: removed broken `build:` block in the ruvllm
  profile that referenced `ruvllm_py/Dockerfile` (which does not
  exist). The service now requires a pre-built `RUVLLM_IMAGE` —
  documented in the comment block.
- `Dockerfile`: fixed comments that overclaimed "full experience" —
  the image is core-only (aiohttp + python-dotenv). Comments now
  say "CORE-ONLY image" and explain how to extend it with extras.
- `pyproject.toml` extras: added `models` extra (llama-cpp-python,
  onnxruntime, huggingface_hub, tokenizers, duckduckgo_search,
  wikipedia) matching `requirements-models.txt`. Expanded `sandbox`
  extra to include `wasmtime` and `cryptography` matching
  `requirements-sandbox.txt`. Updated `all` extra accordingly.
- `README.md`: fixed Python version requirement from "3.10+" to
  "3.11+" to match `requires-python = ">=3.11"` in pyproject.toml.
- `tests/test_sandbox_network_enforcement.py`: softened the
  ENFORCED_GATEWAY test docstring from "the only mode that IS a
  security boundary" to "the only mode *designed* to be a security
  boundary, but that design is not yet validated" — matches the
  experimental/unproven status in all other docs.

### Added (demos — not part of release scope)
- `examples/demo_complex_pipeline.py`: multi-step combinatorics problem
  (ordered triples with a+b+c=100, a<b<c → answer 784) exercising
  routing, CLR scoring, math verifier, schema verifier, and code
  verifier (static analysis + sandbox fallback).
- `examples/demo_coding_task.py`: LRU cache implementation task with
  5 candidate solutions verified in real Docker sandbox (correct
  passes, 4 buggy/dangerous rejected by unit test assertions).
- `examples/demo_constraint_satisfaction.py`: meeting scheduling CSP
  (4 meetings, 3 slots, 3 rooms) verified by 4 different verifiers:
  Z3/SMT (LogicVerifier), JSON schema (SchemaVerifier), NLI judge
  (FactualVerifier with citation-backed claims), and Docker sandbox
  (CodeVerifier with Z3-based scheduler code).

### Still experimental
- Enforced sandbox gateway egress (`NetworkMode.ENFORCED_GATEWAY`).
- RuvLLM (Rust inference engine — default build is stub).
- Distributed federation (Redis-backed HA).

## v0.4.5a0

Stabilization-only release. No new features. Release-engineering
polish addressing the v29 validation report.

### Fixed
- `release_gate.sh` now supports separately callable phases
  (`build`, `install-smoke`, `core`, `all`) so failures are easier to
  isolate. pip output is quieted by default (set
  `RELEASE_GATE_VERBOSE=1` for full logs).
- `build_clean_zip.py` now accepts `--no-tests` (skip the pytest gate,
  compileall only) and `--tests` (force the gate, fail if pytest is
  absent). The default remains best-effort: run if pytest is available,
  skip with a clear warning if not.
- `test_zip_release.sh` now quiets pip/build output and uses `-q` for
  the core pytest run so the self-test completes reliably without
  timeout from log volume. Set `ZIP_TEST_VERBOSE=1` for full logs.
- CLI help text: removed all version-tag annotations (v0.3.9, v1.2,
  v3.0, v3.2, v3.2.1) from user-visible `--help` output. Internal
  code comments retain historical version notes.
- README: removed stale version tags from section headers and body
  text. "RuFlo integration abstractions (v0.3.9)" → "RuFlo integration
  abstractions"; "v3.2.1" references cleaned.
- Sandbox/docs: removed remaining "production" overclaims from
  `sandbox/envoy_sidecar.py`, `sandbox/local_executor.py`,
  `sandbox/sni_proxy.py`, `docs/roadmap.md`. All egress paths now
  consistently described as EXPERIMENTAL / not production-safe.
- `pyproject.toml`: fixed deprecated `license = {file = "LICENSE"}`
  table syntax → `license = "MIT"` + `license-files = ["LICENSE"]`
  (eliminates `SetuptoolsDeprecationWarning` during `python -m build`).

### Still experimental
- Enforced sandbox gateway egress (`NetworkMode.ENFORCED_GATEWAY`).
- RuvLLM (Rust inference engine — default build is stub).
- Distributed federation (Redis-backed HA).

## v0.4.4a0

Stabilization-only release. No new features. Release-engineering fixes
addressing the v28 validation report.

### Fixed
- `release_gate.sh` now creates an isolated build venv (with `build`
  installed) before `python -m build` — no longer assumes the ambient
  Python has `build`.
- `build_clean_zip.py` no longer requires `pytest-timeout` (removed
  `--timeout` flags; relies on subprocess timeout). Skips the pytest
  gate gracefully with a warning when pytest is absent.
- `build_clean_zip.py` now includes the full project source:
  `.github/`, `Dockerfile`, `docker-compose.yml`, `.dockerignore`,
  `docker/`, `ruvllm_py/`, `rust/` (excluding multi-GB `target/` and
  `vendor/` build artifacts). The ZIP is now a complete source release,
  not a Python-only subset.
- `httpx` added to the `web` optional dependency group
  (FastAPI/Starlette TestClient requires it; the web test layer was
  not reproducible without it).
- Web/federation test skipping: replaced module-level
  `pytest.importorskip` with `pytestmark.skipif` so direct execution
  without optional deps collects tests and skips them (exit 0), not
  "no tests collected" (exit code 5).
- CLI/docs: removed "production egress path" and "production-safe"
  overclaims. Enforced egress is now consistently described as
  EXPERIMENTAL / not production-safe until real bypass tests pass.
- `sandbox/Dockerfile`: "production executor" → "default executor".

### Still experimental
- Enforced sandbox gateway egress (`NetworkMode.ENFORCED_GATEWAY`).
- RuvLLM (Rust inference engine — default build is stub).
- Distributed federation (Redis-backed HA).

## v0.4.3a0

Stabilization-only release. No new features.

### Fixed
- CLI `--version` support (now reports the installed package version).
- `release_gate.sh` uses fully isolated venvs for both the clean-wheel
  smoke stage and the dev/test core-gate stage (no ambient Python state).
- `build_clean_zip.py` runs the core-test marker filter (not full pytest)
  and derives the ZIP filename from `pyproject.toml` (no hardcoded
  versioned names, no stale v0.3 docstring).
- `test_web_federation.py` skips cleanly via `importorskip` when the
  optional `web`/`federation` extras are absent (direct execution no
  longer errors with `ModuleNotFoundError: fastapi`).
- Trajectory synthesis: the orchestrator now uses a runtime embedding
  capability check (`embeddings_available()`) instead of a module-level
  constant captured at import time, and accepts an injectable
  `trajectory_store` / `embedder` for tests. Synthesis with fake
  embeddings now works in partial environments; missing-store paths
  return a stable `empty_synthesis_result` instead of `TypeError`.
- Release ZIP self-test reliability (`test_zip_release.sh` checks
  `--version`).

### Still experimental
- Enforced sandbox gateway egress (`NetworkMode.ENFORCED_GATEWAY`).
- RuvLLM (Rust inference engine — default build is stub).
- Distributed federation (Redis-backed HA).

## v0.4.2a0

Stabilization-only release. No new features.

### Fixed
- Executable script permissions preserved in ZIP builds.
- Core release gate reliability (one-shot `test_core.sh`).
- Optional embedding dependency detection (`EMBEDDINGS_AVAILABLE`
  now requires numpy + scikit-learn + sentence-transformers; no stub
  class used for availability detection).
- Vector-store partial dependency behavior (empty embeddings are
  skipped, not upserted as zero-dimensional vectors).
- Trajectory synthesis skip/pass behavior (stable result contract
  on all return paths; skips honestly without embeddings).
- Smoke warning noise (empty cache files handled silently).
- Sandbox documentation accuracy (overclaims removed;
  `NetworkMode` wired into `DockerSandboxExecutor`).
- `test_warm_pool.py` coroutine warning fixed.
- Flake8 E-class lint errors fixed in test files.

### Still experimental
- Enforced gateway egress (`NetworkMode.ENFORCED_GATEWAY`).
- RuvLLM (Rust inference engine — default build is stub).
- Distributed federation (Redis-backed HA).

### Requirements split
- `requirements.txt` now contains only core deps (aiohttp +
  python-dotenv).
- Split files: `requirements-{core,dev,embeddings,federation,
  sandbox,models,legacy-full}.txt`.
- Use `pyproject.toml` extras for editable installs:
  `pip install -e '.[dev,test,embeddings]'`.
