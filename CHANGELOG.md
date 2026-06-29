# Changelog

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
