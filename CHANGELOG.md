# Changelog

## v0.4.6a9 (build 46)

Demo hardening pass — addresses all findings from the build-45 and
build-46 audits. No production modules changed; all fixes are in the
demo script, changelog wording, and release packaging.

### Fixed (demo)
- `demo_verified_swarm.py`: every phase now computes `ok` from actual
  sub-check return values. `sub()` returns `bool` so failures propagate
  to phase status. No more `ok = True` hardcodes.
- Phase 0 `ok` now includes `compileall` and import-check results. The
  old code used `r.returncode` from the smoke test (wrong variable) and
  ignored the import check entirely.
- `LocalSubprocessExecutor` explicitly labeled as NOT a security
  sandbox in Phases 1, 2, and 9. The `/etc/passwd` test label now
  honestly says "verifier rejects output (not sandbox isolation)".
  The final checklist says "verifier-rejected (not sandbox-isolated)"
  instead of "blocked".
- Federation zombie reaper: `await state.reap_stale_claims(...)` (was
  missing `await`, causing `TypeError` on `len(coroutine)`).
- `AgentDBVectorStore`: `url=` → `base_url=`, added `collection=` param.
- `is_cache_entry_trustworthy()`: removed invalid `min_score=` kwarg,
  added required `best_answer` + `schema_version=3` fields to test
  fixture.
- `RuvLLMHTTPBackend.recommended_start_command()`: handle list return
  value instead of calling `.lower()` on a string.
- Body-size 422: import `Request` at module level (PEP 563 stringifies
  annotations; FastAPI can't resolve function-local imports).
- Phase 4 switched from sync `TestClient` to `httpx.AsyncClient` with
  `ASGITransport` (demo runs inside an async function).
- Scheduler examples use backtracking instead of a weak greedy
  assignment that couldn't fill the night shift.

### Fixed (packaging)
- CHANGELOG wording: "No new surface area" was false — the demo script
  was new. Corrected to acknowledge the addition and document it as
  source-checkout-only (not in `pyproject.toml py-modules`, not in the
  wheel).
- Release ZIP built through `scripts/build_clean_zip.py`: 0
  `__pycache__` entries, 0 `.pyc` files, all 15 `.sh` files have +x
  bit. Verified by `scripts/test_zip_release.sh`.

### Added
- `scripts/demo_setup.sh` — installs all optional extras needed to run
  `demo_verified_swarm.py` (`dev,test,web,federation,sandbox,
  embeddings,logic`). Supports `--venv` flag for isolated installation.
- `scripts/release_zip.sh` — the single supported release-archive path.
  Chains `build_clean_zip.py --self-contained` (temp venv + core pytest
  gate + clean ZIP) with `test_zip_release.sh` (extract, reject junk,
  fresh-venv install, doctor, smoke, `test_core.sh`). `--skip-test`
  builds without self-test (non-release artifact). Addresses audit fix
  #1: "Build the ZIP only through build_clean_zip.py --self-contained."
- `scripts/test_gate_matrix.sh` — runs every release-gate profile
  (core/local/release/zip-release/compose/docker/embeddings/federation/
  ruvllm), records PASS/FAIL/SKIP, writes `dist/gate_matrix_<ts>.log`
  + per-gate logs. Gates with missing prerequisites (Docker/Redis/Rust/
  optional extras) SKIP rather than FAIL. `--keep-going` runs all gates.
  Addresses audit fix #5: "Run and capture the full gate matrix."
- `tests/test_release_zip_hygiene.py` — automated pytest guard that
  scans every `dist/vibe-thinker-v*.zip` and fails on
  `__pycache__`/`.pyc`/`.pyo`/`.pytest_cache`/`.egg-info`/`build`/
  `.DS_Store` entries or any `.sh` missing the +x bit in
  `external_attr`. Skips when no release ZIP exists. Mirrors the junk
  check in `test_zip_release.sh` and `EXCLUDE_PATTERNS` in
  `build_clean_zip.py`. Addresses audit fix #2.

### Verified (gate matrix on macOS, 2026-07-01)
- Core gate (`scripts/test_core.sh`): 280 passed in 45s
- Web security (`tests/test_web_security.py`): 21 passed
- Job queue (`tests/test_job_queue.py`): 20 passed
- Static checks (`test_static_missing_self_methods.py`,
  `test_static_unreachable_code.py`): 2 passed
- RuvLLM (`scripts/check_ruvllm.sh`): PASSED (cargo + maturin + import)
- ZIP release self-test (`scripts/test_zip_release.sh`): PASSED
  (280 core tests in fresh venv from extracted ZIP)
- Demo (`demo_verified_swarm.py`): 10/10 phases, 67/67 sub-checks

## v0.4.6a9 (build 45)

Production-code fixes for the issues flagged in the build-44 audit. Every
production-code change closes an existing gap. A new source-only demo script
(`demo_verified_swarm.py`) was also added — it is not packaged in the wheel
and is intended for source-checkout verification only.

### Fixed
- `run_ui.py` now uses a real `argparse.parse_known_args()` parser for
  UI-specific flags instead of a manual token walk. `--port=8000`,
  `--host=...`, and all equals-style args now work correctly. Unknown
  args are forwarded to the orchestrator's `build_argparser`.
- `run_ui.py` now exposes CLI flags for the web security layer:
  `--api-key`, `--allowed-origins`, `--rate-limit-per-minute`, and
  `--max-request-body-bytes` (with env-var fallbacks
  `VIBE_THINKER_API_KEY`, `VIBE_UI_ALLOWED_ORIGINS`, `VIBE_UI_RATE_LIMIT`,
  `VIBE_UI_MAX_BODY_BYTES`). These are passed through to
  `web.app.create_app()`.
- `web_security.py` body-size enforcement now guards the actual request
  stream, not just the `Content-Length` header. A pure ASGI middleware
  (`_BodySizeLimitMiddleware`) wraps the `receive` callable to count
  real body bytes, catching chunked transfers and missing/lying
  `Content-Length` headers. The previous `BaseHTTPMiddleware` approach
  could not catch exceptions from the receive callable (Starlette task
  machinery swallows them); the pure ASGI middleware propagates them
  synchronously.
- `VIBE_NETWORK_MODE=disabled` now maps explicitly to
  `NetworkMode.DISABLED` in `_sandbox_network_mode()`, and `disabled`
  is an accepted `--sandbox-network` choice. Previously `disabled` fell
  through to `None` (auto behavior), which could become best-effort
  proxy when an allow-list was present instead of hard network-off.
  `profiles/mac-local.env` sets `VIBE_NETWORK_MODE=disabled`, so this
  fixes a real silent-escalation bug.
- Version drift: `README.md` and `AGENTS.md` now match `pyproject.toml`.
- `rfsn_job_queue.py` `_dispatch_loop` now bounds task creation by
  `max_concurrent` instead of creating a task for every pending job and
  relying on the semaphore to queue execution. Submitting a large batch
  no longer creates unbounded coroutines — only `max_concurrent` tasks
  exist at any time. The semaphore inside `_run_job` is retained as a
  safety net.

### Added
- `tests/test_web_security.py::test_body_size_limit_without_content_length`
  — verifies the stream guard catches a lying `Content-Length` header
  (header says 5 bytes, actual body is 100 bytes).
- `tests/test_job_queue.py::test_task_creation_is_bounded` — verifies
  that submitting 20 jobs with `max_concurrent=2` creates at most 2
  in-flight tasks, not 20.

### Verified
- Full release gate matrix passes: core (280), broad local (1382),
  federation/web (142), Docker sandbox (13), compose, embeddings (95),
  RuvLLM Rust build, wheel install, ZIP release self-test (280).

## v0.4.6a8

Fixes for the critical AgentDB cutover path, network enforcement, and UI option
forwarding flagged in the v0.4.6a7 audit.

### Fixed
- `CLRResultCache.lookup()` and `VerifiedTrajectoryStore.retrieve()` now search
  the AgentDB vector store *before* the `if not self.entries` local-JSON early
  return, so `agentdb_only=True` works even after local JSON files are archived.
- Restored the missing `sims = cosine_similarity(...)` assignment in
  `CLRResultCache.lookup()` so the local embeddings semantic cache path no longer
  raises `NameError: name 'sims' is not defined`.
- `sni_proxy.py` now applies CIDR port restrictions to concrete IPs inside the
  CIDR (e.g. `10.0.0.0/24:443` blocks `10.0.0.5:80`).
- HTTP proxy path in `sni_proxy.py` now uses the absolute-URL target host for
  allow-list and port decisions, and rejects requests where the absolute URL
  host mismatches the `Host` header.
- `run_ui.py` now forwards every orchestrator option exposed by `rfsn_cli.py`
  (including `--agentdb-only`, `--use-structured-output`, `--specialist-*`,
  `--sandbox-network`, `--proxy-egress`, `--sona-*`, etc.).
- Removed stale "shadow-mode" and "no-op" wording from CLI help, README,
  AGENTS.md, and tests so the docs match the actual dual-write / cut-over
  behavior.

### Added
- `tests/test_agentdb_only.py` with empty-local-JSON + AgentDB-result tests for
  both `CLRResultCache` and `VerifiedTrajectoryStore`.
- Regression test for the local embeddings semantic path in `CLRResultCache`.
- Tests for CIDR port restrictions and HTTP absolute-URL / Host-header mismatch
  in `tests/test_sni_proxy.py`.

### Changed
- `agentdb_only=True` now prints a warning when no embedding model is available
  (instead of silently returning nothing) so the dependency requirement is
  explicit for the AgentDB vector-store path.

### Verified
- Full release gate matrix passes on this environment: core (280 tests), broad
  local (1121 tests), Docker sandbox, compose, federation/web, embeddings,
  RuvLLM Rust build, wheel install, ZIP release self-test, and release gate.

## v0.4.6a7

Packaging, enforcement, and AgentDB consistency fixes from the build-40
audit report.

### Fixed
- Wheel packaging now includes `web/static/index.html`, `sandbox/entrypoint.sh`,
  shell scripts, Docker files, and compose files via `package-data` + `data-files`.
- Refactored `finalize-migration` to import from the new `agentdb_migration`
  module instead of loading `scripts/migrate_to_agentdb.py` from the filesystem.
- IP/CIDR allow-list entries are now enforced by `sni_proxy.py` in both CONNECT
  and HTTP paths.
- Compose sandbox networking: added `RFSN_DOCKER_NETWORK` and `--docker-network` so
  executor-spawned containers join the compose network and can reach `sni-proxy`.
- `agentdb_only=True` now drives cache/trajectory lookup through the AgentDB
  vector store instead of silently using the local embeddings matrix.
- Removed stale "shadow-mode" comments from `persistent_cache.py`.

### Added
- `vibe-thinker-ui` console entry point for `run_ui.py`.
- `agentdb_migration.py` module.
- `tests/test_wheel_install.py` integration test: builds the wheel, installs it
  in a fresh venv, and verifies CLI + web UI entry points.

## v0.4.6a6

Compose proxy hardening and validation.

### Fixed
- Hardened docker-compose `sni-proxy` service with read-only root, dropped
capabilities, no-new-privileges, non-root user, PID/memory limits, and tmpfs.
- Clarified that Envoy sidecar is transparent-routing only, not HTTP_PROXY.
- Added compose smoke test for HTTP allow, HTTPS allow, and blocked-domain
behavior.

### Added
- `scripts/test_compose.sh`
- `tests/test_compose_config.py`

## v0.4.6a5

Fail-closed federation encryption, wildcard allow-list fix, gate
finalization, and test-suite cleanup. The first build with a credible
full-gate pass across all non-local gates (Docker, embeddings,
federation/web, RuvLLM Rust) on the actual Mac environment.

### Security
- `federation_server.py`: **fail-closed encryption**. When
  `federation_secret` is configured but the `cryptography` package
  (Fernet) is unavailable, `create_federation_app()` now raises
  `RuntimeError` at app creation time. Previously the `except
  ImportError: pass` silently set `_fernet = None`, causing `_encrypt()`
  to return plaintext responses under a configured secret — a security
  downgrade that would leak query data over the federation without any
  error. The new contract: no secret → plaintext is intentional; secret
  + cryptography → encrypted; secret + no cryptography → startup
  failure (never silent plaintext).
- `sandbox/sni_proxy.py`: **wildcard allow-list fix**. The `main()`
  extraction checked `entry.host.startswith("*.")` to detect wildcard
  entries, but `NetworkAllowList._parse_entry` already strips the `*.`
  prefix and stores `entry.host="example.com"` with
  `entry.wildcard=True`. The check was never true for wildcard entries,
  so they silently became exact-domain rules — inverting the semantics
  (`*.example.com:443` allowed `example.com:443` but denied
  `foo.example.com:443`). Fixed by checking `entry.wildcard` and
  reconstructing the `*.host` pattern. Also made `_is_port_allowed()`
  wildcard-aware so port restrictions on wildcard entries (e.g.
  `*.example.com:443` rejecting `foo.example.com:80`) are enforced.

### Fixed
- `scripts/check_ruvllm.sh`: made env-aware (reuses active venv or
  creates `.venv-ruvllm`) and pins a single `PYTHON` interpreter for
  all commands (`pip install`, `maturin develop`, `import check`).
  Previously `maturin develop` could install into one Python while the
  import check ran in another, causing a false "stub build" failure.
  Now prints the Python path at the top for auditability.
- `scripts/test_local.sh`: dynamic dep-aware marker filter. Optional-
  dep markers (logic, embeddings, federation, web, nli) are only
  excluded when their deps are NOT installed, so a full-deps venv runs
  ~1333 tests instead of ~1096. The sandbox/integration/
  requires_docker_gateway markers remain always excluded (need a
  running Docker daemon).
- `web/app.py`: added `federated` flag to `/api/query` endpoint. When
  `federated=true`, the job stays "pending" for an external worker to
  claim via `/api/jobs/claim` — no background task races the claim.
  This is both a production feature (federation mode) and a test fix
  (eliminates the race-condition `pytest.skip()` in
  `test_federation_zombie.py`).
- `pyproject.toml`: added `cryptography` to the `federation` extra
  (was only in `sandbox` and `all`). The federation encryption tests
  require it.
- `pyproject.toml`: added `filterwarnings` to suppress the upstream
  Starlette/httpx deprecation warning (not our code; pending fastapi
  httpx2 migration).
- `tests/test_routing.py`: wrapped the `_static_analysis_fallback`
  deprecation warning in `pytest.warns(DeprecationWarning)` so it's
  asserted instead of bubbling up unasserted.

### Added
- `tests/test_sni_proxy.py`: 12 new tests for wildcard allow-list
  extraction and wildcard port enforcement (4 verdict scenarios).
- `tests/test_sona_gossip.py`: `TestEncryptionFailClosed` — 2 tests
  verifying that `create_federation_app(federation_secret=...)` raises
  `RuntimeError` when cryptography is unavailable (mocked via
  sys.modules), and that no-secret + no-cryptography does NOT raise.
- `tests/test_schema_verifier.py`: mock-based companion test for the
  z3-unavailable fail-closed path (runs when z3 IS installed).
- `tests/test_turboquant_ppl.py`: mock-based companion tests for the
  ruvllm_py-unavailable fail-closed path (runs when ruvllm_py IS built).
- `tests/test_sona_gossip.py`: `skipif` guards on encryption test
  classes for environments where cryptography is genuinely absent.

### Gate results (Mac, Apple Silicon, all optional deps installed)
- Core gate (`test_core.sh`): 248 passed, 32 skipped
- Local gate (`test_local.sh`): 1333 passed, 4 skipped, 13 deselected
- Docker gate (`test_docker.sh`): 13 passed
- Embeddings gate (`test_embeddings.sh`): 90 passed, 5 skipped
- Federation gate (`test_federation.sh`): 121 passed (incl. 2 fail-closed)
- RuvLLM gate (`check_ruvllm.sh`): PASSED (inference-metal, SUPPORTS_INFERENCE=True)
- Full suite (no marker filter): 1348 passed, 4 skipped, 0 warnings

## v0.4.6a4

Port-specific egress enforcement in the SNI proxy. Addresses the
build-31 audit verdict's remaining real issue: the SNI proxy ignored
port restrictions in allow-list entries (e.g. `pypi.org:443` allowed
CONNECT to `pypi.org:<any port>`).

### Fixed
- `sandbox/sni_proxy.py`: the proxy now enforces port restrictions from
  allow-list entries. `SNIEgressProxy` accepts an `allowed_ports` dict
  mapping hostname → set of allowed ports. When an allow-list entry
  specifies a port (e.g. `pypi.org:443`), only that port is allowed for
  that host. When no port is specified (e.g. `pypi.org`), all ports are
  allowed (backward compat). The port check is applied in both
  `_handle_connect` (CONNECT target port) and `_handle_http` (Host
  header port or URL port, defaulting to 80). `main()` extracts port
  info from `AllowListEntry.port` and passes it to the proxy.
- `CHANGELOG.md`: fixed wording drift in the v0.4.6a3 section — said
  AGENTS.md was fixed to `v0.4.6a2` but the package was already
  `v0.4.6a3` at the time of that build.

### Added
- `tests/test_sni_proxy.py` `TestPortEnforcement`: 11 new tests for
  port-specific enforcement:
  - Unit tests for `_is_port_allowed`: no restriction (all ports),
    with restriction (only specified port), case-insensitive host,
    multiple ports, different host isolation.
  - Integration tests: CONNECT to allowed port (not 403), CONNECT to
    wrong port (403), CONNECT with no port restriction (200), HTTP
    to allowed port 80 (not 403), HTTP to wrong port 443 (403),
    HTTP default port 80 with no port in Host header (not 403).

## v0.4.6a3

CONNECT proxy fix + gateway hardening + real HTTPS/DockerSandboxExecutor
integration tests. Addresses the build-28/build-29 audit verdict.

### Fixed
- `sandbox/sni_proxy.py` `_handle_connect()`: rewrote CONNECT handling to
  fix two bugs: (1) the proxy read the TLS ClientHello BEFORE sending
  `200 Connection Established` — standard HTTPS clients wait for the 200
  first, so the proxy would hang; (2) the proxy wrote the ClientHello
  back to the client (incorrect). The new flow uses the CONNECT target
  host (the authoritative destination) for the allow-list decision,
  connects to the remote upstream FIRST, then sends `200 Connection
  Established`, then tunnels bidirectionally. After tunneling begins, it
  peeks at the first client bytes (the ClientHello) to extract the SNI
  and verifies it matches the CONNECT target (defense-in-depth against
  SNI spoofing — closes the connection on mismatch). Returns `502 Bad
  Gateway` when the upstream is unreachable (instead of `200`). Added
  `_is_ip_literal()` helper to skip the SNI check for IP-literal CONNECT
  targets.
- `sandbox/docker_executor.py` `_start_gateway()`: hardened the gateway
  container with the same flags as the sandbox container: `--read-only`,
  `--cap-drop ALL`, `--security-opt no-new-privileges`, `--user 1000:1000`,
  `--pids-limit 64`, `--memory 128m`, `--tmpfs /tmp:rw,size=10m`. The
  gateway is a network-facing security boundary component and must be
  hardened accordingly.
- `AGENTS.md`: fixed stale "currently `v0.4.6a1`" reference →
  `v0.4.6a3` (the package version in `pyproject.toml`).

### Added
- `tests/test_sni_proxy.py`: 4 new tests for the corrected CONNECT
  behavior:
  - `test_connect_allowed_domain_no_sni_gets_200`: CONNECT to an
    allowlisted domain with no SNI is allowed (SNI is defense-in-depth,
    not required).
  - `test_connect_sni_mismatch_closes_connection`: CONNECT to an
    allowlisted domain with a mismatched SNI is closed (spoofing
    detection).
  - `test_connect_matching_sni_tunnels_to_remote`: CONNECT with matching
    SNI tunnels data to the remote.
  - `test_connect_unreachable_remote_returns_502`: CONNECT to an
    unreachable upstream returns 502 Bad Gateway.
  - `TestIsIpLiteral`: tests for the `_is_ip_literal()` helper.
- `tests/test_sandbox_network_enforcement.py`: 4 new integration tests:
  - `test_gateway_allows_https_allowlisted_domain`: real HTTPS request
    to `https://example.com` through the gateway proxy (validates the
    CONNECT fix with a real TLS client).
  - `test_gateway_container_uses_hardening_flags`: wiring test that
    verifies the gateway container is started with `--read-only`,
    `--cap-drop ALL`, `--security-opt no-new-privileges`, `--user`,
    `--pids-limit`, `--memory`, `--tmpfs`.
  - `TestDockerSandboxExecutorGatewayIntegration`: 3 tests that use the
    real `DockerSandboxExecutor.execute()` with
    `NetworkMode.ENFORCED_GATEWAY` — the exact production path
    (executor → _ensure_gateway_network → _start_gateway → connect
    gateway → inject proxy env vars → run sandbox → cleanup). Tests
    HTTP allowlisted, HTTPS allowlisted, and non-allowlisted blocked.

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
