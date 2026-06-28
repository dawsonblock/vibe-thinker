# Changelog

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
