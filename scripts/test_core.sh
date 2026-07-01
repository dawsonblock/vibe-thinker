#!/usr/bin/env bash
# Fast core test gate — lightweight test deps required.
#
# This is the crisp, fast confidence gate (target: well under 5 minutes).
# It runs compile + doctor + smoke + a curated fast subset of the test
# suite that covers the historically-regressing failure classes:
#   - the orchestrator runtime spine (run() -> _run_clr_with_cache)
#   - the anti-regression static AST checks (missing-self / unreachable)
#   - routing, REPL, cache, scoring, signers, deterministic check,
#     math verifier, format enforcer, trajectory store
#
# The [dev,test] extras install the lightweight runtime deps needed for
# these tests (numpy, scikit-learn, cryptography, z3-solver). Heavy extras
# (sentence-transformers, faiss-cpu, transformers/torch) are NOT included.
#
# For the BROAD local suite (all ~1000+ core-marker tests, ~15 min) use
# scripts/test_local.sh instead. The fast gate here is what
# release_gate.sh core and the ZIP self-test run for iteration speed.
#
# Env-aware: if a virtualenv is already active (VIRTUAL_ENV set), it is
# reused as-is and no venv is created and no deps are installed — the
# caller is responsible for having prepared it. This avoids nested venv
# creation when invoked from release_gate.sh or test_zip_release.sh,
# both of which create their own isolated venv and install deps before
# calling this script. Standalone invocation (no active venv) creates
# and reuses a controlled .venv-core as before.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "[test_core] using active venv: $VIRTUAL_ENV"
else
  VENV="$ROOT/.venv-core"
  if [ ! -d "$VENV" ]; then
    echo "[test_core] creating controlled venv at $VENV"
    python3 -m venv "$VENV"
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -m pip install -U pip >/dev/null
  # Editable install + dev/test deps. Pins pytest<9 and installs
  # pytest-timeout (registers the `timeout` marker), hypothesis, ruff.
  python -m pip install -e "$ROOT[dev,test]" >/dev/null
fi

cd "$ROOT"
python -m compileall -q .
python rfsn_cli.py doctor
python rfsn_cli.py smoke

# Fast core subset — the anti-regression + spine + core unit tests.
# Target: 280 passed / 0 skipped. See header comment.
FAST_CORE_TESTS="
tests/test_orchestrator_runtime_spine.py
tests/test_static_missing_self_methods.py
tests/test_static_unreachable_code.py
tests/test_trajectory_store.py
tests/test_routing.py
tests/test_repl.py
tests/test_cache.py
tests/test_scoring.py
tests/test_signers.py
tests/test_deterministic_check.py
tests/test_math_verifier.py
tests/test_format_enforcer.py
"
python -m pytest --strict-markers $FAST_CORE_TESTS
python rfsn_cli.py --help >/dev/null
deactivate 2>/dev/null || true
