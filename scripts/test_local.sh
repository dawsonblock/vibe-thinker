#!/usr/bin/env bash
# Broad local test suite — the full core-marker test selection (~1000+
# tests, ~15 min). This is the pre-release confidence gate, NOT the
# fast iteration gate. For fast iteration use scripts/test_core.sh.
#
# Runs the same marker exclusion as the former test_core.sh: every test
# that does NOT require logic/embeddings/federation/web/sandbox/nli/
# integration optional deps. Optional-dep tests skip honestly when
# their deps are absent, so this is safe to run in a core-only venv.
#
# Env-aware: if a virtualenv is already active (VIRTUAL_ENV set), it is
# reused as-is. Standalone invocation creates and reuses .venv-local.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "[test_local] using active venv: $VIRTUAL_ENV"
else
  VENV="$ROOT/.venv-local"
  if [ ! -d "$VENV" ]; then
    echo "[test_local] creating controlled venv at $VENV"
    python3 -m venv "$VENV"
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -m pip install -U pip >/dev/null
  python -m pip install -e "$ROOT[dev]" >/dev/null
fi

cd "$ROOT"
python -m compileall -q .
python rfsn_cli.py doctor
python rfsn_cli.py smoke
python -m pytest --strict-markers \
  -m "not logic and not embeddings and not federation and not web and not sandbox and not nli and not integration"
python rfsn_cli.py --help >/dev/null
deactivate 2>/dev/null || true
