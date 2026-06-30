#!/usr/bin/env bash
# Federation/web gate — runs tests requiring Redis/fakeredis + FastAPI +
# WebSocket deps. Tests marked `federation` or `web`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Env-aware: if a virtualenv is already active (VIRTUAL_ENV set), reuse
# it as-is — the caller is responsible for having prepared it. This
# avoids nested venv creation when invoked from CI or other scripts
# that create their own venv and install deps before calling this gate.
if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "[test_federation] using active venv: $VIRTUAL_ENV"
else
  VENV="$ROOT/.venv-federation"
  if [ ! -d "$VENV" ]; then
    echo "[test_federation] creating controlled venv at $VENV"
    python3 -m venv "$VENV"
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -m pip install -U pip >/dev/null
  python -m pip install -e "$ROOT[dev,test,federation,web]" >/dev/null
fi

cd "$ROOT"
python -m pytest --strict-markers -m "federation or web"
deactivate 2>/dev/null || true
