#!/usr/bin/env bash
# Docker sandbox gate — runs only Docker/sandbox-network tests.
# Requires Docker (and the sandbox extra) installed in an isolated venv.
# Tests marked `sandbox` or `requires_docker_gateway`. The
# requires_docker_gateway tests need a real enforced-gateway Docker fixture;
# they skip if Docker is unavailable, so this gate is green on machines
# without Docker (all-selected tests skip) and authoritative on machines
# with Docker.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Env-aware: if a virtualenv is already active (VIRTUAL_ENV set), reuse
# it as-is — the caller is responsible for having prepared it. This
# avoids nested venv creation when invoked from CI or other scripts
# that create their own venv and install deps before calling this gate.
if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "[test_docker] using active venv: $VIRTUAL_ENV"
else
  VENV="$ROOT/.venv-docker"
  if [ ! -d "$VENV" ]; then
    echo "[test_docker] creating controlled venv at $VENV"
    python3 -m venv "$VENV"
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -m pip install -U pip >/dev/null
  python -m pip install -e "$ROOT[dev,test,sandbox]" >/dev/null
fi

cd "$ROOT"
python -m pytest --strict-markers \
  -m "sandbox or requires_docker_gateway"
deactivate 2>/dev/null || true
