#!/usr/bin/env bash
# Embeddings gate — runs tests requiring numpy + scikit-learn +
# sentence-transformers (+ faiss). Tests marked `embeddings`. The
# trajectory-store UNIT tests inject a fake embedding model and only need
# numpy + sklearn (NOT sentence-transformers); real-embedding integration
# tests need the full embeddings extra installed here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Env-aware: if a virtualenv is already active (VIRTUAL_ENV set), reuse
# it as-is — the caller is responsible for having prepared it. This
# avoids nested venv creation when invoked from CI or other scripts
# that create their own venv and install deps before calling this gate.
if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "[test_embeddings] using active venv: $VIRTUAL_ENV"
else
  VENV="$ROOT/.venv-embeddings"
  if [ ! -d "$VENV" ]; then
    echo "[test_embeddings] creating controlled venv at $VENV"
    python3 -m venv "$VENV"
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -m pip install -U pip >/dev/null
  python -m pip install -e "$ROOT[dev,test,embeddings]" >/dev/null
fi

cd "$ROOT"
python -m pytest --strict-markers -m "embeddings"
deactivate 2>/dev/null || true
