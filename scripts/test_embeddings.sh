#!/usr/bin/env bash
# Embeddings gate — runs tests requiring numpy + scikit-learn +
# sentence-transformers (+ faiss). Tests marked `embeddings`. The
# trajectory-store UNIT tests inject a fake embedding model and only need
# numpy + sklearn (NOT sentence-transformers); real-embedding integration
# tests need the full embeddings extra installed here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv-embeddings"

if [ ! -d "$VENV" ]; then
  echo "[test_embeddings] creating controlled venv at $VENV"
  python3 -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install -U pip >/dev/null
python -m pip install -e "$ROOT[dev,test,embeddings]" >/dev/null

cd "$ROOT"
python -m pytest --strict-markers -m "embeddings"
deactivate 2>/dev/null || true
