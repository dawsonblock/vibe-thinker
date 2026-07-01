#!/usr/bin/env bash
# Broad local test suite — the full core-marker test selection (~1000+
# tests, ~15 min). This is the pre-release confidence gate, NOT the
# fast iteration gate. For fast iteration use scripts/test_core.sh.
#
# Env-aware: if a virtualenv is already active (VIRTUAL_ENV set), it is
# reused as-is. Standalone invocation creates and reuses .venv-local.
#
# Standalone installs the [dev,test] extras so the lightweight core-test
# deps (numpy, scikit-learn, cryptography, z3-solver) are always present.
# Additional optional-dep markers (federation, web, nli) are only included
# when their heavier deps are installed. The sandbox/integration/
# requires_docker_gateway markers are always excluded (they need a running
# Docker daemon + gateway container, not just Python packages).
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
  python -m pip install -e "$ROOT[dev,test]" >/dev/null
fi

cd "$ROOT"
python -m compileall -q .
python rfsn_cli.py doctor
python rfsn_cli.py smoke

# Build a dynamic marker filter: only exclude markers whose optional
# deps are NOT installed. sandbox/integration/requires_docker_gateway
# are always excluded (need a running Docker daemon, not just packages).
# When a dep IS installed, we do NOT add its marker to the exclusion
# list, so those tests run. When a dep is NOT installed, we add the
# marker to the exclusion list so those tests are deselected (they
# would skip honestly, but deselecting is faster and cleaner).
EXCLUDE="sandbox and not integration and not requires_docker_gateway"

python -c "import z3" 2>/dev/null || EXCLUDE="$EXCLUDE and not logic"
python -c "import sentence_transformers, faiss, numpy, sklearn" 2>/dev/null || EXCLUDE="$EXCLUDE and not embeddings"
python -c "import redis, fakeredis" 2>/dev/null || EXCLUDE="$EXCLUDE and not federation"
python -c "import fastapi, uvicorn, websockets, pydantic, httpx" 2>/dev/null || EXCLUDE="$EXCLUDE and not web"
python -c "import transformers, torch" 2>/dev/null || EXCLUDE="$EXCLUDE and not nli"

echo "[test_local] marker filter: not $EXCLUDE"
python -m pytest --strict-markers -m "not $EXCLUDE"
python rfsn_cli.py --help >/dev/null
deactivate 2>/dev/null || true
