#!/usr/bin/env bash
# Core test suite — no optional dependencies required.
# This is the non-negotiable green gate.
#
# Creates/reuses a controlled virtual environment (.venv-core) so the gate
# does NOT depend on whatever happens to be installed globally. The project's
# own [dev] extra pins pytest<9 and pulls in pytest-timeout (which registers
# the `timeout` marker for --strict-markers), eliminating the collection
# failures caused by environment drift (e.g. a global pytest 9, or a missing
# pytest-timeout). A fresh clone with no global test deps must pass this.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv-core"

if [ ! -d "$VENV" ]; then
  echo "[test_core] creating controlled venv at $VENV"
  python3 -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install -U pip >/dev/null
# Editable install + dev/test deps. Pins pytest<9 and installs pytest-timeout
# (registers the `timeout` marker), hypothesis, ruff, mypy.
python -m pip install -e "$ROOT[dev]" >/dev/null

cd "$ROOT"
python -m compileall -q .
python -m pytest --strict-markers \
  -m "not logic and not embeddings and not federation and not web and not sandbox and not nli and not integration"
python rfsn_cli.py --help >/dev/null
deactivate 2>/dev/null || true
