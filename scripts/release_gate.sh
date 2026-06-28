#!/usr/bin/env bash
# Release gate — must pass before any release.
# Run: ./scripts/release_gate.sh
#
# Uses fully isolated venvs for every stage so the gate never depends on
# ambient/system/user Python state:
#   1. Build wheel/sdist (fresh venv with build installed).
#   2. Clean wheel install + CLI smoke (separate fresh venv, wheel only).
#   3. Dev/test editable install + core tests (separate fresh venv).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[release-gate] Cleaning generated artifacts"
rm -rf build dist *.egg-info vibe_thinker.egg-info .pytest_cache
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -name "*.pyc" -delete

echo "[release-gate] Compile check"
python3 -m compileall -q .

echo "[release-gate] Build wheel/sdist (isolated build venv)"
BUILD_VENV="$(mktemp -d)/venv"
python3 -m venv "$BUILD_VENV"
# shellcheck disable=SC1091
source "$BUILD_VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel build
python -m build
deactivate
rm -rf "$(dirname "$BUILD_VENV")"

echo "[release-gate] Clean installed wheel smoke"
INSTALL_VENV="$(mktemp -d)/venv"
python3 -m venv "$INSTALL_VENV"
# shellcheck disable=SC1091
source "$INSTALL_VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install dist/*.whl
vibe-thinker --help
vibe-thinker --version
vibe-thinker doctor
vibe-thinker smoke
deactivate
rm -rf "$(dirname "$INSTALL_VENV")"

echo "[release-gate] Isolated dev/test core gate"
TEST_VENV="$(mktemp -d)/venv"
python3 -m venv "$TEST_VENV"
# shellcheck disable=SC1091
source "$TEST_VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel build
python -m pip install -e ".[dev,test]"
./scripts/test_core.sh
deactivate
rm -rf "$(dirname "$TEST_VENV")"

echo ""
echo "[release-gate] PASS"
