#!/usr/bin/env bash
# Release gate — must pass before any release.
# Run: ./scripts/release_gate.sh
set -euo pipefail

echo "=== Cleaning build artifacts ==="
rm -rf build dist *.egg-info vibe_thinker.egg-info .pytest_cache
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -name "*.pyc" -delete

echo "=== Compile check ==="
python3 -m compileall -q .

echo "=== Build wheel ==="
python3 -m build

echo "=== Fresh venv install test ==="
rm -rf /tmp/vibe-thinker-release-gate
python3 -m venv /tmp/vibe-thinker-release-gate
source /tmp/vibe-thinker-release-gate/bin/activate
python -m pip install --upgrade pip
python -m pip install dist/*.whl

echo "=== CLI help ==="
vibe-thinker --help

echo "=== Doctor ==="
vibe-thinker doctor

echo "=== Smoke ==="
vibe-thinker smoke

deactivate
rm -rf /tmp/vibe-thinker-release-gate

echo "=== Core tests ==="
python3 -m pip install -e ".[dev,test]"
bash scripts/test_core.sh

echo ""
echo "Release gate PASSED."
