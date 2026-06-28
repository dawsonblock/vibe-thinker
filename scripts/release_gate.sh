#!/usr/bin/env bash
# Release gate — must pass before any release.
# Run: ./scripts/release_gate.sh
set -euo pipefail
python3 -m compileall -q .
python3 -m pytest -m "not logic and not embeddings and not federation and not web and not sandbox and not nli and not integration"
python3 -m build
python3 -m pip install --force-reinstall dist/*.whl
vibe-thinker --help
vibe-thinker doctor
vibe-thinker smoke
echo ""
echo "Release gate PASSED."
