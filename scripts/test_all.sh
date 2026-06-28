#!/usr/bin/env bash
# Full test suite + wheel build + install verification.
# Requires all optional dependencies installed.
set -euo pipefail
python3 -m compileall -q .
python3 -m pytest
python3 -m build
python3 -m pip install --force-reinstall dist/*.whl
vibe-thinker --help
