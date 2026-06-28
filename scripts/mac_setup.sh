#!/usr/bin/env bash
# Mac local setup — create a venv and install core deps (no Docker/Redis/Rust).
set -euo pipefail
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel build
python -m pip install -e ".[dev,test]"
python -m compileall -q .
python rfsn_cli.py --help
echo ""
echo "Mac local setup complete."
echo "Activate with: source .venv/bin/activate"
echo "Smoke test with: ./scripts/mac_smoke.sh"
