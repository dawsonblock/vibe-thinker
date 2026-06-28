#!/usr/bin/env bash
# Mac local smoke test — run CLI + core tests with the mac-local profile.
set -euo pipefail
source .venv/bin/activate
export $(grep -v '^#' profiles/mac-local.env | xargs)
python rfsn_cli.py --help
pytest -m "not logic and not embeddings and not federation and not web and not sandbox and not integration"
echo ""
echo "Mac local smoke test passed."
