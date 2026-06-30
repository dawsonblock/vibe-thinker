#!/usr/bin/env bash
# Mac local smoke test — run CLI + fast core gate with the mac-local profile.
# Uses the same fast curated gate as release_gate.sh core and CI.
set -euo pipefail
source .venv/bin/activate
export $(grep -v '^#' profiles/mac-local.env | xargs)
python rfsn_cli.py --help
# test_core.sh is env-aware: it detects the active .venv and reuses it
# (no nested venv creation). Runs compile + doctor + smoke + ~250 curated
# tests in ~30s. For the broad ~1000+ test gate, use test_local.sh.
bash scripts/test_core.sh
echo ""
echo "Mac local smoke test passed."
