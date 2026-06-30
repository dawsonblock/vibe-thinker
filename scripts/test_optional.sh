#!/usr/bin/env bash
# Optional test suite — requires optional dependencies installed.
# Runs logic/embeddings/federation/web/nli tests. Sandbox tests are
# handled by scripts/test_docker.sh (they need Docker + the sandbox
# extra, not just pip packages).
# Run: pip install -e ".[dev,test,logic,embeddings,federation,web,nli]"
set -euo pipefail
python3 -m pytest -m "logic or embeddings or federation or web or nli"
