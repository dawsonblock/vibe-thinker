#!/usr/bin/env bash
# Optional test suite — requires optional dependencies installed.
# Run: pip install -e ".[dev,test,logic,embeddings,federation,web,sandbox]"
set -euo pipefail
python3 -m pytest -m "logic or embeddings or federation or web or sandbox"
