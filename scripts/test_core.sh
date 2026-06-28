#!/usr/bin/env bash
# Core test suite — no optional dependencies required.
# This is the non-negotiable green gate.
set -euo pipefail
python3 -m compileall -q .
python3 -m pytest --strict-markers \
  -m "not logic and not embeddings and not federation and not web and not sandbox and not nli and not integration"
python3 rfsn_cli.py --help
