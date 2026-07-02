#!/usr/bin/env bash
# demo_setup.sh — install all dependencies needed to run demo_verified_swarm.py
#
# The demo exercises every major subsystem: core reasoning, sandbox, web UI,
# web security, memory/trajectory, AgentDB, federation, and RuvLLM. This
# script installs the full set of optional extras so every phase can run.
#
# Usage:
#   bash scripts/demo_setup.sh          # install into current env
#   bash scripts/demo_setup.sh --venv   # create .venv-demo and install there
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

USE_VENV=false
if [[ "${1:-}" == "--venv" ]]; then
    USE_VENV=true
fi

if $USE_VENV; then
    VENV_DIR="$ROOT/.venv-demo"
    echo "Creating isolated venv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip setuptools wheel
fi

echo "Installing vibe-thinker with all extras for demo..."
cd "$ROOT"
pip install -e ".[dev,test,web,federation,sandbox,embeddings,logic]"

echo ""
echo "Demo dependencies installed."
echo "Run the demo with:"
echo "  python3 demo_verified_swarm.py --verbose"
