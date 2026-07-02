#!/usr/bin/env bash
# demo_setup.sh — install the dependencies needed to run demo_verified_swarm.py
#
# The demo exercises every major subsystem: core reasoning, sandbox, web UI,
# web security, memory/trajectory, AgentDB, federation, and RuvLLM. This
# script installs ONLY the extras the demo actually imports — not every
# optional heavy dependency.
#
# Installed extras:
#   dev         pytest (the demo shells out to pytest for several phases)
#   test        numpy, scikit-learn, cryptography, z3-solver, pyyaml
#               (covers the logic verifier + vector store + agentdb test)
#   web         fastapi, uvicorn, websockets, pydantic, httpx
#   federation  redis, fakeredis, cryptography
#   sandbox     docker, wasmtime (Docker tests skip gracefully if Docker
#               is absent; wasmtime powers the static-analysis fallback)
#
# Deliberately NOT installed (heavy, not imported by any demo code path):
#   embeddings  sentence-transformers + torch + faiss-cpu (~2+ GB)
#   models      llama-cpp-python, onnxruntime, huggingface_hub, ...
#   nli         transformers + torch
#   rust        maturin (only needed to BUILD the ruvllm_py binding)
# The demo's AgentDB phase uses a FakeVectorStore / dead-sidecar fail-closed
# path, so the embeddings extra is not required.
#
# Usage:
#   bash scripts/demo_setup.sh          # install into current env
#   bash scripts/demo_setup.sh --venv   # create .venv-demo and install there
#
# OFFICIAL MAC SETUP PATH (the freeze plan's standardized sequence):
#   python3 -m venv .venv
#   source .venv/bin/activate
#   python -m pip install -U pip setuptools wheel
#   bash scripts/demo_setup.sh --venv
#
# Then verify + run the demo:
#   python -m compileall -q .
#   python -m pytest tests/test_release_zip_hygiene.py tests/test_web_security.py tests/test_job_queue.py -q
#   python demo_verified_swarm.py --verbose --json-out gate_results/demo_verified_swarm.json
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

echo "Installing vibe-thinker with the demo's required extras..."
echo "  (dev, test, web, federation, sandbox — NOT embeddings/models/nli/rust)"
cd "$ROOT"
pip install -e ".[dev,test,web,federation,sandbox]"

echo ""
echo "Demo dependencies installed."
echo "Run the demo with:"
echo "  python3 demo_verified_swarm.py --verbose"
echo ""
echo "Note: Docker-dependent sandbox tests skip gracefully if Docker is"
echo "not running. The embeddings/models/nli/rust extras are not needed"
echo "for any demo code path."
