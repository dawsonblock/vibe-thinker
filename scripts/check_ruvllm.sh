#!/usr/bin/env bash
# RuvLLM validation — check the Rust extension builds and imports.
# Only after this passes can RuvLLM be promoted from experimental.
#
# The default ruvllm_py build is a STUB — SUPPORTS_INFERENCE is False
# unless built with --features candle (CPU) or --features inference-metal
# (Apple Silicon). This script uses the correct feature flags.
#
# Env-aware: if a virtualenv is already active (VIRTUAL_ENV set), it is
# reused as-is. Standalone invocation creates and reuses .venv-ruvllm.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "[ruvllm] using active venv: $VIRTUAL_ENV"
else
  VENV="$ROOT/.venv-ruvllm"
  if [ ! -d "$VENV" ]; then
    echo "[ruvllm] creating controlled venv at $VENV"
    python3 -m venv "$VENV"
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -m pip install -U pip >/dev/null
fi

# Pin the Python interpreter so every command uses the same one.
# This prevents the classic env-drift bug where maturin installs into
# one Python but the import check runs in another.
PYTHON="$(command -v python)"
echo "[ruvllm] Python: $PYTHON"
"$PYTHON" -c "import sys; print(f'[ruvllm] sys.executable: {sys.executable}')"

cd "$ROOT/ruvllm_py"

echo "[ruvllm] cargo check..."
cargo check

echo "[ruvllm] installing maturin..."
"$PYTHON" -m pip install maturin

# Detect Apple Silicon for inference-metal; otherwise use candle (CPU).
ARCH="$(uname -m)"
if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
    echo "[ruvllm] building with inference-metal (Apple Silicon)..."
    maturin develop --release --features inference-metal
else
    echo "[ruvllm] building with candle (CPU)..."
    maturin develop --release --features candle
fi

echo "[ruvllm] import check..."
"$PYTHON" -c "import ruvllm_py; print('ruvllm import ok')"

echo "[ruvllm] inference support check..."
"$PYTHON" -c "import ruvllm_py; assert getattr(ruvllm_py, 'SUPPORTS_INFERENCE', False), 'stub build — rebuild with --features candle or --features inference-metal'; print('ruvllm inference ok')"

echo ""
echo "RuvLLM validation PASSED."
