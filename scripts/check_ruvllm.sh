#!/usr/bin/env bash
# RuvLLM validation — check the Rust extension builds and imports.
# Only after this passes can RuvLLM be promoted from experimental.
#
# The default ruvllm_py build is a STUB — SUPPORTS_INFERENCE is False
# unless built with --features candle (CPU) or --features inference-metal
# (Apple Silicon). This script uses the correct feature flags.
set -euo pipefail
cd ruvllm_py

echo "[ruvllm] cargo check..."
cargo check

echo "[ruvllm] installing maturin..."
python -m pip install maturin

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
python -c "import ruvllm_py; print('ruvllm import ok')"

echo "[ruvllm] inference support check..."
python -c "import ruvllm_py; assert getattr(ruvllm_py, 'SUPPORTS_INFERENCE', False), 'stub build — rebuild with --features candle or --features inference-metal'; print('ruvllm inference ok')"

echo ""
echo "RuvLLM validation PASSED."
