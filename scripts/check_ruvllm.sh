#!/usr/bin/env bash
# RuvLLM validation — check the Rust extension builds and imports.
# Only after this passes can RuvLLM be promoted from experimental.
set -euo pipefail
cd ruvllm_py
cargo check
python -m pip install maturin
maturin develop
python -c "import ruvllm_py; print('ruvllm import ok')"
python -c "import ruvllm_py; assert getattr(ruvllm_py, 'SUPPORTS_INFERENCE', False), 'stub build — rebuild with --features candle'; print('ruvllm inference ok')"
echo ""
echo "RuvLLM validation PASSED."
