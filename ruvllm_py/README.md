# ruvllm_py — Python bindings for RuvLLM

Python bindings for the [RuvLLM](https://github.com/ruvnet/ruvllm) Rust
inference engine with TurboQuant KV cache compression. Built with
[PyO3](https://pyo3.rs) and [maturin](https://www.maturin.rs/).

## Status: EXPERIMENTAL (v0.2.0)

The `ruvllm` 2.3 crate is referenced in `Cargo.toml`. The default build
(`cargo build --release`) compiles as a **stub** — `Engine.complete()`
returns an empty response because the real inference path
(`CandleBackend` + `candle-core`) is gated behind the `candle` feature.
Enable real inference with:

```bash
# CPU inference:
cargo build --release --features candle

# Apple Silicon Metal acceleration:
cargo build --release --features inference-metal
```

Without one of these features, the binding compiles and installs but
does NOT perform inference. This binding is **not included in the main
vibe-thinker Python wheel** — it must be built separately with a Rust
toolchain + maturin. The HTTP sidecar path (`--ruvllm-url`) is the
recommended integration mode for most users.

## Why PyO3 instead of HTTP?

The HTTP sidecar approach (`--ruvllm-url`) works today and requires no
Python compilation. The PyO3 binding eliminates HTTP overhead for
ultra-tiny models (e.g. ruvltra 0.5B, ~100+ tok/s) where HTTP latency
dominates inference time:

| Path | Latency overhead |
|---|---|
| HTTP sidecar (aiohttp) | ~2-5ms per call (TCP + JSON) |
| PyO3 in-process | ~0ms (direct function call) |

For a 0.5B model generating 128 tokens at 100 tok/s (1.3s), the HTTP
overhead is 0.15-0.4% — negligible. But for short completions (16 tokens,
0.16s), it's 1.2-3% — measurable. The PyO3 path eliminates it entirely.

## Build

```bash
# Prerequisites: Rust toolchain + maturin
cargo install maturin

# Build and install into the current Python environment:
cd ruvllm_py
maturin develop --release

# Or build a wheel for distribution:
maturin build --release
```

## Usage

```python
from ruvllm_py import Engine

# Same calling convention as llama_cpp.Llama
engine = Engine(
    model_path="~/models/vibethinker-3b-q4_k_m.gguf",
    n_ctx=8192,
    n_threads=6,
    cache_type_k="q8_0",   # TurboQuant K cache
    cache_type_v="turbo3", # TurboQuant V cache
)

# Drop-in replacement for llama_cpp.Llama.__call__
resp = engine(
    prompt="What is 2+2?",
    max_tokens=128,
    temperature=0.7,
    stop=["</s>"],
    grammar='root ::= ...',  # GBNF grammar string
)
print(resp["choices"][0]["text"])
```

## GIL release

The `complete()` method releases the Python GIL during the forward pass
via `py.allow_threads()`. This means the orchestrator's asyncio event
loop is not blocked during generation — other coroutines can run
concurrently while the Rust inference engine generates tokens.

## Thread safety

The model and KV cache state are held in `Arc<Mutex<...>>` for
thread-safe access from pool mode (multiple Python threads calling
`complete()` concurrently via `loop.run_in_executor`).

## Integration with vibe-thinker

When `ruvllm_py` is installed, `vibe_clr_async._init_local_backend`
automatically prefers it over `llama-cpp-python`:

```python
# From vibe_clr_async.py:
from ruvllm_adapter import is_ruvllm_binding_available
if is_ruvllm_binding_available():
    # Use ruvllm_py (zero HTTP overhead, native TurboQuant)
    from ruvllm_py import Engine as RuvLLMEngine
else:
    # Fall back to llama-cpp-python
    from llama_cpp import Llama
```

No configuration needed — the binding is auto-detected.
