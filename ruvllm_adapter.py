"""
RuvLLM inference backend adapter.

RuvLLM (ruvnet/ruvllm) is a Rust-based LLM inference engine that supports
TurboQuant KV cache compression (asymmetric K/V quantization — see
AGENTS.md "TurboQuant+"). On 16GB machines it lets long-context models
fit that would otherwise blow up RAM with the stock llama-server.

vibe-thinker talks to inference servers over an OpenAI-compatible HTTP
API (the ``/completion`` endpoint). RuvLLM exposes the same API, so the
simplest integration is to point ``--vibe`` (or
``VIBE_THINKER_URL``) at the RuvLLM server port — no Python changes
needed. This is the integration plan's "Step 1: Inference Swap (Low
Risk)".

This module provides two things:

1. **HTTP sidecar mode** (zero-code, recommended start):
   :class:`RuvLLMHTTPBackend` is a thin wrapper that documents the
   RuvLLM HTTP endpoint and the recommended TurboQuant KV cache flags.
   It is NOT used directly by the orchestrator — instead, the CLI flag
   ``--ruvllm-url`` sets ``VIBE_THINKER_URL`` to the RuvLLM port, and
   the existing HTTP path in ``vibe_clr_async._call_model_http``
   handles the rest. This class exists for documentation, health
   checks, and configuration validation.

2. **In-process PyO3 binding mode** (zero-HTTP-overhead, optional):
   :class:`RuvLLMBinding` wraps a hypothetical ``ruvllm_py`` Python
   extension built from the Rust crate via PyO3/maturin. When the
   binding is installed (``import ruvllm_py``), it can be used as a
   drop-in replacement for the ``llama-cpp-python`` in-process backend
   in ``vibe_clr_async._init_local_backend``, maintaining the
   zero-HTTP-overhead pool mode introduced in v0.3.6. When the binding
   is NOT installed, constructing :class:`RuvLLMBinding` raises
   ``ImportError`` with build instructions — the caller falls back to
   the HTTP path or llama-cpp-python.

The binding does not exist yet (the ruvllm crate would need a PyO3
wrapper). This module documents the integration contract so that when
the binding is built, it drops in without orchestrator changes.

Integration plan reference: Phase 2.1 — "Integrate RuvLLM for the
In-Process Backend". The plan's action items:
  1. Run ruvllm as the primary local inference server with TurboQuant.
  2. Update _init_local_backend() to target the RuvLLM API endpoints.
  3. (Optional) Wrap the ruvllm crate via PyO3/maturin for direct binding.

v1.1 direction decision — sync ``__call__``-compatible, NOT async batched:
  The original integration plan conflated two directions:
    (a) A sync ``__call__``-compatible binding that drops into the
        existing ``_init_local_backend`` pool (thread) as a
        Llama replacement. The orchestrator's ``_call_model_inprocess``
        already runs sync calls in a thread/process executor, so a sync
        binding gets parallelism for free — no new async surface needed.
    (b) An async batched engine that exposes ``async def generate_batch``
        and bypasses the executor entirely, calling the Rust engine's
        batch API directly.

  Decision: pursue (a), not (b). Rationale:
    - The existing pool infrastructure (thread queue, process pool with
      RAM guardrail, per-instance grammar) already handles parallelism.
      A sync binding reuses all of it — zero new code paths, zero new
      failure modes.
    - An async batched engine would require a new ``_call_model_batch``
      dispatch path, new tests, and a new concurrency model that
      bypasses the semaphore/executor guardrails. The marginal latency
      win (avoiding thread context switches) is negligible compared to
      the inference time itself (100ms+ for a 0.5B model).
    - The sync binding is a strict subset of the async engine's
      functionality — if a batched API is needed later, it can be added
      on top of the sync binding without breaking the pool integration.
    - The process-pool mode (v1.1) already gives each worker its own
      GIL, so Python-side lock contention is not a reason to go async.

  What this means for the binding contract:
    - ``RuvLLMBinding.__call__(prompt, max_tokens, temperature, stop,
      grammar)`` must return ``{"choices": [{"text": "..."}]}`` — the
      same dict shape as ``llama_cpp.Llama.__call__``.
    - The ``grammar`` parameter accepts a raw GBNF string (not a
      ``LlamaGrammar`` object) — this is already handled in
      ``_call_model_inprocess`` and ``_process_pool_worker_call``.
    - No async methods needed. The binding is used identically in
      thread-pool and process-pool modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TurboQuantConfig:
    """Recommended TurboQuant KV cache configuration for RuvLLM.

    See AGENTS.md "TurboQuant+" for the asymmetric compression findings:
    V tolerates aggressive compression, K does not. Never start with
    symmetric turbo K compression (both sides turbo*) — that's where
    models break.

    Attributes:
        cache_type_k: K cache type. Default ``q8_0`` (safe). The K cache
            is sensitive to compression — do not use turbo* for K
            without testing.
        cache_type_v: V cache type. Default ``turbo3`` (recommended).
            V tolerates aggressive compression with <1.5% PPL loss.
    """
    cache_type_k: str = "q8_0"
    cache_type_v: str = "turbo3"

    def as_cli_args(self) -> list:
        """Return llama-server/RuvLLM CLI flags for this config."""
        return [
            "--cache-type-k", self.cache_type_k,
            "--cache-type-v", self.cache_type_v,
        ]


# Recommended presets from AGENTS.md (start light, then compress)
TURBOQUANT_SAFE = TurboQuantConfig(cache_type_k="f16", cache_type_v="turbo4")
TURBOQUANT_CONSERVATIVE = TurboQuantConfig(cache_type_k="q8_0", cache_type_v="turbo4")
TURBOQUANT_DEFAULT = TurboQuantConfig(cache_type_k="q8_0", cache_type_v="turbo3")
TURBOQUANT_AGGRESSIVE_V = TurboQuantConfig(cache_type_k="q8_0", cache_type_v="turbo2")


class RuvLLMHTTPBackend:
    """Documents and validates the RuvLLM HTTP sidecar integration.

    RuvLLM exposes an OpenAI-compatible HTTP API (the same ``/completion``
    and ``/chat/completions`` endpoints that llama-server provides). The
    orchestrator's existing HTTP path
    (``vibe_clr_async.VibeThinkerCLRAsync._call_model_http``) works
    unchanged — you just point ``VIBE_THINKER_URL`` at the RuvLLM port.

    This class provides:
      - :meth:`health_check`: verify the RuvLLM server is up.
      - :meth:`recommended_start_command`: the CLI invocation with
        TurboQuant KV cache flags, suitable for the local setup
        documented in AGENTS.md.
      - :attr:`turboquant`: the TurboQuant config to use.

    Usage:
      backend = RuvLLMHTTPBackend(port=8080, model_path="~/models/vibethinker-3b-q4_k_m.gguf")
      cmd = backend.recommended_start_command()
      # Run cmd in a terminal, then set VIBE_THINKER_URL=http://127.0.0.1:8080
    """

    def __init__(
        self,
        port: int = 8080,
        host: str = "127.0.0.1",
        model_path: Optional[str] = None,
        n_ctx: int = 8192,
        n_threads: int = 6,
        turboquant: Optional[TurboQuantConfig] = None,
    ):
        self.port = port
        self.host = host
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.turboquant = turboquant or TURBOQUANT_DEFAULT

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def recommended_start_command(self) -> list:
        """Return the recommended RuvLLM/llama-server start command.

        This is the command from AGENTS.md "TurboQuant+" with the
        recommended asymmetric KV cache config. Replace ``ruvllm-server``
        with ``llama-server`` if using the TurboQuant+ fork directly.
        """
        if not self.model_path:
            raise ValueError("model_path is required to build the start command")
        cmd = [
            "ruvllm-server",  # or: llama-server (TurboQuant+ fork)
            "-m", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(self.n_ctx),
            "-t", str(self.n_threads),
            "--jinja",
        ]
        cmd.extend(self.turboquant.as_cli_args())
        return cmd

    async def health_check(self) -> bool:
        """Check if the RuvLLM server is responding.

        Returns True if the server is up and reports OK, False otherwise.
        """
        import aiohttp
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=3.0)
            ) as session:
                async with session.get(f"{self.base_url}/health") as resp:
                    return resp.status == 200
        except (aiohttp.ClientError, OSError):
            return False


# Cache for the ONNX embedding model (loaded once, reused across calls).
_ONNX_MODEL_CACHE: dict = {}


def _onnx_embed(text: str) -> List[float]:
    """Generate an embedding using the ONNX all-MiniLM-L6-v2 model.

    Downloads the quantized ONNX model from HuggingFace Hub on first use
    (cached by huggingface_hub in ~/.cache/huggingface). Runs on CPU via
    onnxruntime — no PyTorch dependency.

    Returns a 384-dim L2-normalized embedding vector.

    Raises ImportError if onnxruntime or huggingface_hub is not installed.
    Raises RuntimeError if the model cannot be downloaded or loaded.
    """
    import onnxruntime as ort  # type: ignore

    cache_key = "model"
    cached = _ONNX_MODEL_CACHE.get(cache_key)
    if cached is not None:
        session, tokenizer = cached
    else:
        # Download the quantized ONNX model + tokenizer from HuggingFace.
        from huggingface_hub import hf_hub_download  # type: ignore

        repo_id = "sentence-transformers/all-MiniLM-L6-v2"
        onnx_path = hf_hub_download(repo_id, "onnx/model_quantized.onnx")
        # The tokenizer files are in the same repo.
        tokenizer_dir = hf_hub_download(repo_id, "tokenizer.json")
        import os as _os
        tokenizer_dir = _os.path.dirname(tokenizer_dir)

        # Load the ONNX session (CPU, single thread for portability).
        session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )

        # Load the tokenizer (HuggingFace tokenizers library).
        from tokenizers import Tokenizer  # type: ignore
        tokenizer = Tokenizer.from_file(
            _os.path.join(tokenizer_dir, "tokenizer.json")
        )
        tokenizer.enable_padding(length=None, pad_id=0, pad_token="[PAD]")
        _ONNX_MODEL_CACHE[cache_key] = (session, tokenizer)

    # Tokenize and run inference.
    import numpy as np  # type: ignore

    encoded = tokenizer.encode(text)
    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    # Some models use token_type_ids, some don't.
    if "token_type_ids" in {i.name for i in session.get_inputs()}:
        inputs["token_type_ids"] = token_type_ids

    outputs = session.run(None, inputs)
    # Mean-pool over the sequence dimension using the attention mask.
    token_embeddings = outputs[0]  # (1, seq_len, hidden_dim)
    mask = attention_mask[:, :, None].astype(np.float32)
    summed = (token_embeddings * mask).sum(axis=1)
    counts = mask.sum(axis=1)
    counts = np.clip(counts, a_min=1e-9, a_max=None)
    mean_pooled = summed / counts  # (1, hidden_dim)

    # L2 normalize.
    norm = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
    norm = np.clip(norm, a_min=1e-9, a_max=None)
    normalized = mean_pooled / norm
    return normalized[0].tolist()


class RuvLLMBinding:
    """In-process RuvLLM binding via PyO3 (zero HTTP overhead).

    Wraps a hypothetical ``ruvllm_py`` Python extension built from the
    Rust ruvllm crate via PyO3/maturin. When available, this provides
    the same zero-HTTP-overhead in-process inference as
    ``llama-cpp-python``, but with TurboQuant KV cache compression
    native to the Rust engine.

    The binding does NOT exist yet — the ruvllm crate would need a PyO3
    wrapper. This class documents the integration contract:

      - ``ruvllm_py.Engine(model_path, n_ctx, n_threads, cache_type_k,
         cache_type_v)`` — construct an inference engine.
      - ``engine.complete(prompt, max_tokens, temperature, stop, grammar)``
        — synchronous completion, returns a dict with ``choices[0].text``
        (same shape as llama_cpp.Llama).
      - The engine is NOT thread-safe; pool mode uses multiple instances
        (same pattern as ``vibe_clr_async._init_local_backend`` pool mode).

    When ``ruvllm_py`` is not installed, constructing this class raises
    ``ImportError`` with build instructions. The caller (e.g.
    ``vibe_clr_async._init_local_backend``) should catch this and fall
    back to ``llama-cpp-python`` or HTTP.

    Build instructions (when the PyO3 wrapper is written):
      cd ruvllm-py  # the PyO3 wrapper repo
      maturin develop --release  # builds and installs the extension

    Args:
        model_path: path to the .gguf model file.
        n_ctx: context window size.
        n_threads: number of CPU threads.
        turboquant: KV cache compression config.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = 8,
        turboquant: Optional[TurboQuantConfig] = None,
    ):
        try:
            import ruvllm_py  # type: ignore
            # v0.4.0: check for the compiled Engine class, not just the
            # import. A bare directory named ruvllm_py (the Rust project
            # scaffold) is importable but doesn't have Engine.
            if not hasattr(ruvllm_py, "Engine"):
                raise AttributeError(
                    "ruvllm_py is importable but has no 'Engine' class — "
                    "the PyO3 extension is not compiled. Run "
                    "`maturin develop --release` in the ruvllm_py directory."
                )
        except ImportError as e:
            raise ImportError(
                "RuvLLMBinding requires the 'ruvllm_py' PyO3 extension, "
                "which is not yet published. To use RuvLLM in-process, "
                "build the PyO3 wrapper from the ruvllm Rust crate:\n"
                "  cd ruvllm-py && maturin develop --release\n"
                "Until then, use RuvLLM as an HTTP sidecar via --ruvllm-url "
                "(points VIBE_THINKER_URL at the RuvLLM server port)."
            ) from e
        except AttributeError as e:
            raise ImportError(
                "RuvLLMBinding requires the 'ruvllm_py' PyO3 extension, "
                "which is not yet published. To use RuvLLM in-process, "
                "build the PyO3 wrapper from the ruvllm Rust crate:\n"
                "  cd ruvllm-py && maturin develop --release\n"
                "Until then, use RuvLLM as an HTTP sidecar via --ruvllm-url "
                "(points VIBE_THINKER_URL at the RuvLLM server port)."
            ) from e

        self._ruvllm_py = ruvllm_py
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.turboquant = turboquant or TURBOQUANT_DEFAULT
        self._engine = ruvllm_py.Engine(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            cache_type_k=self.turboquant.cache_type_k,
            cache_type_v=self.turboquant.cache_type_v,
        )

    def __call__(
        self,
        prompt: str,
        max_tokens: int = 8192,
        temperature: float = 1.0,
        stop=None,
        grammar=None,
    ) -> dict:
        """Synchronous completion — same return shape as llama_cpp.Llama.

        Returns a dict with ``{"choices": [{"text": "..."}]}`` so it can
        be parsed by ``vibe_clr_async._parse_inprocess_response``.
        """
        stop_list = stop if stop is not None else ["<|im_end|>"]
        grammar_str = None
        if grammar is not None:
            # If it's a LlamaGrammar-like object with a serialized form,
            # extract the string; otherwise pass through.
            grammar_str = getattr(grammar, "raw", str(grammar)) if grammar else None
        return self._engine.complete(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_list,
            grammar=grammar_str,
        )

    def get_embeddings(self, text: str, dim: int = 384) -> List[float]:
        """Generate a semantic text embedding vector (v3.1).

        Embedding source priority (fail-closed — never returns fake
        vectors):
          1. **Rust-native** — if the ruvllm_py Engine exposes an
             ``embed`` method (Phase 4.2 of the integration plan), use
             it directly. Zero Python overhead.
          2. **sentence-transformers** — if the ``sentence-transformers``
             package is installed, load ``all-MiniLM-L6-v2`` (384-dim)
             and encode. This is the same model the trajectory store
             uses, so embeddings are compatible.
          3. **ONNX** — if ``onnxruntime`` is installed, download the
             quantized ``all-MiniLM-L6-v2`` ONNX model via
             ``huggingface_hub`` and run it on CPU. No PyTorch
             dependency — runs in milliseconds.
          4. **Fail-closed** — if none of the above are available,
             return ``[]`` (empty list). The caller (the orchestrator's
             SONA recorder) treats ``[]`` as "no embedding" and skips
             the recording. NEVER returns fake/hash-based vectors —
             those would silently corrupt SONA memory retrieval and
             clustering with non-semantic similarity.

        The v3.0 n-gram hashing fallback was removed because it
        produced non-semantic vectors that broke SONA's clustering
        (structurally similar but semantically unrelated texts clustered
        together). Fail-closed is the honest behavior.

        Args:
            text: The text to embed.
            dim: Embedding dimension (default 384, matching
                all-MiniLM-L6-v2). Ignored by the Rust engine and
                sentence-transformers (they use the model's native dim).

        Returns:
            A list of floats representing the semantic embedding, or
            ``[]`` if no embedding source is available (fail-closed).
        """
        if not text or not text.strip():
            return []

        # 1. Rust-native embeddings (v1.0 Phase 2.2 — implemented). The
        # ruvllm_py Engine.embed() runs a real BERT model
        # (all-MiniLM-L6-v2) via candle-transformers with GIL release.
        # When the binding is built with --features candle this is the
        # zero-overhead primary path; the stub build returns [] and
        # falls through to the Python sources below.
        engine = getattr(self, "_engine", None)
        if engine is not None and hasattr(engine, "embed"):
            try:
                return list(engine.embed(text))
            except Exception as e:
                print(f"[RuvLLMBinding] Rust embed() failed: {e} — "
                      f"falling back to Python embedding sources")

        # 2. sentence-transformers (same model as the trajectory store).
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            model = getattr(self, "_st_model", None)
            if model is None:
                model = SentenceTransformer("all-MiniLM-L6-v2")
                self._st_model = model  # cache for reuse
            emb = model.encode([text], normalize_embeddings=True)
            # Handle both numpy arrays (real) and plain lists (mocks).
            if hasattr(emb, "tolist"):
                emb = emb.tolist()
            return list(emb[0])
        except ImportError:
            pass  # fall through to ONNX
        except Exception as e:
            print(f"[RuvLLMBinding] sentence-transformers failed: {e} — "
                  f"trying ONNX fallback")

        # 3. ONNX fallback (onnxruntime + huggingface_hub).
        try:
            return _onnx_embed(text)
        except Exception as e:
            print(f"[RuvLLMBinding] ONNX embedding failed: {e} — "
                  f"fail-closed (no embedding)")

        # 4. Fail-closed — no fake vectors.
        return []


def is_ruvllm_binding_available() -> bool:
    """Check if the ruvllm_py in-process binding is installed AND compiled.

    A bare directory named ``ruvllm_py`` (e.g. the Rust project scaffold)
    is importable but doesn't have the compiled ``Engine`` class. We check
    for the ``Engine`` attribute to distinguish the compiled extension from
    the scaffold.
    """
    try:
        import ruvllm_py  # type: ignore  # noqa: F401
        # The compiled extension exposes an Engine class. The Rust
        # scaffold directory does not (it's just source code, not built).
        return hasattr(ruvllm_py, "Engine")
    except ImportError:
        return False
