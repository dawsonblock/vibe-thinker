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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


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
        except ImportError as e:
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


def is_ruvllm_binding_available() -> bool:
    """Check if the ruvllm_py in-process binding is installed."""
    try:
        import ruvllm_py  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False
