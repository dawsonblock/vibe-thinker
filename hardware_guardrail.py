"""Dynamic hardware guardrails for model loading (Phase 4.1).

Prevents OOM crashes by checking model size against available RAM before
loading. When the estimated RAM cost exceeds the available RAM (with a
safety margin), the guardrail refuses to load and returns a clear error
message — instead of letting the process crash with an opaque OOM kill.

The guardrail is intentionally conservative:
  - Model file size is a LOWER bound on actual RAM (KV cache + runtime
    overhead add ~20-50% on top). We apply a safety multiplier to account
    for this.
  - When psutil is not installed, we fall back to a conservative static
    cap (4GB available) so the guardrail still prevents egregious OOM on
    small machines. The user can install psutil for accurate measurement.
  - When the model path is not a local file (e.g. a HuggingFace repo_id),
    we can't estimate the size without a network fetch — the guardrail
    is skipped (returns OK with a note).

Usage:
    from hardware_guardrail import check_model_fits_ram, GuardrailResult
    result = check_model_fits_ram("model.gguf", n_ctx=4096, pool_size=1)
    if not result.ok:
        print(result.error)
        return  # refuse to load
    # proceed with loading
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


# Safety multiplier: actual RAM usage is typically 1.2-1.5x the model
# file size (KV cache + runtime overhead). We use 1.5x to be conservative
# and avoid OOM on edge-case hardware.
_RAM_SAFETY_MULTIPLIER = 1.5

# KV cache RAM per token per layer (bytes). This is intentionally
# conservative — actual KV cache size depends on the model's hidden
# dimension, number of attention heads, and quantization. For a typical
# 7B model with 4096 hidden dim, 32 layers, and q8_0 K + turbo3 V
# cache, the actual KV cache is ~0.5-2 KB per token per layer. We use
# 1 KB as a round, conservative estimate that overestimates slightly
# (safe — better to refuse a load than to OOM-crash).
_KV_CACHE_BYTES_PER_TOKEN_PER_LAYER = 1024

# Default layer count for estimating KV cache (most 3B-7B models have
# 32-40 layers). Used only when we can't determine the actual count.
_DEFAULT_LAYERS = 32

# Conservative static RAM cap when psutil is not installed.
_FALLBACK_AVAILABLE_RAM_MB = 4096


@dataclass
class GuardrailResult:
    """Result of a hardware guardrail check.

    Attributes:
        ok: True if the model is estimated to fit in available RAM.
        model_ram_mb: estimated RAM cost of loading the model (MB), or
            None if it could not be estimated.
        available_ram_mb: available RAM (MB), or None if unknown.
        error: a human-readable error message when ok is False, or None.
        warning: a human-readable warning when ok is True but there is
            a concern (e.g. couldn't estimate model size).
    """
    ok: bool
    model_ram_mb: Optional[int] = None
    available_ram_mb: Optional[int] = None
    error: Optional[str] = None
    warning: Optional[str] = None


def _estimate_model_ram_mb(
    model_path: str,
    n_ctx: int = 4096,
    pool_size: int = 1,
) -> Optional[int]:
    """Estimate the total RAM cost of loading a model.

    Returns the estimated RAM in MB, or None if the path is not a local
    file (can't estimate without a network fetch).

    The estimate includes:
      - Model file size (weights) x pool_size (each instance loads its
        own copy in RAM).
      - KV cache: roughly n_ctx * layers * bytes_per_token_per_layer,
        x pool_size.
      - Safety multiplier (1.5x) for runtime overhead.
    """
    if not os.path.exists(model_path):
        return None
    file_size_mb = int(os.path.getsize(model_path) / (1024 * 1024))
    # KV cache estimate (MB): n_ctx * layers * bytes / 1MB
    kv_cache_mb = int(
        n_ctx * _DEFAULT_LAYERS * _KV_CACHE_BYTES_PER_TOKEN_PER_LAYER
        / (1024 * 1024)
    )
    # Total = (weights + kv_cache) * pool_size * safety_multiplier
    total = int((file_size_mb + kv_cache_mb) * pool_size * _RAM_SAFETY_MULTIPLIER)
    return total


def _available_ram_mb() -> Optional[int]:
    """Return available RAM in MB, or a conservative fallback."""
    try:
        import psutil
        return int(psutil.virtual_memory().available / (1024 * 1024))
    except ImportError:
        return _FALLBACK_AVAILABLE_RAM_MB


def check_model_fits_ram(
    model_path: str,
    n_ctx: int = 4096,
    pool_size: int = 1,
    available_ram_mb: Optional[int] = None,
) -> GuardrailResult:
    """Check whether a model is estimated to fit in available RAM.

    Args:
        model_path: path to the .gguf model file (or a HuggingFace
            repo_id — size can't be estimated for non-local paths).
        n_ctx: context window size (affects KV cache RAM).
        pool_size: number of model instances to load (each adds its
            own RAM cost).
        available_ram_mb: override available RAM (for testing). If None,
            measured via psutil or the conservative fallback.

    Returns:
        A GuardrailResult. When ok is False, the error message explains
        the RAM shortfall and suggests remediation. When the model size
        can't be estimated (non-local path), ok is True with a warning.
    """
    model_ram = _estimate_model_ram_mb(model_path, n_ctx, pool_size)
    if model_ram is None:
        # Can't estimate — skip the guardrail but warn.
        return GuardrailResult(
            ok=True,
            warning=(
                f"Could not estimate model size for '{model_path}' "
                f"(not a local file). Skipping RAM guardrail — proceed "
                f"with caution. If the model is large, watch for OOM."
            ),
        )

    if available_ram_mb is None:
        available_ram_mb = _available_ram_mb()

    if model_ram > available_ram_mb:
        shortfall = model_ram - available_ram_mb
        return GuardrailResult(
            ok=False,
            model_ram_mb=model_ram,
            available_ram_mb=available_ram_mb,
            error=(
                f"Model estimated to need ~{model_ram}MB RAM "
                f"({pool_size}x instance(s) x "
                f"~{model_ram // max(pool_size, 1)}MB each + KV cache "
                f"for n_ctx={n_ctx}), but only ~{available_ram_mb}MB "
                f"is available (shortfall: ~{shortfall}MB). "
                f"Options: (1) use a smaller model, (2) reduce "
                f"--local-specialist-pool-size (currently {pool_size}), "
                f"(3) reduce --local-specialist-n-ctx (currently "
                f"{n_ctx}), (4) free RAM by closing other processes, "
                f"(5) install psutil for accurate measurement "
                f"(pip install psutil)."
            ),
        )

    # Fits — but warn if we're using the fallback RAM estimate.
    warning = None
    try:
        import psutil  # noqa: F401
    except ImportError:
        warning = (
            f"Using fallback RAM estimate ({_FALLBACK_AVAILABLE_RAM_MB}MB) "
            f"because psutil is not installed. Install with "
            f"pip install psutil for accurate measurement."
        )

    return GuardrailResult(
        ok=True,
        model_ram_mb=model_ram,
        available_ram_mb=available_ram_mb,
        warning=warning,
    )
