"""Centralized configuration for vibe-thinker.

Extracts hardcoded timeouts, intervals, and limits into a single module
with environment variable overrides. This makes operational parameters
tunable without code changes.

Usage:
    from vt_config import config

    await asyncio.sleep(config.claim_poll_interval)
    timeout = config.job_execution_timeout

All values can be overridden via environment variables:
  VIBE_THINKER_CLAIM_POLL_INTERVAL=2.0
  VIBE_THINKER_HEARTBEAT_INTERVAL=60.0
  VIBE_THINKER_JOB_EXECUTION_TIMEOUT=600.0
  VIBE_THINKER_REAPER_CLAIM_TIMEOUT=180.0
  VIBE_THINKER_SANDBOX_TIMEOUT=5.0
  VIBE_THINKER_HTTP_TIMEOUT=5.0
  VIBE_THINKER_RAM_SAFETY_MULTIPLIER=1.5
  VIBE_THINKER_FALLBACK_RAM_MB=4096
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


@dataclass(frozen=True)
class Config:
    """Operational configuration with env var overrides."""

    # --- Federation timing ---
    claim_poll_interval: float = 2.0  # seconds between claim polls
    heartbeat_interval: float = 60.0  # seconds between heartbeats
    job_execution_timeout: float = 600.0  # max job runtime (10 min)
    reaper_claim_timeout: float = 180.0  # zombie claim timeout (3x heartbeat)

    # --- Sandbox ---
    sandbox_timeout: float = 5.0  # code execution timeout

    # --- HTTP ---
    http_timeout: float = 5.0  # default HTTP request timeout
    http_job_timeout: float = 600.0  # long-running job HTTP timeout

    # --- Hardware guardrail ---
    ram_safety_multiplier: float = 1.5
    kv_cache_bytes_per_token_per_layer: int = 1024
    default_layers: int = 32
    fallback_available_ram_mb: int = 4096

    # --- Input validation ---
    max_query_length: int = 10000
    max_job_id_length: int = 128
    max_worker_id_length: int = 128
    max_error_length: int = 5000
    max_priority: int = 1000


# Build config from env vars.
config = Config(
    claim_poll_interval=_env_float("VIBE_THINKER_CLAIM_POLL_INTERVAL", 2.0),
    heartbeat_interval=_env_float("VIBE_THINKER_HEARTBEAT_INTERVAL", 60.0),
    job_execution_timeout=_env_float("VIBE_THINKER_JOB_EXECUTION_TIMEOUT", 600.0),
    reaper_claim_timeout=_env_float("VIBE_THINKER_REAPER_CLAIM_TIMEOUT", 180.0),
    sandbox_timeout=_env_float("VIBE_THINKER_SANDBOX_TIMEOUT", 5.0),
    http_timeout=_env_float("VIBE_THINKER_HTTP_TIMEOUT", 5.0),
    http_job_timeout=_env_float("VIBE_THINKER_HTTP_JOB_TIMEOUT", 600.0),
    ram_safety_multiplier=_env_float("VIBE_THINKER_RAM_SAFETY_MULTIPLIER", 1.5),
    kv_cache_bytes_per_token_per_layer=_env_int(
        "VIBE_THINKER_KV_CACHE_BYTES", 1024
    ),
    default_layers=_env_int("VIBE_THINKER_DEFAULT_LAYERS", 32),
    fallback_available_ram_mb=_env_int("VIBE_THINKER_FALLBACK_RAM_MB", 4096),
    max_query_length=_env_int("VIBE_THINKER_MAX_QUERY_LENGTH", 10000),
    max_job_id_length=_env_int("VIBE_THINKER_MAX_JOB_ID_LENGTH", 128),
    max_worker_id_length=_env_int("VIBE_THINKER_MAX_WORKER_ID_LENGTH", 128),
    max_error_length=_env_int("VIBE_THINKER_MAX_ERROR_LENGTH", 5000),
    max_priority=_env_int("VIBE_THINKER_MAX_PRIORITY", 1000),
)
