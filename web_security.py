"""Shared security middleware for vibe-thinker FastAPI apps.

Provides:
  - API key authentication (Header-based, constant-time comparison)
  - Simple in-memory rate limiting (per-IP, sliding window)
  - CORS configuration with explicit allowed origins
  - Request body size limiting

Usage:
    from web_security import configure_security

    app = FastAPI(...)
    configure_security(
        app,
        api_key="my-secret-key",       # None = no auth (dev mode)
        allowed_origins=["http://localhost:3000"],
        rate_limit_per_minute=60,
        max_request_body_bytes=10 * 1024 * 1024,  # 10 MB
    )

When ``api_key`` is None, auth is skipped (for local dev). In production,
set the key via environment variable (e.g. ``VIBE_THINKER_API_KEY``).
"""

from __future__ import annotations

import hmac
import os
import time
from collections import defaultdict, deque
from typing import Callable, List, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


def _get_api_key_from_env() -> Optional[str]:
    """Read API key from environment. Returns None if not set."""
    return os.environ.get("VIBE_THINKER_API_KEY") or None


def _constant_time_eq(a: Optional[str], b: Optional[str]) -> bool:
    """Constant-time string comparison to prevent timing attacks."""
    if a is None or b is None:
        return a is b
    return hmac.compare_digest(a, b)


class _RateLimiter:
    """Simple in-memory sliding-window rate limiter (per-IP).

    Not suitable for multi-process deployments — use Redis-backed rate
    limiting for HA. Sufficient for single-process federation/web servers.
    """

    def __init__(self, max_requests: int, window_seconds: int = 60):
        self._max = max_requests
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> bool:
        """Returns True if the request is allowed, False if rate-limited."""
        now = time.time()
        cutoff = now - self._window
        hits = self._hits[key]
        # Evict expired entries.
        while hits and hits[0] < cutoff:
            hits.popleft()
        if len(hits) >= self._max:
            return False
        hits.append(now)
        return True


def configure_security(
    app: FastAPI,
    *,
    api_key: Optional[str] = None,
    allowed_origins: Optional[List[str]] = None,
    rate_limit_per_minute: int = 0,
    max_request_body_bytes: int = 0,
    exempt_paths: Optional[set] = None,
) -> None:
    """Configure security middleware on a FastAPI app.

    Args:
        api_key: If set, all requests must include ``X-API-Key`` header
            matching this value. If None, checks the
            ``VIBE_THINKER_API_KEY`` env var. If neither is set, auth
            is skipped (dev mode).
        allowed_origins: List of allowed CORS origins. If None, defaults
            to localhost origins only. Use ["*"] to allow all (not
            recommended for production).
        rate_limit_per_minute: Max requests per IP per minute. 0 = disabled.
        max_request_body_bytes: Max request body size in bytes. 0 = disabled.
        exempt_paths: Set of paths that skip auth (e.g. {"/health"}).
    """
    # Resolve API key: explicit param > env var > None (no auth).
    effective_key = api_key or _get_api_key_from_env()
    exempt = exempt_paths or set()

    # --- CORS ---
    if allowed_origins is None:
        allowed_origins = [
            "http://localhost",
            "http://localhost:3000",
            "http://localhost:8000",
            "http://127.0.0.1",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000",
        ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # --- Rate limiter ---
    limiter = _RateLimiter(rate_limit_per_minute) if rate_limit_per_minute > 0 else None

    # --- Auth + rate limit + body size middleware ---
    @app.middleware("http")
    async def security_middleware(request: Request, call_next):
        path = request.url.path

        # Skip auth for exempt paths (e.g. /health).
        if path not in exempt and effective_key is not None:
            provided = request.headers.get("X-API-Key")
            if not _constant_time_eq(provided, effective_key):
                return JSONResponse(
                    {"error": "unauthorized", "message": "missing or invalid API key"},
                    status_code=401,
                )

        # Rate limiting (per-IP).
        if limiter is not None:
            client_ip = request.client.host if request.client else "unknown"
            if not limiter.check(client_ip):
                return JSONResponse(
                    {"error": "rate_limited", "message": "too many requests"},
                    status_code=429,
                    headers={"Retry-After": "60"},
                )

        # Request body size check.
        if max_request_body_bytes > 0:
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > max_request_body_bytes:
                return JSONResponse(
                    {"error": "payload_too_large",
                     "message": f"request body exceeds {max_request_body_bytes} bytes"},
                    status_code=413,
                )

        return await call_next(request)
