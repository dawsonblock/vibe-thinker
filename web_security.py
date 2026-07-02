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


class _PayloadTooLarge(Exception):
    """Raised by the guarded receive callable when the request body
    exceeds the configured limit. Caught by the ASGI body-size
    middleware to produce a 413 response."""


class _BodySizeLimitMiddleware:
    """Pure ASGI middleware that enforces request body size limits.

    ``BaseHTTPMiddleware`` (the ``@app.middleware("http")`` decorator)
    cannot catch exceptions raised inside the ``receive`` callable —
    they propagate through Starlette's task machinery as unhandled
    errors. A pure ASGI middleware wraps the app directly, so the
    exception propagates synchronously through ``await self.app(...)``
    and can be caught before any response is sent.

    Two layers of defense:
      1. Fast path: reject early when ``Content-Length`` advertises a
         body larger than the limit (avoids reading the stream).
      2. Stream guard: wrap the ASGI ``receive`` callable so that the
         *actual* accumulated body bytes are tracked. This catches
         requests that omit ``Content-Length`` (chunked transfers) or
         that lie about it. Without this, a client can bypass the
         limit simply by not sending the header.
    """

    def __init__(self, app, max_bytes: int = 0):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or self.max_bytes <= 0:
            await self.app(scope, receive, send)
            return

        # Fast path: check Content-Length header from the ASGI scope.
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    if int(value) > self.max_bytes:
                        await _send_413(send, self.max_bytes)
                        return
                except (ValueError, TypeError):
                    pass  # malformed — let the stream guard decide
                break

        # Stream guard: wrap receive to count actual body bytes.
        received = 0
        response_started = False

        async def guarded_receive():
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _PayloadTooLarge(
                        f"request body exceeds {self.max_bytes} bytes")
            return message

        async def guarded_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, guarded_receive, guarded_send)
        except _PayloadTooLarge:
            if not response_started:
                await _send_413(send, self.max_bytes)
            # If the response already started we can't replace it; the
            # connection will be terminated by the unhandled exception.


async def _send_413(send, max_bytes: int) -> None:
    """Send a 413 Payload Too Large response via raw ASGI send."""
    import json as _json
    body = _json.dumps(
        {"error": "payload_too_large",
         "message": f"request body exceeds {max_bytes} bytes"}
    ).encode("utf-8")
    await send({
        "type": "http.response.start",
        "status": 413,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
    })
    await send({"type": "http.response.body", "body": body})


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

    # --- Body size limit (pure ASGI middleware) ---
    # Added before the auth/rate-limit HTTP middleware so it runs as the
    # outermost layer and can short-circuit before the request body is
    # read by any downstream handler. Uses a pure ASGI middleware because
    # BaseHTTPMiddleware cannot catch exceptions raised inside the
    # receive callable (Starlette task machinery swallows them).
    if max_request_body_bytes > 0:
        app.add_middleware(_BodySizeLimitMiddleware, max_bytes=max_request_body_bytes)

    # --- Auth + rate limit middleware ---
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

        return await call_next(request)
