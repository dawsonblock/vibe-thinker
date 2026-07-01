"""Tests for web_security.py."""

import importlib.util
import os
from unittest.mock import patch

import pytest

def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ImportError):
        return False

_FASTAPI_AVAILABLE = _has_module("fastapi")

pytestmark = [
    pytest.mark.web,
    pytest.mark.skipif(
        not _FASTAPI_AVAILABLE,
        reason="requires fastapi web extra (pip install -e '.[web]')",
    ),
]

if _FASTAPI_AVAILABLE:
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from web_security import (
        _get_api_key_from_env,
        _constant_time_eq,
        _RateLimiter,
        configure_security,
    )
else:
    FastAPI = None
    Request = None
    TestClient = None
    _get_api_key_from_env = None
    _constant_time_eq = None
    _RateLimiter = None
    configure_security = None


class TestApiKeyFromEnv:
    def test_reads_from_env(self):
        with patch.dict(os.environ, {"VIBE_THINKER_API_KEY": "secret-key"}, clear=True):
            assert _get_api_key_from_env() == "secret-key"

    def test_returns_none_if_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _get_api_key_from_env() is None

    def test_returns_none_if_empty(self):
        with patch.dict(os.environ, {"VIBE_THINKER_API_KEY": ""}, clear=True):
            assert _get_api_key_from_env() is None


class TestConstantTimeEq:
    def test_equal_strings(self):
        assert _constant_time_eq("secret", "secret") is True

    def test_different_strings(self):
        assert _constant_time_eq("secret", "wrong") is False

    def test_one_none(self):
        assert _constant_time_eq(None, "secret") is False
        assert _constant_time_eq("secret", None) is False

    def test_both_none(self):
        assert _constant_time_eq(None, None) is True


class TestRateLimiter:
    def test_allows_under_limit(self):
        limiter = _RateLimiter(max_requests=3, window_seconds=60)
        assert limiter.check("1.1.1.1") is True
        assert limiter.check("1.1.1.1") is True
        assert limiter.check("1.1.1.1") is True

    def test_blocks_over_limit(self):
        limiter = _RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.check("1.1.1.1") is True
        assert limiter.check("1.1.1.1") is True
        assert limiter.check("1.1.1.1") is False

    def test_independent_by_ip(self):
        limiter = _RateLimiter(max_requests=1, window_seconds=60)
        assert limiter.check("1.1.1.1") is True
        assert limiter.check("1.1.1.1") is False
        assert limiter.check("2.2.2.2") is True

    def test_evicts_expired_entries(self, monkeypatch):
        import time
        limiter = _RateLimiter(max_requests=1, window_seconds=10)

        current_time = 100.0
        monkeypatch.setattr(time, "time", lambda: current_time)

        assert limiter.check("1.1.1.1") is True
        assert limiter.check("1.1.1.1") is False

        # Advance time beyond window to trigger eviction.
        current_time = 111.0
        assert limiter.check("1.1.1.1") is True


class TestConfigureSecurity:
    def _make_app(self, **kwargs):
        app = FastAPI()

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.get("/protected")
        async def protected():
            return {"data": "secret"}

        @app.post("/upload")
        async def upload(request: Request):
            body = await request.body()
            return {"size": len(body)}

        configure_security(app, **kwargs)
        return app

    def test_no_auth_allows_all(self):
        app = self._make_app(api_key=None)
        client = TestClient(app)
        assert client.get("/protected").status_code == 200

    def test_auth_blocks_missing_key(self):
        app = self._make_app(api_key="secret-key")
        client = TestClient(app)
        assert client.get("/protected").status_code == 401

    def test_auth_blocks_wrong_key(self):
        app = self._make_app(api_key="secret-key")
        client = TestClient(app)
        assert client.get("/protected", headers={"X-API-Key": "wrong"}).status_code == 401

    def test_auth_allows_correct_key(self):
        app = self._make_app(api_key="secret-key")
        client = TestClient(app)
        assert client.get("/protected", headers={"X-API-Key": "secret-key"}).status_code == 200

    def test_exempt_paths_skip_auth(self):
        app = self._make_app(api_key="secret-key", exempt_paths={"/health"})
        client = TestClient(app)
        assert client.get("/health").status_code == 200
        assert client.get("/protected").status_code == 401

    def test_rate_limiting_middleware(self):
        app = self._make_app(rate_limit_per_minute=2)
        client = TestClient(app)
        assert client.get("/protected").status_code == 200
        assert client.get("/protected").status_code == 200
        resp = client.get("/protected")
        assert resp.status_code == 429
        assert resp.json()["error"] == "rate_limited"

    def test_body_size_limit_middleware(self):
        app = self._make_app(max_request_body_bytes=10)
        client = TestClient(app)

        # Small payload should pass.
        assert client.post("/upload", content=b"12345").status_code == 200

        # Large payload should be blocked.
        resp = client.post("/upload", content=b"12345678901")
        assert resp.status_code == 413
        assert resp.json()["error"] == "payload_too_large"

    def test_cors_headers_present(self):
        app = self._make_app(allowed_origins=["http://testserver"])
        client = TestClient(app)
        # Preflight request to check CORS.
        resp = client.options(
            "/protected",
            headers={
                "Origin": "http://testserver",
                "Access-Control-Request-Method": "GET",
            }
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "http://testserver"

    def test_auth_reads_from_env_fallback(self):
        with patch.dict(os.environ, {"VIBE_THINKER_API_KEY": "env-secret"}, clear=True):
            app = self._make_app()
            client = TestClient(app)
            assert client.get("/protected").status_code == 401
            assert client.get("/protected", headers={"X-API-Key": "env-secret"}).status_code == 200
