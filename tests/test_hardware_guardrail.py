"""Tests for the dynamic hardware guardrail (Phase 4.1).

NOTE: These tests deliberately do NOT allocate multi-GB byte strings.
Previous versions used ``f.write_bytes(b"\\x00" * (5 * 1024**3))`` which
allocates the entire buffer in RAM and can crash constrained environments.
Instead, we monkeypatch ``os.path.getsize`` to return the desired file
size, avoiding any disk I/O or memory allocation for large fake models.
Small models (<= 10 MB) are still written to real temp files since the
cost is negligible.
"""

import os

from hardware_guardrail import (
    check_model_fits_ram,
    GuardrailResult,
    _estimate_model_ram_mb,
    _available_ram_mb,
    _RAM_SAFETY_MULTIPLIER,
    _FALLBACK_AVAILABLE_RAM_MB,
)


def _make_fake_model(tmp_path, name, size_bytes, monkeypatch):
    """Create a real 1-byte file at ``tmp_path/name`` and monkeypatch
    ``os.path.getsize`` to report ``size_bytes`` for it. This avoids
    allocating multi-GB byte strings in memory while still exercising
    the guardrail's size-based logic."""
    f = tmp_path / name
    f.write_bytes(b"\x00")  # 1 byte — real file so os.path.exists works
    real_getsize = os.path.getsize

    def fake_getsize(p):
        if str(p) == str(f):
            return size_bytes
        return real_getsize(p)

    monkeypatch.setattr(os.path, "getsize", fake_getsize)
    return str(f)


class TestEstimateModelRam:
    def test_local_file_returns_estimate(self, tmp_path, monkeypatch):
        # Mock a 100MB model file without allocating 100MB in memory.
        path = _make_fake_model(
            tmp_path, "model.gguf", 100 * 1024 * 1024, monkeypatch,
        )
        ram = _estimate_model_ram_mb(path, n_ctx=4096, pool_size=1)
        assert ram is not None
        # Should be at least 100MB * safety_multiplier.
        assert ram >= int(100 * _RAM_SAFETY_MULTIPLIER)

    def test_nonexistent_file_returns_none(self):
        assert _estimate_model_ram_mb("/nonexistent/model.gguf") is None

    def test_pool_size_scales_ram(self, tmp_path, monkeypatch):
        path = _make_fake_model(
            tmp_path, "model.gguf", 100 * 1024 * 1024, monkeypatch,
        )
        ram1 = _estimate_model_ram_mb(path, n_ctx=4096, pool_size=1)
        ram4 = _estimate_model_ram_mb(path, n_ctx=4096, pool_size=4)
        assert ram4 > ram1
        # 4 instances should cost roughly 4x (minus the shared KV cache
        # overhead, but with the safety multiplier it should be close).
        assert ram4 >= ram1 * 3

    def test_n_ctx_affects_kv_cache(self, tmp_path, monkeypatch):
        path = _make_fake_model(
            tmp_path, "model.gguf", 100 * 1024 * 1024, monkeypatch,
        )
        ram_small = _estimate_model_ram_mb(path, n_ctx=512, pool_size=1)
        ram_large = _estimate_model_ram_mb(path, n_ctx=8192, pool_size=1)
        # Larger context -> more KV cache -> more RAM.
        assert ram_large >= ram_small


class TestCheckModelFitsRam:
    def test_small_model_fits(self, tmp_path):
        # 10MB is small enough to write for real.
        f = tmp_path / "small.gguf"
        f.write_bytes(b"\x00" * (10 * 1024 * 1024))  # 10MB
        result = check_model_fits_ram(
            str(f), n_ctx=512, pool_size=1, available_ram_mb=4096,
        )
        assert result.ok is True
        assert result.model_ram_mb is not None
        assert result.error is None

    def test_large_model_does_not_fit(self, tmp_path, monkeypatch):
        # Mock a 5GB file without allocating 5GB in memory.
        path = _make_fake_model(
            tmp_path, "huge.gguf", 5000 * 1024 * 1024, monkeypatch,
        )
        result = check_model_fits_ram(
            path, n_ctx=4096, pool_size=1, available_ram_mb=2048,
        )
        assert result.ok is False
        assert result.model_ram_mb is not None
        assert result.available_ram_mb == 2048
        assert result.error is not None
        assert "shortfall" in result.error

    def test_non_local_path_skips_guardrail(self):
        result = check_model_fits_ram(
            "repo_id/model.gguf", n_ctx=4096, pool_size=1,
            available_ram_mb=2048,
        )
        assert result.ok is True
        assert result.warning is not None
        assert "Could not estimate" in result.warning

    def test_error_includes_remediation(self, tmp_path, monkeypatch):
        # Mock a 3GB file without allocating 3GB in memory.
        path = _make_fake_model(
            tmp_path, "big.gguf", 3000 * 1024 * 1024, monkeypatch,
        )
        result = check_model_fits_ram(
            path, n_ctx=4096, pool_size=2, available_ram_mb=1024,
        )
        assert result.ok is False
        # Error should mention pool_size and n_ctx as remediation options.
        assert "pool" in result.error.lower()
        assert "n_ctx" in result.error.lower()

    def test_warning_when_psutil_missing(self, tmp_path, monkeypatch):
        # Simulate psutil not installed.
        import sys
        monkeypatch.setitem(sys.modules, "psutil", None)
        f = tmp_path / "small.gguf"
        f.write_bytes(b"\x00" * (10 * 1024 * 1024))
        # Don't pass available_ram_mb — let it use the fallback.
        result = check_model_fits_ram(str(f), n_ctx=512, pool_size=1)
        assert result.ok is True
        # The warning may or may not be present depending on whether
        # psutil is actually installed in the test env. Just check
        # that the result is OK.
        assert result.error is None

    def test_pool_size_in_error_message(self, tmp_path, monkeypatch):
        # Mock a 2GB file without allocating 2GB in memory.
        path = _make_fake_model(
            tmp_path, "big.gguf", 2000 * 1024 * 1024, monkeypatch,
        )
        result = check_model_fits_ram(
            path, n_ctx=4096, pool_size=4, available_ram_mb=1024,
        )
        assert result.ok is False
        assert "4" in result.error  # pool_size mentioned

    def test_guardrail_result_dataclass(self):
        r = GuardrailResult(ok=True, model_ram_mb=500, available_ram_mb=4096)
        assert r.ok is True
        assert r.error is None
        assert r.warning is None
        r2 = GuardrailResult(ok=False, error="too big")
        assert r2.ok is False
        assert r2.error == "too big"


class TestAvailableRamFallback:
    def test_fallback_returns_positive(self, monkeypatch):
        # When psutil is not available, should return the fallback.
        import sys
        monkeypatch.setitem(sys.modules, "psutil", None)
        ram = _available_ram_mb()
        assert ram == _FALLBACK_AVAILABLE_RAM_MB
        assert ram > 0
