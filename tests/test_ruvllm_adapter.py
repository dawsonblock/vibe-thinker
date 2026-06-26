"""Tests for the RuvLLM adapter (ruvllm_adapter.py).

Covers:
  - TurboQuantConfig presets and CLI arg generation
  - RuvLLMHTTPBackend (URL, start command, health check)
  - RuvLLMBinding (ImportError when ruvllm_py not installed)
  - is_ruvllm_binding_available()
  - CLI flag integration (--ruvllm-url, --fast-code-specialist)
"""

import pytest

from ruvllm_adapter import (
    TurboQuantConfig,
    TURBOQUANT_SAFE,
    TURBOQUANT_CONSERVATIVE,
    TURBOQUANT_DEFAULT,
    TURBOQUANT_AGGRESSIVE_V,
    RuvLLMHTTPBackend,
    RuvLLMBinding,
    is_ruvllm_binding_available,
)


class TestTurboQuantConfig:
    """Tests for the TurboQuant KV cache config."""

    def test_default_config(self):
        tq = TurboQuantConfig()
        assert tq.cache_type_k == "q8_0"
        assert tq.cache_type_v == "turbo3"

    def test_cli_args(self):
        tq = TurboQuantConfig(cache_type_k="f16", cache_type_v="turbo4")
        args = tq.as_cli_args()
        assert "--cache-type-k" in args
        assert "f16" in args
        assert "--cache-type-v" in args
        assert "turbo4" in args

    def test_presets(self):
        assert TURBOQUANT_SAFE.cache_type_k == "f16"
        assert TURBOQUANT_SAFE.cache_type_v == "turbo4"
        assert TURBOQUANT_CONSERVATIVE.cache_type_k == "q8_0"
        assert TURBOQUANT_CONSERVATIVE.cache_type_v == "turbo4"
        assert TURBOQUANT_DEFAULT.cache_type_k == "q8_0"
        assert TURBOQUANT_DEFAULT.cache_type_v == "turbo3"
        assert TURBOQUANT_AGGRESSIVE_V.cache_type_v == "turbo2"


class TestRuvLLMHTTPBackend:
    """Tests for the HTTP sidecar backend."""

    def test_base_url(self):
        b = RuvLLMHTTPBackend(port=9090, host="0.0.0.0")
        assert b.base_url == "http://0.0.0.0:9090"

    def test_recommended_start_command(self):
        b = RuvLLMHTTPBackend(
            port=8080, model_path="~/models/test.gguf", n_ctx=8192, n_threads=6
        )
        cmd = b.recommended_start_command()
        assert "ruvllm-server" in cmd
        assert "-m" in cmd
        assert "~/models/test.gguf" in cmd
        assert "--port" in cmd
        assert "8080" in cmd
        # TurboQuant flags
        assert "--cache-type-k" in cmd
        assert "--cache-type-v" in cmd

    def test_start_command_requires_model_path(self):
        b = RuvLLMHTTPBackend(port=8080)
        with pytest.raises(ValueError, match="model_path"):
            b.recommended_start_command()

    @pytest.mark.asyncio
    async def test_health_check_returns_false_when_down(self):
        b = RuvLLMHTTPBackend(port=1)  # invalid port
        ok = await b.health_check()
        assert ok is False


class TestRuvLLMBinding:
    """Tests for the in-process PyO3 binding (not yet published)."""

    def test_is_ruvllm_binding_available_returns_false(self):
        # The ruvllm_py extension is not published yet
        assert is_ruvllm_binding_available() is False

    def test_binding_raises_import_error_when_not_installed(self):
        with pytest.raises(ImportError, match="ruvllm_py"):
            RuvLLMBinding(model_path="test.gguf")


class TestCLIFlags:
    """Tests that the CLI flags are wired correctly."""

    def test_ruvllm_url_flag_accepted(self):
        from rfsn_cli import build_argparser
        args = build_argparser().parse_args(["--ruvllm-url", "http://127.0.0.1:9090"])
        assert args.ruvllm_url == "http://127.0.0.1:9090"

    def test_ruvllm_url_defaults_empty(self):
        from rfsn_cli import build_argparser
        args = build_argparser().parse_args([])
        assert args.ruvllm_url == ""

    def test_fast_code_specialist_flag_accepted(self):
        from rfsn_cli import build_argparser
        args = build_argparser().parse_args(["--fast-code-specialist"])
        assert args.fast_code_specialist is True

    def test_fast_code_specialist_defaults_false(self):
        from rfsn_cli import build_argparser
        args = build_argparser().parse_args([])
        assert args.fast_code_specialist is False

    def test_ruvllm_url_via_env(self, monkeypatch):
        from rfsn_cli import build_argparser
        monkeypatch.setenv("RUVLLM_URL", "http://env:7070")
        args = build_argparser().parse_args([])
        assert args.ruvllm_url == "http://env:7070"
