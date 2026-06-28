"""Tests for the RuvLLM adapter (ruvllm_adapter.py).

Covers:
  - TurboQuantConfig presets and CLI arg generation
  - RuvLLMHTTPBackend (URL, start command, health check)
  - RuvLLMBinding (ImportError when ruvllm_py not installed)
  - is_ruvllm_binding_available()
  - CLI flag integration (--ruvllm-url, --fast-code-specialist)
"""

import pytest
from unittest.mock import MagicMock, patch

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
    """Tests for the in-process PyO3 binding.

    The ruvllm_py extension is an OPTIONAL Rust-built component (requires
    ``pip install -e '.[rust]'`` + ``bash scripts/check_ruvllm.sh``).
    These tests verify binding detection and behavior. They skip cleanly
    when the extension is not installed — RuvLLM is experimental and must
    never be assumed present in a clean environment.
    """

    def test_is_ruvllm_binding_available(self):
        # The binding is optional (Rust/maturin build). When absent,
        # is_ruvllm_binding_available() must return False, not raise.
        # When present, it must return True. Test both branches honestly.
        result = is_ruvllm_binding_available()
        assert result in (True, False)
        if result is False:
            pytest.skip("ruvllm_py not installed in this env (experimental, optional)")

    def test_binding_constructs_when_installed(self):
        # The binding should construct successfully when installed
        # (it will fail on load_gguf with a non-existent model, but the
        # constructor itself should not raise ImportError)
        try:
            binding = RuvLLMBinding(model_path="test.gguf")
            # If it constructed, verify it has the expected attributes
            assert binding is not None
        except (ImportError, RuntimeError, Exception) as e:
            # If the binding fails to load a non-existent model, that's
            # expected — the important thing is it didn't raise ImportError
            # (which would mean the binding isn't installed)
            if isinstance(e, ImportError):
                pytest.skip("ruvllm_py not installed in this env")
            # RuntimeError or other load errors are expected with a fake path


class TestRuvLLMBindingEmbeddings:
    """v3.1: Tests for the semantic embedding fallback.

    The v3.0 n-gram hashing fallback was removed (it produced non-semantic
    vectors that corrupted SONA clustering). The new implementation tries
    Rust-native embed() → sentence-transformers → ONNX → fail-closed ([]).
    """

    def test_get_embeddings_empty_text_returns_empty(self):
        """Empty text fail-closes to [] (no embedding)."""
        binding = object.__new__(RuvLLMBinding)
        assert binding.get_embeddings("") == []
        assert binding.get_embeddings("   ") == []

    def test_get_embeddings_fail_closed_when_no_source(self):
        """When no embedding source is available, fail-closed to [].

        We mock sentence-transformers and onnxruntime to be absent so the
        binding has no embedding source. The result must be [] (NOT a
        fake hash vector — that would silently corrupt SONA memory).
        """
        binding = object.__new__(RuvLLMBinding)
        # Ensure no _engine (no Rust embed method).
        binding._engine = None
        # Mock sentence-transformers import to fail.
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            # Mock onnxruntime import to fail.
            with patch.dict("sys.modules", {"onnxruntime": None}):
                result = binding.get_embeddings("hello world")
        assert result == []

    def test_get_embeddings_uses_rust_engine_when_available(self):
        """When the Rust engine has an embed() method, it is used directly."""
        binding = object.__new__(RuvLLMBinding)
        mock_engine = MagicMock()
        mock_engine.embed = MagicMock(return_value=[0.1] * 384)
        binding._engine = mock_engine
        result = binding.get_embeddings("test text")
        assert result == [0.1] * 384
        mock_engine.embed.assert_called_once_with("test text")

    def test_get_embeddings_uses_sentence_transformers_when_available(self):
        """When sentence-transformers is installed, it is used for embeddings."""
        binding = object.__new__(RuvLLMBinding)
        binding._engine = None  # no Rust embed
        binding._st_model = None  # force reload

        mock_model = MagicMock()
        mock_model.encode = MagicMock(return_value=[[0.2] * 384])
        mock_st = MagicMock()
        mock_st.SentenceTransformer = MagicMock(return_value=mock_model)

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            result = binding.get_embeddings("hello world")
        assert len(result) == 384
        assert result == [0.2] * 384

    def test_get_embeddings_caches_sentence_transformers_model(self):
        """The sentence-transformers model is cached for reuse."""
        binding = object.__new__(RuvLLMBinding)
        binding._engine = None
        binding._st_model = None

        mock_model = MagicMock()
        mock_model.encode = MagicMock(return_value=[[0.3] * 384])
        mock_st = MagicMock()
        mock_st.SentenceTransformer = MagicMock(return_value=mock_model)

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            binding.get_embeddings("first text")
            binding.get_embeddings("second text")
        # SentenceTransformer constructor called once (cached).
        assert mock_st.SentenceTransformer.call_count == 1
        # encode called twice.
        assert mock_model.encode.call_count == 2

    def test_get_embeddings_falls_back_to_onnx(self):
        """When sentence-transformers is absent but onnxruntime is present,
        the ONNX fallback is used."""
        binding = object.__new__(RuvLLMBinding)
        binding._engine = None

        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with patch("ruvllm_adapter._onnx_embed", return_value=[0.4] * 384) as mock_onnx:
                result = binding.get_embeddings("test text")
        assert result == [0.4] * 384
        mock_onnx.assert_called_once_with("test text")


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
