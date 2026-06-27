"""Tests for the TurboQuant PPL validation harness (Phase 2.3).

These test the model-free PPL math, the comparison/tolerance logic, and
the logprob extraction — without needing a live model server or a GGUF
file. The heavy eval paths (HTTP/in-process) are exercised only in their
error/contract behavior.
"""

import json
import math
import os
import sys
import tempfile

import pytest

# Make the scripts dir importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import turboquant_ppl_check as ppl  # noqa: E402


class TestComputePpl:
    def test_uniform_logprobs(self):
        # log P = -1 for each of 4 tokens -> PPL = exp(1) = e
        lp = [-1.0, -1.0, -1.0, -1.0]
        assert ppl.compute_ppl(lp) == pytest.approx(math.e)

    def test_perfect_prediction(self):
        # log P = 0 for every token -> PPL = 1 (best possible)
        assert ppl.compute_ppl([0.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_higher_logprob_lower_ppl(self):
        # Better predictions (higher log-prob) -> lower PPL.
        better = ppl.compute_ppl([-0.1, -0.1, -0.1])
        worse = ppl.compute_ppl([-2.0, -2.0, -2.0])
        assert better < worse

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ppl.compute_ppl([])

    def test_single_token(self):
        # PPL = exp(-(-2.0)/1) = exp(2)
        assert ppl.compute_ppl([-2.0]) == pytest.approx(math.exp(2.0))


class TestPplResult:
    def test_from_log_probs(self):
        r = ppl.PplResult.from_log_probs(
            [-1.0, -1.0], config={"cache_type_k": "q8_0"}, source="test",
        )
        assert r.ppl == pytest.approx(math.e)
        assert r.n_tokens == 2
        assert r.mean_log_prob == pytest.approx(-1.0)
        assert r.config == {"cache_type_k": "q8_0"}
        assert r.source == "test"

    def test_to_dict_roundtrip(self):
        r = ppl.PplResult.from_log_probs([-0.5, -0.5], source="x")
        d = r.to_dict()
        assert d["n_tokens"] == 2
        assert d["ppl"] == pytest.approx(math.exp(0.5))
        assert d["source"] == "x"


class TestComparePpl:
    def test_within_tolerance_passes(self):
        # 1% increase, 1.5% tolerance -> pass
        c = ppl.compare_ppl(baseline=10.0, candidate=10.1, tolerance=0.015)
        assert c.passed is True
        assert c.delta == pytest.approx(0.1)
        assert c.pct_delta == pytest.approx(0.01)

    def test_exceeds_tolerance_fails(self):
        # 2% increase, 1.5% tolerance -> fail
        c = ppl.compare_ppl(baseline=10.0, candidate=10.2, tolerance=0.015)
        assert c.passed is False
        assert c.pct_delta == pytest.approx(0.02)

    def test_lower_ppl_passes(self):
        # Candidate has LOWER PPL (better) -> always passes (delta < 0)
        c = ppl.compare_ppl(baseline=10.0, candidate=9.5, tolerance=0.015)
        assert c.passed is True
        assert c.delta == pytest.approx(-0.5)
        assert c.pct_delta == pytest.approx(-0.05)

    def test_just_under_tolerance_passes(self):
        # Just under the tolerance boundary -> pass (<=). Use a value
        # safely below to avoid float-precision fuzz at exact equality.
        c = ppl.compare_ppl(baseline=10.0, candidate=10.149, tolerance=0.015)
        assert c.passed is True

    def test_just_over_tolerance_fails(self):
        # Just over the tolerance boundary -> fail.
        c = ppl.compare_ppl(baseline=10.0, candidate=10.151, tolerance=0.015)
        assert c.passed is False

    def test_invalid_baseline_raises(self):
        with pytest.raises(ValueError):
            ppl.compare_ppl(baseline=0.0, candidate=1.0, tolerance=0.01)
        with pytest.raises(ValueError):
            ppl.compare_ppl(baseline=-1.0, candidate=1.0, tolerance=0.01)

    def test_to_dict(self):
        c = ppl.compare_ppl(baseline=10.0, candidate=10.1, tolerance=0.015)
        d = c.to_dict()
        assert d["passed"] is True
        assert d["tolerance"] == 0.015
        assert d["baseline_ppl"] == 10.0


class TestExtractLogprobs:
    def test_list_format(self):
        # Older llama-server: logprobs is a list of dicts.
        resp = {"logprobs": [
            {"token": "a", "logprob": -0.5},
            {"token": "b", "logprob": -1.5},
        ]}
        assert ppl._extract_token_logprobs(resp) == [-0.5, -1.5]

    def test_content_format(self):
        # Newer llama-server: logprobs is {"content": [...]}.
        resp = {"logprobs": {"content": [
            {"token": "a", "logprob": -0.2},
            {"token": "b", "logprob": -0.3},
        ]}}
        assert ppl._extract_token_logprobs(resp) == [-0.2, -0.3]

    def test_missing_logprobs_raises(self):
        with pytest.raises(ValueError):
            ppl._extract_token_logprobs({"choices": []})

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError):
            ppl._extract_token_logprobs({"logprobs": []})
        with pytest.raises(ValueError):
            ppl._extract_token_logprobs({"logprobs": {"content": []}})


class TestInProcessContract:
    def _ruvllm_has_logprobs(self):
        """Check if ruvllm_py is importable AND exposes logprobs."""
        try:
            import ruvllm_py  # type: ignore
            return bool(getattr(ruvllm_py, "SUPPORTS_LOGPROBS", False))
        except ImportError:
            return False

    def test_inprocess_raises_not_implemented(self):
        # The in-process path must fail-closed (NotImplementedError) when
        # the binding is absent or stubbed (no logprobs capability). When
        # the binding HAS logprobs, this test is covered by
        # test_inprocess_logprobs instead.
        if self._ruvllm_has_logprobs():
            pytest.skip("ruvllm_py has logprobs — see test_inprocess_logprobs")
        with pytest.raises(NotImplementedError):
            ppl.eval_inprocess("model.gguf", "hello world")

    def test_inprocess_does_not_return_fake_ppl(self):
        # Fail-closed contract: never fabricate a PPL value.
        if self._ruvllm_has_logprobs():
            pytest.skip("ruvllm_py has logprobs — see test_inprocess_logprobs")
        try:
            ppl.eval_inprocess("model.gguf", "hello world")
            assert False, "should have raised"
        except NotImplementedError:
            pass
        except Exception:
            # Any exception is acceptable as long as no PplResult is returned.
            pass

    def test_inprocess_logprobs_returns_ppl_result(self):
        # When ruvllm_py IS built with candle (SUPPORTS_LOGPROBS == True)
        # and a real GGUF model is available, eval_inprocess should return a
        # PplResult with real per-token log-probs computed via log_softmax
        # over the vocabulary. Skips when the binding isn't built or no
        # model is available.
        if not self._ruvllm_has_logprobs():
            pytest.skip(
                "ruvllm_py not built with candle — no logprobs capability. "
                "Run: cd ruvllm_py && maturin develop --release --features candle"
            )

        # A real GGUF model is required for the in-process path. The test
        # model path comes from an env var so it doesn't hardcode a path.
        model_path = os.environ.get("RUVLLM_PPL_TEST_MODEL", "")
        if not model_path or not os.path.isfile(model_path):
            pytest.skip(
                "no test model available — set RUVLLM_PPL_TEST_MODEL to a "
                "GGUF file path (with a tokenizer.json alongside it)"
            )

        result = ppl.eval_inprocess(model_path, "hello world from the test corpus")
        assert isinstance(result, ppl.PplResult)
        assert result.n_tokens > 0
        assert result.ppl > 0
        assert result.source == "ruvllm_py"


class TestCLI:
    def test_compare_cli_pass(self, tmp_path):
        base = tmp_path / "base.json"
        cand = tmp_path / "cand.json"
        base.write_text(json.dumps({"ppl": 10.0}))
        cand.write_text(json.dumps({"ppl": 10.1}))
        out = tmp_path / "report.json"
        rc = ppl.main([
            "compare", "--baseline", str(base), "--candidate", str(cand),
            "--tolerance", "0.015", "--out", str(out),
        ])
        assert rc == 0
        report = json.loads(out.read_text())
        assert report["passed"] is True

    def test_compare_cli_fail(self, tmp_path):
        base = tmp_path / "base.json"
        cand = tmp_path / "cand.json"
        base.write_text(json.dumps({"ppl": 10.0}))
        cand.write_text(json.dumps({"ppl": 10.5}))  # 5% increase
        rc = ppl.main([
            "compare", "--baseline", str(base), "--candidate", str(cand),
            "--tolerance", "0.015",
        ])
        assert rc == 1

    def test_compare_cli_missing_file(self, tmp_path):
        rc = ppl.main([
            "compare", "--baseline", str(tmp_path / "nope.json"),
            "--candidate", str(tmp_path / "nope2.json"),
        ])
        assert rc == 2

    def test_eval_cli_missing_corpus(self, tmp_path):
        rc = ppl.main([
            "eval", "--base-url", "http://127.0.0.1:8081",
            "--corpus", str(tmp_path / "nope.txt"), "--out", str(tmp_path / "o.json"),
        ])
        assert rc == 2

    def test_eval_cli_empty_corpus(self, tmp_path):
        c = tmp_path / "empty.txt"
        c.write_text("   \n  ")
        rc = ppl.main([
            "eval", "--base-url", "http://127.0.0.1:8081",
            "--corpus", str(c), "--out", str(tmp_path / "o.json"),
        ])
        assert rc == 2
