"""Tests for the optional encoder NLI judge (v1.1).

These tests do NOT download a model from HuggingFace — they test:
  - is_available() reflects transformers/torch presence.
  - The prompt parser extracts SOURCE and CLAIM correctly.
  - Construction raises ImportError when deps are absent (mocked).
  - The verdict mapping logic (label → canonical verdict).
  - Low-confidence downgrading to NEUTRAL.
  - select_verifier fail-closed fallback to the LLM judge.
  - The CLI flag / orchestrator wiring (prefer_encoder_nli).

The actual model inference is tested via mocked probabilities, not a
real model load (which would require a network download).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from verifiers.nli_encoder import EncoderNLIJudge, is_available


class TestAvailability:
    def test_is_available_returns_bool(self):
        assert isinstance(is_available(), bool)

    def test_is_available_true_when_deps_present(self):
        # transformers + torch are installed on this machine, so this
        # should be True. On a machine without them, it'd be False.
        assert is_available() is True


class TestPromptParsing:
    """The encoder judge must extract SOURCE and CLAIM from the same
    prompt the LLM judge receives (_NLI_JUDGE_PROMPT)."""

    def test_extract_source_and_claim(self):
        prompt = (
            "You are a strict entailment judge. Given a SOURCE text and a "
            "CLAIM, determine the relationship between them.\n\n"
            "- ENTAILMENT: ...\n"
            "- CONTRADICTION: ...\n"
            "- NEUTRAL: ...\n\n"
            "SOURCE: Paris is the capital of France.\n"
            "CLAIM: The capital of France is Paris.\n\n"
            "Respond with ONLY a JSON object..."
        )
        source, claim = EncoderNLIJudge._extract_source_claim(prompt)
        assert "Paris is the capital of France" in source
        assert "The capital of France is Paris" in claim

    def test_extract_handles_multiline_source(self):
        prompt = (
            "SOURCE: Paris is the capital.\n"
            "It is also the largest city.\n"
            "CLAIM: Paris is the capital.\n\n"
            "Respond..."
        )
        source, claim = EncoderNLIJudge._extract_source_claim(prompt)
        assert "Paris is the capital" in source

    def test_extract_missing_fields_returns_empty(self):
        source, claim = EncoderNLIJudge._extract_source_claim("no labels here")
        assert source == ""
        assert claim == ""

    def test_extract_missing_claim_returns_empty_claim(self):
        prompt = "SOURCE: some text\nRespond with JSON"
        source, claim = EncoderNLIJudge._extract_source_claim(prompt)
        assert "some text" in source
        assert claim == ""


class TestVerdictMapping:
    """Test the _pick_verdict logic with mocked probabilities."""

    def _judge_with_labels(self, id2label):
        """Build an EncoderNLIJudge without loading a model, then inject
        mock labels."""
        if not is_available():
            pytest.skip("transformers/torch not installed")
        # Construct without loading (lazy) — __init__ only stores config.
        judge = EncoderNLIJudge.__new__(EncoderNLIJudge)
        judge._id2label = id2label
        return judge

    def test_entailment_label_mapped(self):
        judge = self._judge_with_labels({0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"})
        import torch
        probs = torch.tensor([0.9, 0.05, 0.05])
        verdict, conf = judge._pick_verdict(probs)
        assert verdict == "ENTAILMENT"
        assert 0.85 < conf < 0.95

    def test_contradiction_label_mapped(self):
        judge = self._judge_with_labels({0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"})
        import torch
        probs = torch.tensor([0.1, 0.1, 0.8])
        verdict, conf = judge._pick_verdict(probs)
        assert verdict == "CONTRADICTION"

    def test_neutral_label_mapped(self):
        judge = self._judge_with_labels({0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"})
        import torch
        probs = torch.tensor([0.2, 0.7, 0.1])
        verdict, conf = judge._pick_verdict(probs)
        assert verdict == "NEUTRAL"

    def test_lowercase_labels_mapped(self):
        judge = self._judge_with_labels({0: "entailment", 1: "neutral", 2: "contradiction"})
        import torch
        probs = torch.tensor([0.9, 0.05, 0.05])
        verdict, _ = judge._pick_verdict(probs)
        assert verdict == "ENTAILMENT"

    def test_unknown_labels_fallback_to_position(self):
        """When labels don't match expected names, fall back to position
        heuristics (0=entailment, 2=contradiction)."""
        judge = self._judge_with_labels({0: "label_0", 1: "label_1", 2: "label_2"})
        import torch
        probs = torch.tensor([0.9, 0.05, 0.05])
        verdict, _ = judge._pick_verdict(probs)
        assert verdict == "ENTAILMENT"  # position 0


class TestFailClosed:
    """The encoder judge must fail-closed on any error."""

    @pytest.mark.asyncio
    async def test_missing_source_and_claim_returns_neutral(self):
        """When the prompt has no SOURCE/CLAIM, return NEUTRAL (fail-closed
        in the verifier: all-NEUTRAL → nli_neutral, score 0.0)."""
        if not is_available():
            pytest.skip("transformers/torch not installed")
        judge = EncoderNLIJudge.__new__(EncoderNLIJudge)
        judge._model = None
        judge._tokenizer = None
        judge._id2label = None
        judge._device = "cpu"
        judge._deterministic = True
        judge._model_name = "unused"
        judge._threshold = 0.6
        # Mock _ensure_loaded to do nothing (don't load a real model).
        with patch.object(judge, "_ensure_loaded"):
            result = await judge("no source or claim here")
        parsed = json.loads(result)
        assert parsed["verdict"] == "NEUTRAL"
        assert parsed["supporting_quote"] == ""


class TestSelectVerifierFallback:
    """select_verifier must fall back to the LLM judge when the encoder
    is unavailable or prefer_encoder_nli is False."""

    def test_default_prefers_llm_judge(self):
        from hybrid_orchestrator import select_verifier
        from verifiers.factual_verifier import FactualVerifier
        v = select_verifier("factual", llm_judge="mock_judge")
        assert isinstance(v, FactualVerifier)
        assert v._llm_judge == "mock_judge"

    def test_conversation_returns_none(self):
        from hybrid_orchestrator import select_verifier
        assert select_verifier("conversation") is None

    def test_math_returns_math_verifier(self):
        from hybrid_orchestrator import select_verifier
        from verifiers import MathVerifier
        assert isinstance(select_verifier("math"), MathVerifier)

    def test_factual_with_encoder_unavailable_falls_back(self):
        """When prefer_encoder_nli=True but the encoder can't be constructed,
        fall back to the LLM judge."""
        from hybrid_orchestrator import select_verifier
        from verifiers.factual_verifier import FactualVerifier
        with patch("verifiers.nli_encoder.is_available", return_value=False):
            v = select_verifier("factual", llm_judge="mock", prefer_encoder_nli=True)
        assert isinstance(v, FactualVerifier)
        assert v._llm_judge == "mock"

    def test_factual_with_encoder_construction_error_falls_back(self):
        """When the encoder judge raises on construction, fall back."""
        from hybrid_orchestrator import select_verifier
        from verifiers.factual_verifier import FactualVerifier

        def fake_init(self, *a, **kw):
            raise RuntimeError("model download failed")

        with patch("verifiers.nli_encoder.is_available", return_value=True), \
             patch.object(EncoderNLIJudge, "__init__", fake_init):
            v = select_verifier("factual", llm_judge="mock", prefer_encoder_nli=True)
        assert isinstance(v, FactualVerifier)
        assert v._llm_judge == "mock"
