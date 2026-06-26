"""Pytest tests for the factual verifier (NLI judge + lexical fallback)."""

import pytest
from unittest.mock import AsyncMock

from verifiers.factual_verifier import FactualVerifier


@pytest.fixture
def verifier():
    """Lexical-only verifier (no LLM judge)."""
    return FactualVerifier()


@pytest.fixture
def nli_verifier():
    """Verifier with a mock LLM judge."""
    judge = AsyncMock()
    return FactualVerifier(llm_judge=judge), judge


class TestFactualVerifierLexical:
    @pytest.mark.asyncio
    async def test_no_sources_returns_unverified(self, verifier):
        result = await verifier.verify(
            "What is the capital of France?",
            "The capital of France is Paris.",
            context={},
        )
        assert result.verified is False
        assert result.method == "unsupported_factual"
        assert "no retrieval sources" in (result.error or "")

    @pytest.mark.asyncio
    async def test_with_supporting_source(self, verifier):
        result = await verifier.verify(
            "What is the capital of France?",
            "The capital of France is Paris, a major European city.",
            context={"sources": ["Paris is the capital and largest city of France. "
                                  "It is situated on the Seine River."]},
        )
        assert result.verified is True
        assert result.method == "retrieval_overlap"
        assert result.score == 0.7  # weak — overlap is not entailment

    @pytest.mark.asyncio
    async def test_with_contradicting_source(self, verifier):
        result = await verifier.verify(
            "What is the capital of France?",
            "The capital of France is Berlin.",
            context={"sources": ["London is the capital of England."]},
        )
        assert result.verified is False
        assert "not supported" in (result.error or "")

    @pytest.mark.asyncio
    async def test_string_source_accepted(self, verifier):
        result = await verifier.verify(
            "Is Python typed?",
            "Python is dynamically typed programming language.",
            context={"sources": "Python is a dynamically typed programming language."},
        )
        assert result.verified is True

    @pytest.mark.asyncio
    async def test_negation_contradiction_rejected(self, verifier):
        """'Paris is NOT the capital' must NOT pass just because 'Paris'
        and 'capital' overlap with the source."""
        result = await verifier.verify(
            "What is the capital of France?",
            "Paris is not the capital of France.",
            context={"sources": ["Paris is the capital and largest city of France."]},
        )
        assert result.verified is False
        assert "negation" in (result.error or "")

    @pytest.mark.asyncio
    async def test_source_negation_answer_affirms_rejected(self, verifier):
        """Source says 'X is not Y', answer says 'X is Y' — contradiction."""
        result = await verifier.verify(
            "Is Pluto a planet?",
            "Pluto is a planet in our solar system.",
            context={"sources": ["Pluto is not a planet; it is a dwarf planet."]},
        )
        assert result.verified is False


class TestFactualVerifierNLI:
    @pytest.mark.asyncio
    async def test_nli_entailment_verifies(self, nli_verifier):
        verifier, judge = nli_verifier
        judge.return_value = "ENTAILMENT"
        result = await verifier.verify(
            "What is the capital of France?",
            "The capital of France is Paris.",
            context={"sources": ["Paris is the capital of France."]},
        )
        assert result.verified is True
        assert result.method == "nli_llm_judge"
        assert result.score == 0.85

    @pytest.mark.asyncio
    async def test_nli_contradiction_rejects(self, nli_verifier):
        verifier, judge = nli_verifier
        judge.return_value = "CONTRADICTION"
        result = await verifier.verify(
            "What is the capital of France?",
            "The capital of France is Berlin.",
            context={"sources": ["Paris is the capital of France."]},
        )
        assert result.verified is False
        assert result.method == "nli_llm_judge"
        assert "contradicts" in (result.error or "")

    @pytest.mark.asyncio
    async def test_nli_neutral_falls_back_to_lexical(self, nli_verifier):
        verifier, judge = nli_verifier
        judge.return_value = "NEUTRAL"
        result = await verifier.verify(
            "What is the capital of France?",
            "The capital of France is Paris, a major European city.",
            context={"sources": ["Paris is the capital and largest city of France."]},
        )
        # All sources NEUTRAL → falls back to lexical, which should pass
        assert result.verified is True
        assert result.method == "retrieval_overlap"

    @pytest.mark.asyncio
    async def test_nli_judge_failure_falls_back_to_lexical(self, nli_verifier):
        verifier, judge = nli_verifier
        judge.side_effect = RuntimeError("LLM unavailable")
        result = await verifier.verify(
            "What is the capital of France?",
            "The capital of France is Paris, a major European city.",
            context={"sources": ["Paris is the capital and largest city of France."]},
        )
        assert result.verified is True
        assert result.method == "retrieval_overlap"

    @pytest.mark.asyncio
    async def test_nli_parses_lowercase_verdict(self, nli_verifier):
        verifier, judge = nli_verifier
        judge.return_value = "entailment"
        result = await verifier.verify(
            "What is the capital of France?",
            "The capital of France is Paris.",
            context={"sources": ["Paris is the capital of France."]},
        )
        assert result.verified is True
        assert result.method == "nli_llm_judge"

    @pytest.mark.asyncio
    async def test_nli_unparseable_response_treated_as_neutral(self, nli_verifier):
        verifier, judge = nli_verifier
        judge.return_value = "I think the answer is correct."
        result = await verifier.verify(
            "What is the capital of France?",
            "The capital of France is Berlin.",
            context={"sources": ["London is the capital of England."]},
        )
        # Unparseable → NEUTRAL → falls back to lexical → no overlap → reject
        assert result.verified is False
