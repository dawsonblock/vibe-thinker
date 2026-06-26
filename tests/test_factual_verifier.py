"""Pytest tests for the factual verifier (NLI judge, fail-closed v0.4.0)."""

import pytest
from unittest.mock import AsyncMock

from verifiers.factual_verifier import FactualVerifier


@pytest.fixture
def verifier():
    """Verifier without an LLM judge (fail-closed since v0.4.0)."""
    return FactualVerifier()


@pytest.fixture
def nli_verifier():
    """Verifier with a mock LLM judge."""
    judge = AsyncMock()
    return FactualVerifier(llm_judge=judge), judge


class TestFactualVerifierNoJudge:
    """Tests for the fail-closed path when no LLM judge is configured.

    v0.4.0: the lexical overlap fallback was removed. Without an LLM
    judge, all factual claims fail-closed (verified=False). Word
    counting cannot approximate semantic truth.
    """

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
    async def test_with_sources_but_no_judge_fails_closed(self, verifier):
        """v0.4.0: sources present but no judge → fail-closed."""
        result = await verifier.verify(
            "What is the capital of France?",
            "The capital of France is Paris, a major European city.",
            context={"sources": ["Paris is the capital and largest city of France."]},
        )
        assert result.verified is False
        assert result.method == "nli_unavailable"
        assert result.score == 0.0
        assert "no LLM judge" in (result.error or "")

    @pytest.mark.asyncio
    async def test_string_source_without_judge_fails_closed(self, verifier):
        """v0.4.0: string source but no judge → fail-closed."""
        result = await verifier.verify(
            "Is Python typed?",
            "Python is dynamically typed programming language.",
            context={"sources": "Python is a dynamically typed programming language."},
        )
        assert result.verified is False
        assert result.method == "nli_unavailable"


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
    async def test_nli_neutral_fails_closed(self, nli_verifier):
        """v0.4.0: all sources NEUTRAL → fail-closed (no lexical fallback)."""
        verifier, judge = nli_verifier
        judge.return_value = "NEUTRAL"
        result = await verifier.verify(
            "What is the capital of France?",
            "The capital of France is Paris, a major European city.",
            context={"sources": ["Paris is the capital and largest city of France."]},
        )
        # All sources NEUTRAL → fail-closed (lexical fallback removed in v0.4.0)
        assert result.verified is False
        assert result.method == "nli_neutral"
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_nli_judge_failure_fails_closed(self, nli_verifier):
        """v0.4.0: LLM judge failure → fail-closed (no lexical fallback)."""
        verifier, judge = nli_verifier
        judge.side_effect = RuntimeError("LLM unavailable")
        result = await verifier.verify(
            "What is the capital of France?",
            "The capital of France is Paris, a major European city.",
            context={"sources": ["Paris is the capital and largest city of France."]},
        )
        assert result.verified is False
        assert result.method == "nli_judge_error"
        assert result.score == 0.0

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
        # Unparseable → NEUTRAL → fail-closed (v0.4.0: no lexical fallback)
        assert result.verified is False
        assert result.method == "nli_neutral"
