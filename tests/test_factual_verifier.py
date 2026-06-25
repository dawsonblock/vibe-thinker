"""Pytest tests for the factual verifier (honest placeholder)."""

import pytest

from verifiers.factual_verifier import FactualVerifier


@pytest.fixture
def verifier():
    return FactualVerifier()


class TestFactualVerifier:
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
