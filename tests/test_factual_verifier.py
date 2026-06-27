"""Pytest tests for the factual verifier (citation-backed NLI, v1.1).

v1.1 changes:
  - The judge prompt now requires JSON: {"verdict", "supporting_quote"}.
  - ENTAILMENT with a verified quote → nli_citation_backed, score 0.8.
  - ENTAILMENT with a fabricated quote → nli_citation_mismatch, score 0.0.
  - Old-style single-word verdicts (no citation) still parse but score
    0.7 (below the 0.75 cache threshold) so un-cited entailment cannot
    poison the CLR cache.
  - Fail-closed paths (no judge, judge error, all NEUTRAL) unchanged.
"""

import json

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


class TestFactualVerifierCitationBacked:
    """v1.1: citation-backed NLI — the judge provides a supporting quote
    that is verified to actually appear in the source."""

    @pytest.mark.asyncio
    async def test_entailment_with_verified_quote_succeeds(self, nli_verifier):
        """JSON verdict + quote that exists in source → verified, 0.8."""
        verifier, judge = nli_verifier
        judge.return_value = json.dumps({
            "verdict": "ENTAILMENT",
            "supporting_quote": "Paris is the capital of France.",
        })
        result = await verifier.verify(
            "What is the capital of France?",
            "The capital of France is Paris.",
            context={"sources": ["Paris is the capital of France."]},
        )
        assert result.verified is True
        assert result.method == "nli_citation_backed"
        assert result.score == 0.8
        assert result.evidence["quote"] == "Paris is the capital of France."

    @pytest.mark.asyncio
    async def test_entailment_with_normalized_quote_succeeds(self, nli_verifier):
        """Quote with different casing/whitespace still matches after
        normalization."""
        verifier, judge = nli_verifier
        judge.return_value = json.dumps({
            "verdict": "ENTAILMENT",
            "supporting_quote": "  PARIS  is the  Capital  of France.  ",
        })
        result = await verifier.verify(
            "What is the capital of France?",
            "The capital of France is Paris.",
            context={"sources": ["Paris is the capital of France."]},
        )
        assert result.verified is True
        assert result.method == "nli_citation_backed"
        assert result.score == 0.8

    @pytest.mark.asyncio
    async def test_entailment_with_fabricated_quote_fails_closed(self, nli_verifier):
        """JSON verdict + quote NOT in source → nli_citation_mismatch, 0.0.

        This is the key v1.1 hardening: a hallucinating judge cannot
        fabricate support because the quote must exist in the source.
        """
        verifier, judge = nli_verifier
        judge.return_value = json.dumps({
            "verdict": "ENTAILMENT",
            "supporting_quote": "Berlin is the capital of Germany.",
        })
        result = await verifier.verify(
            "What is the capital of France?",
            "The capital of France is Berlin.",
            context={"sources": ["Paris is the capital of France."]},
        )
        assert result.verified is False
        assert result.method == "nli_citation_mismatch"
        assert result.score == 0.0
        assert "citation verification failed" in (result.error or "")

    @pytest.mark.asyncio
    async def test_entailment_with_paraphrased_quote_fails_closed(self, nli_verifier):
        """A paraphrased quote (not a verbatim substring) does NOT match —
        fail-closed. The citation check requires a real substring, not a
        semantic approximation."""
        verifier, judge = nli_verifier
        judge.return_value = json.dumps({
            "verdict": "ENTAILMENT",
            "supporting_quote": "The French capital is Paris.",
        })
        result = await verifier.verify(
            "What is the capital of France?",
            "The capital of France is Paris.",
            context={"sources": ["Paris is the capital of France."]},
        )
        assert result.verified is False
        assert result.method == "nli_citation_mismatch"
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_entailment_empty_quote_falls_back_to_uncited(self, nli_verifier):
        """JSON verdict with empty quote → old-style un-cited path, 0.7
        (below the 0.75 cache threshold)."""
        verifier, judge = nli_verifier
        judge.return_value = json.dumps({
            "verdict": "ENTAILMENT",
            "supporting_quote": "",
        })
        result = await verifier.verify(
            "What is the capital of France?",
            "The capital of France is Paris.",
            context={"sources": ["Paris is the capital of France."]},
        )
        assert result.verified is True
        assert result.method == "nli_llm_judge"
        assert result.score == 0.7

    @pytest.mark.asyncio
    async def test_contradiction_with_quote_rejects(self, nli_verifier):
        """CONTRADICTION with a quote → rejected, nli_llm_judge.

        Note: the citation is NOT verified for CONTRADICTION (only
        ENTAILMENT gets the normalized substring check), so the method
        tag is nli_llm_judge, not nli_citation_backed. The quote is still
        included in evidence for debugging.
        """
        verifier, judge = nli_verifier
        judge.return_value = json.dumps({
            "verdict": "CONTRADICTION",
            "supporting_quote": "Paris is the capital of France.",
        })
        result = await verifier.verify(
            "What is the capital of France?",
            "The capital of France is Berlin.",
            context={"sources": ["Paris is the capital of France."]},
        )
        assert result.verified is False
        assert result.method == "nli_llm_judge"
        assert "contradicts" in (result.error or "")
        assert result.evidence.get("quote") == "Paris is the capital of France."

    @pytest.mark.asyncio
    async def test_neutral_fails_closed(self, nli_verifier):
        """All sources NEUTRAL → fail-closed (no lexical fallback)."""
        verifier, judge = nli_verifier
        judge.return_value = json.dumps({
            "verdict": "NEUTRAL",
            "supporting_quote": "",
        })
        result = await verifier.verify(
            "What is the capital of France?",
            "The capital of France is Paris, a major European city.",
            context={"sources": ["Paris is the capital and largest city of France."]},
        )
        assert result.verified is False
        assert result.method == "nli_neutral"
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_judge_failure_fails_closed(self, nli_verifier):
        """LLM judge failure → fail-closed (no lexical fallback)."""
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


class TestFactualVerifierBackwardCompat:
    """Old-style single-word verdicts (no JSON/citation) still parse for
    backward compatibility with judges that don't output JSON. They score
    0.7 — below the 0.75 cache threshold — so un-cited entailment cannot
    poison the CLR cache (v1.1 hardening)."""

    @pytest.mark.asyncio
    async def test_old_style_entailment_scores_below_cache_threshold(self, nli_verifier):
        verifier, judge = nli_verifier
        judge.return_value = "ENTAILMENT"
        result = await verifier.verify(
            "What is the capital of France?",
            "The capital of France is Paris.",
            context={"sources": ["Paris is the capital of France."]},
        )
        assert result.verified is True
        assert result.method == "nli_llm_judge"
        assert result.score == 0.7  # below the 0.75 cache threshold

    @pytest.mark.asyncio
    async def test_old_style_contradiction_rejects(self, nli_verifier):
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
    async def test_old_style_neutral_fails_closed(self, nli_verifier):
        verifier, judge = nli_verifier
        judge.return_value = "NEUTRAL"
        result = await verifier.verify(
            "What is the capital of France?",
            "The capital of France is Paris, a major European city.",
            context={"sources": ["Paris is the capital and largest city of France."]},
        )
        assert result.verified is False
        assert result.method == "nli_neutral"
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_lowercase_verdict_parses(self, nli_verifier):
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
    async def test_unparseable_response_treated_as_neutral(self, nli_verifier):
        verifier, judge = nli_verifier
        judge.return_value = "I think the answer is correct."
        result = await verifier.verify(
            "What is the capital of France?",
            "The capital of France is Berlin.",
            context={"sources": ["London is the capital of England."]},
        )
        assert result.verified is False
        assert result.method == "nli_neutral"

    @pytest.mark.asyncio
    async def test_malformed_json_treated_as_old_style(self, nli_verifier):
        """Malformed JSON falls back to the single-word parser."""
        verifier, judge = nli_verifier
        judge.return_value = '{"verdict": "ENTAILMENT", broken'
        result = await verifier.verify(
            "What is the capital of France?",
            "The capital of France is Paris.",
            context={"sources": ["Paris is the capital of France."]},
        )
        # The broken JSON has "ENTAILMENT" in the text, so the fallback
        # single-word parser picks it up as an un-cited verdict (0.7).
        assert result.verified is True
        assert result.method == "nli_llm_judge"
        assert result.score == 0.7


class TestNormalization:
    """Unit tests for the citation normalization helpers."""

    def test_normalize_collapses_whitespace(self):
        assert FactualVerifier._normalize_span("  a    b  ") == "a b"

    def test_normalize_casefolds(self):
        assert FactualVerifier._normalize_span("PARIS") == "paris"

    def test_normalize_strips_quotes_and_punctuation(self):
        assert FactualVerifier._normalize_span('"Paris."') == "paris"

    def test_verify_quote_exact_match(self):
        assert FactualVerifier._verify_quote_in_source(
            "Paris is the capital", "Paris is the capital of France."
        )

    def test_verify_quote_case_insensitive(self):
        assert FactualVerifier._verify_quote_in_source(
            "PARIS IS THE CAPITAL", "paris is the capital of france."
        )

    def test_verify_quote_whitespace_tolerant(self):
        assert FactualVerifier._verify_quote_in_source(
            "Paris   is   the   capital", "Paris is the capital of France."
        )

    def test_verify_quote_paraphrase_fails(self):
        assert not FactualVerifier._verify_quote_in_source(
            "The French capital is Paris", "Paris is the capital of France."
        )

    def test_verify_quote_empty_fails(self):
        assert not FactualVerifier._verify_quote_in_source("", "anything")

    def test_verify_quote_not_in_source_fails(self):
        assert not FactualVerifier._verify_quote_in_source(
            "Berlin is the capital", "Paris is the capital of France."
        )
