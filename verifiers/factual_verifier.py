"""Factual/retrieval verifier — citation-backed NLI with fail-closed semantics.

For now, do NOT pretend factual claims are verified unless sources exist.
The honesty matters: returning ``verified=True`` without evidence would
make this system a liar. When retrieval sources are available in the
context, this verifier checks claims against them.

Verification path (v1.1 — citation-backed NLI):
  1. **LLM-judge NLI with citation** (when ``llm_judge`` is configured):
     prompts the judge to classify the answer against each source as
     ENTAILMENT, CONTRADICTION, or NEUTRAL — AND to extract the exact
     supporting quote from the source. The verifier then performs a
     normalized substring check: if the judge's quote does not actually
     appear in the source, the verdict is voided (fail-closed, score 0.0).
     This prevents a hallucinating judge from fabricating support.
       - Citation verified → ``nli_citation_backed``, score 0.8 (above the
         0.75 cache trust threshold — safe because the quote is real).
       - Citation mismatch → ``nli_citation_mismatch``, score 0.0.
       - Old-style single-word verdict (no JSON/citation) →
         ``nli_llm_judge``, score 0.7 (BELOW the 0.75 cache threshold, so
         un-cited entailment can no longer poison the CLR cache).

  2. **Fail-closed** (when no judge, judge fails, or all sources NEUTRAL):
     returns ``verified=False``. The lexical overlap fallback was removed
     in v0.4.0 because word counting cannot approximate semantic truth —
     it produced false positives (e.g. "Berlin is a beautiful capital"
     passed verification against a source about Germany because of >40%
     word overlap). Factual truth requires NLI, not word counting.

Behavior:
  - No retrieval source in context -> verified=False, method="unsupported_factual"
  - LLM judge available -> citation-backed NLI classification per source
  - No LLM judge -> fail-closed (nli_unavailable)
"""

import json
import re
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from verifiers.base import VerificationResult


# v1.1: the judge must now output JSON with a verdict AND the exact
# supporting quote from the source. The verifier checks that the quote
# actually appears in the source (normalized) before trusting an
# ENTAILMENT verdict — this is the "citation-backed entailment" check
# that prevents a hallucinating judge from fabricating support.
_NLI_JUDGE_PROMPT = (
    "You are a strict entailment judge. Given a SOURCE text and a CLAIM, "
    "determine the relationship between them.\n\n"
    "- ENTAILMENT: the SOURCE directly supports the CLAIM (the CLAIM "
    "follows logically from the SOURCE).\n"
    "- CONTRADICTION: the SOURCE directly contradicts the CLAIM (the CLAIM "
    "is the opposite of what the SOURCE says).\n"
    "- NEUTRAL: the SOURCE neither supports nor contradicts the CLAIM.\n\n"
    "SOURCE: {source}\n"
    "CLAIM: {claim}\n\n"
    "Respond with ONLY a JSON object in this exact format:\n"
    '{{"verdict": "ENTAILMENT" | "CONTRADICTION" | "NEUTRAL", '
    '"supporting_quote": "the exact substring of the SOURCE that supports '
    'or contradicts the claim, or empty string if NEUTRAL"}}\n'
    "The supporting_quote MUST be copied verbatim from the SOURCE text — "
    "do not paraphrase. If the verdict is NEUTRAL, leave supporting_quote "
    "empty."
)


class FactualVerifier:
    """Factual verifier with citation-backed LLM-judge NLI.

    Args:
        llm_judge: optional async callable that takes a prompt string and
            returns the LLM's text response. When provided, the verifier
            uses it as an NLI judge (entailment/contradiction/neutral) with
            citation-backed verification (v1.1). When None, fail-closed.
    """

    name = "factual_verifier"

    def __init__(
        self,
        llm_judge: Optional[Callable[[str], Awaitable[str]]] = None,
    ):
        self._llm_judge = llm_judge

    async def verify(
        self, query: str, answer: str, context: Dict[str, Any]
    ) -> VerificationResult:
        sources = context.get("sources")
        if not sources:
            return VerificationResult(
                verified=False,
                score=0.0,
                method="unsupported_factual",
                evidence={"answer": answer[:200]},
                error="no retrieval sources provided; factual claims cannot be verified",
            )

        if isinstance(sources, str):
            sources = [sources]

        # Prefer the LLM-judge NLI path when available.
        if self._llm_judge is not None:
            return await self._verify_with_nli(answer, sources)

        # No LLM judge available — fail-closed. Factual truth cannot be
        # approximated by word counting (the lexical fallback was removed
        # in v0.4.0 because it produced false positives: answers with high
        # word overlap but wrong semantics passed verification).
        return VerificationResult(
            verified=False,
            score=0.0,
            method="nli_unavailable",
            evidence={"answer": answer[:200], "source_count": len(sources)},
            error="no LLM judge configured; factual claims require NLI verification",
        )

    async def _verify_with_nli(
        self, answer: str, sources: list
    ) -> VerificationResult:
        """Use the LLM judge to classify entailment per source (v1.1:
        citation-backed).

        For each source, the judge returns a JSON verdict + a supporting
        quote. For ENTAILMENT, the quote is verified to actually appear in
        the source (normalized substring check). If the quote is absent,
        the verdict is voided — fail-closed (nli_citation_mismatch, 0.0).

        Fail-closed: if the judge fails or all sources return NEUTRAL,
        returns verified=False. Lexical overlap is NOT used as a fallback
        — word counting cannot approximate semantic truth (v0.4.0).
        """
        for src in sources:
            prompt = _NLI_JUDGE_PROMPT.format(
                source=str(src)[:2000], claim=answer[:1000],
            )
            try:
                raw = await self._llm_judge(prompt)
            except Exception as e:
                # Judge call failed — fail-closed. Do NOT fall back to
                # lexical overlap (it produces false positives).
                print(f"[FactualVerifier] LLM judge failed ({e}) — fail-closed")
                return VerificationResult(
                    verified=False,
                    score=0.0,
                    method="nli_judge_error",
                    evidence={"answer": answer[:200], "source_count": len(sources)},
                    error=f"LLM judge failed: {e}",
                )

            verdict, quote = self._parse_nli_verdict(raw)
            if verdict == "ENTAILMENT":
                # Citation-backed check: verify the quote actually appears
                # in the source. This is the key v1.1 hardening — a
                # hallucinating judge cannot fabricate support because the
                # quote must exist in the source text (normalized).
                if quote and self._verify_quote_in_source(quote, str(src)):
                    return VerificationResult(
                        verified=True,
                        score=0.8,  # Above the 0.75 cache threshold — safe
                                    # because the supporting quote is real.
                        method="nli_citation_backed",
                        evidence={"source": str(src)[:100],
                                  "verdict": verdict,
                                  "quote": quote[:200],
                                  "source_count": len(sources)},
                    )
                if quote and not self._verify_quote_in_source(quote, str(src)):
                    # The judge's quote does NOT appear in the source —
                    # the judge fabricated the support. Fail-closed.
                    print(f"[FactualVerifier] citation mismatch: judge's "
                          f"quote not found in source — fail-closed")
                    return VerificationResult(
                        verified=False,
                        score=0.0,
                        method="nli_citation_mismatch",
                        evidence={"source": str(src)[:100],
                                  "verdict": verdict,
                                  "quote": quote[:200],
                                  "source_count": len(sources)},
                        error="judge's supporting_quote not found in source "
                              "(normalized); citation verification failed",
                    )
                # ENTAILMENT but no quote provided — old-style or
                # non-citing judge. Accept but score below the cache
                # threshold so un-cited entailment cannot poison the cache.
                return VerificationResult(
                    verified=True,
                    score=0.7,  # Below the 0.75 cache trust threshold —
                                # un-cited entailment is NOT cached.
                    method="nli_llm_judge",
                    evidence={"source": str(src)[:100],
                              "verdict": verdict,
                              "source_count": len(sources)},
                )
            if verdict == "CONTRADICTION":
                # Note: we do NOT verify the citation for CONTRADICTION
                # (only ENTAILMENT gets the normalized substring check).
                # The method tag is therefore always nli_llm_judge, not
                # nli_citation_backed — the latter implies the quote was
                # verified, which it wasn't here. The quote is still
                # included in evidence for debugging.
                return VerificationResult(
                    verified=False,
                    score=0.0,
                    method="nli_llm_judge",
                    evidence={"source": str(src)[:100],
                              "verdict": verdict,
                              "quote": quote[:200] if quote else None,
                              "source_count": len(sources)},
                    error="answer contradicts the provided source",
                )
            # NEUTRAL: try the next source

        # All sources were NEUTRAL — fail-closed. The answer is neither
        # supported nor contradicted by the sources. Lexical overlap
        # cannot substitute for semantic entailment (v0.4.0).
        return VerificationResult(
            verified=False,
            score=0.0,
            method="nli_neutral",
            evidence={"answer": answer[:200], "source_count": len(sources)},
            error="all sources returned NEUTRAL; answer is not supported by any source",
        )

    @staticmethod
    def _parse_nli_verdict(raw: str) -> Tuple[str, str]:
        """Extract (verdict, supporting_quote) from the LLM response.

        v1.1: prefers the JSON shape
        ``{"verdict": "...", "supporting_quote": "..."}``. Falls back to
        the old single-word verdict (with empty quote) for backward
        compatibility with judges that don't output JSON.
        """
        text = raw.strip()
        # Try JSON first.
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(text[start:end + 1])
                if isinstance(data, dict):
                    verdict = str(data.get("verdict", "")).strip().upper()
                    quote = str(data.get("supporting_quote", "") or "").strip()
                    for v in ("ENTAILMENT", "CONTRADICTION", "NEUTRAL"):
                        if v in verdict:
                            return v, quote
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        # Fallback: old-style single-word verdict (no citation possible).
        upper = text.upper()
        first_token = upper.split()[0] if upper.split() else ""
        for v in ("ENTAILMENT", "CONTRADICTION", "NEUTRAL"):
            if v in first_token or v in upper[:50]:
                return v, ""
        return "NEUTRAL", ""

    @staticmethod
    def _normalize_span(s: str) -> str:
        """Normalize text for citation substring matching.

        Casefolds, collapses whitespace, and strips surrounding quotes and
        common punctuation. This makes the citation check robust to
        trivial surface differences (capitalization, extra spaces, quote
        wrapping) while still requiring the quote to be a real substring
        of the source — not a paraphrase.
        """
        s = s.casefold()
        # Collapse all whitespace runs to a single space.
        s = re.sub(r"\s+", " ", s)
        # Strip surrounding quotes and common wrapper punctuation.
        s = s.strip(' "\'`.,;:!?()[]{}')
        return s.strip()

    @classmethod
    def _verify_quote_in_source(cls, quote: str, source: str) -> bool:
        """Verify that the judge's supporting quote actually appears in the
        source text (normalized).

        This is the citation-backed invariant: the model quotes a span it
        asserts supports the claim; we verify that span appears in the
        source after normalization (casefold, whitespace collapse, quote
        strip). If it doesn't, the judge fabricated the support —
        fail-closed. Normalization is deliberately conservative (no
        lemmatization, no synonym mapping) so a real match is meaningful.
        """
        if not quote:
            return False
        n_quote = cls._normalize_span(quote)
        n_source = cls._normalize_span(source)
        if not n_quote:
            return False
        return n_quote in n_source
