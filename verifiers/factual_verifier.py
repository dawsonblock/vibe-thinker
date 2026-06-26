"""Factual/retrieval verifier — NLI judge with fail-closed semantics.

For now, do NOT pretend factual claims are verified unless sources exist.
The honesty matters: returning ``verified=True`` without evidence would
make this system a liar. When retrieval sources are available in the
context, this verifier checks claims against them.

Verification path (v0.4.0):
  1. **LLM-judge NLI** (when ``llm_judge`` is configured): prompts a local
     LLM to classify the answer against each source as ENTAILMENT,
     CONTRADICTION, or NEUTRAL. This catches contradictions that lexical
     overlap misses (e.g. "Paris is NOT the capital" vs a source saying
     "Paris is the capital"). Uses the orchestrator's existing
     ``_call_generalist`` — no new dependencies.

  2. **Fail-closed** (when no judge, judge fails, or all sources NEUTRAL):
     returns ``verified=False``. The lexical overlap fallback was removed
     in v0.4.0 because word counting cannot approximate semantic truth —
     it produced false positives (e.g. "Berlin is a beautiful capital"
     passed verification against a source about Germany because of >40%
     word overlap). Factual truth requires NLI, not word counting.

Behavior:
  - No retrieval source in context -> verified=False, method="unsupported_factual"
  - LLM judge available -> NLI classification per source
  - No LLM judge -> hardened lexical overlap (weaker, but catches negation)
"""

import re
from typing import Any, Awaitable, Callable, Dict, Optional

from verifiers.base import VerificationResult


# Words that flip the polarity of a claim. If the answer contains one of
# these but the source does not, the answer likely contradicts the source
# even when the non-negated content overlaps.
_NEGATION_WORDS = frozenset({
    "not", "no", "never", "neither", "nor", "none", "cannot", "cant",
    "isnt", "wasnt", "arent", "werent", "doesnt", "dont", "didnt",
    "wouldnt", "couldnt", "shouldnt", "hasnt", "havent", "hadnt",
})


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
    "Respond with exactly one word: ENTAILMENT, CONTRADICTION, or NEUTRAL."
)


class FactualVerifier:
    """Factual verifier with optional LLM-judge NLI and lexical fallback.

    Args:
        llm_judge: optional async callable that takes a prompt string and
            returns the LLM's text response. When provided, the verifier
            uses it as an NLI judge (entailment/contradiction/neutral).
            When None, falls back to hardened lexical overlap.
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
        """Use the LLM judge to classify entailment per source.

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

            verdict = self._parse_nli_verdict(raw)
            if verdict == "ENTAILMENT":
                return VerificationResult(
                    verified=True,
                    score=0.85,  # LLM judge is stronger than overlap but
                                 # not deterministic, so below 1.0
                    method="nli_llm_judge",
                    evidence={"source": str(src)[:100],
                              "verdict": verdict,
                              "source_count": len(sources)},
                )
            if verdict == "CONTRADICTION":
                return VerificationResult(
                    verified=False,
                    score=0.0,
                    method="nli_llm_judge",
                    evidence={"source": str(src)[:100],
                              "verdict": verdict,
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
    def _parse_nli_verdict(raw: str) -> str:
        """Extract ENTAILMENT / CONTRADICTION / NEUTRAL from the LLM response."""
        text = raw.strip().upper()
        # Check the first line / first word for the verdict.
        first_token = text.split()[0] if text else ""
        for verdict in ("ENTAILMENT", "CONTRADICTION", "NEUTRAL"):
            if verdict in first_token or verdict in text[:50]:
                return verdict
        return "NEUTRAL"

    @staticmethod
    def _verify_with_lexical(answer: str, sources: list) -> VerificationResult:
        """Hardened lexical overlap with negation detection.

        Checks if key fragments of the answer appear in a source. If the
        answer contains negation words absent from the source, treats it
        as a likely contradiction and rejects — this catches cases like
        "Paris is NOT the capital" passing because "Paris" and "capital"
        overlap with the source.
        """
        answer_lower = answer.lower().strip()
        answer_tokens = set(re.split(r'\W+', answer_lower))
        answer_negations = answer_tokens & _NEGATION_WORDS

        supported_by = []
        for src in sources:
            src_lower = str(src).lower()
            src_tokens = set(re.split(r'\W+', src_lower))
            src_negations = src_tokens & _NEGATION_WORDS

            words = [w for w in re.split(r'\W+', answer_lower) if len(w) > 3]
            if not words:
                continue
            overlap_ratio = sum(1 for w in words if w in src_lower) / len(words)
            if overlap_ratio < 0.4:
                continue

            # Negation polarity check: if the answer negates something the
            # source affirms (or vice versa), the overlap is likely a
            # contradiction, not support. Reject conservatively.
            if answer_negations and not src_negations:
                # Answer says "X is NOT Y", source says "X is Y" — contradiction.
                continue
            if src_negations and not answer_negations:
                # Source says "X is NOT Y", answer says "X is Y" — contradiction.
                continue

            supported_by.append(src[:100])

        if supported_by:
            return VerificationResult(
                verified=True,
                score=0.7,  # weak — overlap is not entailment
                method="retrieval_overlap",
                evidence={"supported_by": supported_by,
                          "source_count": len(sources)},
            )

        return VerificationResult(
            verified=False,
            score=0.0,
            method="retrieval_overlap",
            evidence={"source_count": len(sources)},
            error="answer not supported by any provided source "
                  "(overlap < 40% or negation mismatch)",
        )
