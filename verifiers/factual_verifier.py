"""Factual/retrieval verifier — honest placeholder.

For now, do NOT pretend factual claims are verified unless sources exist.
The honesty matters: returning ``verified=True`` without evidence would
make this system a liar. When retrieval sources are available in the
context, this verifier can be extended to check claims against them.

Behavior:
  - No retrieval source in context -> verified=False, method="unsupported_factual"
  - Sources provided -> check if the answer is supported by at least one source
"""

from typing import Any, Dict

from verifiers.base import VerificationResult


class FactualVerifier:
    """Honest factual verifier that does not fake verification."""

    name = "factual_verifier"

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

        # If sources are provided, do a basic substring check.
        # This is a weak heuristic — a real implementation would use
        # NLI/entailment or a retrieval-augmented check. But at least
        # it requires SOME evidence, unlike pure self-agreement.
        if isinstance(sources, str):
            sources = [sources]

        answer_lower = answer.lower().strip()
        supported_by = []
        for src in sources:
            src_lower = str(src).lower()
            # Check if key fragments of the answer appear in a source
            # (very conservative — only confirms overlap, not correctness)
            # Strip punctuation so "paris," matches "paris"
            import re as _re
            words = [w for w in _re.split(r'\W+', answer_lower) if len(w) > 3]
            if words and sum(1 for w in words if w in src_lower) / len(words) >= 0.4:
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
            error="answer not supported by any provided source (overlap < 40%)",
        )
