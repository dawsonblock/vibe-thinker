"""Confidence scoring with separated fields.

A single score is too vague — it conflates model self-agreement with
deterministic verification. This module splits confidence into distinct
fields so callers can tell the difference between:

  - model_confidence: the model's own claim-level self-verification score
  - claim_consistency: how well the claims agree with each other
  - deterministic_verification: independent verifier result (0.0 or 1.0)
  - final_score: the blended score used for routing/caching decisions

Key rule: self_claims_only confidence is HARD CAPPED at 0.65. Model
self-agreement must never become fake certainty.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Maximum confidence allowed when the only verification is model self-agreement.
SELF_CLAIMS_ONLY_CAP = 0.65


@dataclass
class ConfidenceBreakdown:
    """Separated confidence fields for a CLR result.

    Attributes:
        model_confidence: raw self-verification score (mean^5 over claims).
        claim_consistency: fraction of claims that passed verification.
        deterministic_verification: 1.0 if a deterministic verifier confirmed
            the answer, 0.0 if it refuted it, None if no verifier was run.
        final_score: the blended score used for decisions.
        verified: True if a deterministic verifier confirmed the answer.
        verification_method: how the answer was verified.
    """
    model_confidence: float = 0.0
    claim_consistency: float = 0.0
    deterministic_verification: Optional[float] = None
    final_score: float = 0.0
    verified: bool = False
    verification_method: str = "self_claims_only"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_confidence": round(self.model_confidence, 4),
            "claim_consistency": round(self.claim_consistency, 4),
            "deterministic_verification": (
                round(self.deterministic_verification, 4)
                if self.deterministic_verification is not None
                else None
            ),
            "final_score": round(self.final_score, 4),
            "verified": self.verified,
            "verification_method": self.verification_method,
        }


def compute_confidence(
    model_score: float,
    claim_consistency: float,
    deterministic_verification: Optional[float] = None,
    verification_method: str = "self_claims_only",
) -> ConfidenceBreakdown:
    """Compute a separated confidence breakdown.

    Scoring rules:
      - If deterministic verification is available:
          final_score = deterministic_verification * 0.7 + claim_consistency * 0.3
      - If no deterministic verification:
          final_score = min(claim_consistency, SELF_CLAIMS_ONLY_CAP)
      - Self-claims-only is hard-capped at 0.65

    Args:
        model_score: the raw self-verification score (e.g. mean^5 over claims).
        claim_consistency: fraction of claims that passed (0.0–1.0).
        deterministic_verification: 1.0 (confirmed), 0.0 (refuted), or None.
        verification_method: how the answer was verified.

    Returns:
        A :class:`ConfidenceBreakdown` with all fields populated.
    """
    verified = (
        deterministic_verification is not None
        and deterministic_verification >= 1.0
    )

    if deterministic_verification is not None:
        final_score = (
            deterministic_verification * 0.7
            + claim_consistency * 0.3
        )
    else:
        # No deterministic verifier — cap at SELF_CLAIMS_ONLY_CAP
        final_score = min(claim_consistency, SELF_CLAIMS_ONLY_CAP)

    # If a deterministic verifier explicitly refuted the answer (0.0),
    # score must be 0. Partial scores (0 < x < 1) blend normally.
    if deterministic_verification is not None and deterministic_verification <= 0.0:
        final_score = 0.0
        verified = False

    return ConfidenceBreakdown(
        model_confidence=model_score,
        claim_consistency=claim_consistency,
        deterministic_verification=deterministic_verification,
        final_score=final_score,
        verified=verified,
        verification_method=verification_method,
    )
