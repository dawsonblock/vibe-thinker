"""Base verifier protocol and result model.

A verifier provides INDEPENDENT evidence that an answer is correct —
not model self-agreement. The distinction matters: self-agreement is
not verification, it is consensus. A deterministic verifier can actually
prove a math answer or execute a code snippet.

The protocol is async so verifiers that need I/O (subprocess, network
retrieval) can implement it naturally.

v0.4.0-alpha (stabilization): VerificationResult now includes optional
``confidence``, ``reason``, ``warnings``, and ``errors`` fields for
stricter contract enforcement. The existing ``score`` and ``error``
fields are retained for backward compat. Rules:
  1. Verifier failure is not the same as answer failure.
  2. Missing dependency must return verified=False (not raise).
  3. Self-claims cannot raise confidence above threshold.
  4. Unsupported verification must not be labeled verified.
  5. External-source factual verification must fail closed when sources
     are missing.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, runtime_checkable


@dataclass
class VerificationResult:
    """Result of an independent verification attempt.

    Attributes:
        verified: True only if the verifier has positive evidence the
            answer is correct. False means the verifier has evidence the
            answer is wrong OR could not verify it. Check ``error`` to
            distinguish "wrong" from "unverifiable".
        score: 0.0–1.0. 1.0 = fully verified, 0.0 = wrong or unverifiable.
            Retained for backward compat; prefer ``confidence``.
        method: short tag identifying the verification method
            (e.g. "python_eval", "unit_tests", "numeric_comparison",
            "unsupported_factual").
        evidence: structured details about what was checked.
        error: None if no error, else a description of why verification
            could not be completed (timeout, parse failure, etc.).
            Retained for backward compat; prefer ``errors`` (list).
        confidence: 0.0–1.0. The verifier's confidence in its verdict.
            Distinct from ``score`` (which is the answer's score) —
            confidence measures how sure the VERIFIER is, not how good
            the answer is. Defaults to ``score`` for backward compat.
        reason: human-readable explanation of the verdict.
        warnings: non-fatal issues encountered during verification.
        errors: list of errors that prevented verification (empty if
            verification completed successfully).
    """
    verified: bool
    score: float
    method: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    confidence: float | None = None
    reason: str = ""
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@runtime_checkable
class Verifier(Protocol):
    """Protocol for deterministic verifier adapters."""
    name: str

    async def verify(
        self, query: str, answer: str, context: Dict[str, Any]
    ) -> VerificationResult:
        """Independently verify ``answer`` to ``query``.

        Args:
            query: the original problem/query.
            answer: the candidate answer to verify.
            context: optional context (e.g. expected answer, unit tests,
                reference solution).

        Returns:
            A :class:`VerificationResult`. Must be honest: return
            ``verified=False`` if you cannot verify — never fake it.
        """
        ...
