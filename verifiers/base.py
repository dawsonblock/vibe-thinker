"""Base verifier protocol and result model.

A verifier provides INDEPENDENT evidence that an answer is correct —
not model self-agreement. The distinction matters: self-agreement is
not verification, it is consensus. A deterministic verifier can actually
prove a math answer or execute a code snippet.

The protocol is async so verifiers that need I/O (subprocess, network
retrieval) can implement it naturally.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, runtime_checkable


@dataclass
class VerificationResult:
    """Result of an independent verification attempt.

    Attributes:
        verified: True only if the verifier has positive evidence the
            answer is correct. False means the verifier has evidence the
            answer is wrong OR could not verify it. Check ``error`` to
            distinguish "wrong" from "unverifiable".
        score: 0.0–1.0. 1.0 = fully verified, 0.0 = wrong or unverifiable.
        method: short tag identifying the verification method
            (e.g. "python_eval", "unit_tests", "numeric_comparison",
            "unsupported_factual").
        evidence: structured details about what was checked.
        error: None if no error, else a description of why verification
            could not be completed (timeout, parse failure, etc.).
    """
    verified: bool
    score: float
    method: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    error: str | None = None


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
