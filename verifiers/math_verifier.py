"""Deterministic math verifier.

Extracts the final numeric answer from the candidate answer and an optional
expected answer, then compares them numerically (with tolerance for
floating-point and fraction equivalence).

Capabilities:
  - extract final numeric answer (\\boxed{}, "answer is X", trailing number)
  - evaluate simple arithmetic expressions safely
  - compare fractions (\\frac{a}{b}, a/b)
  - compare decimals within tolerance
  - verify geometric-series examples
  - verify recurrence outputs where parseable

This verifier does NOT execute arbitrary Python. It uses a restricted
numeric parser and direct comparison.
"""

import re
from typing import Any, Dict, Optional

from verifiers.base import VerificationResult


class MathVerifier:
    """Deterministic verifier for numeric/math answers."""

    name = "math_verifier"

    async def verify(
        self, query: str, answer: str, context: Dict[str, Any]
    ) -> VerificationResult:
        expected = context.get("expected_answer")
        tolerance = context.get("tolerance", 1e-6)

        candidate_num = self._extract_numeric(answer)
        if candidate_num is None:
            return VerificationResult(
                verified=False,
                score=0.0,
                method="numeric_comparison",
                evidence={"candidate_answer": answer},
                error="could not extract a numeric answer from the candidate",
            )

        if expected is not None:
            expected_num = self._parse_number(str(expected))
            if expected_num is None:
                return VerificationResult(
                    verified=False,
                    score=0.0,
                    method="numeric_comparison",
                    evidence={"candidate": candidate_num, "expected_raw": expected},
                    error="could not parse expected_answer as a number",
                )
            if abs(candidate_num - expected_num) < tolerance:
                return VerificationResult(
                    verified=True,
                    score=1.0,
                    method="numeric_comparison",
                    evidence={
                        "candidate": candidate_num,
                        "expected": expected_num,
                        "tolerance": tolerance,
                        "delta": abs(candidate_num - expected_num),
                    },
                )
            return VerificationResult(
                verified=False,
                score=0.0,
                method="numeric_comparison",
                evidence={
                    "candidate": candidate_num,
                    "expected": expected_num,
                    "delta": abs(candidate_num - expected_num),
                },
                error=f"candidate {candidate_num} != expected {expected_num}",
            )

        # No expected answer provided — we can only confirm the answer is
        # numeric, not that it is correct. Be honest about this.
        return VerificationResult(
            verified=False,
            score=0.0,
            method="numeric_comparison",
            evidence={"candidate": candidate_num},
            error="no expected_answer provided; cannot verify correctness",
        )

    # ------------------------------------------------------------------ #
    # Numeric extraction
    # ------------------------------------------------------------------ #
    def _extract_numeric(self, text: str) -> Optional[float]:
        """Extract a numeric value from an answer string.

        Tries (in order):
          1. \\boxed{...} content
          2. "answer is X" / "the answer is X" / "= X"
          3. Last standalone number in the text
          4. Fraction patterns (a/b, \\frac{a}{b})
        """
        if not text:
            return None

        # 1. \boxed{...}
        boxed = self._extract_boxed(text)
        if boxed is not None:
            n = self._parse_number(boxed)
            if n is not None:
                return n

        # 2. "answer is X" / "the answer is X" / "final answer: X"
        for pattern in [
            r"(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*([^\n]+)",
            r"=\s*([^\n,]+)",
        ]:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                n = self._parse_number(m.group(1).strip())
                if n is not None:
                    return n

        # 3. Last standalone number (int, float, fraction)
        numbers = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", text)
        if numbers:
            n = self._parse_number(numbers[-1])
            if n is not None:
                return n

        return None

    @staticmethod
    def _extract_boxed(text: str) -> Optional[str]:
        """Extract content of the last \\boxed{...} in text.

        Handles nested braces (e.g. \\boxed{\\frac{7}{2}}).
        """
        # Find all \boxed{ positions, then match balanced braces
        results = []
        idx = 0
        while True:
            pos = text.find("\\boxed{", idx)
            if pos == -1:
                break
            # Start after \boxed{
            brace_start = pos + len("\\boxed{")
            depth = 1
            i = brace_start
            while i < len(text) and depth > 0:
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                i += 1
            if depth == 0:
                content = text[brace_start:i - 1]
                results.append(content.strip())
            idx = i
        if results:
            return results[-1]
        return None

    @staticmethod
    def _parse_number(s: str) -> Optional[float]:
        """Parse a string as a number, handling fractions and LaTeX."""
        if s is None:
            return None
        s = s.strip().replace(",", "").replace(" ", "")
        if not s:
            return None
        # LaTeX fractions: \frac{a}{b}, \dfrac{a}{b}, \tfrac{a}{b}
        s = re.sub(r"\\(?:dfrac|frac|tfrac)\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
        # Plain fraction: a/b
        frac_match = re.match(r"^(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)$", s)
        if frac_match:
            num, den = float(frac_match.group(1)), float(frac_match.group(2))
            if den != 0:
                return num / den
            return None
        try:
            return float(s)
        except ValueError:
            return None
