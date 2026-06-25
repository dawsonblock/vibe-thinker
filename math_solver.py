"""Deterministic math problem solver for verifier context.

This module derives expected answers for simple math problems so the
MathVerifier has something to compare against. It does NOT solve
arbitrary math — it handles a narrow set of patterns:

  1. Simple arithmetic: "What is 2+2?" -> 4
  2. Finite sums: "Compute the sum of 1 + 2 + 3 + 4 + 5" -> 15
  3. Explicit recurrences: "a_1=2, a_{n+1}=a_n^2-a_n+1, find a_5" -> 1807
  4. Geometric series (finite, explicit terms)

If a problem doesn't match any pattern, it returns None — do NOT fake
an expected answer. The verifier will then return verified=False with
"no expected_answer provided", which is the honest result.

This is a deterministic computation, not a model call. The whole point
is that it provides INDEPENDENT evidence the model answer is correct.
"""

import re
from typing import Optional


def solve(problem: str) -> Optional[str]:
    """Attempt to deterministically solve a math problem.

    Args:
        problem: the math problem string.

    Returns:
        The expected answer as a string, or None if the problem doesn't
        match any solvable pattern.
    """
    result = _try_recurrence(problem)
    if result is not None:
        return result

    result = _try_finite_sum(problem)
    if result is not None:
        return result

    result = _try_arithmetic(problem)
    if result is not None:
        return result

    return None


def _try_recurrence(problem: str) -> Optional[str]:
    """Parse and solve explicit recurrence relations.

    Handles patterns like:
      a_1=2, a_{n+1}=a_n^2-a_n+1, find a_5
      a_1=1, a_{n+1}=2*a_n+1, find a_4

    The recurrence must be explicit (a_{n+1} = f(a_n)) with a single
    variable. Polynomial expressions in a_n are supported.
    """
    # Extract initial condition: a_1=2 or a_1 = 2
    init_match = re.search(r"a_1\s*=\s*(-?\d+(?:\.\d+)?)", problem)
    if not init_match:
        return None

    a0 = float(init_match.group(1))

    # Extract recurrence: a_{n+1} = <expression> or a_(n+1) = <expr>
    rec_match = re.search(
        r"a_\{?n\+1\}?\s*=\s*(.+?)(?:[,;.]|\bfind\b|$)",
        problem, re.IGNORECASE,
    )
    if not rec_match:
        return None

    expr = rec_match.group(1).strip()
    # Clean up the expression — replace a_n with a placeholder
    # We'll evaluate by substituting a_n with the current value
    expr = expr.replace("a_{n}", "x").replace("a_n", "x").replace("a", "x")
    # Remove any remaining LaTeX formatting
    expr = expr.replace("{", "").replace("}", "")
    # Replace ^ with ** for Python exponentiation
    expr = expr.replace("^", "**")

    # Verify the expression is safe (only contains x, numbers, operators)
    if not re.match(r"^[\dx\s+\-*/().]+(\*\*)?[\dx\s+\-*/().]*$", expr):
        return None

    # Extract target: find a_5, find a_{5}
    target_match = re.search(r"find\s+a_?\{?(\d+)\}?", problem, re.IGNORECASE)
    if not target_match:
        return None

    target_n = int(target_match.group(1))
    if target_n < 1 or target_n > 20:  # safety limit
        return None

    # Iterate the recurrence
    current = a0
    for i in range(1, target_n):
        try:
            current = eval(expr, {"x": current, "__builtins__": {}}, {})
        except Exception:
            return None
        # Safety: prevent overflow
        if abs(current) > 1e15:
            return None

    # Return as integer if it's a whole number
    if current == int(current):
        return str(int(current))
    return str(current)


def _try_finite_sum(problem: str) -> Optional[str]:
    """Parse and compute finite sums.

    Handles:
      "Compute the sum of 1 + 2 + 3 + 4 + 5"
      "sum of 1 + 1/2 + 1/4 + 1/8 + 1/16"
    """
    # Look for "sum of <numbers with +>"
    sum_match = re.search(
        r"sum\s+of\s+([\d\s+/\-\.]+)",
        problem, re.IGNORECASE,
    )
    if not sum_match:
        return None

    terms_str = sum_match.group(1).strip()
    # Split by +
    parts = [p.strip() for p in terms_str.split("+") if p.strip()]
    if len(parts) < 2:
        return None

    total = 0.0
    for part in parts:
        try:
            # Handle fractions: 1/2
            if "/" in part:
                num, den = part.split("/")
                total += float(num) / float(den)
            else:
                total += float(part)
        except (ValueError, ZeroDivisionError):
            return None

    if total == int(total):
        return str(int(total))
    return str(total)


def _try_arithmetic(problem: str) -> Optional[str]:
    """Parse and compute simple arithmetic expressions.

    Handles:
      "What is 2+2?"
      "What is 3 * 4?"
      "Calculate 15 - 7"
    """
    # Look for "what is <expr>" or "calculate <expr>" or "compute <expr>"
    arith_match = re.search(
        r"(?:what\s+is|calculate|compute)\s+([\d\s+\-*/().]+)",
        problem, re.IGNORECASE,
    )
    if not arith_match:
        return None

    expr = arith_match.group(1).strip().rstrip("?")
    # Verify it's a safe arithmetic expression
    if not re.match(r"^[\d\s+\-*/().]+$", expr):
        return None

    try:
        result = eval(expr, {"__builtins__": {}}, {})
        if result == int(result):
            return str(int(result))
        return str(result)
    except Exception:
        return None
