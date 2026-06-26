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

Security: arithmetic expressions are evaluated with a strict AST
whitelist (``_safe_eval``) instead of ``eval()``. Only numbers, the
four binary operators, unary +/-, parenthesization, exponentiation
(``**``), and a single variable name (for recurrences) are permitted.
Any other AST node (calls, attributes, comprehensions, etc.) causes
the expression to be rejected (returns None).
"""

import ast
import re
from typing import Optional


# AST node types allowed in safe arithmetic evaluation.
_SAFE_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)
_SAFE_UNARYOPS = (ast.UAdd, ast.USub)


def _safe_eval(expr: str, names: Optional[dict] = None) -> Optional[float]:
    """Evaluate a strictly arithmetic expression without ``eval()``.

    Only the following are permitted:
      - numeric literals (int/float)
      - binary +, -, *, /, **, %, //
      - unary +, -
      - parenthesization
      - a single variable name (when ``names`` is provided, e.g. {"x": 2.0})

    Any other construct (function calls, attribute access, subscripts,
    comprehensions, boolean ops, comparisons, etc.) is rejected and
    ``None`` is returned. This is a defense-in-depth replacement for
    ``eval(expr, {"__builtins__": {}}, {})`` which is unsafe even with
    stripped builtins.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    return _eval_node(tree.body, names or {})


def _eval_node(node, names: dict) -> Optional[float]:
    """Recursively evaluate an AST node, returning None on any disallowed node."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        return None
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, _SAFE_BINOPS):
        left = _eval_node(node.left, names)
        right = _eval_node(node.right, names)
        if left is None or right is None:
            return None
        try:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
        except (ZeroDivisionError, OverflowError):
            return None
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _SAFE_UNARYOPS):
        operand = _eval_node(node.operand, names)
        if operand is None:
            return None
        return +operand if isinstance(node.op, ast.UAdd) else -operand
    return None


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
        val = _safe_eval(expr, {"x": current})
        if val is None:
            return None
        current = val
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

    result = _safe_eval(expr)
    if result is None:
        return None
    if result == int(result):
        return str(int(result))
    return str(result)
