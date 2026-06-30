#!/usr/bin/env python3
"""Hard challenge demo for the vibe-thinker verification system.

This demo presents a multi-stage reasoning problem that exercises ALL
the deterministic verifiers the system has — without requiring a live
LLM. The problem is deliberately difficult: it combines a non-linear
recurrence, a constraint-satisfaction sub-problem, a structural/schema
sub-problem, and a code-correctness sub-problem. Each stage is verified
with hard evidence (not "the model said so"):

  Stage 1 — Math (recurrence): a chaotic quadratic recurrence that
    requires 7 iterations. Verified by the deterministic math_solver
    (independent computation) + math_verifier (numeric comparison).

  Stage 2 — Logic (constraint satisfaction): find integer values that
    satisfy a system of non-linear constraints. Verified by Z3/SMT
    (proof tool, not a model). Requires z3-solver.

  Stage 3 — Schema (structural): the answer from stage 2 must be
    packaged as a JSON object matching a strict schema. Verified by
    the schema_verifier (deterministic JSON-schema validation).

  Stage 4 — Code (algorithmic correctness): a candidate Python
    implementation of the recurrence from stage 1 is executed in a
    Docker sandbox with unit tests. Verified by the code_verifier
    (real sandboxed execution, not static analysis). Requires Docker.

  Stage 5 — Orchestrator spine: the full orchestrator.run() ->
    _run_clr_with_cache -> reasoner.run path is exercised with a
    fake CLR backend (same as the smoke test), proving the runtime
    spine is intact.

The demo PASSES only if every stage produces verified=True with
evidence. If any stage fails, the demo reports which stage failed
and why, and exits with a non-zero code.

Usage:
    python3 demo_hard_challenge.py
    python3 demo_hard_challenge.py --verbose   # show full evidence
    python3 demo_hard_challenge.py --stage 1   # run only stage N
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import os
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def _c(color: str, text: str) -> str:
    return f"{color}{text}{Colors.RESET}"


def _print_stage(num: int, title: str, description: str) -> None:
    print()
    print(_c(Colors.CYAN, _c(Colors.BOLD, f"  Stage {num}: {title}")))
    print(_c(Colors.DIM, f"  {'—' * 60}"))
    # Word-wrap the description at 58 chars for readability.
    words = description.split()
    line = "  "
    for word in words:
        if len(line) + len(word) + 1 > 62:
            print(_c(Colors.DIM, line))
            line = "  " + word
        else:
            line += " " + word if line.strip() else word
    if line.strip():
        print(_c(Colors.DIM, line))


def _print_result(
    passed: bool, summary: str, evidence: Optional[Dict] = None,
    verbose: bool = False,
) -> None:
    status = _c(Colors.GREEN, "PASS") if passed else _c(Colors.RED, "FAIL")
    print(f"  Result: [{status}] {summary}")
    if evidence and verbose:
        print(_c(Colors.DIM, "  Evidence:"))
        for key, value in evidence.items():
            val_str = str(value)
            if len(val_str) > 120:
                val_str = val_str[:120] + "..."
            print(_c(Colors.DIM, f"    {key}: {val_str}"))


# ---------------------------------------------------------------------------
# Stage 1: Math — chaotic quadratic recurrence
# ---------------------------------------------------------------------------

async def stage1_math_recurrence(verbose: bool) -> Tuple[bool, str, Dict]:
    """Solve and verify a non-linear recurrence relation.

    Problem: a_1 = 3, a_{n+1} = a_n^2 - 2*a_n + 1, find a_7.

    This is the recurrence a_{n+1} = (a_n - 1)^2, which grows
    extremely fast (double-exponential). The math_solver computes
    this independently, and the math_verifier confirms the answer.

    The answer is (3-1)^2 = 4, then (4-1)^2 = 9, then (9-1)^2 = 64,
    then (64-1)^2 = 3969, then (3969-1)^2 = 15745024, then
    (15745024-1)^2 = 247905749270529. So a_7 = 247905749270529.
    """
    from math_solver import solve
    from verifiers.math_verifier import MathVerifier

    problem = "a_1=3, a_{n+1}=a_n^2-2*a_n+1, find a_7"

    # Step 1: deterministic solver computes the expected answer.
    expected = solve(problem)
    if expected is None:
        return False, "math_solver could not solve the recurrence", {}

    # Step 2: simulate a "model answer" (in a real run, the LLM would
    # produce this). We format it as a boxed answer.
    model_answer = f"The answer is \\boxed{{{expected}}}"

    # Step 3: verify with the deterministic math verifier.
    verifier = MathVerifier()
    result = await verifier.verify(
        problem, model_answer,
        context={"expected_answer": expected},
    )

    return result.verified, (
        f"recurrence a_1=3, a_{{n+1}}=(a_n-1)^2, a_7 = {expected} "
        f"(verified={result.verified}, method={result.method})"
    ), result.evidence


# ---------------------------------------------------------------------------
# Stage 2: Logic — constraint satisfaction via Z3/SMT
# ---------------------------------------------------------------------------

async def stage2_logic_constraints(verbose: bool) -> Tuple[bool, str, Dict]:
    """Find integers satisfying a system of non-linear constraints.

    Problem: find integers x, y, z such that:
      - x + y + z = 30
      - x * y * z = 336
      - x < y < z
      - x > 0

    The unique solution is x=4, y=8, z=18 (4+8+18=30, 4*8*18=576...
    wait, let me recalculate. 4*8*18 = 576, not 336. Let me find the
    right problem.)

    Actually: x=6, y=8, z=16: 6+8+16=30, 6*8*16=768. No.
    x=4, y=7, z=19: 4+7+19=30, 4*7*19=532. No.
    x=3, y=8, z=19: 3+8+19=30, 3*8*19=456. No.
    x=2, y=12, z=16: 2+12+16=30, 2*12*16=384. No.
    x=4, y=6, z=20: 4+6+20=30, 4*6*20=480. No.
    x=5, y=6, z=19: 5+6+19=30, 5*6*19=570. No.
    x=2, y=7, z=21: 2+7+21=30, 2*7*21=294. No.
    x=1, y=12, z=17: 1+12+17=30, 1*12*17=204. No.
    x=3, y=10, z=17: 3+10+17=30, 3*10*17=510. No.
    x=4, y=9, z=17: 4+9+17=30, 4*9*17=612. No.
    x=2, y=14, z=14: not strictly increasing.
    x=2, y=10, z=18: 2+10+18=30, 2*10*18=360. No.
    x=2, y=8, z=20: 2+8+20=30, 2*8*20=320. No.
    x=2, y=6, z=22: 2+6+22=30, 2*6*22=264. No.
    x=4, y=8, z=18: already tried = 576.
    x=6, y=7, z=17: 6+7+17=30, 6*7*17=714. No.
    x=7, y=8, z=15: 7+8+15=30, 7*8*15=840. No.
    x=4, y=10, z=16: 4+10+16=30, 4*10*16=640. No.
    x=4, y=11, z=15: 4+11+15=30, 4*11*15=660. No.
    x=5, y=8, z=17: 5+8+17=30, 5*8*17=680. No.
    x=5, y=10, z=15: 5+10+15=30, 5*10*15=750. No.
    x=6, y=9, z=15: 6+9+15=30, 6*9*15=810. No.
    x=7, y=9, z=14: 7+9+14=30, 7*9*14=882. No.
    x=7, y=10, z=13: 7+10+13=30, 7*10*13=910. No.
    x=8, y=9, z=13: 8+9+13=30, 8*9*13=936. No.
    x=8, y=10, z=12: 8+10+12=30, 8*10*12=960. No.
    x=8, y=11, z=11: not strictly increasing.

    Let me use a product that works: x=2, y=3, z=25: 2*3*25=150. No.
    x=1, y=14, z=15: 1*14*15=210. No.
    x=1, y=15, z=14: not increasing.
    x=3, y=7, z=20: 3*7*20=420. No.
    x=3, y=5, z=22: 3*5*22=330. No.
    x=3, y=4, z=23: 3*4*23=276. No.
    x=4, y=5, z=21: 4*5*21=420. No.
    x=4, y=3, z=23: not increasing.

    OK let me just pick constraints where I KNOW the answer:
    x=4, y=6, z=20: sum=30, product=480. Use product=480.
    Actually, let me use a cleaner problem.

    Problem: x + y + z = 25, x * y = 24, x < y < z, x > 0, z is prime.
    x=3, y=8: 3*8=24, z=14 (not prime).
    x=4, y=6: 4*6=24, z=15 (not prime).
    x=2, y=12: 2*12=24, z=11 (prime!). 2 < 12 < 11? No, 12 > 11.
    x=1, y=24: 1*24=24, z=0 (not > 0).
    x=6, y=4: not increasing.

    Let me use: x + y + z = 20, x * y = 24, x < y, z > x, z > y, all > 0.
    x=3, y=8, z=9: 3+8+9=20, 3*8=24. 3 < 8 < 9. All > 0. Works!

    So the problem is: find positive integers x, y, z where
    x + y + z = 20, x * y = 24, x < y < z.

    The solution: x=3, y=8, z=9. Verified by Z3.
    """
    try:
        import z3
    except ImportError:
        return False, "z3-solver not installed (pip install z3-solver)", {}

    from verifiers.logic_verifier import LogicVerifier

    constraints = [
        "x + y + z == 20",
        "x * y == 24",
        "x < y",
        "y < z",
        "x > 0",
    ]
    variables = {"x": "Int", "y": "Int", "z": "Int"}
    # The "model answer" — in a real run the LLM would produce this.
    values = {"x": 3, "y": 8, "z": 9}
    answer_str = f"x={values['x']}, y={values['y']}, z={values['z']}"

    verifier = LogicVerifier()
    result = await verifier.verify(
        "Find positive integers x, y, z where x+y+z=20, x*y=24, x<y<z",
        answer_str,
        context={
            "constraints": constraints,
            "variables": variables,
            "values": values,
        },
    )

    return result.verified, (
        f"constraint satisfaction: x=3, y=8, z=9 "
        f"(sum=20, product=24, strictly increasing) "
        f"(verified={result.verified}, method={result.method})"
    ), result.evidence


# ---------------------------------------------------------------------------
# Stage 3: Schema — structural validation
# ---------------------------------------------------------------------------

async def stage3_schema_validation(verbose: bool) -> Tuple[bool, str, Dict]:
    """Verify a JSON answer against a strict schema.

    The "model" must produce a JSON object describing the solution from
    stage 2, with specific types, constraints, and a required structure.
    The schema enforces:
      - type: object
      - required: solution, verification, metadata
      - solution: object with x, y, z as integers with min/max bounds
      - verification: string matching a pattern
      - metadata: object with method and timestamp

    A subtly wrong answer (e.g., z as a string, or missing a required
    field) must be REJECTED. We test both the correct answer and a
    wrong answer to prove the verifier is actually checking.
    """
    from verifiers.schema_verifier import SchemaVerifier

    schema = {
        "type": "object",
        "required": ["solution", "verification", "metadata"],
        "properties": {
            "solution": {
                "type": "object",
                "required": ["x", "y", "z"],
                "properties": {
                    "x": {"type": "integer", "minimum": 1, "maximum": 20},
                    "y": {"type": "integer", "minimum": 1, "maximum": 20},
                    "z": {"type": "integer", "minimum": 1, "maximum": 20},
                },
            },
            "verification": {
                "type": "string",
                "pattern": "^(smt_verified|smt_refuted|smt_unknown)$",
            },
            "metadata": {
                "type": "object",
                "required": ["method"],
                "properties": {
                    "method": {"type": "string"},
                    "solver": {"type": "string"},
                },
            },
        },
    }

    # Correct answer — should pass.
    correct_answer = json.dumps({
        "solution": {"x": 3, "y": 8, "z": 9},
        "verification": "smt_verified",
        "metadata": {
            "method": "z3_constraint_check",
            "solver": "Z3 4.x",
        },
    })

    verifier = SchemaVerifier()
    result_correct = await verifier.verify(
        "Package the constraint solution as a structured JSON object",
        correct_answer,
        context={"schema": schema, "format": "json"},
    )

    if not result_correct.verified:
        return False, (
            f"correct answer was REJECTED by schema verifier: "
            f"{result_correct.error}"
        ), result_correct.evidence

    # Wrong answer — z as a string instead of integer. Must be rejected.
    wrong_answer = json.dumps({
        "solution": {"x": 3, "y": 8, "z": "9"},
        "verification": "smt_verified",
        "metadata": {"method": "z3_constraint_check"},
    })

    result_wrong = await verifier.verify(
        "Package the constraint solution as a structured JSON object",
        wrong_answer,
        context={"schema": schema, "format": "json"},
    )

    if result_wrong.verified:
        return False, (
            "wrong answer (z as string) was INCORRECTLY ACCEPTED — "
            "schema verifier is not checking types!"
        ), result_wrong.evidence

    return True, (
        f"correct JSON accepted, wrong JSON (z as string) rejected "
        f"(method={result_correct.method})"
    ), result_correct.evidence


# ---------------------------------------------------------------------------
# Stage 4: Code — sandboxed execution with unit tests
# ---------------------------------------------------------------------------

async def stage4_code_verification(verbose: bool) -> Tuple[bool, str, Dict]:
    """Verify a candidate Python implementation in a Docker sandbox.

    The candidate code implements the recurrence from stage 1 as a
    function. Unit tests check:
      - base case (a_1 = 3)
      - first iteration (a_2 = 4)
      - the full a_7 value
      - edge case: a_1 with a different starting value
      - type check: returns int, not float

    The code is executed in a real Docker container (not static
    analysis). A subtly broken implementation (off-by-one, float
    division, etc.) would fail the tests.
    """
    from verifiers.code_verifier import CodeVerifier

    # Candidate code — a correct implementation.
    candidate_code = '''
def recurrence(start, n):
    """Compute a_n where a_1=start, a_{k+1} = (a_k - 1)^2."""
    if n < 1:
        raise ValueError("n must be >= 1")
    current = start
    for _ in range(1, n):
        current = (current - 1) ** 2
    return current
'''

    # Unit tests — these run inside the sandbox against the candidate.
    # NOTE: We use plain assert statements (not unittest.main()) because
    # the test harness wraps the tests in a try/except that catches
    # Exception. unittest.main() calls sys.exit() which raises
    # SystemExit (a BaseException, not Exception) and would bypass the
    # harness's nonce-marker print. Plain asserts raise AssertionError
    # which IS caught by the harness.
    unit_tests = '''
# Test 1: base case
assert recurrence(3, 1) == 3, f"base case: expected 3, got {recurrence(3, 1)}"

# Test 2: first iteration
assert recurrence(3, 2) == 4, f"a_2: expected 4, got {recurrence(3, 2)}"

# Test 3: third term
assert recurrence(3, 3) == 9, f"a_3: expected 9, got {recurrence(3, 3)}"

# Test 4: the full a_7 value (computed independently by math_solver)
assert recurrence(3, 7) == 247905749270529, \\
    f"a_7: expected 247905749270529, got {recurrence(3, 7)}"

# Test 5: different starting value
# a_1=5: a_2=(5-1)^2=16, a_3=(16-1)^2=225
assert recurrence(5, 3) == 225, f"start=5 a_3: expected 225, got {recurrence(5, 3)}"

# Test 6: return type is int, not float
result = recurrence(3, 5)
assert isinstance(result, int), f"return type: expected int, got {type(result).__name__}"

# Test 7: n < 1 raises ValueError
try:
    recurrence(3, 0)
    assert False, "expected ValueError for n=0"
except ValueError:
    pass  # expected
'''

    verifier = CodeVerifier(timeout=15.0)
    if verifier.executor is None:
        return False, (
            "no sandbox executor available (install Docker to run "
            "this stage)"
        ), {}

    result = await verifier.verify(
        "Implement the recurrence a_1=start, a_{n+1}=(a_n-1)^2 as a "
        "Python function",
        candidate_code,
        context={"unit_tests": unit_tests},
    )

    return result.verified, (
        f"recurrence implementation executed in Docker sandbox with "
        f"7 unit tests (verified={result.verified}, "
        f"method={result.method})"
    ), result.evidence


# ---------------------------------------------------------------------------
# Stage 5: Orchestrator spine
# ---------------------------------------------------------------------------

async def stage5_orchestrator_spine(verbose: bool) -> Tuple[bool, str, Dict]:
    """Exercise the full orchestrator.run() -> CLR path with no model.

    This is the same check as the smoke test: instantiate the real
    HybridReasoningOrchestrator, fake the CLR backend at the layer
    BELOW _run_clr_with_cache, and verify the full spine works.
    """
    from unittest.mock import AsyncMock
    from vibe_clr_async import CLRResult
    from hybrid_orchestrator import HybridReasoningOrchestrator

    try:
        orchestrator = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            use_clr=True,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
            code_verifier=None,
            retrieval_backend=None,
        )
    except Exception as e:
        return False, f"orchestrator construction failed: {e}", {}

    if "_run_clr_with_cache" not in vars(HybridReasoningOrchestrator):
        return False, "_run_clr_with_cache missing from class", {}

    fake_clr = CLRResult(
        best_answer="247905749270529",
        best_score=0.95,
        best_raw_trace="",
        verified=True,
        verification_method="math_verifier",
    )
    orchestrator.reasoner.run = AsyncMock(return_value=fake_clr)

    try:
        result = await orchestrator.run(
            "a_1=3, a_{n+1}=a_n^2-2*a_n+1, find a_7"
        )
    except Exception as e:
        return False, f"orchestrator.run() failed: {type(e).__name__}: {e}", {}

    if result is None:
        return False, "orchestrator.run() returned None", {}

    if result.final_answer != "247905749270529":
        return False, (
            f"unexpected final_answer: {result.final_answer!r}"
        ), {}

    if result.route_taken != "specialist_clr":
        return False, (
            f"unexpected route_taken: {result.route_taken!r}"
        ), {}

    if not orchestrator.reasoner.run.await_count:
        return False, "_run_clr_with_cache did not reach reasoner.run", {}

    return True, (
        f"orchestrator.run() -> _run_clr_with_cache -> reasoner.run "
        f"(answer={result.final_answer}, route={result.route_taken})"
    ), {
        "final_answer": result.final_answer,
        "route_taken": result.route_taken,
        "clr_score": result.clr_score,
        "reasoner_called": orchestrator.reasoner.run.await_count,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STAGES = [
    ("Math Recurrence", stage1_math_recurrence,
     "a_1=3, a_{n+1}=(a_n-1)^2 — a chaotic quadratic recurrence "
     "requiring 7 iterations. The math_solver computes the expected "
     "answer independently; the math_verifier confirms it via numeric "
     "comparison. The answer grows double-exponentially to "
     "247,905,749,270,529."),
    ("Logic Constraints", stage2_logic_constraints,
     "Find positive integers x, y, z where x+y+z=20, x*y=24, x<y<z. "
     "Verified by Z3/SMT — a proof tool, not a model. The solution "
     "(3, 8, 9) is checked against the constraints with a Z3 model "
     "as evidence."),
    ("Schema Validation", stage3_schema_validation,
     "The constraint solution must be packaged as a JSON object "
     "matching a strict schema (typed fields, required keys, regex "
     "pattern, min/max bounds). Both a correct answer and a subtly "
     "wrong answer (z as string) are tested to prove the verifier "
     "is actually checking."),
    ("Code Verification", stage4_code_verification,
     "A candidate Python implementation of the stage-1 recurrence is "
     "executed in a Docker sandbox with 7 unit tests. The tests cover "
     "the base case, iterations, the full a_7 value, a different "
     "starting value, return type, and error handling. Requires "
     "Docker."),
    ("Orchestrator Spine", stage5_orchestrator_spine,
     "The full orchestrator.run() -> _run_clr_with_cache -> "
     "reasoner.run path is exercised with a fake CLR backend. This "
     "proves the runtime spine is intact — the same check as the "
     "smoke test, but with the stage-1 problem."),
]


async def run_demo(verbose: bool, only_stage: Optional[int] = None) -> int:
    print()
    print(_c(Colors.BOLD, "  ╔══════════════════════════════════════════════════════════╗"))
    print(_c(Colors.BOLD, "  ║     Vibe-Thinker Hard Challenge Demo                     ║"))
    print(_c(Colors.BOLD, "  ║     5-stage verification gauntlet                        ║"))
    print(_c(Colors.BOLD, "  ╚══════════════════════════════════════════════════════════╝"))
    print()
    print(_c(Colors.DIM, "  Each stage is verified with hard evidence (not 'the model "
                         "said so')."))
    print(_c(Colors.DIM, "  The demo PASSES only if every stage produces "
                         "verified=True."))

    results: List[Tuple[int, str, bool, str]] = []
    overall_pass = True

    for i, (title, fn, desc) in enumerate(STAGES, 1):
        if only_stage is not None and only_stage != i:
            continue
        _print_stage(i, title, desc)
        try:
            passed, summary, evidence = await fn(verbose)
        except Exception as e:
            passed = False
            summary = f"exception: {type(e).__name__}: {e}"
            evidence = {}
        _print_result(passed, summary, evidence, verbose)
        results.append((i, title, passed, summary))
        if not passed:
            overall_pass = False

    # Summary
    print()
    print(_c(Colors.BOLD, "  ═══════════════════════════════════════════════════════════"))
    print(_c(Colors.BOLD, "  Summary"))
    print(_c(Colors.BOLD, "  ═══════════════════════════════════════════════════════════"))
    for num, title, passed, _summary in results:
        status = _c(Colors.GREEN, "PASS") if passed else _c(Colors.RED, "FAIL")
        print(f"    Stage {num} ({title}): [{status}]")
    print()
    if overall_pass:
        print(_c(Colors.GREEN, _c(Colors.BOLD,
              "  ✓ ALL STAGES PASSED — the verification system is intact.")))
    else:
        failed = [r for r in results if not r[2]]
        print(_c(Colors.RED, _c(Colors.BOLD,
              f"  ✗ {len(failed)} STAGE(S) FAILED:")))
        for num, title, _, summary in failed:
            print(_c(Colors.RED, f"    Stage {num} ({title}): {summary}"))
    print()
    return 0 if overall_pass else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Vibe-Thinker hard challenge demo — 5-stage "
                    "verification gauntlet")
    parser.add_argument("--verbose", action="store_true",
                        help="show full evidence for each stage")
    parser.add_argument("--stage", type=int, default=None,
                        help="run only a specific stage (1-5)")
    args = parser.parse_args()
    return asyncio.run(run_demo(verbose=args.verbose,
                                only_stage=args.stage))


if __name__ == "__main__":
    sys.exit(main())
