#!/usr/bin/env python3
"""Demo 3 — TurboQuant KV-cache compression perplexity validation.

Shows the PPL (perplexity) validation pipeline that checks whether
TurboQuant KV-cache compression degrades model output quality:

  1. Core PPL math: compute_ppl() from per-token log-probabilities
  2. Comparison + tolerance: compare_ppl() with pass/fail threshold
  3. Logprob extraction: parsing llama-server /completion responses
  4. End-to-end: baseline vs candidate PPL with a real tolerance check

No model server required — uses synthetic logprob data. To run against
a real model:

    # HTTP path (llama-server with --logprobs):
    llama-server -m model.gguf --port 8081 --logprobs
    python demo_turboquant_ppl.py --live --base-url http://127.0.0.1:8081

    # In-process path (ruvllm_py with candle):
    python demo_turboquant_ppl.py --inprocess --model model.gguf --metal

    python demo_turboquant_ppl.py
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import turboquant_ppl_check as ppl


def header(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


def test_ppl_math():
    """Test the core PPL computation with known values."""
    header("1. PPL MATH (compute_ppl from log-probabilities)")

    cases = [
        ("Perfect prediction (log P = 0)", [0.0, 0.0, 0.0, 0.0], 1.0),
        ("Uniform (log P = -1)", [-1.0, -1.0, -1.0, -1.0], math.e),
        ("Poor prediction (log P = -3)", [-3.0, -3.0], math.exp(3.0)),
        ("Mixed", [-0.5, -1.0, -0.1, -2.0], None),  # Just check it's positive
    ]

    passed = 0
    for label, logprobs, expected in cases:
        result = ppl.compute_ppl(logprobs)
        if expected is not None:
            ok = abs(result - expected) < 0.001
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            print(f"  [{status:4}] PPL={result:.4f}  expected={expected:.4f}  {label}")
        else:
            ok = result > 1.0
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            print(f"  [{status:4}] PPL={result:.4f}  {label}")

    print(f"\n  Results: {passed}/{len(cases)} passed")
    return passed == len(cases)


def test_ppl_comparison():
    """Test the comparison + tolerance logic."""
    header("2. PPL COMPARISON (tolerance-based pass/fail)")

    cases = [
        ("Within tolerance (0.5% < 1.5%)", 10.0, 10.05, 0.015, True),
        ("Exceeds tolerance (2% > 1.5%)", 10.0, 10.20, 0.015, False),
        ("Candidate better (lower PPL)", 10.0, 9.50, 0.015, True),
        ("Just under tolerance (1.4%)", 10.0, 10.14, 0.015, True),
        ("Just over tolerance (1.6%)", 10.0, 10.16, 0.015, False),
        ("Large baseline, small delta", 100.0, 100.5, 0.015, True),
    ]

    passed = 0
    for label, baseline, candidate, tol, expected_pass in cases:
        comp = ppl.compare_ppl(baseline, candidate, tol)
        status = "PASS" if comp.passed == expected_pass else "FAIL"
        if status == "PASS":
            passed += 1
        verdict = "PASS" if comp.passed else "FAIL"
        print(f"  [{status:4}] {verdict:4}  baseline={baseline:.1f} candidate={candidate:.2f} "
              f"delta={comp.pct_delta*100:+.2f}%  tol={tol*100:.1f}%  {label}")

    print(f"\n  Results: {passed}/{len(cases)} passed")
    return passed == len(cases)


def test_logprob_extraction():
    """Test extraction of logprobs from llama-server response formats."""
    header("3. LOGPROB EXTRACTION (llama-server response parsing)")

    # Old format: logprobs is a list
    old_format = {
        "logprobs": [
            {"token": "The", "logprob": -0.1},
            {"token": " quick", "logprob": -0.5},
            {"token": " brown", "logprob": -0.3},
        ]
    }

    # New format: logprobs is {"content": [...]}
    new_format = {
        "logprobs": {
            "content": [
                {"token": "The", "logprob": -0.1},
                {"token": " quick", "logprob": -0.5},
                {"token": " brown", "logprob": -0.3},
            ]
        }
    }

    cases = [
        ("Old format (list)", old_format, [-0.1, -0.5, -0.3]),
        ("New format (dict with content)", new_format, [-0.1, -0.5, -0.3]),
    ]

    passed = 0
    for label, response, expected in cases:
        result = ppl._extract_token_logprobs(response)
        ok = result == expected
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  [{status:4}] extracted {len(result)} logprobs: {result}  {label}")

    print(f"\n  Results: {passed}/{len(cases)} passed")
    return passed == len(cases)


def test_end_to_end():
    """Simulate a full baseline-vs-candidate PPL comparison."""
    header("4. END-TO-END SIMULATION (baseline vs TurboQuant candidate)")

    # Simulate per-token logprobs for a 128-token sequence
    # Baseline (f16/f16): slightly better predictions
    import random
    random.seed(42)
    baseline_logprobs = [-random.expovariate(2.0) for _ in range(128)]

    # Candidate (q8_0/turbo3): slightly worse due to KV compression
    # Add a small noise to simulate compression-induced degradation
    candidate_logprobs = [lp - random.gauss(0.001, 0.002) for lp in baseline_logprobs]

    baseline_result = ppl.PplResult.from_log_probs(
        baseline_logprobs,
        config={"cache_type_k": "f16", "cache_type_v": "f16"},
        source="simulated_baseline",
    )
    candidate_result = ppl.PplResult.from_log_probs(
        candidate_logprobs,
        config={"cache_type_k": "q8_0", "cache_type_v": "turbo3"},
        source="simulated_candidate",
    )

    comp = ppl.compare_ppl(baseline_result.ppl, candidate_result.ppl, tolerance=0.015)

    print(f"  Baseline:  PPL={baseline_result.ppl:.4f}  "
          f"({baseline_result.n_tokens} tokens, {baseline_result.config})")
    print(f"  Candidate: PPL={candidate_result.ppl:.4f}  "
          f"({candidate_result.n_tokens} tokens, {candidate_result.config})")
    print(f"  Delta:     {comp.delta:+.4f} ({comp.pct_delta*100:+.2f}%)")
    print(f"  Tolerance: {comp.tolerance*100:.1f}%")
    print(f"  Verdict:   {'PASS — compression is safe' if comp.passed else 'FAIL — do NOT ship'}")

    print()
    print("  JSON output (what would be saved to disk):")
    report = comp.to_dict()
    report["baseline_config"] = baseline_result.config
    report["candidate_config"] = candidate_result.config
    print(json.dumps(report, indent=2))

    return comp.passed


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Demo 3: TurboQuant PPL validation")
    parser.add_argument("--live", action="store_true", help="Run against a live llama-server")
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--inprocess", action="store_true", help="Use ruvllm_py in-process")
    parser.add_argument("--model", help="GGUF model path (for --inprocess)")
    parser.add_argument("--metal", action="store_true", help="Use Apple Silicon Metal")
    args = parser.parse_args()

    if args.live:
        header("DEMO 3 (LIVE): TurboQuant PPL via llama-server")
        corpus = os.path.join(os.path.dirname(__file__), "ppl_corpus.txt")
        if not os.path.exists(corpus):
            corpus = os.path.join(os.path.dirname(__file__), "..", "scripts", "ppl_corpus.txt")
        with open(corpus) as f:
            text = f.read()
        result = ppl.eval_http(args.base_url, text)
        print(f"  PPL={result.ppl:.4f} over {result.n_tokens} tokens (source={result.source})")
        return

    if args.inprocess:
        header("DEMO 3 (IN-PROCESS): TurboQuant PPL via ruvllm_py")
        corpus = os.path.join(os.path.dirname(__file__), "ppl_corpus.txt")
        if not os.path.exists(corpus):
            corpus = os.path.join(os.path.dirname(__file__), "..", "scripts", "ppl_corpus.txt")
        with open(corpus) as f:
            text = f.read()
        result = ppl.eval_inprocess(args.model, text, use_metal=args.metal)
        print(f"  PPL={result.ppl:.4f} over {result.n_tokens} tokens (source={result.source})")
        print(f"  Config: {result.config}")
        return

    header("DEMO 3: TurboQuant KV-cache Compression PPL Validation")
    print("  No model server required — uses synthetic logprob data.")
    print()

    ok1 = test_ppl_math()
    ok2 = test_ppl_comparison()
    ok3 = test_logprob_extraction()
    ok4 = test_end_to_end()

    header("SUMMARY")
    print(f"  PPL math:           {'PASS' if ok1 else 'FAIL'}")
    print(f"  Comparison logic:   {'PASS' if ok2 else 'FAIL'}")
    print(f"  Logprob extraction: {'PASS' if ok3 else 'FAIL'}")
    print(f"  End-to-end sim:     {'PASS' if ok4 else 'FAIL'}")
    print()
    print("  To run against a REAL model:")
    print("    HTTP:   llama-server -m model.gguf --port 8081 --logprobs")
    print("            python demo_turboquant_ppl.py --live --base-url http://127.0.0.1:8081")
    print("    In-proc: python demo_turboquant_ppl.py --inprocess --model model.gguf --metal")


if __name__ == "__main__":
    main()
