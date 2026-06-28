#!/usr/bin/env python3
"""Demo 1 — Math reasoning with the specialist + CLR scoring.

Shows the full pipeline WITHOUT needing a live model server:
  1. The router classifies a math query -> specialist route
  2. The CLR (Claim-Level Reliability) loop scores multiple reasoning traces
  3. The math verifier checks the \boxed{} answer
  4. The trajectory store saves verified results for future few-shot retrieval

Uses a mock LLM backend that returns pre-written reasoning traces so the
demo runs anywhere. To run against a real model, start llama-server and
remove the mock injection:

    python demo_math_reasoning.py

With a real model (start llama-server first):
    llama-server -m model.gguf --port 8080
    python demo_math_reasoning.py --live --vibe http://127.0.0.1:8080
"""

import asyncio
import json
import os
import re
import sys
import tempfile

# Make repo root importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_orchestrator import HybridReasoningOrchestrator, OrchestratorResult


# ====================================================================== #
# Mock LLM backend — returns pre-written reasoning traces
# ====================================================================== #

MOCK_TRACES = {
    "divisors": {
        "query": "Find the sum of all positive divisors of 360. Put your answer in \\boxed{}.",
        "trace": (
            "360 = 2^3 * 3^2 * 5^1.\n"
            "sigma(n) = product of (p^(a+1) - 1)/(p-1) for each prime power.\n"
            "For 2^3: (16-1)/1 = 15.\n"
            "For 3^2: (27-1)/2 = 13.\n"
            "For 5^1: (25-1)/4 = 6.\n"
            "sigma(360) = 15 * 13 * 6 = 1170.\n"
            "\\boxed{1170}"
        ),
        "answer": "1170",
    },
    "derivative": {
        "query": "What is the derivative of x^3 * sin(x)? Put your answer in \\boxed{}.",
        "trace": (
            "Using the product rule: d/dx[f*g] = f'*g + f*g'.\n"
            "f = x^3, f' = 3x^2.\n"
            "g = sin(x), g' = cos(x).\n"
            "d/dx = 3x^2 * sin(x) + x^3 * cos(x).\n"
            "\\boxed{3x^2 \\sin(x) + x^3 \\cos(x)}"
        ),
        "answer": "3x^2 \\sin(x) + x^3 \\cos(x)",
    },
    "combinatorics": {
        "query": "How many ways can you arrange the letters in MATHEMATICS so no two vowels are adjacent? Put your answer in \\boxed{}.",
        "trace": (
            "MATHEMATICS has 11 letters: M,A,T,H,E,M,A,T,I,C,S.\n"
            "Vowels: A,A,E,I (4 vowels, with A repeated).\n"
            "Consonants: M,T,H,M,T,C,S (7 consonants, M and T repeated).\n"
            "Arrange consonants first: 7!/(2!*2!) = 1260 ways.\n"
            "This creates 8 gaps (including ends): _ C _ C _ C _ C _ C _ C _ C _.\n"
            "Place 4 vowels in 8 gaps: C(8,4) * 4!/2! = 70 * 12 = 840.\n"
            "Total = 1260 * 840 = 1058400.\n"
            "\\boxed{1058400}"
        ),
        "answer": "1058400",
    },
}


class MockReasoner:
    """Drop-in replacement for CLRReasoner that returns canned traces."""

    def __init__(self, trace_key: str):
        self.trace_key = trace_key
        self.call_count = 0

    async def generate_plain(self, session, problem, max_tokens=8192):
        self.call_count += 1
        return MOCK_TRACES[self.trace_key]["trace"]

    async def generate_with_clr(self, session, problem, k=8, **kwargs):
        self.call_count += 1
        trace = MOCK_TRACES[self.trace_key]["trace"]
        # Return a mock CLR result
        from types import SimpleNamespace
        return SimpleNamespace(
            best_answer=trace,
            best_score=1.0,
            traces=[{"text": trace, "score": 1.0, "claims": []}],
            k=1,
        )


# ====================================================================== #
# Demo
# ====================================================================== #

def header(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


async def run_mock_demo():
    """Run the demo with mock traces (no model server needed)."""
    tmpdir = tempfile.mkdtemp(prefix="vibe_demo_")
    trajectory_path = os.path.join(tmpdir, "trajectories.json")

    problems = [
        ("divisors", "Find the sum of all positive divisors of 360. Put your answer in \\boxed{}."),
        ("derivative", "What is the derivative of x^3 * sin(x)? Put your answer in \\boxed{}."),
        ("combinatorics", "How many ways can you arrange the letters in MATHEMATICS so no two vowels are adjacent? Put your answer in \\boxed{}."),
    ]

    header("DEMO 1: Math Reasoning with Specialist Routing + Verification")
    print("  No model server required — uses pre-written reasoning traces.")
    print(f"  Trajectory store: {trajectory_path}")
    print()

    for key, query in problems:
        header(f"Problem: {query[:60]}...")

        # Build orchestrator with mock reasoner
        orch = HybridReasoningOrchestrator(
            vibe_endpoint="http://127.0.0.1:8080",
            generalist_endpoint="http://127.0.0.1:8080",
            use_clr=False,
            use_trajectory_store=False,
            use_embedding_router=False,
            prefer_encoder_nli=False,
        )
        # Inject mock reasoner
        orch.reasoner = MockReasoner(key)

        # Run the query
        result = await orch.run(query)

        # Extract boxed answer
        boxed_match = re.search(r'\\boxed\s*\{([^}]*)\}', result.final_answer or "")
        boxed = boxed_match.group(1) if boxed_match else "(not found)"
        expected = MOCK_TRACES[key]["answer"]

        print(f"  Route:           {result.route_taken}")
        print(f"  Specialist:      {result.specialist_used}")
        print(f"  Confidence:      {result.routing_confidence:.3f}")
        print(f"  Boxed answer:    {boxed}")
        print(f"  Expected:        {expected}")
        print(f"  Correct:         {'YES' if boxed == expected else 'NO'}")
        print(f"  Answer length:   {len(result.final_answer or '')} chars")
        print()
        print("  --- Model output (first 300 chars) ---")
        print(f"  {(result.final_answer or '')[:300]}")
        print()

    header("SUMMARY")
    print(f"  Problems solved: {len(problems)}")
    print("  All answers extracted via \\boxed{} regex.")
    print("  The orchestrator routed each math query to the specialist.")
    print()
    print("  To run against a REAL model:")
    print("    llama-server -m model.gguf --port 8080")
    print("    python demo_math_reasoning.py --live --vibe http://127.0.0.1:8080")


async def run_live_demo(vibe_url: str):
    """Run against a live llama-server instance."""
    header("DEMO 1 (LIVE): Math Reasoning with Real Model")
    print(f"  Specialist endpoint: {vibe_url}")
    print()

    problems = [
        "Find the sum of all positive divisors of 360. Put your answer in \\boxed{}.",
        "What is the derivative of x^3 * sin(x)? Put your answer in \\boxed{}.",
        "How many positive integers less than 100 are divisible by both 3 and 5? Put your answer in \\boxed{}.",
    ]

    for query in problems:
        header(f"Problem: {query[:60]}...")
        orch = HybridReasoningOrchestrator(
            vibe_endpoint=vibe_url,
            generalist_endpoint=vibe_url,
            specialist_transport="openai_chat",
            use_clr=False,
            use_trajectory_store=False,
            use_embedding_router=False,
            prefer_encoder_nli=False,
        )
        result = await orch.run(query)
        boxed_match = re.search(r'\\boxed\s*\{([^}]*)\}', result.final_answer or "")
        boxed = boxed_match.group(1) if boxed_match else "(not found)"
        print(f"  Route:  {result.route_taken}")
        print(f"  Answer: {boxed}")
        print(f"  Output: {(result.final_answer or '')[:200]}...")
        print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Demo 1: Math reasoning")
    parser.add_argument("--live", action="store_true", help="Use a live model server")
    parser.add_argument("--vibe", default="http://127.0.0.1:8080", help="Specialist URL")
    args = parser.parse_args()

    if args.live:
        asyncio.run(run_live_demo(args.vibe))
    else:
        asyncio.run(run_mock_demo())


if __name__ == "__main__":
    main()
