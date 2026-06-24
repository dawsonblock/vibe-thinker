"""
End-to-end CLR test against the running local llama-server.

Runs the async CLR wrapper with a small k (default 4) on a verifiable math
problem, prints the best answer + per-trajectory scores, and saves the full
trace to clr_trace.json.

Usage:
    python test_clr.py            # k=4, default problem
    python test_clr.py 8          # k=8
"""

import asyncio
import json
import sys

from vibe_clr_async import VibeThinkerCLRAsync


DEFAULT_PROBLEM = (
    "Solve this step by step:\n\n"
    "A sequence is defined by a_1 = 2, a_{n+1} = (a_n)^2 - a_n + 1 for n >= 1.\n"
    "Find the value of a_5."
)

# Ground truth for verification (a_1=2, a_2=3, a_3=7, a_4=43, a_5=1807)
EXPECTED_ANSWER = "1807"


async def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    problem = DEFAULT_PROBLEM

    print(f"Server: http://127.0.0.1:8080")
    print(f"k = {k}")
    print(f"Problem: {problem}\n")

    clr = VibeThinkerCLRAsync(
        server_url="http://127.0.0.1:8080",
        k=k,
        max_concurrent=4,  # keep modest for a single M2 Pro
    )

    result = await clr.run(problem, max_tokens_per_trace=4096)

    print("\n" + "=" * 60)
    print("FINAL BEST ANSWER:", result.best_answer)
    print("RELIABILITY SCORE:", round(result.best_score, 4))
    print("EXPECTED ANSWER:", EXPECTED_ANSWER)
    print(
        "CORRECT:",
        EXPECTED_ANSWER in (result.best_answer or ""),
    )
    print("=" * 60)

    print("\nPer-trajectory summary:")
    for i, t in enumerate(result.all_trajectories):
        print(
            f"  [{i}] score={t['score']:.4f} "
            f"verdicts={t['verdicts']} "
            f"answer={t['answer']!r}"
        )

    # Save full trace
    out = {
        "problem": problem,
        "expected": EXPECTED_ANSWER,
        "best_answer": result.best_answer,
        "best_score": result.best_score,
        "k": result.k,
        "trajectories": result.all_trajectories,
    }
    with open("clr_trace.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nFull trace saved to clr_trace.json")


if __name__ == "__main__":
    asyncio.run(main())
