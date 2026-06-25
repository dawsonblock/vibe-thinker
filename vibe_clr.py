"""
VibeThinker-3B Claim-Level Reliability (CLR) wrapper — synchronous facade.

This is a thin synchronous wrapper around ``VibeThinkerCLRAsync``. The old
synchronous implementation used a broken scoring model (``mean ** 5`` over
raw verdicts with no claim filtering, no answer-present check, and no
deterministic verification) and silently swallowed model-call failures by
returning an empty string. Both defects are gone: the async engine is the
single source of truth for scoring and fail-closed behavior.

Requires a running llama-server (e.g. on http://127.0.0.1:8080) serving the
VibeThinker-3B GGUF model with the patched reasoning chat template.

Install:  pip install aiohttp
"""

import asyncio
from typing import Optional

from vibe_clr_async import CLRResult, VibeThinkerCLRAsync


class VibeThinkerCLR:
    """Synchronous facade over :class:`VibeThinkerCLRAsync`.

    Delegates all scoring, claim filtering, deterministic checking, and
    fail-closed behavior to the async implementation so there is exactly
    one reliability engine in the repo.
    """

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8080",
        k: int = 8,
        max_concurrent: int = 6,
    ):
        self._async = VibeThinkerCLRAsync(
            server_url=server_url, k=k, max_concurrent=max_concurrent
        )
        # Expose commonly accessed attributes for backward compatibility.
        self.server_url = self._async.server_url
        self.k = self._async.k

    def run(self, problem: str, max_tokens_per_trace: int = 16384) -> CLRResult:
        """Generate k trajectories, score them, and return the best one.

        Raises ``RuntimeError`` if all trajectories fail (dead endpoint),
        matching the async implementation's fail-closed contract.
        """
        return asyncio.run(self._async.run(problem, max_tokens_per_trace))

    # Convenience pass-throughs for callers that used the old sync API.
    def generate_plain(self, problem: str, max_tokens: int = 8192) -> str:
        async def _go():
            import aiohttp
            async with aiohttp.ClientSession() as session:
                return await self._async.generate_plain(session, problem, max_tokens)
        return asyncio.run(_go())

    # Expose scoring helpers so external code/tests can use the same rules.
    def _calculate_reliability(self, *args, **kwargs) -> float:
        return self._async._calculate_reliability(*args, **kwargs)

    def _is_meaningful_claim(self, claim: str) -> bool:
        return self._async._is_meaningful_claim(claim)

    def _parse_verdict(self, raw: str) -> int:
        return self._async._parse_verdict(raw)


# ====================== EXAMPLE USAGE ======================

if __name__ == "__main__":
    clr = VibeThinkerCLR(k=8)

    problem = (
        "Solve this step by step:\n\n"
        "A sequence is defined by a_1 = 2, a_{n+1} = (a_n)^2 - a_n + 1 for n >= 1.\n"
        "Find the value of a_5."
    )

    result = clr.run(problem)

    print("\n" + "=" * 60)
    print("FINAL BEST ANSWER:", result.best_answer)
    print("RELIABILITY SCORE:", round(result.best_score, 4))
    print("=" * 60)
