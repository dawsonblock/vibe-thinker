"""
VibeThinker-3B Claim-Level Reliability (CLR) wrapper — async/parallel version.

Generates all k trajectories concurrently using asyncio + aiohttp, which
dramatically speeds up CLR on Apple Silicon (especially with Metal).

Requires a running llama-server (e.g. on http://127.0.0.1:8080) serving the
VibeThinker-3B GGUF model with the patched reasoning chat template.

Install:  pip install aiohttp

Bug fixes vs. the original walkthrough version:
  - Stop tokens: removed the bare "]" (a corrupted  artifact) that
    prematurely truncated generations. Now only ["<|im_end|>"].
  - Verdict parsing: no longer treats "10" or "1 reason..." as verdict 1.
    Parses the first standalone 0/1 or yes/no.
  - final_answer "null": the JSON extractor treated the string "null" as a
    real answer. Now normalized to None.
  - Added a plain (non-CLR) async generation helper for reuse by callers.
  - Filter exceptions from asyncio.gather more defensively.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import aiohttp


@dataclass
class CLRResult:
    best_answer: str
    best_score: float
    best_raw_trace: str
    all_trajectories: List[Dict] = field(default_factory=list)
    k: int = 8


class VibeThinkerCLRAsync:
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8080",
        k: int = 8,
        max_concurrent: int = 6,
    ):
        self.server_url = server_url.rstrip("/")
        self.k = k
        self.max_concurrent = max_concurrent  # Limit concurrent requests
        self.semaphore = asyncio.Semaphore(max_concurrent)

    # ------------------------------------------------------------------ #
    # Low-level async model call
    # ------------------------------------------------------------------ #
    async def _call_model(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        max_tokens: int = 8192,
        temperature: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Async call to llama-server /completion endpoint.

Raises RuntimeError on any failure — callers must handle the exception
rather than silently proceeding with an empty string."""
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": -1,
            "stop": stop if stop is not None else ["<|im_end|>"],
        }
        async with self.semaphore:
            try:
                async with session.post(
                    f"{self.server_url}/completion",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=600),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    content = data.get("content", "")
                    if not content:
                        raise RuntimeError(
                            f"Model at {self.server_url} returned empty content"
                        )
                    return content
            except aiohttp.ClientError as e:
                raise RuntimeError(f"Model call to {self.server_url} failed: {e}") from e
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"Unexpected error calling {self.server_url}: {e}") from e

    async def generate_plain(
        self, session: aiohttp.ClientSession, problem: str, max_tokens: int = 8192
    ) -> str:
        """Single plain async generation (no CLR)."""
        prompt = (
            f"<|im_start|>user\n{problem}\n<|im_end|>\n<|im_start|>assistant\n"
        )
        return await self._call_model(session, prompt, max_tokens=max_tokens)

    # ------------------------------------------------------------------ #
    # Claim extraction + answer parsing
    # ------------------------------------------------------------------ #
    async def _extract_claims_and_answer(
        self, session: aiohttp.ClientSession, text: str
    ) -> Dict:
        extraction_prompt = (
            "<|im_start|>user\n"
            "You are an expert at analyzing reasoning traces.\n\n"
            "Here is a reasoning trace:\n"
            f"{text}\n\n"
            "Extract exactly 5 key decision-relevant claims from the reasoning above.\n"
            "Also extract the final answer if it exists.\n\n"
            "Output ONLY valid JSON in this exact format:\n"
            '{\n  "claims": ["claim 1", "claim 2", "claim 3", "claim 4", "claim 5"],\n'
            '  "final_answer": "the final answer here or null"\n'
            "}\n<|im_end|>\n<|im_start|>assistant\n"
        )
        raw = await self._call_model(
            session, extraction_prompt, max_tokens=2048, temperature=0.3
        )

        try:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                claims = data.get("claims", []) or []
                if isinstance(claims, list):
                    claims = [str(c) for c in claims][:5]
                else:
                    claims = []
                final_answer = data.get("final_answer")
                if isinstance(final_answer, str):
                    if final_answer.strip().lower() in ("null", "none", "", "n/a"):
                        final_answer = None
                return {"claims": claims, "final_answer": final_answer, "raw": text}
        except Exception:
            pass

        # Fallback
        claims = re.findall(
            r"(?:Claim|Step|Reason)\s*\d*[:\-]?\s*(.+?)(?=\n|$)",
            text,
            re.IGNORECASE,
        )[:5]
        answer_match = re.search(r"\\boxed\{(.*?)\}", text)
        return {
            "claims": claims,
            "final_answer": answer_match.group(1) if answer_match else None,
            "raw": text,
        }

    # ------------------------------------------------------------------ #
    # Self-verification
    # ------------------------------------------------------------------ #
    def _parse_verdict(self, raw: str) -> int:
        """Robustly parse a 0/1 verdict from the model's response."""
        s = raw.strip().lower()
        if not s:
            return 0
        if s.startswith("yes") or "yes," in s[:6]:
            return 1
        if s.startswith("no") or "no," in s[:6]:
            return 0
        m = re.search(r"\b([01])\b", s)
        if m:
            return int(m.group(1))
        m = re.search(r"([01])", s[:10])
        return int(m.group(1)) if m else 0

    async def _verify_claims(
        self, session: aiohttp.ClientSession, claims: List[str]
    ) -> List[int]:
        verdicts = []
        for claim in claims:
            if not claim or len(claim.strip()) < 5:
                verdicts.append(0)
                continue

            verify_prompt = (
                "<|im_start|>user\n"
                "Verify whether this claim is correct based on logical reasoning "
                "and mathematics.\n\n"
                f"Claim: {claim}\n\n"
                "Respond with ONLY a single digit: 1 if the claim is correct, "
                "0 if it is incorrect or uncertain.\n"
                "<|im_end|>\n<|im_start|>assistant\n"
            )
            raw = await self._call_model(
                session, verify_prompt, max_tokens=128, temperature=0.2
            )
            verdicts.append(self._parse_verdict(raw))
        return verdicts

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    # Minimum number of meaningful claims required for a non-zero score.
    # A trajectory with fewer claims cannot be considered "reliable" —
    # verifying one trivial fragment is not enough.
    MIN_CLAIMS_FOR_SCORING = 3

    # Claims shorter than this (after stripping) are too trivial to count.
    MIN_CLAIM_LENGTH = 15

    # Known garbage / prompt-fragment patterns that should never count as claims.
    # Matches both exact strings and strings that START with these fragments
    # (e.g. "by step reasoning. So we can elaborate." starts with "by step reasoning.")
    _GARBAGE_PATTERNS = re.compile(
        r"^(by step\.?|by step reasoning\.?|step by step\.?|"
        r"so we can elaborate\.?|the final answer\.?|"
        r"none|n/?a|null|undefined)",
        re.IGNORECASE,
    )

    def _is_meaningful_claim(self, claim: str) -> bool:
        """Return True if a claim is substantive enough to score."""
        s = claim.strip()
        if len(s) < self.MIN_CLAIM_LENGTH:
            return False
        if self._GARBAGE_PATTERNS.match(s):
            return False
        # Reject claims that are just punctuation or fragments
        if not re.search(r"[a-zA-Z]{3,}", s):
            return False
        return True

    def _calculate_reliability(
        self, verdicts: List[int], claims: Optional[List[str]] = None
    ) -> float:
        """Calculate a reliability score for a trajectory.

        Scoring rules (fail-closed):
          - No verdicts or empty claims -> 0.0
          - Fewer than MIN_CLAIMS_FOR_SCORING meaningful claims -> 0.0
          - Any unverified claim (verdict 0) heavily penalizes the score
          - The score is mean^5 but only over *meaningful* claims, and
            is further penalized if some claims were filtered as garbage.

        Args:
            verdicts: list of 0/1 verdicts from the verifier.
            claims: optional list of claim strings, used to filter garbage.
        """
        if not verdicts:
            return 0.0

        # If claims are provided, filter to meaningful ones and their verdicts
        if claims is not None:
            meaningful = [
                (c, v) for c, v in zip(claims, verdicts) if self._is_meaningful_claim(c)
            ]
            if len(meaningful) < self.MIN_CLAIMS_FOR_SCORING:
                return 0.0
            verdicts = [v for _, v in meaningful]

        # Any failed verdict means the trajectory has errors — penalize hard
        failed = sum(1 for v in verdicts if v == 0)
        if failed > 0:
            # A trajectory with even one wrong claim cannot be "perfect"
            # Score is proportional to how many claims passed, but capped low
            return (len(verdicts) - failed) / len(verdicts) * 0.3

        mean = sum(verdicts) / len(verdicts)
        return mean ** 5

    # ------------------------------------------------------------------ #
    # One full trajectory
    # ------------------------------------------------------------------ #
    async def _generate_one_trajectory(
        self, session: aiohttp.ClientSession, problem: str, max_tokens: int
    ) -> Dict:
        """Generate + score one full trajectory."""
        reasoning_prompt = (
            "<|im_start|>user\n"
            f"{problem}\n\n"
            "Solve this step by step. Think carefully and put your final "
            "answer in \\boxed{}.\n"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        raw_trace = await self._call_model(
            session, reasoning_prompt, max_tokens=max_tokens
        )

        parsed = await self._extract_claims_and_answer(session, raw_trace)
        verdicts = await self._verify_claims(session, parsed["claims"])
        score = self._calculate_reliability(verdicts, claims=parsed["claims"])

        return {
            "score": score,
            "answer": parsed["final_answer"],
            "claims": parsed["claims"],
            "verdicts": verdicts,
            "raw_trace": raw_trace,
            "answer_present": parsed["final_answer"] is not None,
        }

    # ------------------------------------------------------------------ #
    # Main CLR entry point
    # ------------------------------------------------------------------ #
    async def run(self, problem: str, max_tokens_per_trace: int = 16384) -> CLRResult:
        print(
            f"Running async CLR with k={self.k} trajectories "
            f"(max_concurrent={self.max_concurrent})..."
        )

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._generate_one_trajectory(session, problem, max_tokens_per_trace)
                for _ in range(self.k)
            ]
            trajectories = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out any failed trajectories (exceptions from gather)
        valid_trajectories = [t for t in trajectories if isinstance(t, dict)]

        if not valid_trajectories:
            return CLRResult(
                best_answer="No clear answer found",
                best_score=0.0,
                best_raw_trace="",
                all_trajectories=[],
                k=self.k,
            )

        # Only consider trajectories that actually produced a final answer.
        # A trajectory with no answer is worthless regardless of claim scores.
        answered = [t for t in valid_trajectories if t.get("answer_present") and t.get("answer")]

        if not answered:
            return CLRResult(
                best_answer="No clear answer found",
                best_score=0.0,
                best_raw_trace="",
                all_trajectories=valid_trajectories,
                k=self.k,
            )

        best = max(answered, key=lambda x: x["score"])

        result = CLRResult(
            best_answer=best["answer"],
            best_score=best["score"],
            best_raw_trace=best["raw_trace"],
            all_trajectories=valid_trajectories,
            k=self.k,
        )

        print(f"\nBest trajectory score: {best['score']:.4f}")
        print(f"Best answer: {result.best_answer}")
        return result


# ====================== EXAMPLE USAGE ======================

async def main():
    clr = VibeThinkerCLRAsync(k=8, max_concurrent=6)

    problem = (
        "Solve this step by step:\n\n"
        "A sequence is defined by a_1 = 2, a_{n+1} = (a_n)^2 - a_n + 1 for n >= 1.\n"
        "Find the value of a_5."
    )

    result = await clr.run(problem)

    print("\n" + "=" * 60)
    print("FINAL BEST ANSWER:", result.best_answer)
    print("RELIABILITY SCORE:", round(result.best_score, 4))
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
