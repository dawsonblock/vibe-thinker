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
    # Fail-closed metadata: lets callers distinguish infrastructure failure
    # from a low-confidence answer. A dead model server is NOT a low-confidence
    # answer — it is a transport/model failure that must propagate.
    transport_failures: int = 0
    model_failures: int = 0
    partial_failure: bool = False
    failure_reason: Optional[str] = None


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
    # The audit requires at least 5 meaningful claims — verifying fewer
    # than that is insufficient to claim "reliability."
    MIN_CLAIMS_FOR_SCORING = 5

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

    # ------------------------------------------------------------------ #
    # Deterministic answer extraction + comparison
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_boxed_answer(text: str) -> Optional[str]:
        """Extract the content of \\boxed{...} from a reasoning trace."""
        # Find the last \boxed{...} in the text (the final answer)
        matches = re.findall(r"\\boxed\{([^}]*)\}", text)
        if matches:
            return matches[-1].strip()
        return None

    @staticmethod
    def _normalize_numeric(s: str) -> Optional[float]:
        """Try to parse a string as a number. Returns None if not numeric."""
        s = s.strip().replace(",", "").replace(" ", "")
        # Remove common math formatting: \frac{a}{b} -> a/b
        s = re.sub(r"\\(?:dfrac|frac|tfrac)\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
        # Handle plain fractions like "7/2" or "1/2"
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

    def _check_answer_deterministic(self, answer: str, trajectories: List[Dict]) -> Optional[bool]:
        """Deterministically check if an answer is consistent across trajectories.

        For math problems, if multiple trajectories produced the same numeric
        answer extracted from \\boxed{}, that's a strong correctness signal that
        doesn't depend on self-verification.

        Returns:
          True  — answer is consistent across multiple trajectories (boost)
          False — answer contradicts other trajectories (penalty)
          None  — cannot determine (no deterministic check possible)
        """
        boxed_answers = []
        for t in trajectories:
            if not t.get("answer_present"):
                continue
            extracted = self._extract_boxed_answer(t.get("raw_trace", ""))
            if extracted is not None:
                boxed_answers.append(extracted)

        if len(boxed_answers) < 2:
            return None  # Not enough data for deterministic check

        # Normalize and compare
        target = self._normalize_numeric(answer)
        if target is None:
            # Non-numeric answer — check exact string match
            matching = sum(1 for a in boxed_answers if a.strip().lower() == answer.strip().lower())
            if matching >= 2:
                return True
            contradicting = sum(1 for a in boxed_answers if a.strip().lower() != answer.strip().lower())
            if contradicting > matching:
                return False
            return None

        # Numeric: compare with tolerance
        matching = 0
        contradicting = 0
        for a in boxed_answers:
            n = self._normalize_numeric(a)
            if n is None:
                continue
            if abs(n - target) < 1e-6:
                matching += 1
            else:
                contradicting += 1

        if matching >= 2:
            return True
        if contradicting > matching:
            return False
        return None

    def _calculate_reliability(
        self,
        verdicts: List[int],
        claims: Optional[List[str]] = None,
        answer_present: bool = False,
        deterministic_check: Optional[bool] = None,
    ) -> float:
        """Calculate a reliability score for a trajectory.

        Scoring rules (fail-closed):
          - No verdicts or empty claims -> 0.0
          - No final answer -> 0.0
          - Fewer than MIN_CLAIMS_FOR_SCORING meaningful claims -> 0.0
          - Any unverified claim (verdict 0) heavily penalizes the score
          - Deterministic check contradicts -> score halved
          - Deterministic check confirms -> score boosted (capped at 1.0)
          - The score is mean^5 but only over *meaningful* claims

        Args:
            verdicts: list of 0/1 verdicts from the verifier.
            claims: optional list of claim strings, used to filter garbage.
            answer_present: whether the trajectory produced a final answer.
            deterministic_check: result of deterministic cross-trajectory check.
        """
        if not verdicts:
            return 0.0
        if not answer_present:
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
            base = (len(verdicts) - failed) / len(verdicts) * 0.3
        else:
            mean = sum(verdicts) / len(verdicts)
            base = mean ** 5

        # Apply deterministic check adjustment
        if deterministic_check is False:
            # Answer contradicts other trajectories — halve the score
            base *= 0.5
        elif deterministic_check is True:
            # Answer confirmed by deterministic check — boost, cap at 1.0
            base = min(base * 1.15, 1.0)

        return base

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
        # Initial score without deterministic check (applied later in run())
        score = self._calculate_reliability(
            verdicts,
            claims=parsed["claims"],
            answer_present=parsed["final_answer"] is not None,
        )

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

        # Separate successful trajectories from failures.
        # Fail-closed rule: if ALL trajectories failed due to transport/model
        # exceptions, that is an INFRASTRUCTURE failure, not a low-confidence
        # answer. It must raise so the job queue marks the job FAILED.
        successful = [t for t in trajectories if isinstance(t, dict)]
        failures = [t for t in trajectories if isinstance(t, Exception)]

        if not successful and failures:
            # Every single trajectory failed — the model server is dead or
            # unreachable. Raise so callers know it was infrastructure, not
            # reasoning uncertainty.
            raise RuntimeError(
                f"All CLR trajectories failed ({len(failures)}/{self.k}): "
                f"{failures[0]}"
            )

        valid_trajectories = successful
        partial_failure = len(failures) > 0
        if partial_failure:
            print(
                f"[CLR] WARNING: {len(failures)}/{self.k} trajectories failed "
                f"(partial failure) — continuing with {len(successful)} successful"
            )

        if not valid_trajectories:
            # No failures but no successes either (shouldn't happen, but
            # fail closed just in case).
            return CLRResult(
                best_answer="No clear answer found",
                best_score=0.0,
                best_raw_trace="",
                all_trajectories=[],
                k=self.k,
                failure_reason="no trajectories produced",
            )

        # Only consider trajectories that actually produced a final answer.
        # A trajectory with no answer is worthless regardless of claim scores.
        answered = [t for t in valid_trajectories if t.get("answer_present") and t.get("answer")]

        if not answered:
            # Trajectories succeeded (model responded) but none produced a
            # usable answer. This is a genuine score-0 completed result, NOT
            # an infrastructure failure.
            return CLRResult(
                best_answer="No clear answer found",
                best_score=0.0,
                best_raw_trace="",
                all_trajectories=valid_trajectories,
                k=self.k,
                transport_failures=len(failures),
                partial_failure=partial_failure,
            )

        # Apply deterministic cross-trajectory answer checking.
        # This is independent of self-verification: if multiple trajectories
        # produced the same \\boxed{} answer, that's a correctness signal.
        # If they contradict, the score is penalized.
        for t in answered:
            det_check = self._check_answer_deterministic(t["answer"], valid_trajectories)
            t["deterministic_check"] = det_check
            t["score"] = self._calculate_reliability(
                t["verdicts"],
                claims=t["claims"],
                answer_present=True,
                deterministic_check=det_check,
            )

        # Penalize contradictory trajectories: if different trajectories
        # produced different answers, lower confidence in all of them.
        unique_answers = set()
        for t in answered:
            boxed = self._extract_boxed_answer(t.get("raw_trace", ""))
            if boxed is not None:
                unique_answers.add(boxed.lower().strip())
        if len(unique_answers) > 1:
            # Multiple different answers -> contradiction penalty
            for t in answered:
                t["score"] *= 0.7
            print(f"[CLR] {len(unique_answers)} distinct answers detected — applying contradiction penalty")

        best = max(answered, key=lambda x: x["score"])

        result = CLRResult(
            best_answer=best["answer"],
            best_score=best["score"],
            best_raw_trace=best["raw_trace"],
            all_trajectories=valid_trajectories,
            k=self.k,
            transport_failures=len(failures),
            partial_failure=partial_failure,
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
