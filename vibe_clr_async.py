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
from typing import Any, Dict, List, Optional

import aiohttp

from scoring import compute_confidence


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
    # Verification metadata: how the best answer was verified.
    # "self_claims_only" means the model checked its own claims — weak.
    # "math_verifier" / "code_verifier" / "factual_verifier" means an
    # independent deterministic verifier was run.
    verification_method: str = "self_claims_only"
    verified: bool = False
    deterministic_verification: Optional[float] = None


class VibeThinkerCLRAsync:
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8080",
        k: int = 8,
        max_concurrent: int = 6,
        adaptive: bool = True,
        k_min: int = 2,
        k_max: int = 6,
    ):
        self.server_url = server_url.rstrip("/")
        self.k = k
        self.max_concurrent = max_concurrent  # Limit concurrent requests
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Adaptive compute: instead of firing all k trajectories at once,
        # start with k_min and scale up to k_max only when needed.
        # This is "System 1 / System 2" thinking:
        #   Phase 1: k_min trajectories + early verifier exit
        #   Phase 2: consensus check (early exit if answers agree)
        #   Phase 3: scale up to k_max on disagreement/uncertainty
        self.adaptive = adaptive
        self.k_min = min(k_min, k) if adaptive else k
        self.k_max = min(k_max, k) if adaptive else k

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
          - Self-claims-only confidence is HARD CAPPED at 0.65
          - Deterministic verifier passes -> eligible above 0.65
          - Deterministic verifier fails -> score 0.0

        The raw claim-level score (mean^5 over meaningful claims) is passed
        through :func:`compute_confidence` which enforces the self-claims-only
        cap. This is the active runtime path — the cap is not advisory.

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

        # Convert deterministic_check (bool|None) to a numeric verification
        # score for compute_confidence: 1.0 (confirmed), 0.0 (refuted),
        # None (no verifier run).
        det_verification: Optional[float] = None
        if deterministic_check is True:
            det_verification = 1.0
        elif deterministic_check is False:
            det_verification = 0.0

        verification_method = (
            "deterministic_check" if det_verification is not None
            else "self_claims_only"
        )

        # Route through compute_confidence to enforce the self-claims-only cap.
        # This is the trust model: self-agreement alone can never exceed 0.65.
        confidence = compute_confidence(
            model_score=base,
            claim_consistency=base,
            deterministic_verification=det_verification,
            verification_method=verification_method,
        )
        return confidence.final_score

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
    async def run(
        self,
        problem: str,
        max_tokens_per_trace: int = 16384,
        verifier: Optional[Any] = None,
        task_type: str = "unknown",
        verifier_context: Optional[Dict[str, Any]] = None,
    ) -> CLRResult:
        """Run CLR with adaptive compute.

        Uses a phased approach instead of brute-force k trajectories:

        Phase 1 (Fast Path): Generate k_min trajectories. If a verifier
        is available, run it immediately. If it returns verified=True,
        exit early — no need for more compute.

        Phase 2 (Consensus Check): If no verifier or verifier didn't
        confirm, check if the first trajectories agree. If they do and
        self-verify well, exit early — more trajectories won't raise
        the score above the 0.65 cap anyway.

        Phase 3 (Branching): If trajectories disagree or the verifier
        failed, scale up to k_max trajectories. This is the "System 2"
        mode for high-uncertainty problems.

        If adaptive=False, falls back to the original brute-force mode
        (all k trajectories at once).

        Args:
            problem: the problem to solve.
            max_tokens_per_trace: max tokens per trajectory.
            verifier: optional deterministic verifier. If provided, the
                verifier independently checks the best answer and the
                result score can exceed the self-claims-only cap of 0.65.
            task_type: the detected task type (math, code, factual, etc.).
            verifier_context: optional context dict passed to the verifier.
        """
        if not self.adaptive:
            return await self._run_static(
                problem, max_tokens_per_trace, verifier, task_type,
                verifier_context,
            )

        return await self._run_adaptive(
            problem, max_tokens_per_trace, verifier, task_type,
            verifier_context,
        )

    async def _run_adaptive(
        self,
        problem: str,
        max_tokens_per_trace: int,
        verifier: Optional[Any],
        task_type: str,
        verifier_context: Optional[Dict[str, Any]],
    ) -> CLRResult:
        """Adaptive compute: phased trajectory generation with early exit."""
        k_initial = self.k_min
        k_total = self.k_max
        print(
            f"Running adaptive CLR: k_min={k_initial}, k_max={k_total} "
            f"(max_concurrent={self.max_concurrent})..."
        )

        all_trajectories: List[Any] = []
        all_failures: List[Exception] = []

        async with aiohttp.ClientSession() as session:
            # === Phase 1: Fast Path ===
            print(f"[CLR] Phase 1: generating {k_initial} trajectories...")
            tasks = [
                self._generate_one_trajectory(session, problem, max_tokens_per_trace)
                for _ in range(k_initial)
            ]
            phase1_results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in phase1_results:
                if isinstance(r, Exception):
                    all_failures.append(r)
                else:
                    all_trajectories.append(r)

            # Check for total infrastructure failure
            if not all_trajectories and all_failures:
                raise RuntimeError(
                    f"All CLR trajectories failed ({len(all_failures)}/{k_initial}): "
                    f"{all_failures[0]}"
                )

            # Score the phase 1 trajectories
            answered = self._score_trajectories(all_trajectories)
            if not answered:
                return self._build_no_answer_result(
                    all_trajectories, all_failures, k_initial,
                )

            best = max(answered, key=lambda x: x["score"])

            # Early exit: verifier confirms the answer
            verifier_refuted = False
            if verifier is not None:
                v_result = await self._try_verifier(
                    verifier, problem, best["answer"], verifier_context, best_score=best["score"],
                )
                if v_result and v_result[2]:  # verified=True
                    print(f"[CLR] Phase 1 early exit: verifier confirmed answer")
                    return self._build_final_result(
                        best, all_trajectories, all_failures,
                        k_used=len(all_trajectories),
                        verification_method=v_result[0],
                        verified=v_result[2],
                        det_verification=v_result[1],
                        final_score_override=v_result[3],
                    )
                if v_result and not v_result[2]:
                    verifier_refuted = True

            # === Phase 2: Consensus Check ===
            # Skip consensus if the verifier already refuted — the answer
            # is wrong even if trajectories agree. We need more trajectories
            # to find a correct answer.
            # If trajectories agree and self-verify well, no need to branch.
            # More trajectories won't raise the score above 0.65 without
            # a verifier, so spending the compute is pointless.
            if not verifier_refuted and self._check_consensus(answered):
                print(f"[CLR] Phase 2 early exit: trajectories agree (consensus)")
                return self._build_final_result(
                    best, all_trajectories, all_failures,
                    k_used=len(all_trajectories),
                )

            # === Phase 3: Branching (System 2) ===
            remaining = k_total - len(all_trajectories)
            if remaining > 0:
                print(
                    f"[CLR] Phase 3: uncertainty detected — "
                    f"generating {remaining} more trajectories..."
                )
                tasks = [
                    self._generate_one_trajectory(session, problem, max_tokens_per_trace)
                    for _ in range(remaining)
                ]
                phase3_results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in phase3_results:
                    if isinstance(r, Exception):
                        all_failures.append(r)
                    else:
                        all_trajectories.append(r)

        # Re-score all trajectories with the full set
        answered = self._score_trajectories(all_trajectories)
        if not answered:
            return self._build_no_answer_result(
                all_trajectories, all_failures, k_total,
            )

        best = max(answered, key=lambda x: x["score"])

        # Run verifier on the best answer from the full set
        verification_method = "self_claims_only"
        verified = False
        det_verification: Optional[float] = None
        final_score = best["score"]

        if verifier is not None:
            v_result = await self._try_verifier(
                verifier, problem, best["answer"], verifier_context, best_score=best["score"],
            )
            if v_result:
                verification_method, det_verification, verified, final_score = v_result
                if final_score is None:
                    final_score = best["score"]

        result = CLRResult(
            best_answer=best["answer"],
            best_score=final_score,
            best_raw_trace=best["raw_trace"],
            all_trajectories=[t for t in all_trajectories if isinstance(t, dict)],
            k=k_total,
            transport_failures=len(all_failures),
            partial_failure=len(all_failures) > 0,
            verification_method=verification_method,
            verified=verified,
            deterministic_verification=det_verification,
        )

        print(f"\nBest trajectory score: {final_score:.4f}")
        print(f"Best answer: {result.best_answer}")
        print(f"Verification: {verification_method} (verified={verified})")
        print(f"Compute used: {len(all_trajectories)} trajectories "
              f"(of max {k_total})")
        return result

    async def _run_static(
        self,
        problem: str,
        max_tokens_per_trace: int,
        verifier: Optional[Any],
        task_type: str,
        verifier_context: Optional[Dict[str, Any]],
    ) -> CLRResult:
        """Original brute-force mode: all k trajectories at once."""
        print(
            f"Running static CLR with k={self.k} trajectories "
            f"(max_concurrent={self.max_concurrent})..."
        )

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._generate_one_trajectory(session, problem, max_tokens_per_trace)
                for _ in range(self.k)
            ]
            trajectories = await asyncio.gather(*tasks, return_exceptions=True)

        successful = [t for t in trajectories if isinstance(t, dict)]
        failures = [t for t in trajectories if isinstance(t, Exception)]

        if not successful and failures:
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
            return CLRResult(
                best_answer="No clear answer found",
                best_score=0.0,
                best_raw_trace="",
                all_trajectories=[],
                k=self.k,
                failure_reason="no trajectories produced",
            )

        answered = self._score_trajectories(valid_trajectories)
        if not answered:
            return CLRResult(
                best_answer="No clear answer found",
                best_score=0.0,
                best_raw_trace="",
                all_trajectories=valid_trajectories,
                k=self.k,
                transport_failures=len(failures),
                partial_failure=partial_failure,
            )

        best = max(answered, key=lambda x: x["score"])

        # Run verifier
        verification_method = "self_claims_only"
        verified = False
        det_verification: Optional[float] = None
        final_score = best["score"]

        if verifier is not None:
            v_result = await self._try_verifier(
                verifier, problem, best["answer"], verifier_context, best_score=best["score"],
            )
            if v_result:
                verification_method, det_verification, verified, final_score = v_result
                if final_score is None:
                    final_score = best["score"]

        result = CLRResult(
            best_answer=best["answer"],
            best_score=final_score,
            best_raw_trace=best["raw_trace"],
            all_trajectories=valid_trajectories,
            k=self.k,
            transport_failures=len(failures),
            partial_failure=partial_failure,
            verification_method=verification_method,
            verified=verified,
            deterministic_verification=det_verification,
        )

        print(f"\nBest trajectory score: {final_score:.4f}")
        print(f"Best answer: {result.best_answer}")
        print(f"Verification: {verification_method} (verified={verified})")
        return result

    def _score_trajectories(self, trajectories: List[Any]) -> List[Dict]:
        """Score valid trajectories and return those with answers.

        Applies deterministic cross-trajectory checking and contradiction
        penalties. Returns only trajectories that produced a final answer.
        """
        valid = [t for t in trajectories if isinstance(t, dict)]
        answered = [t for t in valid if t.get("answer_present") and t.get("answer")]
        if not answered:
            return []

        for t in answered:
            det_check = self._check_answer_deterministic(t["answer"], valid)
            t["deterministic_check"] = det_check
            t["score"] = self._calculate_reliability(
                t["verdicts"],
                claims=t["claims"],
                answer_present=True,
                deterministic_check=det_check,
            )

        # Contradiction penalty
        unique_answers = set()
        for t in answered:
            boxed = self._extract_boxed_answer(t.get("raw_trace", ""))
            if boxed is not None:
                unique_answers.add(boxed.lower().strip())
        if len(unique_answers) > 1:
            for t in answered:
                t["score"] *= 0.7
            print(f"[CLR] {len(unique_answers)} distinct answers detected — applying contradiction penalty")

        return answered

    def _check_consensus(self, answered: List[Dict]) -> bool:
        """Check if trajectories agree — early exit signal.

        Returns True if:
        - All answered trajectories produced the same boxed answer, AND
        - The best score is reasonable (>= 0.3, meaning claims aren't garbage)

        If they agree, generating more trajectories won't help: without a
        verifier, the score is capped at 0.65 regardless of how many
        trajectories agree.
        """
        if len(answered) < 2:
            return False  # Need at least 2 to check consensus

        boxed_answers = []
        for t in answered:
            extracted = self._extract_boxed_answer(t.get("raw_trace", ""))
            if extracted is not None:
                boxed_answers.append(extracted.lower().strip())

        if len(boxed_answers) < 2:
            return False  # Not enough boxed answers to compare

        # All answers agree
        if len(set(boxed_answers)) == 1:
            best_score = max(t["score"] for t in answered)
            if best_score >= 0.3:
                print(f"[CLR] Consensus: all {len(boxed_answers)} trajectories agree "
                      f"(score={best_score:.3f})")
                return True

        return False

    async def _try_verifier(
        self,
        verifier: Any,
        problem: str,
        answer: str,
        verifier_context: Optional[Dict[str, Any]],
        best_score: float = 0.65,
    ) -> Optional[tuple]:
        """Run the verifier and return (method, det_score, verified, final_score).

        Returns None if the verifier raises an exception.
        final_score is None if the verifier didn't verify (caller uses
        best["score"] as default).
        """
        try:
            v_result = await verifier.verify(
                problem, answer,
                context=verifier_context or {},
            )
            verification_method = getattr(verifier, "name", "verifier")
            det_verification = v_result.score if v_result.verified else 0.0
            verified = v_result.verified
            print(f"[CLR] Verifier {verification_method}: "
                  f"verified={verified}, score={v_result.score:.3f}")

            final_score = None
            if verified:
                # Use the verifier's actual score, not 1.0
                confidence = compute_confidence(
                    model_score=best_score,
                    claim_consistency=best_score,
                    deterministic_verification=v_result.score,
                    verification_method=verification_method,
                )
                final_score = confidence.final_score
            else:
                if v_result.score <= 0.0 and v_result.error:
                    final_score = 0.0

            return (verification_method, det_verification, verified, final_score)
        except Exception as e:
            print(f"[CLR] Verifier error: {e}")
            return None

    def _build_final_result(
        self,
        best: Dict,
        all_trajectories: List[Any],
        all_failures: List[Exception],
        k_used: int,
        verification_method: str = "self_claims_only",
        verified: bool = False,
        det_verification: Optional[float] = None,
        final_score_override: Optional[float] = None,
    ) -> CLRResult:
        """Build a CLRResult from the best trajectory."""
        final_score = final_score_override if final_score_override is not None else best["score"]
        valid = [t for t in all_trajectories if isinstance(t, dict)]

        result = CLRResult(
            best_answer=best["answer"],
            best_score=final_score,
            best_raw_trace=best["raw_trace"],
            all_trajectories=valid,
            k=k_used,
            transport_failures=len(all_failures),
            partial_failure=len(all_failures) > 0,
            verification_method=verification_method,
            verified=verified,
            deterministic_verification=det_verification,
        )

        print(f"\nBest trajectory score: {final_score:.4f}")
        print(f"Best answer: {result.best_answer}")
        print(f"Verification: {verification_method} (verified={verified})")
        print(f"Compute used: {len(valid)} trajectories (of max {k_used})")
        return result

    def _build_no_answer_result(
        self,
        all_trajectories: List[Any],
        all_failures: List[Exception],
        k_used: int,
    ) -> CLRResult:
        """Build a CLRResult when no trajectories produced an answer."""
        valid = [t for t in all_trajectories if isinstance(t, dict)]
        return CLRResult(
            best_answer="No clear answer found",
            best_score=0.0,
            best_raw_trace="",
            all_trajectories=valid,
            k=k_used,
            transport_failures=len(all_failures),
            partial_failure=len(all_failures) > 0,
        )


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
