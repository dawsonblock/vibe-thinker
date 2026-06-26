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


# GBNF grammar for the JSON claim extraction format.
# Forces the model to output valid JSON with "claims" (array of strings)
# and "final_answer" (string or null). This prevents small models from
# producing malformed JSON that causes trajectory scoring to fail.
_CLAIMS_JSON_GRAMMAR = r"""root ::= "{" ws "\"claims\"" ws ":" ws "[" ws string ("," ws string)* ws "]" ws "," ws "\"final_answer\"" ws ":" ws (string | "null") ws "}"
string ::= "\"" ([^"\\] | "\\" .)* "\""
ws ::= [ \t\n]*
"""


@dataclass
class AdaptivePolicy:
    """Policy for adaptive compute (dynamic sampling / early exiting).

    Controls how the CLR runtime scales trajectory generation:
      - Start with initial_k trajectories
      - Verify early if a verifier is available
      - Branch up to max_k on disagreement/uncertainty
      - Self-consensus never exceeds self_claim_cap (0.65)
      - Only external verifiers can exceed the cap
      - High-risk tasks disable self-consensus early exit
    """
    initial_k_with_verifier: int = 1
    initial_k_without_verifier: int = 2
    max_k: int = 6
    self_claim_cap: float = 0.65
    min_claims: int = 5
    early_exit_on_self_consensus: bool = True
    disable_self_consensus_for_high_risk: bool = True
    # Consistency boost within the cap (agreement adds this much, capped)
    consistency_boost: float = 0.05
    # Contradiction penalty (disagreement multiplies by this)
    contradiction_penalty: float = 0.75


# High-risk task types that should NOT early-exit from self-consensus alone.
# These require external verification — the model agreeing with itself
# is not sufficient for code execution, file modification, etc.
HIGH_RISK_TASK_TYPES = {"code", "file_modify", "security", "medical", "legal", "financial"}


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
    # Adaptive compute metadata: how much compute was actually used.
    # trajectories_used <= max_trajectories. When early_exit_reason is set,
    # the system stopped before exhausting the budget.
    adaptive: bool = False
    trajectories_used: int = 0
    max_trajectories: int = 0
    early_exit_reason: Optional[str] = None
    branch_reason: Optional[str] = None
    agreement: Optional[bool] = None
    # Verification status: one of "verified", "refuted", "unsupported",
    # "self_only", "error". More granular than verified (bool).
    verification_status: str = "self_only"


class VibeThinkerCLRAsync:
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8080",
        k: int = 8,
        max_concurrent: int = 6,
        adaptive: bool = True,
        k_min: int = 2,
        k_max: int = 6,
        policy: Optional[AdaptivePolicy] = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.k = k
        self.max_concurrent = max_concurrent  # Limit concurrent requests
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Adaptive compute policy. If not provided, construct from k_min/k_max
        # for backwards compatibility.
        self.adaptive = adaptive
        if policy is None:
            self.policy = AdaptivePolicy(
                initial_k_without_verifier=min(k_min, k) if adaptive else k,
                initial_k_with_verifier=1 if adaptive else k,
                max_k=min(k_max, k) if adaptive else k,
            )
        else:
            self.policy = policy
        # Keep k_min/k_max for backwards compat
        self.k_min = self.policy.initial_k_without_verifier
        self.k_max = self.policy.max_k
        # Store the original max_k so adjust_max_k_for_queue_load can restore it
        self._original_max_k = self.policy.max_k

    def adjust_max_k_for_queue_load(self, queue_load: float) -> None:
        """Adjust max_k based on queue pressure.

        When the job queue is busy, lower max compute to improve throughput.
        Most jobs can early-exit at k=1 or k=2, so lowering max_k for
        high-load periods doesn't hurt reliability for easy problems.

        Queue load thresholds:
          < 0.5  -> max_k = policy.max_k (full budget)
          0.5-0.8 -> max_k = 4 (moderate reduction)
          > 0.8  -> max_k = 2 (minimal, unless verifier required)

        For code/math tasks with a verifier, a single verified answer is
        better than six self-consistent guesses, so this reduction is safe.

        Args:
            queue_load: fraction of queue capacity in use (0.0 to 1.0).
        """
        original_max = self._original_max_k
        if queue_load > 0.8:
            self.policy.max_k = min(2, original_max)
        elif queue_load > 0.5:
            self.policy.max_k = min(4, original_max)
        else:
            self.policy.max_k = original_max
        self.k_max = self.policy.max_k
        if self.policy.max_k != original_max:
            print(f"[CLR] Queue load {queue_load:.0%}: max_k adjusted "
                  f"{original_max} -> {self.policy.max_k}")

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
        grammar: Optional[str] = None,
    ) -> str:
        """Async call to llama-server /completion endpoint.

Raises RuntimeError on any failure — callers must handle the exception
rather than silently proceeding with an empty string.

Args:
    grammar: optional GBNF grammar string to constrain output format.
        When set, llama-server enforces the grammar — the model physically
        cannot output invalid JSON. Used for claim extraction to prevent
        small models from producing malformed JSON.
"""
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": -1,
            "stop": stop if stop is not None else ["<|im_end|>"],
        }
        if grammar:
            payload["grammar"] = grammar
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
            session, extraction_prompt, max_tokens=2048, temperature=0.3,
            grammar=_CLAIMS_JSON_GRAMMAR,
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

    def _check_answer_consistency(self, answer: str, trajectories: List[Dict]) -> Optional[bool]:
        """Check if an answer is consistent across trajectories.

        This is NOT deterministic verification. It is cross-trajectory
        consistency — the model agreeing with itself. Consensus is not
        proof of correctness. This signal can adjust the score WITHIN
        the self-claims-only cap (0.65) but can NEVER exceed it.

        Only an external deterministic verifier (MathVerifier,
        CodeVerifier, FactualVerifier) can produce scores above 0.65.

        Returns:
          True  — answer is consistent across multiple trajectories
          False — answer contradicts other trajectories
          None  — cannot determine (not enough data)
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
        consistency_check: Optional[bool] = None,
    ) -> float:
        """Calculate a reliability score for a trajectory.

        Scoring rules (fail-closed):
          - No verdicts or empty claims -> 0.0
          - No final answer -> 0.0
          - Fewer than MIN_CLAIMS_FOR_SCORING meaningful claims -> 0.0
          - Any unverified claim (verdict 0) heavily penalizes the score
          - Self-claims-only confidence is HARD CAPPED at 0.65
          - Cross-trajectory consistency gives a small boost WITHIN the cap
          - Only an external verifier can exceed 0.65

        The raw claim-level score (mean^5 over meaningful claims) is passed
        through :func:`compute_confidence` which enforces the self-claims-only
        cap. This is the active runtime path — the cap is not advisory.

        Args:
            verdicts: list of 0/1 verdicts from the verifier.
            claims: optional list of claim strings, used to filter garbage.
            answer_present: whether the trajectory produced a final answer.
            consistency_check: result of cross-trajectory consistency check.
                This is NOT deterministic verification — it is the model
                agreeing with itself. It can adjust the score within the
                0.65 cap but can NEVER exceed it.
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
            if len(meaningful) < self.policy.min_claims:
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

        # Cross-trajectory consistency is NOT deterministic verification.
        # It is the model agreeing with itself — consensus is not proof.
        # Consistency gives a small boost within the 0.65 cap:
        #   - Agree: +0.05 (capped at 0.65)
        #   - Disagree: * 0.75 (penalty)
        # It NEVER sets det_verification or verification_method to anything
        # that would bypass the self-claims-only cap.
        if consistency_check is True:
            base = min(base + self.policy.consistency_boost, self.policy.self_claim_cap)
        elif consistency_check is False:
            base *= self.policy.contradiction_penalty

        # Always route through compute_confidence as self_claims_only.
        # The cap is enforced here. Only external verifiers (handled in
        # run() after this method returns) can exceed 0.65.
        confidence = compute_confidence(
            model_score=base,
            claim_consistency=base,
            deterministic_verification=None,
            verification_method="self_claims_only",
        )
        return confidence.final_score

    # ------------------------------------------------------------------ #
    # One full trajectory
    # ------------------------------------------------------------------ #
    async def _generate_one_trajectory(
        self, session: aiohttp.ClientSession, problem: str, max_tokens: int
    ) -> Dict:
        """Generate + score one full trajectory.

        This does the full CLR pipeline: reasoning → claim extraction →
        claim verification → scoring. It is the expensive path (7 LLM calls).
        """
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
        # Initial score without consistency check (applied later in run())
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

    async def _generate_lightweight_trajectory(
        self, session: aiohttp.ClientSession, problem: str, max_tokens: int
    ) -> Dict:
        """Generate a trajectory with reasoning + answer extraction only.

        Skips claim extraction and claim verification (saves 6 LLM calls).
        Used in the fast path when a deterministic verifier can check the
        final answer directly — no need for expensive self-verification
        if an external verifier will confirm or refute.

        The claims/verdicts fields are empty. If the verifier doesn't
        confirm, the caller should re-run with full trajectory generation
        to get self-claim scores.
        """
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

        # Extract just the final answer (no LLM call — regex only)
        final_answer = self._extract_boxed_answer(raw_trace)

        return {
            "score": 0.0,  # Not scored yet — caller scores after verification
            "answer": final_answer,
            "claims": [],
            "verdicts": [],
            "raw_trace": raw_trace,
            "answer_present": final_answer is not None,
            "lightweight": True,
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
        """Adaptive compute: phased trajectory generation with early exit.

        Trust model (NON-NEGOTIABLE):
          - Self-consensus NEVER exceeds 0.65 (self_claim_cap)
          - Only external verifier success can exceed 0.65
          - Consensus saves compute, it does NOT raise trust
          - High-risk tasks cannot early-exit from self-consensus alone

        Phases:
          Phase 1: k=1 if verifier, k=2 if no verifier
            - If verifier: lightweight trajectory (answer only) + verify
            - If no verifier: full trajectories + consensus check
          Phase 2: consensus early exit (capped at 0.65, no high-risk)
          Phase 3: branch up to max_k on disagreement/uncertainty
        """
        policy = self.policy
        is_high_risk = task_type in HIGH_RISK_TASK_TYPES

        # Determine initial k
        if verifier is not None:
            k_initial = policy.initial_k_with_verifier
        else:
            k_initial = policy.initial_k_without_verifier
        k_total = policy.max_k

        print(
            f"Running adaptive CLR: k_initial={k_initial}, k_max={k_total} "
            f"(max_concurrent={self.max_concurrent}, "
            f"verifier={'yes' if verifier else 'no'}, "
            f"high_risk={is_high_risk})..."
        )

        all_trajectories: List[Any] = []
        all_failures: List[Exception] = []
        early_exit_reason: Optional[str] = None
        branch_reason: Optional[str] = None

        async with aiohttp.ClientSession() as session:
            # === Phase 1: Fast Path ===
            # If a verifier exists, use lightweight generation (answer only)
            # and verify immediately. This saves 6 LLM calls per trajectory.
            use_lightweight = verifier is not None
            gen_method = (
                self._generate_lightweight_trajectory if use_lightweight
                else self._generate_one_trajectory
            )
            print(f"[CLR] Phase 1: generating {k_initial} trajectories "
                  f"({'lightweight' if use_lightweight else 'full'})...")
            tasks = [
                gen_method(session, problem, max_tokens_per_trace)
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

            # Get answered trajectories
            valid = [t for t in all_trajectories if isinstance(t, dict)]
            answered = [t for t in valid if t.get("answer_present") and t.get("answer")]
            if not answered:
                return self._build_no_answer_result(
                    all_trajectories, all_failures, k_total,
                    adaptive=True, max_trajectories=k_total,
                )

            best = max(answered, key=lambda x: x.get("score", 0.0))

            # Early exit: verifier confirms the answer
            verifier_refuted = False
            verifier_unsupported = False
            if verifier is not None:
                v_result = await self._try_verifier(
                    verifier, problem, best["answer"], verifier_context,
                    best_score=0.65,  # cap for self-claims baseline
                )
                if v_result and v_result[2]:  # verified=True
                    print(f"[CLR] Phase 1 early exit: verifier confirmed answer")
                    early_exit_reason = "deterministic_verifier_passed"
                    return self._build_final_result(
                        best, all_trajectories, all_failures,
                        k_used=len(all_trajectories),
                        max_k=k_total,
                        verification_method=v_result[0],
                        verified=v_result[2],
                        det_verification=v_result[1],
                        final_score_override=v_result[3],
                        adaptive=True,
                        early_exit_reason=early_exit_reason,
                        verification_status="verified",
                    )
                if v_result and not v_result[2]:
                    if v_result[1] is not None and v_result[1] <= 0.0:
                        verifier_refuted = True
                    else:
                        verifier_unsupported = True

            # === Phase 2: Consensus Check ===
            # Only for non-high-risk tasks, and only when verifier didn't refute.
            # Consensus saves compute but does NOT raise trust above 0.65.
            can_consensus_exit = (
                not verifier_refuted
                and policy.early_exit_on_self_consensus
                and not (is_high_risk and policy.disable_self_consensus_for_high_risk)
            )
            if can_consensus_exit and len(answered) >= 2 and best.get("score", 0.0) > 0.0:
                # Consensus requires full trajectories with real scores.
                # Lightweight trajectories have score=0.0 (no claims/verdicts),
                # so consensus among them is meaningless — proceed to Phase 3.
                agreement = self._check_answer_agreement(answered)
                if agreement:
                    print(f"[CLR] Phase 2 early exit: self-consensus "
                          f"(score capped at {policy.self_claim_cap})")
                    early_exit_reason = "self_consensus_cap_reached"
                    # Score is capped at 0.65 — consensus does NOT raise trust
                    final_score = min(
                        best.get("score", 0.0), policy.self_claim_cap,
                    )
                    return self._build_final_result(
                        best, all_trajectories, all_failures,
                        k_used=len(all_trajectories),
                        max_k=k_total,
                        final_score_override=final_score,
                        adaptive=True,
                        early_exit_reason=early_exit_reason,
                        agreement=True,
                        verification_status="self_only",
                    )

            # === Phase 3: Branching (System 2) ===
            if verifier_refuted:
                branch_reason = "verifier_refuted"
            elif verifier_unsupported and len(answered) < 2:
                branch_reason = "insufficient_trajectories"
            elif not can_consensus_exit and is_high_risk:
                branch_reason = "high_risk_no_verifier"
            else:
                branch_reason = "disagreement_or_uncertainty"

            remaining = k_total - len(all_trajectories)
            if remaining > 0:
                print(
                    f"[CLR] Phase 3: {branch_reason} — "
                    f"generating {remaining} more trajectories..."
                )
                # In phase 3, always use full trajectory generation
                # (we need claim scores for final ranking)
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
                adaptive=True, max_trajectories=k_total,
                branch_reason=branch_reason,
            )

        best = max(answered, key=lambda x: x["score"])

        # Run verifier on the best answer from the full set
        verification_method = "self_claims_only"
        verified = False
        det_verification: Optional[float] = None
        final_score = best["score"]
        verification_status = "self_only"

        if verifier is not None:
            v_result = await self._try_verifier(
                verifier, problem, best["answer"], verifier_context,
                best_score=best["score"],
            )
            if v_result:
                verification_method, det_verification, verified, final_score = v_result
                if final_score is None:
                    final_score = best["score"]
                if verified:
                    verification_status = "verified"
                elif det_verification is not None and det_verification <= 0.0:
                    verification_status = "refuted"
                    final_score = 0.0
                else:
                    verification_status = "unsupported"

        # Enforce the self-claims-only cap
        if not verified:
            final_score = min(final_score, policy.self_claim_cap)

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
            adaptive=True,
            trajectories_used=len([t for t in all_trajectories if isinstance(t, dict)]),
            max_trajectories=k_total,
            early_exit_reason=early_exit_reason,
            branch_reason=branch_reason,
            agreement=self._check_answer_agreement(answered) if len(answered) >= 2 else None,
            verification_status=verification_status,
        )

        print(f"\nBest trajectory score: {final_score:.4f}")
        print(f"Best answer: {result.best_answer}")
        print(f"Verification: {verification_method} (verified={verified}, "
              f"status={verification_status})")
        print(f"Compute used: {result.trajectories_used} trajectories "
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
                adaptive=False,
                trajectories_used=0,
                max_trajectories=self.k,
                verification_status="error",
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
                adaptive=False,
                trajectories_used=len(valid_trajectories),
                max_trajectories=self.k,
                verification_status="error",
            )

        best = max(answered, key=lambda x: x["score"])

        # Run verifier
        verification_method = "self_claims_only"
        verified = False
        det_verification: Optional[float] = None
        final_score = best["score"]
        verification_status = "self_only"

        if verifier is not None:
            v_result = await self._try_verifier(
                verifier, problem, best["answer"], verifier_context, best_score=best["score"],
            )
            if v_result:
                verification_method, det_verification, verified, final_score = v_result
                if final_score is None:
                    final_score = best["score"]
                if verified:
                    verification_status = "verified"
                elif det_verification is not None and det_verification <= 0.0:
                    verification_status = "refuted"
                    final_score = 0.0
                else:
                    verification_status = "unsupported"

        # Enforce self-claims-only cap
        if not verified:
            final_score = min(final_score, self.policy.self_claim_cap)

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
            adaptive=False,
            trajectories_used=len(valid_trajectories),
            max_trajectories=self.k,
            verification_status=verification_status,
        )

        print(f"\nBest trajectory score: {final_score:.4f}")
        print(f"Best answer: {result.best_answer}")
        print(f"Verification: {verification_method} (verified={verified}, "
              f"status={verification_status})")
        return result

    def _score_trajectories(self, trajectories: List[Any]) -> List[Dict]:
        """Score valid trajectories and return those with answers.

        Applies cross-trajectory consistency checking and contradiction
        penalties. Returns only trajectories that produced a final answer.

        Consistency is NOT deterministic verification — it is the model
        agreeing with itself. It can adjust the score within the 0.65 cap
        but can NEVER exceed it.
        """
        valid = [t for t in trajectories if isinstance(t, dict)]
        answered = [t for t in valid if t.get("answer_present") and t.get("answer")]
        if not answered:
            return []

        for t in answered:
            consistency = self._check_answer_consistency(t["answer"], valid)
            t["consistency_check"] = consistency
            t["score"] = self._calculate_reliability(
                t["verdicts"],
                claims=t["claims"],
                answer_present=True,
                consistency_check=consistency,
            )

        # Note: contradiction penalty is already applied per-trajectory via
        # _check_answer_consistency returning False -> _calculate_reliability
        # multiplies by contradiction_penalty. No separate penalty needed here.

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

        if not self._check_answer_agreement(answered):
            return False

        best_score = max(t.get("score", 0.0) for t in answered)
        if best_score >= 0.3:
            print(f"[CLR] Consensus: all trajectories agree "
                  f"(score={best_score:.3f})")
            return True

        return False

    def _check_answer_agreement(self, answered: List[Dict]) -> bool:
        """Check if all answered trajectories produced the same boxed answer.

        This is cross-trajectory consistency — NOT deterministic verification.
        The model agreeing with itself is not proof of correctness.
        """
        if len(answered) < 2:
            return False

        boxed_answers = []
        for t in answered:
            extracted = self._extract_boxed_answer(t.get("raw_trace", ""))
            if extracted is not None:
                boxed_answers.append(extracted.lower().strip())

        if len(boxed_answers) < 2:
            return False

        return len(set(boxed_answers)) == 1

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

        det_score semantics:
          - verified=True: det_score = v_result.score (the verifier's confidence)
          - verified=False, v_result.score == 0.0: refuted (det_score = 0.0)
          - verified=False, v_result.score > 0.0: unsupported (det_score = None)
            This preserves the self-claim score instead of zeroing it.
        """
        try:
            v_result = await verifier.verify(
                problem, answer,
                context=verifier_context or {},
            )
            verification_method = getattr(verifier, "name", "verifier")
            verified = v_result.verified
            print(f"[CLR] Verifier {verification_method}: "
                  f"verified={verified}, score={v_result.score:.3f}")

            if verified:
                det_verification = v_result.score
            elif v_result.score <= 0.0:
                # Explicit refutation
                det_verification = 0.0
            else:
                # Unsupported — verifier couldn't verify but didn't refute.
                # det_verification = None so caller keeps self-claim score.
                det_verification = None

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
            elif det_verification == 0.0:
                # Explicit refutation -> zero the score
                final_score = 0.0
            # else: unsupported -> final_score stays None, caller uses best["score"]

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
        max_k: int = 0,
        verification_method: str = "self_claims_only",
        verified: bool = False,
        det_verification: Optional[float] = None,
        final_score_override: Optional[float] = None,
        adaptive: bool = False,
        early_exit_reason: Optional[str] = None,
        agreement: Optional[bool] = None,
        verification_status: str = "self_only",
    ) -> CLRResult:
        """Build a CLRResult from the best trajectory."""
        final_score = final_score_override if final_score_override is not None else best["score"]
        valid = [t for t in all_trajectories if isinstance(t, dict)]

        # Enforce self-claims-only cap
        if not verified:
            final_score = min(final_score, self.policy.self_claim_cap)

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
            adaptive=adaptive,
            trajectories_used=len(valid),
            max_trajectories=max_k or k_used,
            early_exit_reason=early_exit_reason,
            agreement=agreement,
            verification_status=verification_status,
        )

        print(f"\nBest trajectory score: {final_score:.4f}")
        print(f"Best answer: {result.best_answer}")
        print(f"Verification: {verification_method} (verified={verified}, "
              f"status={verification_status})")
        print(f"Compute used: {len(valid)} trajectories (of max {max_k or k_used})")
        return result

    def _build_no_answer_result(
        self,
        all_trajectories: List[Any],
        all_failures: List[Exception],
        k_used: int,
        adaptive: bool = False,
        max_trajectories: int = 0,
        branch_reason: Optional[str] = None,
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
            adaptive=adaptive,
            trajectories_used=len(valid),
            max_trajectories=max_trajectories or k_used,
            branch_reason=branch_reason,
            verification_status="error",
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
