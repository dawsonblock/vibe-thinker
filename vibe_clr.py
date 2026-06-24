"""
VibeThinker-3B Claim-Level Reliability (CLR) wrapper — synchronous version.

Requires a running llama-server (e.g. on http://127.0.0.1:8080) serving the
VibeThinker-3B GGUF model with the patched reasoning chat template.

Install:  pip install requests

Bug fixes vs. the original walkthrough version:
  - Stop tokens: removed the bare "]" (a corrupted </think> artifact) that
    prematurely truncated generations. Now only ["<|im_end|>"].
  - Verdict parsing: no longer treats "10" or "1 reason..." as verdict 1.
    Parses the first standalone 0/1 or yes/no.
  - final_answer "null": the JSON extractor treated the string "null" as a
    real answer. Now normalized to None.
  - Added a plain (non-CLR) generation helper for reuse by callers.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests


@dataclass
class CLRResult:
    best_answer: str
    best_score: float
    best_raw_trace: str
    all_trajectories: List[Dict] = field(default_factory=list)
    k: int = 8


class VibeThinkerCLR:
    def __init__(self, server_url: str = "http://127.0.0.1:8080", k: int = 8):
        self.server_url = server_url.rstrip("/")
        self.k = k  # Number of trajectories (start with 8, scale to 16-32 for max quality)

    # ------------------------------------------------------------------ #
    # Low-level model call
    # ------------------------------------------------------------------ #
    def _call_model(
        self,
        prompt: str,
        max_tokens: int = 8192,
        temperature: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Call the running llama-server /completion endpoint."""
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": -1,
            "stop": stop if stop is not None else ["<|im_end|>"],
        }
        try:
            resp = requests.post(
                f"{self.server_url}/completion", json=payload, timeout=600
            )
            resp.raise_for_status()
            return resp.json().get("content", "")
        except Exception as e:
            print(f"Model call failed: {e}")
            return ""

    def generate_plain(self, problem: str, max_tokens: int = 8192) -> str:
        """Single plain generation (no CLR). Useful for the orchestrator's
        non-CLR specialist path."""
        prompt = (
            f"<|im_start|>user\n{problem}\n<|im_end|>\n<|im_start|>assistant\n"
        )
        return self._call_model(prompt, max_tokens=max_tokens)

    # ------------------------------------------------------------------ #
    # Claim extraction + answer parsing
    # ------------------------------------------------------------------ #
    def _extract_claims_and_answer(self, text: str) -> Dict:
        """
        Ask the model to extract structured claims + final answer.
        More reliable than pure regex for this model.
        """
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
        raw = self._call_model(extraction_prompt, max_tokens=2048, temperature=0.3)

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
                # Normalize the string "null" / empty to None
                if isinstance(final_answer, str):
                    if final_answer.strip().lower() in ("null", "none", "", "n/a"):
                        final_answer = None
                return {"claims": claims, "final_answer": final_answer, "raw": text}
        except Exception:
            pass

        # Fallback: simple regex extraction
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
        # Explicit yes/no first
        if s.startswith("yes") or "yes," in s[:6]:
            return 1
        if s.startswith("no") or "no," in s[:6]:
            return 0
        # First standalone digit 0 or 1 (not "10", not "1 reason")
        m = re.search(r"\b([01])\b", s)
        if m:
            return int(m.group(1))
        # Last resort: leading digit
        m = re.search(r"([01])", s[:10])
        return int(m.group(1)) if m else 0

    def _verify_claims(self, claims: List[str]) -> List[int]:
        """Self-verification: return list of 0/1 verdicts."""
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
            raw = self._call_model(verify_prompt, max_tokens=128, temperature=0.2)
            verdicts.append(self._parse_verdict(raw))
        return verdicts

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def _calculate_reliability(self, verdicts: List[int]) -> float:
        """Nonlinear scoring — one bad claim heavily penalizes the trajectory."""
        if not verdicts:
            return 0.0
        mean = sum(verdicts) / len(verdicts)
        return mean ** 5  # Strong penalty for weak claims (as in the paper)

    # ------------------------------------------------------------------ #
    # Main CLR entry point
    # ------------------------------------------------------------------ #
    def run(self, problem: str, max_tokens_per_trace: int = 16384) -> CLRResult:
        """
        Generate k trajectories, score them, and return the best one.
        """
        print(f"Running CLR with k={self.k} trajectories...")
        trajectories = []

        for i in range(self.k):
            print(f"  Generating trajectory {i + 1}/{self.k}...")

            reasoning_prompt = (
                "<|im_start|>user\n"
                f"{problem}\n\n"
                "Solve this step by step. Think carefully and put your final "
                "answer in \\boxed{}.\n"
                "<|im_end|>\n<|im_start|>assistant\n"
            )
            raw_trace = self._call_model(
                reasoning_prompt, max_tokens=max_tokens_per_trace
            )

            parsed = self._extract_claims_and_answer(raw_trace)
            verdicts = self._verify_claims(parsed["claims"])
            score = self._calculate_reliability(verdicts)

            trajectories.append(
                {
                    "score": score,
                    "answer": parsed["final_answer"],
                    "claims": parsed["claims"],
                    "verdicts": verdicts,
                    "raw_trace": raw_trace,
                }
            )

        best = max(trajectories, key=lambda x: x["score"])

        result = CLRResult(
            best_answer=best["answer"] or "No clear answer found",
            best_score=best["score"],
            best_raw_trace=best["raw_trace"],
            all_trajectories=trajectories,
            k=self.k,
        )

        print(f"\nBest trajectory score: {best['score']:.4f}")
        print(f"Best answer: {result.best_answer}")
        return result


# ====================== EXAMPLE USAGE ======================

if __name__ == "__main__":
    clr = VibeThinkerCLR(k=8)  # Start with 8 for speed

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

    # Optional: Save full trace for your memory vault
    # with open("clr_trace.json", "w") as f:
    #     json.dump(result.__dict__, f, indent=2)
