#!/usr/bin/env python3
"""TurboQuant KV-cache compression perplexity (PPL) validation harness.

Phase 2.3 of the production plan. Validates that the asymmetric TurboQuant
KV cache compression (``--cache-type-k q8_0 --cache-type-v turbo3``) does
NOT degrade reasoning output quality, by measuring perplexity on a held-out
text corpus and comparing it against an uncompressed baseline.

Why perplexity?
  PPL is the standard information-theoretic measure of how well a language
  model predicts a text. A KV-cache compression scheme that corrupts the
  cache will show up as a PPL increase on long contexts (where the cache
  is actually exercised). A small PPL delta (< tolerance) means the
  compression preserves the model's predictive distribution.

Asymmetric finding (from AGENTS.md "TurboQuant+"):
  The V cache tolerates aggressive compression (turbo2/3/4) with <1.5% PPL
  loss; the K cache does NOT — symmetric turbo K compression is where
  models break. This harness lets you confirm that asymmetry on your
  target hardware before shipping a preset.

Usage (HTTP path — llama-server / RuvLLM sidecar with logprobs):
  # 1. Start the baseline server (uncompressed):
  llama-server -m model.gguf --port 8081 \\
      --cache-type-k f16 --cache-type-v f16
  # 2. Run the baseline eval:
  python3 scripts/turboquant_ppl_check.py eval \\
      --base-url http://127.0.0.1:8081 --corpus corpus.txt \\
      --out baseline_ppl.json
  # 3. Start the candidate server (TurboQuant default):
  llama-server -m model.gguf --port 8081 \\
      --cache-type-k q8_0 --cache-type-v turbo3
  # 4. Run the candidate eval:
  python3 scripts/turboquant_ppl_check.py eval \\
      --base-url http://127.0.0.1:8081 --corpus corpus.txt \\
      --out candidate_ppl.json
  # 5. Compare:
  python3 scripts/turboquant_ppl_check.py compare \\
      --baseline baseline_ppl.json --candidate candidate_ppl.json \\
      --tolerance 0.015

Usage (in-process path — ruvllm_py, needs a logprobs method, see below):
  python3 scripts/turboquant_ppl_check.py eval-inprocess \\
      --model model.gguf --corpus corpus.txt --metal \\
      --cache-type-k q8_0 --cache-type-v turbo3 --out candidate_ppl.json

Exit codes (compare):
  0 — candidate PPL within tolerance of baseline (compression is safe).
  1 — candidate PPL exceeds tolerance (compression degrades quality; do
      NOT ship this preset).
  2 — evaluation error (missing corpus, server unreachable, etc.).

NOTE on the in-process path: ``ruvllm_py.Engine.complete_with_logprobs()``
generates tokens and returns per-token log-probabilities (computed via
``log_softmax`` over the vocabulary by the Rust candle backend). When the
binding is built with the ``candle`` feature and ``SUPPORTS_LOGPROBS`` is
True, the in-process path produces a real PplResult. When the binding is
absent or stubbed (no candle feature), it fail-closed raises
NotImplementedError — use the HTTP path instead.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Core PPL math (model-free, unit-testable)
# ---------------------------------------------------------------------------

def compute_ppl(token_log_probs: List[float]) -> float:
    """Compute perplexity from a list of per-token log-probabilities.

    PPL = exp( -1/N * sum_i log P(token_i | context_i) )

    Args:
        token_log_probs: natural-log probabilities of each token given its
            context. Must be non-empty; values should be <= 0 (log-probs).

    Returns:
        The perplexity (>= 1.0; lower is better).

    Raises:
        ValueError: if the list is empty.
    """
    if not token_log_probs:
        raise ValueError("compute_ppl requires a non-empty list of log-probs")
    mean_neg_log_prob = -sum(token_log_probs) / len(token_log_probs)
    return math.exp(mean_neg_log_prob)


@dataclass
class PplResult:
    """A single PPL evaluation result."""
    ppl: float
    n_tokens: int
    mean_log_prob: float
    config: Dict[str, str] = field(default_factory=dict)
    source: str = ""  # e.g. "llama-server:8081" or "ruvllm_py"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ppl": self.ppl,
            "n_tokens": self.n_tokens,
            "mean_log_prob": self.mean_log_prob,
            "config": self.config,
            "source": self.source,
        }

    @classmethod
    def from_log_probs(
        cls,
        token_log_probs: List[float],
        config: Optional[Dict[str, str]] = None,
        source: str = "",
    ) -> "PplResult":
        ppl = compute_ppl(token_log_probs)
        mean_lp = sum(token_log_probs) / len(token_log_probs)
        return cls(
            ppl=ppl,
            n_tokens=len(token_log_probs),
            mean_log_prob=mean_lp,
            config=config or {},
            source=source,
        )


@dataclass
class PplComparison:
    """Result of comparing a candidate PPL against a baseline."""
    baseline_ppl: float
    candidate_ppl: float
    delta: float  # candidate - baseline (absolute)
    pct_delta: float  # relative increase (candidate/baseline - 1)
    tolerance: float  # max acceptable pct_delta
    passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_ppl": self.baseline_ppl,
            "candidate_ppl": self.candidate_ppl,
            "delta": self.delta,
            "pct_delta": self.pct_delta,
            "tolerance": self.tolerance,
            "passed": self.passed,
        }


def compare_ppl(baseline: float, candidate: float, tolerance: float) -> PplComparison:
    """Compare a candidate PPL against a baseline within a tolerance.

    Args:
        baseline: the uncompressed-baseline perplexity.
        candidate: the TurboQuant-compressed perplexity.
        tolerance: the max acceptable relative PPL increase
            (e.g. 0.015 = 1.5%). The candidate may have LOWER PPL (delta<0)
            without failing — only an increase beyond tolerance fails.

    Returns:
        A PplComparison with `passed` set accordingly.
    """
    if baseline <= 0:
        raise ValueError("baseline PPL must be positive")
    delta = candidate - baseline
    pct_delta = candidate / baseline - 1.0
    passed = pct_delta <= tolerance
    return PplComparison(
        baseline_ppl=baseline,
        candidate_ppl=candidate,
        delta=delta,
        pct_delta=pct_delta,
        tolerance=tolerance,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# HTTP eval path (llama-server / RuvLLM /completion with logprobs)
# ---------------------------------------------------------------------------

def _extract_token_logprobs(completion_response: Dict[str, Any]) -> List[float]:
    """Extract per-token log-probs from a llama-server /completion response.

    llama-server returns ``logprobs`` as a list of dicts when the
    ``logprobs`` request field is set. Each entry has a ``token`` and a
    ``logprob`` (natural log) for the chosen token. We collect the chosen
    token's logprob at each position.
    """
    logprobs = completion_response.get("logprobs")
    if not logprobs:
        raise ValueError(
            "completion response has no 'logprobs' — start llama-server "
            "with --logprobs and request logprobs in the POST body"
        )
    # llama-server: logprobs is a list of {token, token_id, logprob, ...}
    # for each generated position, OR a dict with 'content' (newer format).
    seq = logprobs.get("content") if isinstance(logprobs, dict) else logprobs
    if not seq:
        raise ValueError("logprobs sequence is empty")
    out: List[float] = []
    for entry in seq:
        if isinstance(entry, dict) and "logprob" in entry:
            out.append(float(entry["logprob"]))
    return out


def eval_http(
    base_url: str,
    corpus: str,
    n_predict: int = 0,
    timeout: float = 120.0,
) -> PplResult:
    """Evaluate PPL against a llama-server / RuvLLM HTTP endpoint.

    Sends the corpus as the prompt with ``n_predict=0`` (no generation) and
    ``logprobs`` enabled so the server returns the log-prob of each prompt
    token under the model. This evaluates the model's PPL on the corpus
    text directly (prompt-scoring), which is the standard PPL measurement.

    Args:
        base_url: e.g. "http://127.0.0.1:8081".
        corpus: the text to score.
        n_predict: tokens to generate (0 = prompt-scoring only).
        timeout: HTTP timeout in seconds.

    Returns:
        A PplResult.
    """
    import urllib.request
    import urllib.error

    body = json.dumps({
        "prompt": corpus,
        "n_predict": n_predict,
        "logprobs": 1,  # request log-probs for the top-1 token at each pos
        "temperature": 0.0,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/completion",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"could not reach llama-server at {base_url}: {e}. "
            f"Start it with --logprobs and the desired --cache-type-k/v."
        ) from e
    log_probs = _extract_token_logprobs(data)
    if not log_probs:
        raise RuntimeError("no token log-probs extracted from completion response")
    return PplResult.from_log_probs(log_probs, source=f"llama-server:{base_url}")


def eval_inprocess(
    model_path: str,
    corpus: str,
    cache_type_k: str = "q8_0",
    cache_type_v: str = "turbo3",
    use_metal: bool = False,
    n_threads: int = 8,
) -> PplResult:
    """Evaluate PPL via the in-process ruvllm_py Engine.

    Uses ``ruvllm_py.Engine.complete_with_logprobs()`` to generate tokens
    with per-token log-probabilities (computed via ``log_softmax`` over the
    vocabulary by the Rust candle backend). The logprobs are fed into
    ``_extract_token_logprobs`` and then into ``PplResult.from_log_probs``.

    Fail-closed: when ``ruvllm_py`` is not importable, or is importable but
    does not expose the logprobs capability (``SUPPORTS_LOGPROBS`` is False,
    i.e. built without the ``candle`` feature), this raises
    ``NotImplementedError`` with guidance to use the HTTP path.
    """
    # Fail-closed: check for the binding before doing any work.
    try:
        import ruvllm_py  # type: ignore
    except ImportError:
        raise NotImplementedError(
            "In-process PPL eval requires the ruvllm_py PyO3 extension, "
            "which is not installed. Use the HTTP path: start "
            "llama-server/RuvLLM with --logprobs and run "
            "`turboquant_ppl_check.py eval --base-url ...`. "
            "Build the extension with `maturin develop --release "
            "--features candle` in the ruvllm_py directory."
        ) from None

    # Check the logprobs capability BEFORE creating an Engine (avoids a
    # model-not-found error when the binding is present but stubbed).
    if not getattr(ruvllm_py, "SUPPORTS_LOGPROBS", False):
        raise NotImplementedError(
            "In-process PPL eval requires per-token log-probs, which this "
            "ruvllm_py build does not expose (built without the candle "
            "feature). Use the HTTP path: start llama-server/RuvLLM with "
            "--logprobs and run `turboquant_ppl_check.py eval --base-url "
            "...`. Rebuild with `maturin develop --release --features "
            "candle` to enable the in-process path."
        )

    engine = ruvllm_py.Engine(
        model_path=model_path,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        use_metal=use_metal,
        n_threads=n_threads,
    )

    # Generate tokens with per-token logprobs (greedy / temperature=0 for
    # deterministic scoring). The Rust backend applies log_softmax over the
    # full vocabulary at each step.
    response = engine.complete_with_logprobs(
        prompt=corpus,
        max_tokens=128,
        temperature=0.0,
    )

    # Extract per-token log-probs (same format as llama-server /completion).
    log_probs = _extract_token_logprobs(response)
    if not log_probs:
        raise RuntimeError("no token log-probs extracted from in-process engine")

    config: Dict[str, str] = {
        "cache_type_k": cache_type_k,
        "cache_type_v": cache_type_v,
        "use_metal": str(use_metal),
        "n_threads": str(n_threads),
    }
    return PplResult.from_log_probs(log_probs, config=config, source="ruvllm_py")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_eval(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.corpus):
        print(f"[ppl] corpus file not found: {args.corpus}", file=sys.stderr)
        return 2
    with open(args.corpus, "r", encoding="utf-8") as f:
        corpus = f.read()
    if not corpus.strip():
        print(f"[ppl] corpus is empty: {args.corpus}", file=sys.stderr)
        return 2
    try:
        result = eval_http(args.base_url, corpus, timeout=args.timeout)
    except Exception as e:
        print(f"[ppl] eval failed: {e}", file=sys.stderr)
        return 2
    out = result.to_dict()
    out["corpus"] = args.corpus
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[ppl] PPL={result.ppl:.4f} over {result.n_tokens} tokens "
          f"(source={result.source}) -> {args.out}")
    return 0


def _cmd_eval_inprocess(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.corpus):
        print(f"[ppl] corpus file not found: {args.corpus}", file=sys.stderr)
        return 2
    with open(args.corpus, "r", encoding="utf-8") as f:
        corpus = f.read()
    try:
        result = eval_inprocess(
            args.model, corpus,
            cache_type_k=args.cache_type_k,
            cache_type_v=args.cache_type_v,
            use_metal=args.metal,
            n_threads=args.n_threads,
        )
    except NotImplementedError as e:
        print(f"[ppl] {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"[ppl] eval failed: {e}", file=sys.stderr)
        return 2
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"[ppl] PPL={result.ppl:.4f} over {result.n_tokens} tokens -> {args.out}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    try:
        with open(args.baseline, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        with open(args.candidate, "r", encoding="utf-8") as f:
            candidate = json.load(f)
        comp = compare_ppl(
            baseline["ppl"], candidate["ppl"], args.tolerance,
        )
    except Exception as e:
        print(f"[ppl] compare failed: {e}", file=sys.stderr)
        return 2
    report = comp.to_dict()
    report["baseline_file"] = args.baseline
    report["candidate_file"] = args.candidate
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    verdict = "PASS (compression within tolerance)" if comp.passed else \
              "FAIL (compression degrades PPL beyond tolerance — do NOT ship)"
    print(f"[ppl] baseline={comp.baseline_ppl:.4f} "
          f"candidate={comp.candidate_ppl:.4f} "
          f"delta={comp.delta:+.4f} ({comp.pct_delta*100:+.2f}%) "
          f"tolerance={comp.tolerance*100:.2f}%")
    print(f"[ppl] {verdict}")
    return 0 if comp.passed else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="turboquant_ppl_check",
        description="TurboQuant KV-cache compression perplexity validation.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_eval = sub.add_parser("eval", help="Evaluate PPL via a llama-server HTTP endpoint.")
    p_eval.add_argument("--base-url", required=True, help="llama-server base URL.")
    p_eval.add_argument("--corpus", required=True, help="Text corpus to score.")
    p_eval.add_argument("--out", required=True, help="Output JSON path.")
    p_eval.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout (s).")
    p_eval.set_defaults(func=_cmd_eval)

    p_inproc = sub.add_parser("eval-inprocess", help="Evaluate PPL via ruvllm_py (stub).")
    p_inproc.add_argument("--model", required=True, help="GGUF model path.")
    p_inproc.add_argument("--corpus", required=True, help="Text corpus to score.")
    p_inproc.add_argument("--cache-type-k", default="q8_0")
    p_inproc.add_argument("--cache-type-v", default="turbo3")
    p_inproc.add_argument("--n-threads", type=int, default=8)
    p_inproc.add_argument("--metal", action="store_true", help="Use Apple Silicon Metal.")
    p_inproc.add_argument("--out", required=True, help="Output JSON path.")
    p_inproc.set_defaults(func=_cmd_eval_inprocess)

    p_cmp = sub.add_parser("compare", help="Compare baseline vs candidate PPL.")
    p_cmp.add_argument("--baseline", required=True, help="Baseline PPL JSON.")
    p_cmp.add_argument("--candidate", required=True, help="Candidate PPL JSON.")
    p_cmp.add_argument("--tolerance", type=float, default=0.015,
                       help="Max acceptable relative PPL increase (default 0.015 = 1.5%%).")
    p_cmp.add_argument("--out", help="Optional output JSON for the report.")
    p_cmp.set_defaults(func=_cmd_compare)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
