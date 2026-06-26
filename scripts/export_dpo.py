#!/usr/bin/env python3
"""Export verified trajectories + failed candidates as fine-tuning data.

Turns the system into an automated data-flywheel: the verified-trajectory
store (independently-verified correct answers) becomes the "chosen" column
and failed / unverified completions from the orchestrator memory log become
the "rejected" column, in standard HuggingFace DPO / SFT formats.

Trust model (fail-closed, no epistemic contamination):
  - Chosen is drawn ONLY from verified_trajectories.json entries that are
    verified=True with a deterministic verification_method (not
    self_claims_only). The trajectory store already enforces this on load;
    we re-check defensively here. We never learn from self-claims.
  - Rejected is drawn ONLY from clearly-worse completions in the memory log:
      * CLR trajectories whose score is below --reject-threshold AND whose
        answer differs from the verified chosen answer (a near-tie is NOT
        labeled "rejected" — that would teach the model a wrong preference).
      * Code tasks where raw_traces.verified is False (the sandbox rejected
        the candidate — a genuine failure).
      * CLR runs whose best_score is below --min-score (low-confidence /
        unverified output).
  - A chosen entry with no matching rejected completion is still emitted as
    an SFT (chosen-only) example — verified data is always safe to learn
    from. DPO pairs require both sides.

Outputs:
  --format dpo : one JSON object per line: {"prompt","chosen","rejected"}
  --format sft : one JSON object per line: {"messages":[{"role","content"},...]}
  --format both: write <out>.dpo.jsonl and <out>.sft.jsonl

Usage:
    python3 scripts/export_dpo.py
    python3 scripts/export_dpo.py --trajectories verified_trajectories.json \
        --memory orchestrator_memory.jsonl --out dataset --format both \
        --min-score 0.75 --reject-threshold 0.5 --max-pairs-per-query 3
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Add the project root to sys.path so project modules are importable when
# run from anywhere (e.g. python3 scripts/export_dpo.py).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_query(query: str) -> str:
    """Normalize a query for matching chosen <-> rejected by query.

    Strips, collapses internal whitespace, and lowercases. Matching is
    intentionally exact-after-normalization: fuzzy matching risks pairing
    a verified answer for one problem with a failed attempt at a different
    one, which would poison the preference signal.
    """
    if not query:
        return ""
    return _WHITESPACE_RE.sub(" ", query).strip().lower()


def _is_trustworthy_chosen(entry: Dict[str, Any]) -> bool:
    """A chosen entry must be independently verified, not self-claimed.

    Mirrors the trust filter in VerifiedTrajectoryStore._load so a
    hand-edited or stale file cannot inject unverified "chosen" data.
    """
    if not entry.get("verified"):
        return False
    method = entry.get("verification_method", "self_claims_only")
    if method == "self_claims_only":
        return False
    answer = (entry.get("answer") or entry.get("best_answer") or "").strip()
    if not answer:
        return False
    return True


# ---------------------------------------------------------------------- #
# Loaders
# ---------------------------------------------------------------------- #
def load_chosen(trajectories_path: str) -> List[Dict[str, Any]]:
    """Load verified (chosen) completions from the trajectory store.

    Returns a list of dicts: {query, normalized, answer, score,
    verification_method, task_type}. Only independently-verified entries
    pass the trust filter.
    """
    if not os.path.exists(trajectories_path):
        return []
    try:
        with open(trajectories_path, "r") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        print(f"[export_dpo] WARNING: could not parse {trajectories_path}: {e}",
              file=sys.stderr)
        return []
    entries = data.get("entries", []) if isinstance(data, dict) else []
    chosen: List[Dict[str, Any]] = []
    for e in entries:
        if not isinstance(e, dict) or not _is_trustworthy_chosen(e):
            continue
        answer = (e.get("answer") or e.get("best_answer") or "").strip()
        chosen.append({
            "query": e.get("query", ""),
            "normalized": _normalize_query(e.get("query", "")),
            "answer": answer,
            "score": float(e.get("score", e.get("best_score", 0.0))),
            "verification_method": e.get("verification_method", ""),
            "task_type": e.get("task_type", "unknown"),
        })
    return chosen


def _rejected_from_clr(entry: Dict[str, Any], reject_threshold: float,
                       chosen_answers: set) -> List[str]:
    """Extract rejected completions from a CLR memory-log entry.

    The trimmed CLR result carries a `trajectories` list, each with its own
    `answer` and `score`. A trajectory is "rejected" only if its score is
    below reject_threshold AND its answer differs from the verified chosen
    answer — a near-tie is not a preference signal.
    """
    rejected: List[str] = []
    clr = entry.get("clr_result")
    if not isinstance(clr, dict):
        return rejected
    best_answer = (clr.get("best_answer") or "").strip()
    for t in clr.get("trajectories", []):
        if not isinstance(t, dict):
            continue
        ans = (t.get("answer") or "").strip()
        if not ans:
            continue
        score = t.get("score")
        try:
            score_f = float(score) if score is not None else 0.0
        except (TypeError, ValueError):
            score_f = 0.0
        if score_f >= reject_threshold:
            continue  # too good to label "rejected"
        if ans in chosen_answers:
            continue  # don't reject the verified answer itself
        if best_answer and ans == best_answer:
            continue
        rejected.append(ans)
    return rejected


def load_rejected(memory_path: str, reject_threshold: float,
                  min_score: float) -> Dict[str, List[str]]:
    """Load rejected completions from the orchestrator memory log.

    Returns a mapping of normalized query -> list of rejected answer
    strings (deduplicated, order preserved).
    """
    if not os.path.exists(memory_path):
        return {}
    rejected_map: Dict[str, List[str]] = {}
    with open(memory_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if not isinstance(entry, dict):
                continue
            query = entry.get("query", "")
            if not query:
                continue
            norm = _normalize_query(query)
            if not norm:
                continue
            raw = entry.get("raw_traces")
            # raw_traces may be the string "<non-serializable>" — skip those.
            if not isinstance(raw, dict):
                continue

            rejected: List[str] = []
            chosen_answers: set = set()  # placeholder; refined per-query later

            # CLR path: low-scoring trajectories are rejected completions.
            if isinstance(raw.get("clr_result"), dict):
                rejected.extend(
                    _rejected_from_clr(raw, reject_threshold, chosen_answers)
                )
                # An unverified / low-confidence CLR best_answer is itself a
                # rejected completion for this query.
                clr = raw["clr_result"]
                best_score = clr.get("best_score")
                try:
                    best_score_f = float(best_score) if best_score is not None else 0.0
                except (TypeError, ValueError):
                    best_score_f = 0.0
                if best_score_f < min_score:
                    best_ans = (clr.get("best_answer") or "").strip()
                    if best_ans:
                        rejected.append(best_ans)

            # Code path: an unverified best-effort answer is a rejected
            # completion (the sandbox explicitly rejected it).
            if raw.get("verified") is False and "all_verification_traces" in raw:
                ans = (entry.get("answer") or "").strip()
                if ans:
                    rejected.append(ans)

            if not rejected:
                continue
            bucket = rejected_map.setdefault(norm, [])
            for ans in rejected:
                if ans and ans not in bucket:
                    bucket.append(ans)
    return rejected_map


# ---------------------------------------------------------------------- #
# Pair building
# ---------------------------------------------------------------------- #
def build_dpo_pairs(
    chosen: List[Dict[str, Any]],
    rejected_map: Dict[str, List[str]],
    max_pairs_per_query: int,
) -> List[Dict[str, str]]:
    """Build DPO preference pairs: {prompt, chosen, rejected}.

    For each verified chosen entry, pair it with up to max_pairs_per_query
    rejected completions for the same (normalized) query. A chosen answer
    is never used as its own rejected (guarded in load_rejected + here).
    """
    pairs: List[Dict[str, str]] = []
    for c in chosen:
        rejected = rejected_map.get(c["normalized"], [])
        if not rejected:
            continue
        emitted = 0
        for rej in rejected:
            if rej == c["answer"]:
                continue  # never reject the verified answer
            pairs.append({
                "prompt": c["query"],
                "chosen": c["answer"],
                "rejected": rej,
            })
            emitted += 1
            if emitted >= max_pairs_per_query:
                break
    return pairs


def build_sft_examples(chosen: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build SFT (chosen-only) chat examples: {messages: [user, assistant]}.

    Every verified chosen entry becomes an SFT example — verified data is
    always safe to learn from, even without a matching rejected completion.
    """
    examples: List[Dict[str, Any]] = []
    for c in chosen:
        examples.append({
            "messages": [
                {"role": "user", "content": c["query"]},
                {"role": "assistant", "content": c["answer"]},
            ]
        })
    return examples


def _write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export verified trajectories + failed candidates as "
                    "DPO / SFT fine-tuning data."
    )
    parser.add_argument("--trajectories",
                        default=os.environ.get("TRAJECTORY_STORE_PATH",
                                               "verified_trajectories.json"),
                        help="Path to verified_trajectories.json (chosen).")
    parser.add_argument("--memory",
                        default="orchestrator_memory.jsonl",
                        help="Path to orchestrator_memory.jsonl (rejected).")
    parser.add_argument("--out", default="dataset",
                        help="Output base path. For --format both, writes "
                             "<out>.dpo.jsonl and <out>.sft.jsonl.")
    parser.add_argument("--format", choices=["dpo", "sft", "both"],
                        default="both",
                        help="Output format (default: both).")
    parser.add_argument("--min-score", type=float, default=0.75,
                        help="CLR best_score below this is treated as an "
                             "unverified / rejected completion (default 0.75, "
                             "matching the cache trust threshold).")
    parser.add_argument("--reject-threshold", type=float, default=0.5,
                        help="A CLR trajectory is only 'rejected' if its "
                             "score is below this AND its answer differs from "
                             "the verified chosen (default 0.5 — near-ties "
                             "are NOT labeled rejected).")
    parser.add_argument("--max-pairs-per-query", type=int, default=3,
                        help="Cap DPO pairs per chosen query so one popular "
                             "query cannot dominate the dataset (default 3).")
    args = parser.parse_args(argv)

    chosen = load_chosen(args.trajectories)
    if not chosen:
        print(f"[export_dpo] No verified trajectories found in "
              f"{args.trajectories}. Nothing to export.", file=sys.stderr)
        print("[export_dpo] Tip: run the orchestrator with the trajectory "
              "store enabled to accumulate verified examples first.",
              file=sys.stderr)
        return 1

    rejected_map = load_rejected(args.memory, args.reject_threshold, args.min_score)

    n_dpo = n_sft = 0
    if args.format in ("dpo", "both"):
        pairs = build_dpo_pairs(chosen, rejected_map, args.max_pairs_per_query)
        out_path = (f"{args.out}.dpo.jsonl" if args.format == "both"
                    else (args.out if args.out.endswith(".jsonl")
                          else f"{args.out}.jsonl"))
        n_dpo = _write_jsonl(out_path, pairs)
        print(f"[export_dpo] Wrote {n_dpo} DPO pairs -> {out_path}")

    if args.format in ("sft", "both"):
        examples = build_sft_examples(chosen)
        out_path = (f"{args.out}.sft.jsonl" if args.format == "both"
                    else (args.out if args.out.endswith(".jsonl")
                          else f"{args.out}.jsonl"))
        n_sft = _write_jsonl(out_path, examples)
        print(f"[export_dpo] Wrote {n_sft} SFT examples -> {out_path}")

    matched = sum(1 for c in chosen if rejected_map.get(c["normalized"]))
    print(f"[export_dpo] Summary: {len(chosen)} verified chosen, "
          f"{sum(len(v) for v in rejected_map.values())} rejected completions, "
          f"{matched} chosen queries matched a rejected completion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
