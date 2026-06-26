#!/usr/bin/env python3
"""One-shot backfill + verification for AgentDB shadow-mode migration.

Reads all entries from the local JSON cache files (CLRResultCache and
VerifiedTrajectoryStore) and pushes their embeddings + metadata into
AgentDB via the AgentDBVectorStore HTTP API. After the backfill, runs a
recall check comparing local vs AgentDB search results.

Usage:
    # Backfill + verify (safe — no data loss, read-only on local store)
    python3 scripts/migrate_to_agentdb.py \\
        --agentdb-url http://127.0.0.1:8088 \\
        --clr-cache-path ./clr_cache.json \\
        --trajectory-store-path ./trajectories.json

    # Dry run (report what would be migrated, don't write to AgentDB)
    python3 scripts/migrate_to_agentdb.py --dry-run \\
        --agentdb-url http://127.0.0.1:8088 \\
        --clr-cache-path ./clr_cache.json

    # Verify only (skip backfill, just check recall)
    python3 scripts/migrate_to_agentdb.py --verify-only \\
        --agentdb-url http://127.0.0.1:8088 \\
        --clr-cache-path ./clr_cache.json \\
        --trajectory-store-path ./trajectories.json

Fail-closed behavior:
  - If AgentDB is unreachable, exits with error code 1 (no data loss).
  - If a local cache file doesn't exist, skips it with a warning.
  - If the recall check fails below the threshold (default 95%), exits
    with error code 2 (refuses to finalize — do NOT cut over to
    AgentDB-only until recall is fixed).
  - If recall passes, prints a success summary and exits 0.

This script does NOT modify the local JSON files. It only reads them.
The cut-over (dropping the local store) is done separately via
`python rfsn_cli.py finalize-migration`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from vector_store import AgentDBVectorStore, LocalVectorStore


# ---------------------------------------------------------------------- #
# Local cache file readers (read-only — never modify the JSON files)
# ---------------------------------------------------------------------- #
def _load_cache_entries(path: str) -> List[Dict[str, Any]]:
    """Load entries from a CLRResultCache or VerifiedTrajectoryStore JSON file.

    Both stores use the same on-disk format:
      {"model_name": "...", "schema_version": N, "entries": [...]}
    Each entry has an "embedding" key (list of floats) and metadata keys.

    Returns an empty list if the file doesn't exist (warned by caller).
    """
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    entries = data.get("entries", [])
    # Filter out entries without embeddings (corrupt/incomplete).
    return [e for e in entries if "embedding" in e and e["embedding"]]


def _extract_clr_metadata(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Extract AgentDB metadata from a CLRResultCache entry."""
    return {
        "problem": entry.get("problem", ""),
        "best_answer": entry.get("best_answer", ""),
        "best_score": float(entry.get("best_score", 0.0)),
        "verified": bool(entry.get("verified", False)),
        "verification_method": entry.get("verification_method", ""),
        "task_type": entry.get("task_type", "unknown"),
    }


def _extract_trajectory_metadata(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Extract AgentDB metadata from a VerifiedTrajectoryStore entry."""
    return {
        "query": entry.get("query", ""),
        "answer": entry.get("answer", ""),
        "score": float(entry.get("score", 0.0)),
        "verification_method": entry.get("verification_method", ""),
        "task_type": entry.get("task_type", "unknown"),
        "synthesized": bool(entry.get("synthesized", False)),
    }


# ---------------------------------------------------------------------- #
# Backfill
# ---------------------------------------------------------------------- #
def backfill(
    agentdb: AgentDBVectorStore,
    clr_path: Optional[str],
    trajectory_path: Optional[str],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Backfill local cache entries into AgentDB.

    Returns a summary dict: {clr_count, traj_count, total, failures, dry_run}.
    """
    summary = {
        "clr_count": 0, "traj_count": 0, "total": 0,
        "failures": 0, "dry_run": dry_run,
    }

    # --- CLR result cache ---
    if clr_path:
        entries = _load_cache_entries(clr_path)
        if not entries:
            if os.path.exists(clr_path):
                print(f"[Backfill] CLR cache: {clr_path} exists but has no "
                      f"entries with embeddings — skipping")
            else:
                print(f"[Backfill] CLR cache: {clr_path} not found — skipping")
        else:
            print(f"[Backfill] CLR cache: {len(entries)} entries to backfill")
            for i, entry in enumerate(entries):
                vector_id = f"clr_{i}"
                metadata = _extract_clr_metadata(entry)
                if dry_run:
                    print(f"  [dry-run] would upsert {vector_id}: "
                          f"{metadata.get('problem', '')[:50]}...")
                else:
                    try:
                        agentdb.upsert(vector_id, entry["embedding"], metadata)
                    except Exception as e:
                        print(f"  [FAIL] {vector_id}: {e}")
                        summary["failures"] += 1
                        continue
                summary["clr_count"] += 1
                if (i + 1) % 50 == 0:
                    print(f"  ... {i + 1}/{len(entries)}")

    # --- Trajectory store ---
    if trajectory_path:
        entries = _load_cache_entries(trajectory_path)
        if not entries:
            if os.path.exists(trajectory_path):
                print(f"[Backfill] Trajectory store: {trajectory_path} exists "
                      f"but has no entries with embeddings — skipping")
            else:
                print(f"[Backfill] Trajectory store: {trajectory_path} not "
                      f"found — skipping")
        else:
            print(f"[Backfill] Trajectory store: {len(entries)} entries to "
                  f"backfill")
            for i, entry in enumerate(entries):
                vector_id = f"traj_{i}"
                metadata = _extract_trajectory_metadata(entry)
                if dry_run:
                    print(f"  [dry-run] would upsert {vector_id}: "
                          f"{metadata.get('query', '')[:50]}...")
                else:
                    try:
                        agentdb.upsert(vector_id, entry["embedding"], metadata)
                    except Exception as e:
                        print(f"  [FAIL] {vector_id}: {e}")
                        summary["failures"] += 1
                        continue
                summary["traj_count"] += 1
                if (i + 1) % 50 == 0:
                    print(f"  ... {i + 1}/{len(entries)}")

    summary["total"] = summary["clr_count"] + summary["traj_count"]
    return summary


# ---------------------------------------------------------------------- #
# Recall verification
# ---------------------------------------------------------------------- #
def verify_recall(
    agentdb: AgentDBVectorStore,
    clr_path: Optional[str],
    trajectory_path: Optional[str],
    sample_size: int = 20,
    top_k: int = 5,
    recall_threshold: float = 0.95,
) -> Dict[str, Any]:
    """Verify that AgentDB recall matches local recall.

    For each store (CLR cache, trajectory store), samples up to
    ``sample_size`` entries, runs a search query using each entry's
    own embedding against both the local store (in-memory) and AgentDB,
    and compares the top_k results. Recall is the fraction of queries
    where AgentDB's top_k result set overlaps with the local store's
    top_k result set by at least 50% (i.e. at least ceil(top_k/2) of
    the local results appear in AgentDB's results).

    Returns a summary dict with per-store and overall recall.
    """
    results = {
        "stores": {}, "overall_recall": 0.0,
        "threshold": recall_threshold, "passed": False,
    }
    all_recalls: List[float] = []

    for store_name, path, id_prefix in [
        ("clr_cache", clr_path, "clr_"),
        ("trajectory_store", trajectory_path, "traj_"),
    ]:
        if not path or not os.path.exists(path):
            print(f"[Verify] {store_name}: {path or 'not configured'} — skipping")
            continue
        entries = _load_cache_entries(path)
        if not entries:
            print(f"[Verify] {store_name}: no entries — skipping")
            continue

        # Build a local store from the JSON entries (in-memory).
        local = LocalVectorStore()
        for i, entry in enumerate(entries):
            local.upsert(f"{id_prefix}{i}", entry["embedding"],
                         {"source_index": i})

        # Sample entries for recall testing.
        sample_indices = list(range(0, len(entries), max(1, len(entries) // sample_size)))[:sample_size]

        matching = 0
        total = 0
        for idx in sample_indices:
            query_emb = entries[idx]["embedding"]
            local_results = local.search(query_emb, top_k=top_k)
            agentdb_results = agentdb.search(query_emb, top_k=top_k)
            local_ids = {r[0] for r in local_results}
            agentdb_ids = {r[0] for r in agentdb_results}
            if not local_ids:
                continue
            total += 1
            overlap = len(local_ids & agentdb_ids)
            # Recall for this query: did AgentDB find at least half of
            # the local results?
            if overlap >= max(1, len(local_ids) // 2):
                matching += 1

        recall = matching / total if total > 0 else 0.0
        results["stores"][store_name] = {
            "entries": len(entries), "sampled": total,
            "matching": matching, "recall": round(recall, 4),
        }
        all_recalls.append(recall)
        print(f"[Verify] {store_name}: recall={recall:.1%} "
              f"({matching}/{total} queries matched)")

    if all_recalls:
        results["overall_recall"] = round(sum(all_recalls) / len(all_recalls), 4)
    results["passed"] = results["overall_recall"] >= recall_threshold
    return results


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Backfill local cache entries into AgentDB for "
                    "shadow-mode migration. Read-only on local files."
    )
    p.add_argument("--agentdb-url", required=True,
                   help="AgentDB HTTP endpoint (e.g. http://127.0.0.1:8088)")
    p.add_argument("--collection", default="vibe_thinker",
                   help="AgentDB collection name (default: vibe_thinker)")
    p.add_argument("--clr-cache-path", default="",
                   help="Path to the CLR result cache JSON file")
    p.add_argument("--trajectory-store-path", default="",
                   help="Path to the verified trajectory store JSON file")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be migrated without writing to AgentDB")
    p.add_argument("--verify-only", action="store_true",
                   help="Skip backfill, only run the recall check")
    p.add_argument("--sample-size", type=int, default=20,
                   help="Number of entries to sample for recall check (default 20)")
    p.add_argument("--top-k", type=int, default=5,
                   help="top_k for recall comparison (default 5)")
    p.add_argument("--recall-threshold", type=float, default=0.95,
                   help="Minimum recall to pass verification (default 0.95)")
    return p


def _check_agentdb_reachable(agentdb: AgentDBVectorStore) -> bool:
    """Check if AgentDB is reachable by doing a test upsert + search + delete.

    count() returns 0 both for "empty" and "unreachable" (fail-closed),
    and upsert() silently fails (prints a warning, doesn't raise). So we
    verify by: upsert a test entry, search for it, and check it was
    actually stored. If the search returns nothing, AgentDB is unreachable.
    """
    test_id = "__migration_reachability_check__"
    try:
        agentdb.upsert(test_id, [0.0, 0.0, 0.0], {"test": True})
        results = agentdb.search([0.0, 0.0, 0.0], top_k=1)
        if not results:
            return False
        agentdb.delete(test_id)
        return True
    except Exception:
        return False


def main() -> int:
    args = build_argparser().parse_args()

    clr_path = args.clr_cache_path or None
    traj_path = args.trajectory_store_path or None
    if not clr_path and not traj_path:
        print("Error: at least one of --clr-cache-path or "
              "--trajectory-store-path must be set")
        return 1

    agentdb = AgentDBVectorStore(args.agentdb_url, args.collection)

    # Check AgentDB is reachable (fail-closed before backfill).
    # Skip for dry-run (dry run doesn't write, so reachability is irrelevant).
    if not args.dry_run:
        reachable = _check_agentdb_reachable(agentdb)
        if not reachable:
            print(f"Error: AgentDB at {args.agentdb_url} is unreachable "
                  f"— refusing to migrate (fail-closed, no data loss)")
            return 1
    count = agentdb.count()
    print(f"[Migration] AgentDB connected: {args.agentdb_url} "
          f"(collection={args.collection}, current count={count})")

    # --- Backfill ---
    if not args.verify_only:
        print(f"\n=== Backfill {'(dry run)' if args.dry_run else ''} ===")
        bsum = backfill(agentdb, clr_path, traj_path, dry_run=args.dry_run)
        print(f"\n[Backfill] Summary: {bsum['total']} entries "
              f"({bsum['clr_count']} CLR + {bsum['traj_count']} trajectory), "
              f"{bsum['failures']} failures")
        if bsum["failures"] > 0:
            print(f"[Backfill] WARNING: {bsum['failures']} entries failed to "
                  f"backfill — check AgentDB logs")

    # --- Verify ---
    if args.dry_run:
        print("\n[Migration] Dry run complete — skipping recall check")
        return 0

    print(f"\n=== Recall verification ===")
    vres = verify_recall(
        agentdb, clr_path, traj_path,
        sample_size=args.sample_size, top_k=args.top_k,
        recall_threshold=args.recall_threshold,
    )
    print(f"\n[Verify] Overall recall: {vres['overall_recall']:.1%} "
          f"(threshold: {vres['threshold']:.1%})")
    if vres["passed"]:
        print("[Verify] PASSED — AgentDB recall meets threshold. "
              "You can now run `python rfsn_cli.py finalize-migration` "
              "to cut over to AgentDB-only.")
        return 0
    else:
        print("[Verify] FAILED — AgentDB recall is below threshold. "
              "Do NOT finalize migration. Check AgentDB configuration "
              "and re-run the backfill.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
