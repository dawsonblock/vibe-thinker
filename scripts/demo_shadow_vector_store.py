#!/usr/bin/env python3
"""Demo: shadow-mode vector DB migration with AgentDB.

Shows how the ShadowVectorStore dual-writes to both a local in-memory
store (the source of truth) and an AgentDB HTTP sidecar (the shadow).
Reads come from the local store first; if it returns nothing, they fall
back to AgentDB. This enables zero-downtime migration:

  1. Start with --agentdb-url (shadow mode: dual-write, local-read)
  2. Verify AgentDB recall matches local recall
  3. Cut over to AgentDB-only (drop the local JSON file)

This demo runs without an actual AgentDB sidecar — it shows the
fail-closed behavior (AgentDB down -> reads from local, writes to
AgentDB are silently skipped with a warning).

Usage:
    python3 scripts/demo_shadow_vector_store.py
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from vector_store import (
    LocalVectorStore,
    AgentDBVectorStore,
    ShadowVectorStore,
    make_vector_store,
)


def main() -> int:
    print("=== Shadow-Mode Vector DB Migration Demo ===\n")

    # 1. The local store (source of truth) — what CLRResultCache uses today.
    print("1. Local store (in-memory, source of truth):")
    local = LocalVectorStore()
    local.upsert("prob_1", [1.0, 0.0, 0.0], {"score": 0.95, "task_type": "math"})
    local.upsert("prob_2", [0.0, 1.0, 0.0], {"score": 0.88, "task_type": "code"})
    local.upsert("prob_3", [0.9, 0.1, 0.0], {"score": 0.91, "task_type": "math"})
    print(f"   Inserted {local.count()} entries into the local store.\n")

    # 2. AgentDB sidecar — NOT running (port 1 = connection refused).
    #    This simulates the migration start: AgentDB is being set up.
    print("2. AgentDB sidecar (not running yet — simulating migration start):")
    agentdb = AgentDBVectorStore("http://127.0.0.1:1", "clr_results")
    print(f"   count() = {agentdb.count()} (fail-closed to 0)")
    print(f"   search() = {agentdb.search([1.0, 0.0, 0.0])} (fail-closed to [])\n")

    # 3. Shadow store — dual-writes to local + AgentDB.
    #    Reads come from local first (the source of truth).
    print("3. Shadow store (dual-write: local + AgentDB):")
    shadow = ShadowVectorStore(local, agentdb)

    # Write a new entry — goes to both local and AgentDB (AgentDB write
    # is silently skipped since the sidecar is down).
    print("   Inserting 'prob_4' via shadow store...")
    shadow.upsert("prob_4", [0.1, 0.9, 0.0], {"score": 0.79, "task_type": "code"})
    print(f"   Local count: {local.count()} (write succeeded)")
    print(f"   AgentDB count: {agentdb.count()} (write skipped — sidecar down)\n")

    # Read — comes from local (primary).
    print("4. Reads (from primary/local first):")
    results = shadow.search([1.0, 0.0, 0.0], top_k=3, filters={"task_type": "math"})
    print(f"   search([1,0,0], filters={{task_type: math}}) -> {len(results)} results:")
    for vid, score, meta in results:
        print(f"     {vid}: sim={score:.3f}, score={meta.get('score')}")
    print()

    # 5. Factory: this is what --agentdb-url builds internally.
    print("5. Factory (make_vector_store with agentdb_url):")
    vs = make_vector_store(
        agentdb_url="http://127.0.0.1:1",
        collection="clr_results",
        shadow_primary=LocalVectorStore(),
    )
    print(f"   Type: {type(vs).__name__}")
    print(f"   This is what --agentdb-url builds: ShadowVectorStore(local, agentdb)\n")

    print("=== Migration path ===")
    print("1. Start with --agentdb-url (shadow mode: dual-write, local-read)")
    print("2. Verify AgentDB recall matches local recall (run both queries, compare)")
    print("3. Cut over to AgentDB-only (drop the local JSON file)")
    print()
    print("When AgentDB is down, reads fall back to local (fail-closed).")
    print("When AgentDB comes back, shadow writes resume automatically.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
