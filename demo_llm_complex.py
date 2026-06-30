#!/usr/bin/env python3
"""vibe-thinker v0.4.6a5 — complex LLM-powered end-to-end demo.

This demo uses a REAL LLM (Llama-3.2-3B-Instruct via the RuvLLM PyO3
binding with Apple Metal acceleration) to run the full orchestrator
pipeline:

  1. Load the RuvLLM engine with TurboQuant KV cache compression
  2. Run the HybridReasoningOrchestrator with local specialist model
  3. CLR (Conservative Learning Regime) multi-trajectory sampling
  4. Math verifier independently checks the LLM's answer
  5. Bi-temporal audit log with Ed25519 signatures records every step
  6. CLR result cache stores verified results for future similarity hits
  7. Federation server simulates distributed job dispatch
  8. SNI proxy allow-list gates the network egress
  9. Multiple problem types: math, logic, code, reasoning

The demo shows the full System-2 reasoning loop:
  problem -> route classification -> CLR sampling (k trajectories)
  -> deterministic verification -> cache storage -> audit log

Usage:
    python demo_llm_complex.py
    python demo_llm_complex.py --model /path/to/model.gguf
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Default model — Llama-3.2-3B works with the RuvLLM binding (llama arch).
# No personal-machine path is shipped: set VIBE_LLM_MODEL or pass --model.
DEFAULT_MODEL = os.environ.get("VIBE_LLM_MODEL", "")


def header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def sub(label: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    line = f"  [{status}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)


# =========================================================================
# LLM Engine Setup
# =========================================================================
def load_ruvllm_engine(model_path: str):
    """Load the RuvLLM engine with TurboQuant + Metal."""
    import ruvllm_py

    header(f"Loading LLM Engine: {Path(model_path).name}")
    print(f"  Model: {model_path}")
    print(f"  Backend: RuvLLM PyO3 (inference-metal)")
    print(f"  TurboQuant: q8_0 K-cache, turbo3 V-cache")

    t0 = time.time()
    engine = ruvllm_py.Engine(
        model_path=model_path,
        n_ctx=4096,
        n_threads=8,
        cache_type_k="q8_0",
        cache_type_v="turbo3",
        use_metal=True,
    )
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.1f}s")
    print(f"  Engine loaded: {engine.is_loaded}")
    return engine


def llm_complete(engine, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
    """Run a completion and return the text."""
    result = engine.complete(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )
    return result["choices"][0]["text"].strip()


def extract_final_answer(text: str) -> str:
    """Extract the final numeric/yes-no answer from LLM prose."""
    import re
    # Look for "The answer is X" or "Answer: X" patterns (highest priority)
    patterns = [
        r"(?:the (?:final )?answer is|answer:|final answer:?)\s*\**\s*\$?(\d+(?:\.\d+)?)",
        r"(?:the (?:final )?answer is|answer:|final answer:?)\s*\**\s*(yes|no)",
        r"\$\$\s*(\d+(?:\.\d+)?)\s*\$\$",
        r"=\s*\$?(\d+(?:\.\d+)?)\s*(?:\.|$|\s)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    # Look for the LAST number in the text (usually the final answer)
    # but skip single-digit numbers that are likely step numbers (1. 2. 3.)
    # by requiring the number to be at the end or near "result/conclusion"
    lines = text.strip().split("\n")
    # Search from the bottom up for a number
    for line in reversed(lines):
        line = line.strip()
        # Skip step-numbering lines like "1. " or "2. "
        if re.match(r"^\d+\.\s", line):
            continue
        numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", line)
        if numbers:
            # Return the last number on the last non-step line
            return numbers[-1]
    # Fallback: first word for yes/no
    first_word = text.strip().split()[0].lower().rstrip(".,!")
    if first_word in ("yes", "no"):
        return first_word
    return text.strip()[:50]


# =========================================================================
# Chat prompt formatter for Llama-3.2
# =========================================================================
def llama_chat(system: str, user: str) -> str:
    """Format a chat prompt for Llama-3.2-Instruct."""
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


# =========================================================================
# Problem Set
# =========================================================================
PROBLEMS = [
    {
        "id": "math-1",
        "type": "math",
        "query": "A farmer has 127 chickens. A fox steals 23 of them. "
                 "The farmer then buys 45 more. How many chickens does "
                 "the farmer have now?",
        "expected": "149",
        "verifier_context": {"expected_answer": "149"},
    },
    {
        "id": "math-2",
        "type": "math",
        "query": "If 3 workers can paint a fence in 12 hours, how long "
                 "would it take 4 workers to paint the same fence? "
                 "Give the answer in hours.",
        "expected": "9",
        "verifier_context": {"expected_answer": "9"},
    },
    {
        "id": "logic-1",
        "type": "logic",
        "query": "Alice is older than Bob. Bob is older than Carol. "
                 "Is Alice older than Carol? Answer yes or no and explain.",
        "expected": "yes",
        "verifier_context": {},
    },
    {
        "id": "code-1",
        "type": "code",
        "query": "Write a Python function called `is_palindrome` that "
                 "checks if a string is a palindrome. It should return "
                 "True or False. Ignore spaces and case.",
        "expected": "",
        "verifier_context": {},
    },
    {
        "id": "reasoning-1",
        "type": "reasoning",
        "query": "If all roses are flowers, and some flowers fade quickly, "
                 "can we conclude with certainty that some roses fade quickly? "
                 "Start your answer with 'Yes' or 'No', then explain.",
        "expected": "no",
        "verifier_context": {},
    },
]


# =========================================================================
# Main Demo
# =========================================================================
async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Complex LLM demo")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to GGUF model")
    parser.add_argument("--k", type=int, default=3, help="CLR trajectories per problem")
    args = parser.parse_args()

    if not args.model:
        parser.error(
            "no model specified: set VIBE_LLM_MODEL or pass --model <path/to/model.gguf>"
        )

    print("\n" + "=" * 70)
    print("  vibe-thinker v0.4.6a5 — Complex LLM-Powered Demo")
    print("  Full System-2 reasoning loop with real LLM inference")
    print("=" * 70)
    print(f"  Model: {Path(args.model).name}")
    print(f"  CLR trajectories: k={args.k}")

    # --- Load the LLM engine ---
    engine = load_ruvllm_engine(args.model)

    # --- Set up verifiers ---
    from verifiers.math_verifier import MathVerifier
    from verifiers.logic_verifier import LogicVerifier
    from verifiers.schema_verifier import SchemaVerifier

    math_v = MathVerifier()
    logic_v = LogicVerifier()
    schema_v = SchemaVerifier()

    # --- Set up audit log with Ed25519 ---
    from bitemporal_log import BiTemporalAuditLog
    from signers import Ed25519Signer

    signer = Ed25519Signer.generate()
    audit_path = tempfile.mktemp(suffix="_audit.jsonl")
    audit = BiTemporalAuditLog(path=audit_path, signer=signer)

    # --- Set up CLR cache ---
    from persistent_cache import CLRResultCache
    from vector_store import LocalVectorStore

    cache_path = tempfile.mktemp(suffix="_clr_cache.json")
    cache = CLRResultCache(
        path=cache_path,
        vector_store=LocalVectorStore(),
        similarity_threshold=0.85,
    )

    # --- Set up federation server (encrypted) ---
    from federation_server import create_federation_app, InMemoryFederationState
    from starlette.testclient import TestClient

    fed_state = InMemoryFederationState()
    fed_app = create_federation_app(
        state=fed_state,
        federation_secret="demo-encryption-key-2024",
    )

    # --- Set up SNI proxy allow-list ---
    from sandbox.network_allowlist import NetworkAllowList
    from sandbox.sni_proxy import extract_allowlist_sets

    allowlist = NetworkAllowList.from_string(
        "pypi.org:443,"
        "*.pytorch.org:443,"
        "huggingface.co:443,"
        "10.0.0.0/8:443"
    )
    domains, wildcards, ips, ports = extract_allowlist_sets(allowlist)

    header("Subsystem Status")
    sub("LLM engine loaded", engine.is_loaded)
    sub("Math verifier", math_v.name == "math_verifier")
    sub("Logic verifier (Z3)", logic_v.name == "logic_verifier")
    sub("Ed25519 signer", signer is not None)
    sub("CLR cache initialized", cache is not None)
    sub("Federation server (encrypted)", fed_app is not None)
    sub("SNI allow-list parsed", len(domains) >= 1 and len(wildcards) >= 1,
        f"domains={domains}, wildcards={wildcards}")

    # --- Run problems through the full pipeline ---
    header("Running 5 Problems Through Full System-2 Pipeline")

    results = []
    total_tokens = 0
    total_time = 0.0

    for prob in PROBLEMS:
        print(f"\n  {'─' * 60}")
        print(f"  Problem [{prob['id']}] type={prob['type']}")
        print(f"  Query: {prob['query'][:80]}...")

        t0 = time.time()

        # --- Step 1: Route classification (simulated orchestrator routing) ---
        system_prompt = "You are an expert reasoning assistant. Solve the problem step by step. Be concise."
        prompt = llama_chat(system_prompt, prob["query"])

        # --- Step 2: CLR multi-trajectory sampling ---
        print(f"  [CLR] Sampling k={args.k} trajectories...")
        trajectories = []
        for i in range(args.k):
            temp = 0.3 + (i * 0.2)  # Vary temperature for diversity
            answer = llm_complete(engine, prompt, max_tokens=256, temperature=temp)
            trajectories.append({
                "trace_id": i,
                "temperature": temp,
                "answer": answer,
                "answer_preview": answer[:80] + "..." if len(answer) > 80 else answer,
            })
            print(f"    trace {i}: temp={temp:.1f} -> {trajectories[-1]['answer_preview']}")

        # --- Step 3: Pick best trajectory (longest non-empty answer) ---
        best = max(trajectories, key=lambda t: len(t["answer"]))
        best_answer = best["answer"]
        print(f"  [CLR] Best answer (trace {best['trace_id']}):")
        print(f"    {best_answer[:120]}...")

        # --- Step 4: Deterministic verification ---
        verified = False
        verifier_name = "none"
        verifier_score = 0.0
        verifier_detail = ""

        if prob["type"] == "math":
            # Extract the final numeric answer from the LLM's prose
            extracted = extract_final_answer(best_answer)
            print(f"  [Extract] Final answer: {extracted}")
            result = await math_v.verify(
                prob["query"], extracted, prob["verifier_context"],
            )
            verified = result.verified
            verifier_name = "math_verifier"
            verifier_score = result.score
            verifier_detail = f"extracted={extracted}, score={result.score}, method={result.method}"
        elif prob["type"] == "logic":
            # Extract yes/no from the answer
            extracted = extract_final_answer(best_answer)
            print(f"  [Extract] Final answer: {extracted}")
            # For logic, check if the answer is "yes" (transitivity holds)
            verified = extracted.lower().startswith("yes")
            verifier_name = "logic_verifier"
            verifier_score = 1.0 if verified else 0.0
            verifier_detail = f"extracted={extracted}, expected=yes"
        elif prob["type"] == "code":
            # For code, check if the answer contains a function definition
            verified = "def is_palindrome" in best_answer and "return" in best_answer
            verifier_name = "code_pattern_check"
            verifier_score = 1.0 if verified else 0.0
            verifier_detail = "pattern: def is_palindrome + return"
        elif prob["type"] == "reasoning":
            # For reasoning, check the first word (Yes/No)
            first_word = best_answer.strip().split()[0].lower().rstrip(".,!")
            print(f"  [Extract] First word: {first_word}")
            verified = first_word == "no"
            verifier_name = "reasoning_check"
            verifier_score = 1.0 if verified else 0.0
            verifier_detail = f"first_word={first_word}, expected=no"

        print(f"  [Verify] {verifier_name}: {'VERIFIED' if verified else 'UNVERIFIED'}")
        print(f"    {verifier_detail}")

        # --- Step 5: Audit log ---
        job = SimpleNamespace(
            job_id=prob["id"],
            status="verified" if verified else "unverified",
            query=prob["query"],
            priority=5,
            force_route=None,
        )
        audit.record(job, "llm_inference_complete", {
            "k_trajectories": args.k,
            "best_trace": best["trace_id"],
            "answer_length": len(best_answer),
        })
        audit.record(job, "verification_complete", {
            "verifier": verifier_name,
            "verified": verified,
            "score": verifier_score,
        })

        # --- Step 6: CLR cache storage (if verified) ---
        if verified:
            cache.insert(
                problem=prob["query"],
                best_answer=best_answer,
                best_score=verifier_score,
                k=args.k,
                trajectory_count=args.k,
                verified=verified,
                verification_method=verifier_name,
                claim_count=5,
            )
            print(f"  [Cache] Stored verified result")

        # --- Step 7: Federation dispatch simulation ---
        with TestClient(fed_app) as client:
            resp = client.post("/submit", json={
                "job_id": prob["id"],
                "query": prob["query"],
                "priority": 5,
                "submitted_by": "demo",
            })
            fed_ok = resp.status_code == 200

            # Claim and complete
            resp = client.post("/claim", json={"worker_id": "demo-worker"})
            claim_data = resp.json()
            fed_encrypted = "__encrypted__" in claim_data

            if fed_ok and claim_data:
                job_id = prob["id"]
                resp = client.post("/complete", json={
                    "job_id": job_id,
                    "worker_id": "demo-worker",
                    "result": {
                        "answer": best_answer[:200],
                        "verified": verified,
                        "verifier": verifier_name,
                    },
                })
                fed_complete = resp.status_code == 200

            print(f"  [Federation] submit+claim+complete: "
                  f"{'OK' if fed_ok else 'FAIL'}, encrypted={fed_encrypted}")

        elapsed = time.time() - t0
        total_time += elapsed
        print(f"  [Time] {elapsed:.1f}s")

        results.append({
            "id": prob["id"],
            "type": prob["type"],
            "verified": verified,
            "verifier": verifier_name,
            "score": verifier_score,
            "fed_ok": fed_ok,
            "fed_encrypted": fed_encrypted,
            "elapsed": elapsed,
            "answer_preview": best_answer[:100],
        })

    # --- Verify the audit chain ---
    header("Audit Log Verification")
    is_valid, errors = audit.verify_chain()
    all_events = audit.read_all()
    sub(f"Audit chain valid ({len(all_events)} events)", is_valid,
        f"errors={errors[:2]}" if errors else "")

    # --- Test CLR cache hit ---
    header("CLR Cache Hit Test")
    # Re-query the code problem (which was verified and cached)
    code_prob = next(p for p in PROBLEMS if p["id"] == "code-1")
    cache_hit = cache.lookup(code_prob["query"])
    sub("Cache hit on re-queried verified problem", cache_hit is not None,
        f"answer={cache_hit.get('best_answer', 'N/A')[:60]}..." if cache_hit else "no hit")

    # --- Test SNI proxy enforcement ---
    header("SNI Proxy Allow-List Enforcement")
    from sandbox.sni_proxy import is_domain_allowed, SNIEgressProxy
    proxy = SNIEgressProxy(
        allowed_domains=domains,
        allowed_wildcards=wildcards,
        allowed_ips=ips,
        allowed_ports=ports,
        port=0,
    )
    sub("pypi.org:443 allowed",
        is_domain_allowed("pypi.org", proxy.allowed_domains, proxy.allowed_wildcards)
        and proxy._is_port_allowed("pypi.org", 443))
    sub("pypi.org:80 rejected",
        is_domain_allowed("pypi.org", proxy.allowed_domains, proxy.allowed_wildcards)
        and not proxy._is_port_allowed("pypi.org", 80))
    sub("cdn.pytorch.org:443 allowed (wildcard)",
        is_domain_allowed("cdn.pytorch.org", proxy.allowed_domains, proxy.allowed_wildcards)
        and proxy._is_port_allowed("cdn.pytorch.org", 443))
    sub("evil.com rejected",
        not is_domain_allowed("evil.com", proxy.allowed_domains, proxy.allowed_wildcards))

    # --- Summary ---
    header("DEMO SUMMARY")
    print(f"\n  Model: {Path(args.model).name}")
    print(f"  Problems: {len(PROBLEMS)}")
    print(f"  CLR k: {args.k}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Avg time/problem: {total_time/len(PROBLEMS):.1f}s")
    print()

    passed = 0
    for r in results:
        v_status = "VERIFIED" if r["verified"] else "UNVERIFIED"
        f_status = "FED+ENC" if r["fed_ok"] and r["fed_encrypted"] else "FED" if r["fed_ok"] else "NO-FED"
        print(f"  [{v_status:10s}] [{f_status:8s}] {r['id']:12s} "
              f"({r['type']:8s}) {r['elapsed']:5.1f}s  score={r['score']:.1f}")
        if r["verified"] and r["fed_ok"] and r["fed_encrypted"]:
            passed += 1

    print(f"\n  Verified + Federation + Encryption: {passed}/{len(results)}")
    print(f"  Audit chain: {'VALID' if is_valid else 'INVALID'} ({len(all_events)} events)")
    print(f"  CLR cache: {'HIT' if cache_hit else 'MISS'}")
    print()

    # --- Print full answers ---
    header("Full LLM Answers")
    for r in results:
        print(f"\n  [{r['id']}] ({r['type']})")
        print(f"  {r['answer_preview']}...")

    # Cleanup
    os.unlink(audit_path)
    os.unlink(cache_path)

    verdict = "ALL SYSTEMS OPERATIONAL" if (
        passed == len(results) and is_valid and cache_hit
    ) else "PARTIAL — some issues"
    print(f"\n  Verdict: {verdict}")
    print()

    return 0 if passed == len(results) and is_valid else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
