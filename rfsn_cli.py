"""
Interactive CLI / REPL front-end for the RFSN job queue.

Sits on top of JobQueue + HybridReasoningOrchestrator and gives you an
interactive prompt for submitting queries, watching jobs, inspecting
results, and querying the bi-temporal audit log — all while the async
dispatcher runs in the background.

Usage:
    python rfsn_cli.py
    python rfsn_cli.py --vibe http://127.0.0.1:8080 --generalist http://127.0.0.1:8081
    python rfsn_cli.py --max-concurrent 3 --no-clr

Commands (type `help` in the REPL):
    submit QUERY...        submit a job (aliases: s)
        flags:  -p N       priority (default 0)
                -r ROUTE   force route: specialist|generalist|hybrid
    list                   list all jobs and their status (aliases: ls)
    status JOB_ID          show details for one job
    result JOB_ID          print the final answer for a finished job
    wait JOB_ID            block until the job finishes, then print result
    cancel JOB_ID          cancel a pending job
    history JOB_ID         bi-temporal event history for a job
        flags:  --axis valid|transaction   (default valid)
    asof JOB_ID TIME       state of a job as of an ISO timestamp
        flags:  --axis valid|transaction   (default valid)
    log-state              reconstruct current state of all jobs from the log
    help                   show this help
    quit / exit            stop the queue and exit

Notes:
  - Multi-word queries can be typed verbatim after `submit`:
        submit Solve a_1=2, a_{n+1}=a_n^2-a_n+1, find a_5 -p 5
  - Flags may appear anywhere after the command word.
  - The REPL keeps running while jobs execute; you don't have to wait.
"""

import argparse
import asyncio
import os
import shlex

from hybrid_orchestrator import HybridReasoningOrchestrator
from rfsn_job_queue import JobQueue
from federated_queue import make_job_queue

# Optional: load .env file if python-dotenv is installed.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable.

    Accepts: true/false, 1/0, yes/no (case-insensitive).
    """
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes", "on")


def _build_retrieval_backend(args) -> "object | None":
    """Build a retrieval backend from CLI args / env vars.

    Precedence: --serper-key > --searchapi-key > SERPER_API_KEY env >
    SEARCHAPI_API_KEY env. Returns None when no key is configured (the
    orchestrator then skips retrieval — fail-closed, unchanged behavior).
    """
    from retrieval import make_retrieval_backend
    serper = args.serper_key or None
    searchapi = args.searchapi_key or None
    backend = make_retrieval_backend(serper_key=serper, searchapi_key=searchapi)
    if backend is not None:
        print(f"[CLI] Active retrieval enabled: {backend.name} — "
              f"factual tasks will fetch real sources for NLI verification")
    return backend


# ----------------------------- command parsing ----------------------------- #
def _split_flags(tokens):
    """Pull leading -p/-r style flags out of a token list.

    Returns (flags_dict, remaining_tokens).
    """
    flags = {}
    rest = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("-p", "--priority"):
            if i + 1 < len(tokens):
                flags["priority"] = int(tokens[i + 1])
                i += 2
                continue
        elif tok in ("-r", "--route", "--force-route"):
            if i + 1 < len(tokens):
                flags["force_route"] = tokens[i + 1]
                i += 2
                continue
        elif tok in ("--axis",):
            if i + 1 < len(tokens):
                flags["axis"] = tokens[i + 1]
                i += 2
                continue
        rest.append(tok)
        i += 1
    return flags, rest


HELP = """\
Commands:
  submit QUERY...        submit a job  (alias: s)
      -p N               priority (default 0)
      -r ROUTE           force route: specialist|generalist|hybrid
  list                   list all jobs  (alias: ls)
  status JOB_ID          show details for one job
  result JOB_ID          print the final answer for a finished job
  wait JOB_ID            block until the job finishes, then print result
  cancel JOB_ID          cancel a pending job
  history JOB_ID         bi-temporal event history for a job
      --axis valid|transaction   (default valid)
  asof JOB_ID TIME       state of a job as of an ISO timestamp
      --axis valid|transaction   (default valid)
  log-state              reconstruct current state of all jobs from the log
  help                   show this help
  quit / exit            stop the queue and exit
"""


# ----------------------------- the REPL ----------------------------- #
class JobQueueREPL:
    def __init__(self, queue: JobQueue):
        self.queue = queue

    async def arepl(self) -> None:
        loop = asyncio.get_running_loop()
        print("RFSN job queue REPL. Type 'help' for commands, 'quit' to exit.")
        while True:
            try:
                # input() blocks; run it in a thread so the dispatcher keeps going.
                line = await loop.run_in_executor(None, lambda: input("rfsn> "))
            except (EOFError, KeyboardInterrupt):
                print()
                break
            line = line.strip()
            if not line:
                continue
            try:
                await self._dispatch(line)
            except SystemExit:
                raise
            except Exception as e:
                print(f"error: {e}")

    async def _dispatch(self, line: str) -> None:
        tokens = shlex.split(line)
        if not tokens:
            return
        cmd = tokens[0].lower()
        args = tokens[1:]

        if cmd in ("quit", "exit", "q"):
            raise SystemExit(0)
        if cmd in ("help", "h", "?"):
            print(HELP)
            return

        if cmd in ("submit", "s"):
            flags, rest = _split_flags(args)
            if not rest:
                print("usage: submit QUERY... [-p N] [-r ROUTE]")
                return
            query = " ".join(rest)
            job = self.queue.submit(
                query,
                priority=flags.get("priority", 0),
                force_route=flags.get("force_route"),
            )
            print(f"submitted job {job.job_id} (priority={job.priority}"
                  + (f", force_route={job.force_route}" if job.force_route else "")
                  + ")")
            return

        if cmd in ("list", "ls"):
            jobs = self.queue.list_jobs()
            if not jobs:
                print("(no jobs)")
                return
            print(f"{'job_id':14} {'status':10} {'pri':>3} {'route':16} query")
            for j in jobs:
                q = (j["query"][:48] + "...") if len(j["query"]) > 48 else j["query"]
                route = j.get("force_route") or "-"
                print(f"{j['job_id']:14} {j['status']:10} {j['priority']:>3} "
                      f"{route:16} {q}")
            return

        if cmd == "status":
            if not args:
                print("usage: status JOB_ID")
                return
            job = self.queue.get(args[0])
            if job is None:
                print(f"no such job: {args[0]}")
                return
            import json as _json
            print(_json.dumps(job.to_dict(), indent=2))
            return

        if cmd == "result":
            if not args:
                print("usage: result JOB_ID")
                return
            job = self.queue.get(args[0])
            if job is None:
                print(f"no such job: {args[0]}")
                return
            if job.result is None:
                print(f"job {job.job_id} has no result yet (status={job.status.value})")
                if job.error:
                    print(f"error: {job.error}")
                return
            print(job.result.final_answer)
            return

        if cmd == "wait":
            if not args:
                print("usage: wait JOB_ID")
                return
            jid = args[0]
            timeout = float(args[1]) if len(args) > 1 else None
            print(f"waiting for {jid}...")
            try:
                result = await self.queue.wait_for(jid, timeout=timeout)
            except (TimeoutError, RuntimeError, KeyError) as e:
                print(f"wait failed: {e}")
                return
            print(result.final_answer)
            return

        if cmd == "cancel":
            if not args:
                print("usage: cancel JOB_ID")
                return
            ok = self.queue.cancel(args[0])
            print("cancelled" if ok else f"could not cancel {args[0]} "
                  "(not pending or unknown)")
            return

        if cmd == "history":
            flags, rest = _split_flags(args)
            if not rest:
                print("usage: history JOB_ID [--axis valid|transaction]")
                return
            axis = flags.get("axis", "valid")
            rows = self.queue.job_history(rest[0], axis=axis)
            if not rows:
                print(f"(no history for {rest[0]})")
                return
            print(f"history for {rest[0]} (axis={axis}):")
            for e in rows:
                extra = e.get("extra", {})
                tail = f"  {extra}" if extra else ""
                print(f"  {e['valid_time']}  {e['event']:10} -> {e['status']}{tail}")
            return

        if cmd == "asof":
            flags, rest = _split_flags(args)
            if len(rest) < 2:
                print("usage: asof JOB_ID TIME [--axis valid|transaction]")
                return
            axis = flags.get("axis", "valid")
            e = self.queue.state_as_of(rest[0], rest[1], axis=axis)
            if e is None:
                print(f"(no state for {rest[0]} as of {rest[1]})")
                return
            import json as _json
            print(_json.dumps(e, indent=2))
            return

        if cmd in ("log-state", "logstate"):
            if self.queue.bitemporal is None:
                print("bi-temporal log disabled")
                return
            axis = "valid"
            if args and args[0] == "--axis" and len(args) > 1:
                axis = args[1]
            state = self.queue.bitemporal.current_state(axis=axis)
            if not state:
                print("(log empty)")
                return
            print(f"{'job_id':14} {'status':10} {'event':10} {'valid_time':27} query")
            for jid, e in state.items():
                q = (e["query"][:40] + "...") if len(e["query"]) > 40 else e["query"]
                print(f"{jid:14} {e['status']:10} {e['event']:10} "
                      f"{e['valid_time']:27} {q}")
            return

        print(f"unknown command: {cmd}  (type 'help')")


# ----------------------------- entry point ----------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RFSN job queue REPL.")
    # Precedence: CLI flags > environment variables > defaults
    p.add_argument("--vibe",
                   default=os.environ.get("VIBE_THINKER_URL", "http://127.0.0.1:8080"),
                   help="VibeThinker specialist endpoint")
    p.add_argument("--generalist",
                   default=os.environ.get("GENERALIST_URL", "http://127.0.0.1:8081"),
                   help="Generalist model endpoint")
    p.add_argument("--code-specialist",
                   default=os.environ.get("CODE_SPECIALIST_URL", ""),
                   help="Dedicated code-specialist endpoint (e.g. ruvltra on "
                        "8082). Empty disables code-specialist routing.")
    p.add_argument("--code-candidates", type=int,
                   default=int(os.environ.get("CODE_CANDIDATES", "6")),
                   help="Number of parallel code candidates to generate and "
                        "verify in the sandbox (default 6 — higher for fast "
                        "0.5B models, lower for slower models).")
    p.add_argument("--max-repair-attempts", type=int,
                   default=int(os.environ.get("MAX_REPAIR_ATTEMPTS", "2")),
                   help="When a code candidate fails with a real bug "
                        "(ASSERTION_FAILED/IMPORT_ERROR), feed the failing "
                        "code + error back to the code specialist for a "
                        "targeted repair. Max repair rounds (default 2, 0 "
                        "disables). Fail-closed: unverified if no repair "
                        "passes.")
    p.add_argument("--no-trajectory-store", dest="use_trajectory_store",
                   action="store_false",
                   help="Disable the verified-trajectory store (self-improving "
                        "few-shot memory). Enabled by default when embedding "
                        "deps are available.")
    p.set_defaults(use_trajectory_store=_env_bool("RFSN_USE_TRAJECTORY_STORE", True))
    p.add_argument("--trajectory-store-path",
                   default=os.environ.get("TRAJECTORY_STORE_PATH", "verified_trajectories.json"),
                   help="Path to the verified-trajectory store file.")
    p.add_argument("--max-concurrent", type=int,
                   default=int(os.environ.get("RFSN_MAX_CONCURRENT", "2")),
                   help="Max concurrent jobs")
    p.add_argument("--clr", dest="use_clr", action="store_true",
                   help="Use CLR on the specialist (default)")
    p.add_argument("--no-clr", dest="use_clr", action="store_false",
                   help="Disable CLR on the specialist")
    p.set_defaults(use_clr=_env_bool("RFSN_USE_CLR", True))
    p.add_argument("--clr-k", type=int,
                   default=int(os.environ.get("RFSN_CLR_K", "8")),
                   help="CLR k")
    p.add_argument("--fast-specialist", dest="fast_specialist",
                   action="store_true",
                   help="Use the aggressive adaptive policy (up to 3/5/15 "
                        "trajectories, capped at --clr-k) tuned for an "
                        "ultra-tiny fast specialist (e.g. 0.5B). Do NOT use "
                        "with a 3B+ specialist on 16GB RAM — it will "
                        "thrash/OOM. Default off.")
    p.set_defaults(fast_specialist=_env_bool("RFSN_FAST_SPECIALIST", False))
    p.add_argument("--local-specialist-model",
                   default=os.environ.get("VIBE_THINKER_LOCAL_MODEL", ""),
                   help="Path to a local .gguf (or 'repo_id/filename.gguf') to "
                        "load the specialist in-process via llama-cpp-python, "
                        "bypassing HTTP entirely. Auto-preferred over --vibe "
                        "when set; falls back to HTTP if llama-cpp-python is "
                        "missing or the load fails. Requires llama-cpp-python.")
    p.add_argument("--local-specialist-n-ctx", type=int,
                   default=int(os.environ.get("VIBE_THINKER_LOCAL_N_CTX", "4096")),
                   help="Context window for the in-process specialist (default 4096).")
    p.add_argument("--local-specialist-n-threads", type=int,
                   default=int(os.environ.get("VIBE_THINKER_LOCAL_N_THREADS", "8")),
                   help="CPU threads for the in-process specialist (default 8).")
    p.add_argument("--local-specialist-pool-size", type=int,
                   default=int(os.environ.get("VIBE_THINKER_LOCAL_POOL_SIZE", "1")),
                   help="Number of in-process Llama instances to load for true "
                        "parallel inference (default 1 = single instance + Lock). "
                        "For a 0.5B model (~398MB each), 4 instances cost ~1.6GB "
                        "and enable 4 concurrent trajectories. Each instance gets "
                        "n_threads/pool_size CPU threads.")
    # --- RuvLLM integration (v0.3.9) ---
    # RuvLLM is a Rust inference engine with TurboQuant KV cache compression.
    # It exposes the same OpenAI-compatible HTTP API as llama-server, so the
    # simplest integration is to point --vibe at the RuvLLM port. This flag
    # is a convenience that overrides --vibe with the RuvLLM URL and documents
    # the integration. See ruvllm_adapter.py for details.
    p.add_argument("--ruvllm-url",
                   default=os.environ.get("RUVLLM_URL", ""),
                   help="RuvLLM HTTP endpoint (e.g. http://127.0.0.1:8080). "
                        "When set, overrides --vibe to use the RuvLLM server "
                        "with TurboQuant KV cache compression. The orchestrator's "
                        "existing HTTP path handles it — no other changes needed. "
                        "See ruvllm_adapter.RuvLLMHTTPBackend for the recommended "
                        "start command with TurboQuant flags.")
    # --- Fast code-specialist preset (v0.3.9) ---
    # Bumps CODE_CANDIDATES to 15 for ultra-fast 0.5B code models (ruvltra).
    # Does NOT hardcode a model path — pair with --code-specialist and/or
    # --local-specialist-model pointing at ruvltra-claude-code-0.5b.
    p.add_argument("--fast-code-specialist", dest="fast_code_specialist",
                   action="store_true",
                   help="Preset for ultra-fast 0.5B code specialists (ruvltra): "
                        "sets CODE_CANDIDATES=15 so the multi-candidate loop "
                        "shotgun-samples 15 candidates in parallel. At 0.5B "
                        "speed (~100+ tok/s), 15 candidates cost roughly what 2 "
                        "cost on a 3B model. Pair with --code-specialist pointing "
                        "at a ruvltra-claude-code server, or --local-specialist-"
                        "model pointing at the ruvltra .gguf. Do NOT use with a "
                        "3B+ code model — 15 parallel candidates will thrash.")
    p.set_defaults(fast_code_specialist=_env_bool("RFSN_FAST_CODE_SPECIALIST", False))
    p.add_argument("--audit-log",
                   default=os.environ.get("RFSN_AUDIT_LOG", "rfsn_jobs_bitemporal.jsonl"),
                   help="Bi-temporal audit log path (empty disables logging)")
    # --- Audit-log signing (v0.3.9) ---
    # HMAC-SHA256 (symmetric, stdlib) or Ed25519 (asymmetric, SLSA L2).
    # Ed25519 takes precedence when both are set. When neither is set,
    # the log is tamper-evident only (hash chain, no signatures).
    p.add_argument("--signing-key",
                   default=os.environ.get("RFSN_SIGNING_KEY", ""),
                   help="HMAC-SHA256 shared secret for audit-log signatures "
                        "(symmetric, stdlib). When set, each log entry is "
                        "tamper-proof — an attacker cannot forge signatures "
                        "without the key. Empty = no signing (tamper-evident only).")
    p.add_argument("--ed25519-private-key",
                   default=os.environ.get("RFSN_ED25519_PRIVATE_KEY", ""),
                   help="Hex-encoded Ed25519 private key for asymmetric audit-log "
                        "signatures (SLSA L2 compliant). Stronger than HMAC: the "
                        "public key can verify but cannot forge. Requires the "
                        "'cryptography' package. Takes precedence over --signing-key. "
                        "Generate with: python3 -c \"from signers import Ed25519Signer; "
                        "s=Ed25519Signer.generate(); print(s.private_key_hex)\"")
    p.add_argument("--ed25519-public-key",
                   default=os.environ.get("RFSN_ED25519_PUBLIC_KEY", ""),
                   help="Hex-encoded Ed25519 public key for verify-only mode "
                        "(nodes that read but don't write the log). Takes "
                        "precedence over --signing-key for verification.")
    # --- AgentDB vector store (v0.3.9) ---
    # When set, CLRResultCache and VerifiedTrajectoryStore delegate similarity
    # search to a RuFlo/AgentDB HTTP sidecar (with shadow-mode dual-write to
    # the local JSON file for zero-downtime migration).
    p.add_argument("--agentdb-url",
                   default=os.environ.get("AGENTDB_URL", ""),
                   help="RuFlo/AgentDB HTTP endpoint for vector similarity search "
                        "(e.g. http://127.0.0.1:8088). When set, the CLR result cache "
                        "and trajectory store dual-write to both the local JSON file "
                        "and AgentDB (shadow mode). Reads fall back to local if AgentDB "
                        "is down. Empty = in-memory numpy (default, unchanged).")
    # --- Federated job queue (v0.3.9) ---
    # When set, jobs are published to an exo-federation swarm network over mTLS.
    # Any idle node can claim pending jobs. Fail-closed-fallback to local when
    # the federation is unreachable.
    p.add_argument("--federation-url",
                   default=os.environ.get("FEDERATION_URL", ""),
                   help="exo-federation HTTP endpoint for multi-node job distribution "
                        "(e.g. https://swarm.local:7443). When set, jobs are published "
                        "to the swarm; any idle node can claim them. Requires mTLS certs. "
                        "Empty = local-only single-node queue (default).")
    p.add_argument("--mtls-cert",
                   default=os.environ.get("FEDERATION_MTLS_CERT", ""),
                   help="Path to the mTLS client certificate (PEM) for exo-federation.")
    p.add_argument("--mtls-key",
                   default=os.environ.get("FEDERATION_MTLS_KEY", ""),
                   help="Path to the mTLS client private key (PEM) for exo-federation.")
    p.add_argument("--mtls-ca",
                   default=os.environ.get("FEDERATION_MTLS_CA", ""),
                   help="Path to the mTLS CA certificate (PEM) that signed all node certs.")
    # --- Active retrieval (v0.4.0) ---
    # When configured, factual tasks fetch real source text from a search API
    # and feed it to the FactualVerifier's NLI judge. Fail-closed: no key =
    # no retrieval = unsupported_factual (unchanged honest behavior).
    # Keys are NEVER logged or committed. Precedence: --serper-key >
    # --searchapi-key > SERPER_API_KEY env > SEARCHAPI_API_KEY env.
    p.add_argument("--serper-key",
                   default=os.environ.get("SERPER_API_KEY", ""),
                   help="Serper.dev API key for factual retrieval. When set, "
                        "factual queries fetch real Google search results and "
                        "feed the snippets to the FactualVerifier NLI judge. "
                        "Empty = no retrieval (default, fail-closed).")
    p.add_argument("--searchapi-key",
                   default=os.environ.get("SEARCHAPI_API_KEY", ""),
                   help="SearchApi.io API key for factual retrieval (alternative "
                        "to --serper-key). Same fail-closed behavior.")
    p.add_argument("--embedding-router", dest="use_embedding_router",
                   action="store_true",
                   help="Use embedding-based semantic router (default)")
    p.add_argument("--no-embedding-router", dest="use_embedding_router",
                   action="store_false",
                   help="Disable embedding router, use keyword fallback")
    p.set_defaults(use_embedding_router=_env_bool("RFSN_USE_EMBEDDING_ROUTER", True))
    return p


async def _amain() -> None:
    args = build_argparser().parse_args()

    # --- RuvLLM URL override (v0.3.9) ---
    # When --ruvllm-url is set, it takes precedence over --vibe. The
    # orchestrator's HTTP path handles RuvLLM unchanged (same OpenAI API).
    vibe_endpoint = args.ruvllm_url or args.vibe
    if args.ruvllm_url:
        print(f"[CLI] RuvLLM backend enabled: --vibe overridden to {args.ruvllm_url}")

    # --- Fast code-specialist preset (v0.3.9) ---
    # Bumps code_candidates to 15 for ultra-fast 0.5B code models.
    code_candidates = args.code_candidates
    if args.fast_code_specialist:
        code_candidates = max(args.code_candidates, 15)
        print(f"[CLI] Fast code-specialist preset: CODE_CANDIDATES -> {code_candidates}")
        if not (args.code_specialist or args.local_specialist_model):
            print("[CLI] Warning: --fast-code-specialist is set but no code specialist "
                  "is configured. Pair with --code-specialist <ruvltra-url> or "
                  "--local-specialist-model <ruvltra.gguf> for it to take effect.")

    orchestrator = HybridReasoningOrchestrator(
        vibe_endpoint=vibe_endpoint,
        generalist_endpoint=args.generalist,
        code_specialist_endpoint=args.code_specialist or None,
        code_candidates=code_candidates,
        max_repair_attempts=args.max_repair_attempts,
        use_clr=args.use_clr,
        clr_k=args.clr_k,
        use_embedding_router=args.use_embedding_router,
        use_trajectory_store=args.use_trajectory_store,
        trajectory_store_path=args.trajectory_store_path,
        fast_specialist=args.fast_specialist,
        local_specialist_model=args.local_specialist_model or None,
        local_specialist_n_ctx=args.local_specialist_n_ctx,
        local_specialist_n_threads=args.local_specialist_n_threads,
        local_specialist_pool_size=args.local_specialist_pool_size,
        agentdb_url=args.agentdb_url or None,
        retrieval_backend=_build_retrieval_backend(args),
    )

    # --- Audit-log signing (v0.3.9) ---
    # Ed25519 (asymmetric) takes precedence over HMAC-SHA256 (symmetric).
    signing_key = args.signing_key or None
    ed25519_private = args.ed25519_private_key or None
    ed25519_public = args.ed25519_public_key or None
    if ed25519_private:
        print("[CLI] Ed25519 audit-log signing enabled (asymmetric, SLSA L2)")
    elif signing_key:
        print("[CLI] HMAC-SHA256 audit-log signing enabled (symmetric)")

    # --- Queue construction (v0.3.9) ---
    # Uses make_job_queue so --federation-url switches to FederatedJobQueue
    # (with local fallback). Without --federation-url, a LocalJobQueue is
    # used (wraps JobQueue, same behavior). Signing keys are threaded
    # through to the inner JobQueue's BiTemporalAuditLog.
    queue = make_job_queue(
        orchestrator,
        federation_url=args.federation_url or None,
        max_concurrent=args.max_concurrent,
        mtls_cert=args.mtls_cert or None,
        mtls_key=args.mtls_key or None,
        mtls_ca=args.mtls_ca or None,
        audit_log=args.audit_log or None,
        signing_key=signing_key,
        ed25519_private_key_hex=ed25519_private,
        ed25519_public_key_hex=ed25519_public,
    )
    await queue.start()
    repl = JobQueueREPL(queue)
    try:
        await repl.arepl()
    except SystemExit:
        pass
    finally:
        await queue.stop()


def main() -> None:
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
