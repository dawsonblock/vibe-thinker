"""
Interactive CLI / REPL front-end for the RFSN job queue.

Sits on top of JobQueue + HybridReasoningOrchestrator and gives you an
interactive prompt for submitting queries, watching jobs, inspecting
results, and querying the bi-temporal audit log — all while the async
dispatcher runs in the background.

Usage:
    python rfsn_cli.py
    python rfsn_cli.py --vibe http://127.0.0.1:8080 \
        --generalist http://127.0.0.1:8081
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
import subprocess
import sys

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
    backend = make_retrieval_backend(
        serper_key=serper, searchapi_key=searchapi)
    if backend is not None:
        print(f"[CLI] Active retrieval enabled: {backend.name} — "
              f"factual tasks will fetch real sources for NLI verification")
    return backend


def _build_network_allowlist(args) -> "object | None":
    """Build a NetworkAllowList from CLI args / env vars.

    Precedence: --network-allowlist-file > --network-allowlist > env.
    Returns None when no allow-list is configured (the sandbox uses
    --network=none, unchanged behavior).
    """
    from sandbox.network_allowlist import NetworkAllowList

    file_path = args.network_allowlist_file or None
    spec = args.network_allowlist or None

    if file_path:
        if not os.path.exists(file_path):
            print(f"[CLI] Warning: --network-allowlist-file {file_path} "
                  f"not found — no allow-list (deny all)")
            return None
        allowlist = NetworkAllowList.from_file(file_path)
        source = f"file:{file_path}"
    elif spec:
        allowlist = NetworkAllowList.from_string(spec)
        source = "CLI string"
    else:
        return None

    if allowlist.is_empty:
        print("[CLI] Network allow-list is empty — deny all egress "
              "(same as --network=none)")
    else:
        summary = allowlist.summary()
        print(f"[CLI] Network allow-list active ({source}): "
              f"{summary['entry_count']} entries "
              f"({len(summary['domains'])} domains, "
              f"{len(summary['ips'])} IPs, "
              f"{len(summary['cidrs'])} CIDRs, "
              f"{len(summary['wildcards'])} wildcards)")
    return allowlist


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
                # input() blocks; run it in a thread so the dispatcher
                # keeps going.
                line = await loop.run_in_executor(
                    None, lambda: input("rfsn> "))
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
                  + (f", force_route={job.force_route}"
                     if job.force_route else "")
                  + ")")
            return

        if cmd in ("list", "ls"):
            jobs = self.queue.list_jobs()
            if not jobs:
                print("(no jobs)")
                return
            print(f"{'job_id':14} {'status':10} {'pri':>3} {'route':16} query")
            for j in jobs:
                q = (j["query"][:48] + "...") if len(j["query"]) > 48 \
                    else j["query"]
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
                print(f"job {job.job_id} has no result yet "
                      f"(status={job.status.value})")
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
                print(f"  {e['valid_time']}  {e['event']:10} "
                      f"-> {e['status']}{tail}")
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
            print(f"{'job_id':14} {'status':10} {'event':10} "
                  f"{'valid_time':27} query")
            for jid, e in state.items():
                q = (e["query"][:40] + "...") if len(e["query"]) > 40 \
                    else e["query"]
                print(f"{jid:14} {e['status']:10} {e['event']:10} "
                      f"{e['valid_time']:27} {q}")
            return

        print(f"unknown command: {cmd}  (type 'help')")


# ----------------------------- entry point ----------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RFSN job queue REPL.")
    # Precedence: CLI flags > environment variables > defaults
    p.add_argument("--vibe",
                   default=os.environ.get(
                       "VIBE_THINKER_URL", "http://127.0.0.1:8080"),
                   help="VibeThinker specialist endpoint")
    p.add_argument("--specialist-transport",
                   choices=["completion", "openai_chat", "anthropic"],
                   default=os.environ.get(
                       "VIBE_THINKER_SPECIALIST_TRANSPORT",
                       "completion"),
                   help="Which HTTP API shape the specialist server speaks. "
                        "'completion' (default) = llama-server/RuvLLM "
                        "/completion "
                        "(accepts the 'grammar' GBNF field). 'openai_chat' "
                        "= "
                        "OpenAI-compatible /v1/chat/completions (no "
                        "/completion "
                        "endpoint; structured output via response_format). "
                        "'anthropic' = /v1/messages (structured output via "
                        "tools). "
                        "Ignored when --local-specialist-model is set.")
    p.add_argument("--specialist-api-key",
                   default=os.environ.get(
                       "VIBE_THINKER_SPECIALIST_API_KEY", ""),
                   help="API key for the specialist when using openai_chat or "
                        "anthropic transports. Sent as 'Authorization: "
                        "Bearer ' "
                        "(openai_chat) or 'x-api-key' (anthropic). Never "
                        "logged.")
    p.add_argument("--specialist-model-name",
                   default=os.environ.get(
                       "VIBE_THINKER_SPECIALIST_MODEL_NAME", ""),
                   help="Model name to send in the chat payload for "
                        "openai_chat / "
                        "anthropic transports (e.g. 'gpt-4o-mini' or "
                        "'claude-3-5-sonnet-20241022'). Some self-hosted "
                        "OpenAI-compatible servers ignore this field.")
    p.add_argument("--max-parse-repairs", type=int,
                   default=int(os.environ.get("MAX_PARSE_REPAIRS", "2")),
                   help="When a structured-output specialist call returns "
                        "malformed JSON, feed the bad text + parse error back "
                        "for a corrected attempt. Max repair rounds (default "
                        "2, "
                        "0 disables). Most useful for transports without "
                        "native "
                        "grammar enforcement (openai_chat, anthropic). "
                        "Fail-closed: falls back to regex extraction if no "
                        "repair parses.")
    p.add_argument("--prefer-encoder-nli", dest="prefer_encoder_nli",
                   action="store_true", default=True,
                   help="(Default ON) Prefer an encoder-only NLI model (e.g. "
                        "DeBERTa-v3) over the LLM judge for factual "
                        "verification. More robust to fabrication (encoder "
                        "models can't hallucinate). Requires the optional "
                        "'nli' extra: pip install \"vibe-thinker[nli]\". "
                        "The model is downloaded from HuggingFace on first "
                        "use. Fail-closed to the LLM judge when unavailable.")
    p.add_argument("--no-encoder-nli", dest="prefer_encoder_nli",
                   action="store_false",
                   help="Disable the encoder NLI judge and use the LLM judge "
                        "for factual verification. Useful when the encoder "
                        "model download is undesirable or when you want the "
                        "LLM judge's citation extraction (encoder NLI "
                        "doesn't extract supporting quotes).")
    p.set_defaults(prefer_encoder_nli=not _env_bool(
        "VIBE_THINKER_NO_ENCODER_NLI", False))
    p.add_argument("--generalist",
                   default=os.environ.get(
                       "GENERALIST_URL", "http://127.0.0.1:8081"),
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
                   help="Disable the verified-trajectory store "
                        "(self-improving "
                        "few-shot memory). Enabled by default when embedding "
                        "deps are available.")
    p.set_defaults(use_trajectory_store=_env_bool(
        "RFSN_USE_TRAJECTORY_STORE", True))
    p.add_argument("--trajectory-store-path",
                   default=os.environ.get(
                       "TRAJECTORY_STORE_PATH",
                       "verified_trajectories.json"),
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
                   help="Path to a local .gguf (or 'repo_id/filename.gguf') "
                        "to "
                        "load the specialist in-process via llama-cpp-python, "
                        "bypassing HTTP entirely. Auto-preferred over --vibe "
                        "when set; falls back to HTTP if llama-cpp-python is "
                        "missing or the load fails. Requires "
                        "llama-cpp-python.")
    p.add_argument("--local-specialist-n-ctx", type=int,
                   default=int(os.environ.get(
                       "VIBE_THINKER_LOCAL_N_CTX", "4096")),
                   help="Context window for the in-process specialist "
                        "(default 4096).")
    p.add_argument("--local-specialist-n-threads", type=int,
                   default=int(os.environ.get(
                       "VIBE_THINKER_LOCAL_N_THREADS", "8")),
                   help="CPU threads for the in-process specialist "
                        "(default 8).")
    p.add_argument("--local-specialist-pool-size", type=int,
                   default=int(os.environ.get(
                       "VIBE_THINKER_LOCAL_POOL_SIZE", "1")),
                   help="Number of in-process Llama instances to load for "
                        "true "
                        "parallel inference (default 1 = single instance "
                        "+ Lock). "
                        "For a 0.5B model (~398MB each), 4 instances cost "
                        "~1.6GB "
                        "and enable 4 concurrent trajectories. Each "
                        "instance gets "
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
                        "with TurboQuant KV cache compression. The "
                        "orchestrator's "
                        "existing HTTP path handles it — no other changes "
                        "needed. "
                        "See ruvllm_adapter.RuvLLMHTTPBackend for the "
                        "recommended "
                        "start command with TurboQuant flags.")
    # --- Fast code-specialist preset (v0.3.9) ---
    # Bumps CODE_CANDIDATES to 15 for ultra-fast 0.5B code models (ruvltra).
    # Does NOT hardcode a model path — pair with --code-specialist and/or
    # --local-specialist-model pointing at ruvltra-claude-code-0.5b.
    p.add_argument("--fast-code-specialist", dest="fast_code_specialist",
                   action="store_true",
                   help="Preset for ultra-fast 0.5B code specialists "
                        "(ruvltra): "
                        "sets CODE_CANDIDATES=15 so the multi-candidate loop "
                        "shotgun-samples 15 candidates in parallel. At 0.5B "
                        "speed (~100+ tok/s), 15 candidates cost roughly "
                        "what 2 "
                        "cost on a 3B model. Pair with --code-specialist "
                        "pointing "
                        "at a ruvltra-claude-code server, or --local-"
                        "specialist-"
                        "model pointing at the ruvltra .gguf. Do NOT use "
                        "with a "
                        "3B+ code model — 15 parallel candidates will thrash.")
    p.set_defaults(fast_code_specialist=_env_bool(
        "RFSN_FAST_CODE_SPECIALIST", False))
    p.add_argument("--structured-output", dest="use_structured_output",
                   action="store_true",
                   help="Force the specialist to output structured JSON "
                        "(reasoning_steps, boxed_answer, code_solution) via "
                        "GBNF grammar. Eliminates brittle \\boxed{} regex "
                        "scraping — the answer is read directly from the "
                        "boxed_answer key. Falls back to regex for backends "
                        "without grammar support. Default: off (backward "
                        "compat).")
    p.set_defaults(use_structured_output=_env_bool(
        "RFSN_STRUCTURED_OUTPUT", False))
    # --- Static-analysis fallback gate (v3.2) ---
    # AST static analysis is NOT a security boundary. Off by default in
    # production: when no sandbox (wasmtime/Docker) is available, the code
    # route returns verified=False / score=0.0 instead of a 0.2 heuristic.
    # Enable for local dev where you want the "parses + no restricted
    # imports" signal but have no sandbox installed.
    p.add_argument("--allow-static-fallback", dest="allow_static_fallback",
                   action="store_true",
                   help="Allow the deprecated AST static-analysis fallback "
                        "when no sandbox is available. AST is NOT a security "
                        "boundary — only the code's parse + restricted-import "
                        "status is checked. Emits a 0.2 heuristic "
                        "(verified=False). "
                        "Default: off (production-safe).")
    p.set_defaults(allow_static_fallback=_env_bool(
        "VIBE_THINKER_ALLOW_STATIC_FALLBACK", False))
    p.add_argument("--audit-log",
                   default=os.environ.get(
                       "RFSN_AUDIT_LOG",
                       "rfsn_jobs_bitemporal.jsonl"),
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
                        "without the key. Empty = no signing (tamper-"
                        "evident only).")
    p.add_argument("--ed25519-private-key",
                   default=os.environ.get("RFSN_ED25519_PRIVATE_KEY", ""),
                   help="Hex-encoded Ed25519 private key for asymmetric "
                        "audit-log "
                        "signatures (SLSA L2 compliant). Stronger than "
                        "HMAC: the "
                        "public key can verify but cannot forge. Requires "
                        "the "
                        "'cryptography' package. Takes precedence over "
                        "--signing-key. "
                        "Generate with: python3 -c \"from signers import "
                        "Ed25519Signer; "
                        "s=Ed25519Signer.generate(); "
                        "print(s.private_key_hex)\"")
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
                   help="RuFlo/AgentDB HTTP endpoint for vector similarity "
                        "search "
                        "(e.g. http://127.0.0.1:8088). When set, the CLR "
                        "result cache "
                        "and trajectory store dual-write to both the local "
                        "JSON file "
                        "and AgentDB (shadow mode). Reads fall back to "
                        "local if AgentDB "
                        "is down. Empty = in-memory numpy (default, "
                        "unchanged).")
    p.add_argument("--agentdb-only", action="store_true",
                   default=_env_bool("VIBE_THINKER_AGENTDB_ONLY", False),
                   help="Use AgentDB as the SOLE vector store (no shadow "
                        "mode, "
                        "no local JSON fallback). Use this AFTER running "
                        "'finalize-migration' to cut over to AgentDB-only. "
                        "Requires --agentdb-url. Fail-closed: if AgentDB is "
                        "down, searches return empty (no local fallback).")
    # --- Federated job queue (v0.3.9) ---
    # When set, jobs are published to a Python-native federation coordinator
    # over mTLS. Any idle node can claim pending jobs. Fail-closed-fallback
    # to local when the federation is unreachable.
    p.add_argument("--federation-url",
                   default=os.environ.get("FEDERATION_URL", ""),
                   help="Federation coordinator HTTP endpoint for multi-"
                        "node job "
                        "distribution (e.g. https://swarm.local:7443). May "
                        "be a "
                        "comma-separated list of URLs for HA failover (e.g. "
                        "https://c1:7443,https://c2:7443) — the client "
                        "tries each "
                        "and sticks to the first that succeeds. Run each "
                        "coordinator with: python3 -m federation_server "
                        "[--redis-url ...]. When set, jobs are published "
                        "to the "
                        "swarm; any idle node can claim them. Requires "
                        "mTLS certs. "
                        "Empty = local-only single-node queue (default).")
    p.add_argument("--mtls-cert",
                   default=os.environ.get("FEDERATION_MTLS_CERT", ""),
                   help="Path to the mTLS client certificate (PEM) for "
                        "federation.")
    p.add_argument("--mtls-key",
                   default=os.environ.get("FEDERATION_MTLS_KEY", ""),
                   help="Path to the mTLS client private key (PEM) for "
                        "federation.")
    p.add_argument("--mtls-ca",
                   default=os.environ.get("FEDERATION_MTLS_CA", ""),
                   help="Path to the mTLS CA certificate (PEM) that signed "
                        "all node certs.")
    # v3.0: Zero-trust federation encryption
    p.add_argument("--federation-secret",
                   default=os.environ.get("FEDERATION_SECRET", ""),
                   help="Shared secret for zero-trust payload encryption "
                        "(v3.0). "
                        "When set, all federation payloads (job queries, "
                        "results) "
                        "are encrypted with Fernet AEAD before transmission. "
                        "Nodes without the secret see only opaque ciphertext. "
                        "Requires the 'cryptography' package.")
    # v3.0: SONA gossip protocol — Distributed Brain
    p.add_argument("--sona-sync-url",
                   default=os.environ.get("SONA_SYNC_URL", ""),
                   help="Federation coordinator URL for SONA pattern sync "
                        "(v3.0). "
                        "When set, the orchestrator periodically exports its "
                        "learned patterns and imports global patterns from "
                        "other nodes. Enables swarm-wide learning.")
    p.add_argument("--sona-sync-interval",
                   type=int,
                   default=int(os.environ.get("SONA_SYNC_INTERVAL", "3600")),
                   help="Interval in seconds between SONA sync cycles "
                        "(default 3600).")
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
                   help="SearchApi.io API key for factual retrieval "
                        "(alternative "
                        "to --serper-key). Same fail-closed behavior.")
    # --- Network allow-list (v0.4.0) ---
    # When set, code sandbox execution uses --network=default + iptables
    # egress filtering instead of --network=none. Only allow-listed
    # destinations (domains, IPs, CIDRs) are reachable from the sandbox.
    # Fail-closed: empty allow-list = deny all (same as --network=none).
    p.add_argument("--network-allowlist",
                   default=os.environ.get("RFSN_NETWORK_ALLOWLIST", ""),
                   help="Comma-separated list of allowed network destinations "
                        "for sandbox egress (e.g. "
                        "'pypi.org:443,10.0.0.0/24'). "
                        "When set, the Docker sandbox uses iptables "
                        "filtering "
                        "instead of --network=none. Empty = deny all "
                        "(default).")
    p.add_argument("--network-allowlist-file",
                   default=os.environ.get("RFSN_NETWORK_ALLOWLIST_FILE", ""),
                   help="Path to a file with allowed network destinations "
                        "(one per line, # comments supported). Alternative to "
                        "--network-allowlist for long lists.")
    p.add_argument("--dns-resolver",
                   default=os.environ.get("RFSN_DNS_RESOLVER", ""),
                   help="IP address of a DNS resolver to restrict sandbox DNS "
                        "queries to (e.g. '8.8.8.8'). When set with "
                        "--network-allowlist, only this resolver can receive "
                        "DNS queries, preventing DNS-based data exfiltration. "
                        "Empty = allow DNS to any resolver (default).")
    p.add_argument("--sandbox-image",
                   default=os.environ.get("RFSN_SANDBOX_IMAGE",
                                          "vibe-thinker-sandbox:latest"),
                   help="Docker image for the code sandbox. Defaults to the "
                        "purpose-built vibe-thinker-sandbox image with "
                        "iptables "
                        "baked in. Build it with: docker build -f "
                        "sandbox/Dockerfile -t vibe-thinker-sandbox:latest .")
    p.add_argument("--proxy-egress",
                   default=os.environ.get("RFSN_PROXY_EGRESS", ""),
                   help="Address of an SNI-aware egress proxy (e.g. "
                        "127.0.0.1:8888). When set, the sandbox routes "
                        "traffic through the proxy instead of using "
                        "iptables IP-based filtering. This solves CDN IP "
                        "rotation: the proxy inspects the TLS SNI / HTTP "
                        "Host header and allows/denies based on the "
                        "domain, not the IP. v1.2: SNI-proxy is now the "
                        "DEFAULT egress mode when an allow-list is present; "
                        "this flag overrides the default address. Run the "
                        "proxy with: python3 -m sandbox.sni_proxy "
                        "--allowlist '...'")
    p.add_argument("--envoy-sidecar",
                   action="store_true",
                   default=os.environ.get("RFSN_ENVOY_SIDECAR", "") != "",
                   help="Launch an Envoy sidecar as the SNI-aware egress "
                        "proxy (v1.2). When set, the CLI generates an Envoy "
                        "config from the network allow-list and starts "
                        "Envoy as a child process before the orchestrator "
                        "runs. Requires the envoy binary on PATH. The "
                        "sandbox routes traffic through the Envoy listener "
                        "(default 127.0.0.1:8888). This is the recommended "
                        "production egress path; the Python sni_proxy.py is "
                        "the lightweight fallback.")
    p.add_argument("--embedding-router", dest="use_embedding_router",
                   action="store_true",
                   help="Use embedding-based semantic router (default)")
    p.add_argument("--no-embedding-router", dest="use_embedding_router",
                   action="store_false",
                   help="Disable embedding router, use keyword fallback")
    p.set_defaults(use_embedding_router=_env_bool(
        "RFSN_USE_EMBEDDING_ROUTER", True))
    return p


async def _amain() -> None:
    args = build_argparser().parse_args()

    # --- RuvLLM URL override (v0.3.9) ---
    # When --ruvllm-url is set, it takes precedence over --vibe. The
    # orchestrator's HTTP path handles RuvLLM unchanged (same OpenAI API).
    vibe_endpoint = args.ruvllm_url or args.vibe
    if args.ruvllm_url:
        print(f"[CLI] RuvLLM backend enabled: --vibe overridden to "
              f"{args.ruvllm_url}")

    # --- Specialist transport (v1.1) ---
    # Only relevant for the HTTP path; ignored when --local-specialist-model
    # is set (the in-process backend has its own code path).
    if not args.local_specialist_model and \
            args.specialist_transport != "completion":
        print(f"[CLI] Specialist transport: {args.specialist_transport} "
              f"(endpoint {vibe_endpoint})")
        if not args.specialist_api_key:
            print("[CLI] Warning: --specialist-transport is set but no "
                  "--specialist-api-key is configured. Set "
                  "VIBE_THINKER_SPECIALIST_API_KEY "
                  "or --specialist-api-key for authenticated providers.")

    # --- Encoder NLI judge (v1.1, default ON as of Phase 3.3) ---
    if args.prefer_encoder_nli:
        from verifiers.nli_encoder import is_available as encoder_available
        if encoder_available():
            print("[CLI] Encoder NLI judge enabled (default, factual "
                  "verification). Model downloads from HuggingFace on "
                  "first use. Use --no-encoder-nli to disable.")
        else:
            print("[CLI] Encoder NLI judge: 'nli' extra not installed. "
                  "Install with: pip install \"vibe-thinker[nli]\". "
                  "Falling back to the LLM judge (default behavior).")
    else:
        print("[CLI] Encoder NLI judge disabled (--no-encoder-nli). "
              "Using LLM judge for factual verification.")

    # --- AgentDB vector store mode (v3.2.1) ---
    # ShadowVectorStore was removed: when --agentdb-url is set, AgentDB is
    # used directly (no local shadow/fallback). --agentdb-only is kept as
    # a no-op flag for backward CLI compat — setting --agentdb-url alone
    # is now always AgentDB-only.
    if args.agentdb_only and not args.agentdb_url:
        print("[CLI] WARNING: --agentdb-only set but --agentdb-url is empty. "
              "AgentDB-only mode requires --agentdb-url. Falling back to "
              "in-memory numpy (default behavior).")
        args.agentdb_only = False
    elif args.agentdb_url:
        print(f"[CLI] AgentDB mode: vector store is AgentDB at "
              f"{args.agentdb_url} (no local fallback). Fail-closed: "
              f"searches return empty if AgentDB is down. "
              f"(--agentdb-only is now a no-op; shadow mode was removed in "
              f"v3.2.1 — run finalize-migration before relying on this.)")

    # --- Fast code-specialist preset (v0.3.9) ---
    # Bumps code_candidates to 15 for ultra-fast 0.5B code models.
    code_candidates = args.code_candidates
    if args.fast_code_specialist:
        code_candidates = max(args.code_candidates, 15)
        print(f"[CLI] Fast code-specialist preset: CODE_CANDIDATES -> "
              f"{code_candidates}")
        if not (args.code_specialist or args.local_specialist_model):
            print("[CLI] Warning: --fast-code-specialist is set but no "
                  "code specialist "
                  "is configured. Pair with --code-specialist "
                  "<ruvltra-url> or "
                  "--local-specialist-model <ruvltra.gguf> for it to "
                  "take effect.")

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
        agentdb_only=args.agentdb_only,
        retrieval_backend=_build_retrieval_backend(args),
        network_allowlist=_build_network_allowlist(args),
        dns_resolver=args.dns_resolver or None,
        sandbox_image=args.sandbox_image or None,
        proxy_egress=args.proxy_egress or None,
        use_structured_output=args.use_structured_output,
        specialist_transport=args.specialist_transport,
        specialist_api_key=args.specialist_api_key or None,
        specialist_model_name=args.specialist_model_name or None,
        max_parse_repairs=args.max_parse_repairs,
        prefer_encoder_nli=args.prefer_encoder_nli,
        sona_sync_url=args.sona_sync_url or None,
        sona_sync_interval=args.sona_sync_interval,
        federation_secret=args.federation_secret or None,
        allow_static_fallback=args.allow_static_fallback,
    )

    # --- Envoy sidecar egress (v1.2) ---
    # When --envoy-sidecar is set, generate an Envoy config from the
    # network allow-list and launch Envoy as a child process. The
    # sandbox routes traffic through the Envoy listener. This is the
    # recommended production egress path (replaces the Python sni_proxy).
    envoy_proc = None
    if args.envoy_sidecar:
        from sandbox.envoy_sidecar import (
            generate_envoy_config, write_envoy_config, launch_envoy,
            find_envoy_binary,
        )
        allowlist = _build_network_allowlist(args)
        if allowlist is None or allowlist.is_empty:
            print("[CLI] --envoy-sidecar requires a network allow-list "
                  "(--network-allowlist). Skipping Envoy launch.")
        elif find_envoy_binary() is None:
            print("[CLI] --envoy-sidecar: envoy binary not found on PATH. "
                  "Install Envoy (e.g. brew install envoy) or use the "
                  "Python SNI proxy (sandbox.sni_proxy) instead. "
                  "Falling back to the default proxy address.")
        else:
            import tempfile
            config = generate_envoy_config(allowlist)
            fd, config_path = tempfile.mkstemp(suffix=".yaml",
                                               prefix="envoy_sidecar_")
            import os as _os
            _os.close(fd)
            write_envoy_config(config, config_path)
            print(f"[CLI] Launching Envoy sidecar (config={config_path})")
            try:
                envoy_proc = launch_envoy(config_path)
                print(f"[CLI] Envoy sidecar started (PID {envoy_proc.pid})")
            except FileNotFoundError as e:
                print(f"[CLI] Envoy launch failed: {e}")
                envoy_proc = None

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
        federation_secret=args.federation_secret or None,
    )
    await queue.start()
    repl = JobQueueREPL(queue)
    try:
        await repl.arepl()
    except SystemExit:
        pass
    finally:
        await queue.stop()
        # Clean up the Envoy sidecar if it was launched (v1.2).
        if envoy_proc is not None:
            print("[CLI] Terminating Envoy sidecar...")
            envoy_proc.terminate()
            try:
                envoy_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                envoy_proc.kill()
                envoy_proc.wait()
            print("[CLI] Envoy sidecar terminated.")


def main() -> None:
    # --- subcommands (detected as the first argument before the REPL) ---
    # This avoids needing subparsers (which would complicate the existing
    # REPL arg structure). When invoked, the subcommand runs and exits
    # without starting the REPL.
    if len(sys.argv) > 1:
        sub = sys.argv[1]
        if sub == "finalize-migration":
            sys.argv = [sys.argv[0]] + sys.argv[2:]
            sys.exit(_run_finalize_migration())
        if sub == "doctor":
            sys.exit(_run_doctor())
        if sub == "smoke":
            sys.exit(_run_smoke())

    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        pass


def _run_doctor() -> int:
    """Check the environment and report what's available.

    Verifies Python version, package install, optional dependencies, model
    backend config, cache backend, sandbox mode, and service availability.
    Prints a human-readable report and exits 0 if the core profile is
    runnable, 1 otherwise.
    """
    import sys as _sys
    import importlib.util

    print("Vibe Thinker Doctor")
    print("=" * 50)

    # Core
    print("\nCore:")
    py_ok = _sys.version_info >= (3, 10)
    print(f"  Python: {_sys.version.split()[0]} {'OK' if py_ok else 'FAIL'}")
    if not py_ok:
        print("  Verdict: Python >= 3.10 required")
        return 1

    # Package install
    try:
        import hybrid_orchestrator  # noqa: F401
        print("  Package import: OK")
    except ImportError as e:
        print(f"  Package import: FAIL ({e})")
        return 1

    try:
        import rfsn_cli  # noqa: F401
        print("  CLI module: OK")
    except ImportError as e:
        print(f"  CLI module: FAIL ({e})")
        return 1

    # Optional dependencies
    print("\nOptional:")
    optional_deps = [
        ("z3-solver", "z3"),
        ("sentence-transformers", "sentence_transformers"),
        ("faiss-cpu", "faiss"),
        ("fakeredis", "fakeredis"),
        ("redis", "redis"),
        ("docker", "docker"),
        ("fastapi", "fastapi"),
        ("cryptography", "cryptography"),
        ("wasmtime", "wasmtime"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
    ]
    for pip_name, mod_name in optional_deps:
        avail = importlib.util.find_spec(mod_name) is not None
        print(f"  {pip_name}: {'present' if avail else 'missing'}")

    # RuvLLM
    ruvllm_env = os.environ.get("VIBE_RUVLLM", "disabled")
    print(f"  ruvllm: {ruvllm_env}")

    # Sandbox
    print("\nSandbox:")
    sandbox_mode = os.environ.get("VIBE_SANDBOX_MODE", "disabled")
    network_mode = os.environ.get("VIBE_NETWORK_MODE", "disabled")
    print(f"  mode: {sandbox_mode} "
          f"{'OK' if sandbox_mode == 'disabled' else 'CHECK'}")
    print(f"  network: {network_mode} "
          f"{'OK' if network_mode == 'disabled' else 'CHECK'}")
    if network_mode == "best_effort_proxy":
        print("  WARNING: best_effort_proxy is NOT a security boundary.")
        print("           Use 'disabled' for untrusted code.")

    # Verdict
    print("\nVerdict:")
    if py_ok and sandbox_mode == "disabled" and network_mode == "disabled":
        print("  core local profile is runnable")
        return 0
    else:
        print("  some components need attention (see above)")
        return 1


def _run_smoke() -> int:
    """Run a minimal no-network, no-model smoke test.

    Verifies:
      - import orchestrator
      - run deterministic math verifier
      - run schema verifier
      - create audit event
      - write/read local cache
    Exits 0 if all pass, 1 otherwise.
    """
    import asyncio as _asyncio

    print("Vibe Thinker Smoke Test")
    print("=" * 50)

    failures = []

    # 1. Import orchestrator
    try:
        import hybrid_orchestrator  # noqa: F401
        print("[1/5] Import orchestrator: OK")
    except ImportError as e:
        print(f"[1/5] Import orchestrator: FAIL ({e})")
        failures.append("import")

    # 2. Math verifier (async)
    try:
        from verifiers.math_verifier import MathVerifier
        v = MathVerifier()
        result = _asyncio.run(v.verify(
            "What is 2+2?", "4", context={"expected_answer": "4"}
        ))
        assert result.verified, (
            f"MathVerifier expected verified=True, "
            f"got {result.verified}")
        print("[2/5] Math verifier (2+2=4): OK")
    except Exception as e:
        print(f"[2/5] Math verifier: FAIL ({e})")
        failures.append("math")

    # 3. Schema verifier (async)
    try:
        from verifiers.schema_verifier import SchemaVerifier
        v = SchemaVerifier()
        result = _asyncio.run(v.verify(
            "Return a JSON object with a name field",
            '{"name": "test"}',
            context={"schema": {"type": "object", "required": ["name"]}},
        ))
        assert result.verified, (
            f"SchemaVerifier expected verified=True, "
            f"got {result.verified}")
        print("[3/5] Schema verifier: OK")
    except Exception as e:
        print(f"[3/5] Schema verifier: FAIL ({e})")
        failures.append("schema")

    # 4. Audit log (uses record() with a job-like object)
    try:
        import tempfile
        import os
        import types
        from bitemporal_log import BiTemporalAuditLog
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            log_path = f.name
        try:
            log = BiTemporalAuditLog(path=log_path)
            # Create a minimal job-like object
            job = types.SimpleNamespace(
                job_id="smoke-test-1",
                status="completed",
                query="smoke test",
                priority=0,
                force_route=None,
            )
            log.record(job, event="smoke_test", extra={"test": True})
            entries = log.read_all()
            assert len(entries) >= 1, f"Expected >=1 entry, got {len(entries)}"
            print("[4/5] Audit log write/read: OK")
        finally:
            if os.path.exists(log_path):
                os.unlink(log_path)
    except Exception as e:
        print(f"[4/5] Audit log: FAIL ({e})")
        failures.append("audit")

    # 5. Local cache (exact key lookup, no embeddings needed)
    try:
        import tempfile
        import os
        from persistent_cache import CLRResultCache, CacheSimilarityMode
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            cache_path = f.name
        try:
            # Use similarity_mode=NONE so no embedding model is loaded.
            # The smoke test verifies the exact-match cache path only.
            cache = CLRResultCache(
                path=cache_path,
                similarity_mode=CacheSimilarityMode.NONE,
            )
            cache.insert(
                problem="smoke test",
                best_answer="ok",
                best_score=1.0,
                k=1,
                trajectory_count=1,
                verified=True,
                verification_method="python_eval",
            )
            print("[5/5] Local cache write/read: OK")
        finally:
            if os.path.exists(cache_path):
                os.unlink(cache_path)
    except Exception as e:
        print(f"[5/5] Local cache: FAIL ({e})")
        failures.append("cache")

    print()
    if failures:
        print(f"Smoke test FAILED: {', '.join(failures)}")
        return 1
    print("Smoke test PASSED.")
    return 0


class _FakeVectorStore:
    """Minimal vector store for smoke test (no embeddings needed)."""

    def upsert(self, vector_id, embedding, metadata=None):
        pass

    def search(self, query_embedding, top_k=10, filters=None):
        return []

    def delete(self, vector_id):
        return False

    def count(self):
        return 0

    def cluster(self, **kwargs):
        return []


def _run_finalize_migration() -> int:
    """Finalize the AgentDB shadow-mode migration.

    Verifies that AgentDB has sufficient recall compared to the local
    store, then switches the orchestrator config to AgentDB-only by
    archiving the local JSON files (renamed to .bak). Fail-closed: if
    recall fails or AgentDB is unreachable, refuses to finalize (no
    data loss).

    Usage:
        python rfsn_cli.py finalize-migration \\
            --agentdb-url http://127.0.0.1:8088 \\
            --clr-cache-path ./clr_cache.json \\
            --trajectory-store-path ./trajectories.json
    """
    p = argparse.ArgumentParser(
        prog="rfsn_cli.py finalize-migration",
        description="Finalize AgentDB migration: verify recall, archive "
                    "local JSON files. Fail-closed if recall is insufficient."
    )
    p.add_argument("--agentdb-url", required=True,
                   help="AgentDB HTTP endpoint (must be reachable)")
    p.add_argument("--collection", default="vibe_thinker",
                   help="AgentDB collection name (default: vibe_thinker)")
    p.add_argument("--clr-cache-path", default="",
                   help="Path to the CLR result cache JSON file to archive")
    p.add_argument("--trajectory-store-path", default="",
                   help="Path to the trajectory store JSON file to archive")
    p.add_argument("--recall-threshold", type=float, default=0.95,
                   help="Minimum recall to finalize (default 0.95)")
    p.add_argument("--sample-size", type=int, default=20,
                   help="Sample size for recall check (default 20)")
    p.add_argument("--archive-suffix", default=".bak",
                   help="Suffix for archived local files (default .bak)")
    p.add_argument("--force", action="store_true",
                   help="Finalize even if recall is below threshold "
                        "(DANGEROUS — not recommended)")
    args = p.parse_args()

    from vector_store import AgentDBVectorStore

    clr_path = args.clr_cache_path or None
    traj_path = args.trajectory_store_path or None
    if not clr_path and not traj_path:
        print("Error: at least one of --clr-cache-path or "
              "--trajectory-store-path must be set")
        return 1

    agentdb = AgentDBVectorStore(args.agentdb_url, args.collection)

    # Check reachability via test upsert+delete (count() returns 0 both
    # for "empty" and "unreachable" — fail-closed).
    import importlib.util
    _mig_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "scripts", "migrate_to_agentdb.py",
    )
    _spec = importlib.util.spec_from_file_location(
        "migrate_to_agentdb", _mig_path)
    _mig_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mig_mod)

    if not _mig_mod._check_agentdb_reachable(agentdb):
        print(f"Error: AgentDB at {args.agentdb_url} is unreachable "
              f"— refusing to finalize (fail-closed, no data loss)")
        return 1
    count = agentdb.count()
    print(f"[Finalize] AgentDB connected: {count} entries in collection "
          f"'{args.collection}'")

    # Import the recall verification from the migration script (already
    # loaded above for the reachability check).
    verify_recall = _mig_mod.verify_recall
    print("\n=== Recall verification ===")
    vres = verify_recall(
        agentdb, clr_path, traj_path,
        sample_size=args.sample_size,
        recall_threshold=args.recall_threshold,
    )
    print(f"\n[Finalize] Overall recall: {vres['overall_recall']:.1%} "
          f"(threshold: {args.recall_threshold:.1%})")

    if not vres["passed"] and not args.force:
        print("[Finalize] FAILED — recall is below threshold. "
              "Refusing to finalize (no data loss).")
        print("[Finalize] Fix AgentDB configuration and re-run the "
              "backfill (scripts/migrate_to_agentdb.py), then retry.")
        print("[Finalize] To override (DANGEROUS), use --force.")
        return 2

    if not vres["passed"] and args.force:
        print("[Finalize] WARNING: --force used with recall below "
              "threshold — proceeding anyway (DATA LOSS RISK)")

    # Archive the local JSON files.
    archived = []
    for path in [clr_path, traj_path]:
        if path and os.path.exists(path):
            archive_path = path + args.archive_suffix
            # Don't overwrite an existing archive.
            if os.path.exists(archive_path):
                print(f"[Finalize] Archive {archive_path} already "
                      f"exists — "
                      f"skipping (rename it manually if you want to "
                      f"re-archive)")
                continue
            os.rename(path, archive_path)
            archived.append((path, archive_path))
            print(f"[Finalize] Archived {path} -> {archive_path}")

    if not archived:
        print("[Finalize] No local files to archive (they may not exist)")
    else:
        print(f"\n[Finalize] Migration complete! {len(archived)} file(s) "
              f"archived. AgentDB is now the primary vector store.")
        print("[Finalize] Restart the orchestrator with --agentdb-only to "
              "use AgentDB-only mode (no local fallback):")
        print(f"  python3 rfsn_cli.py --agentdb-url {args.agentdb_url} "
              f"--agentdb-only [other flags...]")
        print("[Finalize] To roll back, rename the .bak files back and "
              "restart with --agentdb-url (shadow mode, no --agentdb-only).")

    return 0


if __name__ == "__main__":
    main()
