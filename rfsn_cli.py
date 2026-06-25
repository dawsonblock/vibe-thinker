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
    p.add_argument("--audit-log",
                   default=os.environ.get("RFSN_AUDIT_LOG", "rfsn_jobs_bitemporal.jsonl"),
                   help="Bi-temporal audit log path (empty disables logging)")
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
    orchestrator = HybridReasoningOrchestrator(
        vibe_endpoint=args.vibe,
        generalist_endpoint=args.generalist,
        use_clr=args.use_clr,
        clr_k=args.clr_k,
        use_embedding_router=args.use_embedding_router,
    )
    queue = JobQueue(
        orchestrator,
        max_concurrent=args.max_concurrent,
        audit_log=args.audit_log or None,
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
