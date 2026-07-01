#!/usr/bin/env python3
"""Launch the vibe-thinker local web UI.

Usage:
  python3 run_ui.py [options]

Options are passed through to the orchestrator. Common ones:
  --vibe URL              Specialist endpoint (default: http://127.0.0.1:8080)
  --generalist URL        Generalist endpoint (default: http://127.0.0.1:8081)
  --code-specialist URL   Dedicated code specialist endpoint
  --ruvllm-url URL        RuvLLM HTTP endpoint (overrides --vibe)
  --local-specialist-model PATH   Local .gguf model for in-process specialist
  --network-allowlist SPEC        Network allow-list for sandbox
  --network-allowlist-file PATH   Allow-list from file
  --dns-resolver IP               Restrict sandbox DNS
  --sandbox-image IMAGE           Docker sandbox image
  --port PORT            UI server port (default: 8000)
  --host HOST            UI bind address (default: 127.0.0.1)
"""

import argparse
import sys

# Parse only the UI-specific flags here; pass the rest to the orchestrator.
# We use parse_known_args so unknown args are forwarded to build_argparser.
def main():
    # First, extract --port and --host before the orchestrator argparser
    # sees them (it doesn't know about these).
    port = 8000
    host = "127.0.0.1"
    redis_url = ""
    remaining = []
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1]); i += 2; continue
        if args[i] == "--host" and i + 1 < len(args):
            host = args[i + 1]; i += 2; continue
        if args[i] == "--redis-url" and i + 1 < len(args):
            redis_url = args[i + 1]; i += 2; continue
        remaining.append(args[i]); i += 1

    # Build the orchestrator argparser to parse the remaining flags.
    from rfsn_cli import build_argparser, _build_network_allowlist, _build_retrieval_backend
    parser = build_argparser()
    # Remove the 'command' positional if present (UI doesn't use REPL commands).
    opts = parser.parse_args(remaining)

    # Resolve endpoints (same logic as rfsn_cli._amain).
    vibe_endpoint = opts.ruvllm_url or opts.vibe
    if opts.ruvllm_url:
        print(f"[UI] RuvLLM backend enabled: --vibe overridden to {opts.ruvllm_url}")

    code_candidates = opts.code_candidates
    if opts.fast_code_specialist:
        code_candidates = max(opts.code_candidates, 15)

    from hybrid_orchestrator import HybridReasoningOrchestrator

    orchestrator = HybridReasoningOrchestrator(
        vibe_endpoint=vibe_endpoint,
        generalist_endpoint=opts.generalist,
        code_specialist_endpoint=opts.code_specialist or None,
        code_candidates=code_candidates,
        max_repair_attempts=opts.max_repair_attempts,
        use_clr=opts.use_clr,
        clr_k=opts.clr_k,
        use_embedding_router=opts.use_embedding_router,
        use_trajectory_store=opts.use_trajectory_store,
        trajectory_store_path=opts.trajectory_store_path,
        fast_specialist=opts.fast_specialist,
        local_specialist_model=opts.local_specialist_model or None,
        local_specialist_n_ctx=opts.local_specialist_n_ctx,
        local_specialist_n_threads=opts.local_specialist_n_threads,
        local_specialist_pool_size=opts.local_specialist_pool_size,
        agentdb_url=opts.agentdb_url or None,
        retrieval_backend=_build_retrieval_backend(opts),
        network_allowlist=_build_network_allowlist(opts),
        dns_resolver=opts.dns_resolver or None,
        sandbox_image=opts.sandbox_image or None,
        docker_network=opts.docker_network or None,
    )

    from web.app import create_app
    import uvicorn

    app = create_app(orchestrator, redis_url=redis_url or None)
    if redis_url:
        print(f"[UI] HA WebSocket fan-out enabled: Redis Pub/Sub ({redis_url})")
    print(f"\n  vibe-thinker UI running at  http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
