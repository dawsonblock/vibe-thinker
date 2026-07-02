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
  --redis-url URL        Redis URL for HA WebSocket fan-out
  --api-key KEY          Require X-API-Key header on all HTTP requests
  --allowed-origins ORIGINS  Comma-separated CORS origins (default: localhost)
  --rate-limit-per-minute N   Max requests per IP per minute (0 = disabled)
  --max-request-body-bytes N  Max request body size in bytes (0 = disabled)
"""

import argparse
import os
import sys


def _build_ui_parser() -> argparse.ArgumentParser:
    """Argparse parser for UI-only flags.

    Uses ``parse_known_args`` so every other flag (``--vibe``,
    ``--generalist``, ``--sandbox-network``, …) is left in ``remaining``
    and forwarded to the orchestrator's ``build_argparser``. This handles
    both ``--port 8000`` and ``--port=8000`` forms correctly, unlike the
    previous manual token walk.
    """
    p = argparse.ArgumentParser(
        prog="run_ui.py",
        description="Launch the vibe-thinker local web UI.",
        add_help=False,
    )
    p.add_argument("--port", type=int,
                   default=int(os.environ.get("VIBE_UI_PORT", "8000")),
                   help="UI server port (default: 8000)")
    p.add_argument("--host",
                   default=os.environ.get("VIBE_UI_HOST", "127.0.0.1"),
                   help="UI bind address (default: 127.0.0.1)")
    p.add_argument("--redis-url",
                   default=os.environ.get("VIBE_UI_REDIS_URL", ""),
                   help="Redis URL for HA WebSocket fan-out")
    p.add_argument("--api-key",
                   default=os.environ.get("VIBE_THINKER_API_KEY", ""),
                   help="Require X-API-Key header on all HTTP requests")
    p.add_argument("--allowed-origins",
                   default=os.environ.get("VIBE_UI_ALLOWED_ORIGINS", ""),
                   help="Comma-separated CORS origins (default: localhost)")
    p.add_argument("--rate-limit-per-minute", type=int,
                   default=int(os.environ.get("VIBE_UI_RATE_LIMIT", "0")),
                   help="Max requests per IP per minute (0 = disabled)")
    p.add_argument("--max-request-body-bytes", type=int,
                   default=int(os.environ.get("VIBE_UI_MAX_BODY_BYTES", "0")),
                   help="Max request body size in bytes (0 = disabled)")
    return p


def main():
    # Parse the UI-specific flags; everything else is forwarded to the
    # orchestrator argparser via parse_known_args.
    ui_parser = _build_ui_parser()
    ui_opts, remaining = ui_parser.parse_known_args()

    # Build the orchestrator argparser to parse the remaining flags.
    from rfsn_cli import (
        build_argparser,
        _build_network_allowlist,
        _build_retrieval_backend,
        _sandbox_network_mode,
    )
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
        agentdb_only=opts.agentdb_only,
        retrieval_backend=_build_retrieval_backend(opts),
        network_allowlist=_build_network_allowlist(opts),
        dns_resolver=opts.dns_resolver or None,
        sandbox_image=opts.sandbox_image or None,
        proxy_egress=opts.proxy_egress or None,
        network_mode=_sandbox_network_mode(opts.sandbox_network),
        docker_network=opts.docker_network or None,
        use_structured_output=opts.use_structured_output,
        specialist_transport=opts.specialist_transport,
        specialist_api_key=opts.specialist_api_key or None,
        specialist_model_name=opts.specialist_model_name or None,
        max_parse_repairs=opts.max_parse_repairs,
        prefer_encoder_nli=opts.prefer_encoder_nli,
        sona_sync_url=opts.sona_sync_url or None,
        sona_sync_interval=opts.sona_sync_interval,
        federation_secret=opts.federation_secret or None,
        allow_static_fallback=opts.allow_static_fallback,
    )

    from web.app import create_app
    import uvicorn

    # Resolve security options for the web layer.
    api_key = ui_opts.api_key or None
    allowed_origins = (
        [o.strip() for o in ui_opts.allowed_origins.split(",") if o.strip()]
        if ui_opts.allowed_origins else None
    )
    rate_limit = ui_opts.rate_limit_per_minute
    max_body = ui_opts.max_request_body_bytes

    app = create_app(
        orchestrator,
        redis_url=ui_opts.redis_url or None,
        api_key=api_key,
        allowed_origins=allowed_origins,
        rate_limit_per_minute=rate_limit,
        max_request_body_bytes=max_body,
    )
    if ui_opts.redis_url:
        print(f"[UI] HA WebSocket fan-out enabled: Redis Pub/Sub ({ui_opts.redis_url})")
    if api_key:
        print("[UI] API key authentication enabled")
    if rate_limit:
        print(f"[UI] Rate limiting enabled: {rate_limit} req/min per IP")
    if max_body:
        print(f"[UI] Request body size limit: {max_body} bytes")
    print(f"\n  vibe-thinker UI running at  http://{ui_opts.host}:{ui_opts.port}\n")
    uvicorn.run(app, host=ui_opts.host, port=ui_opts.port, log_level="info")


if __name__ == "__main__":
    main()
