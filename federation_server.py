"""Python-native federation server for multi-node reasoning swarms.

A lightweight FastAPI app that implements the federation HTTP API.
Any node can run this server to become a federation coordinator.
Workers connect via mTLS and claim/submit jobs.

This replaces the stubbed exo-federation Rust crate with a simple,
maintained Python implementation using standard asyncio + FastAPI.

Run standalone:
    python3 -m federation_server --host 0.0.0.0 --port 7443 \\
        --mtls-cert node.crt --mtls-key node.key --mtls-ca ca.crt

Or via the CLI:
    python3 rfsn_cli.py --federation-server --federation-url https://0.0.0.0:7443 \\
        --mtls-cert node.crt --mtls-key node.key --mtls-ca ca.crt

API:
    POST /submit    — publish a job to the swarm
    POST /claim     — a worker claims a pending job
    POST /complete  — a worker reports a completed job
    GET  /jobs      — list all jobs and their status
    GET  /health    — health check
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


# ---------------------------------------------------------------------------
# Job model
# ---------------------------------------------------------------------------

@dataclass
class FederatedJob:
    """A job in the federation."""
    job_id: str
    query: str
    priority: int = 0
    force_route: Optional[str] = None
    submitted_by: str = ""
    status: str = "pending"  # pending | claimed | done | error
    claimed_by: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    claimed_at: Optional[float] = None
    completed_at: Optional[float] = None


# ---------------------------------------------------------------------------
# Federation state
# ---------------------------------------------------------------------------

class FederationState:
    """In-memory federation state. Single-coordinator design.

    For multi-coordinator deployments, back this with Redis or a shared
    database. For the typical 2-10 node swarm, a single coordinator
    with in-memory state is sufficient and far simpler.
    """

    def __init__(self):
        self._jobs: Dict[str, FederatedJob] = {}
        self._lock = asyncio.Lock()

    async def submit(self, job_id: str, query: str, priority: int = 0,
                     force_route: Optional[str] = None,
                     submitted_by: str = "") -> FederatedJob:
        async with self._lock:
            job = FederatedJob(
                job_id=job_id or uuid.uuid4().hex[:12],
                query=query, priority=priority,
                force_route=force_route,
                submitted_by=submitted_by,
            )
            self._jobs[job.job_id] = job
            return job

    async def claim(self, worker_id: str) -> Optional[FederatedJob]:
        """Claim the highest-priority pending job. Returns None if no jobs."""
        async with self._lock:
            pending = [j for j in self._jobs.values() if j.status == "pending"]
            if not pending:
                return None
            # Sort by priority (desc), then by created_at (asc = FIFO).
            pending.sort(key=lambda j: (-j.priority, j.created_at))
            job = pending[0]
            job.status = "claimed"
            job.claimed_by = worker_id
            job.claimed_at = time.time()
            return job

    async def complete(self, job_id: str, result: Optional[Dict] = None,
                       error: Optional[str] = None) -> bool:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            job.status = "error" if error else "done"
            job.result = result
            job.error = error
            job.completed_at = time.time()
            return True

    async def list_jobs(self) -> List[Dict[str, Any]]:
        async with self._lock:
            return [
                {
                    "job_id": j.job_id, "query": j.query,
                    "priority": j.priority, "status": j.status,
                    "submitted_by": j.submitted_by,
                    "claimed_by": j.claimed_by,
                    "created_at": j.created_at,
                    "completed_at": j.completed_at,
                    "error": j.error,
                }
                for j in self._jobs.values()
            ]

    async def get_job(self, job_id: str) -> Optional[FederatedJob]:
        async with self._lock:
            return self._jobs.get(job_id)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_federation_app() -> FastAPI:
    """Create the federation server FastAPI app."""
    app = FastAPI(title="vibe-thinker federation", docs_url="/docs")
    state = FederationState()

    @app.get("/health")
    async def health():
        return {"status": "ok", "jobs": len(state._jobs)}

    @app.post("/submit")
    async def submit_job(req: Request):
        body = await req.json()
        job = await state.submit(
            job_id=body.get("job_id", ""),
            query=body.get("query", ""),
            priority=body.get("priority", 0),
            force_route=body.get("force_route"),
            submitted_by=body.get("submitted_by", ""),
        )
        return {"job_id": job.job_id, "status": "pending"}

    @app.post("/claim")
    async def claim_job(req: Request):
        body = await req.json()
        worker_id = body.get("worker_id", "unknown")
        job = await state.claim(worker_id)
        if job is None:
            return {"job_id": None, "status": "no_jobs"}
        return {
            "job_id": job.job_id, "query": job.query,
            "priority": job.priority, "force_route": job.force_route,
            "status": "claimed",
        }

    @app.post("/complete")
    async def complete_job(req: Request):
        body = await req.json()
        job_id = body.get("job_id", "")
        result = body.get("result")
        error = body.get("error")
        ok = await state.complete(job_id, result=result, error=error)
        if not ok:
            return JSONResponse({"error": "job not found"}, status_code=404)
        return {"status": "ok"}

    @app.get("/jobs")
    async def list_jobs():
        return await state.list_jobs()

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        job = await state.get_job(job_id)
        if not job:
            return JSONResponse({"error": "not found"}, status_code=404)
        return {
            "job_id": job.job_id, "query": job.query,
            "status": job.status, "result": job.result,
            "error": job.error,
        }

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    import uvicorn
    import ssl

    parser = argparse.ArgumentParser(description="vibe-thinker federation server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7443)
    parser.add_argument("--mtls-cert", default="", help="Server certificate (PEM)")
    parser.add_argument("--mtls-key", default="", help="Server private key (PEM)")
    parser.add_argument("--mtls-ca", default="", help="CA certificate for client verification")
    parser.add_argument("--no-tls", action="store_true", help="Disable TLS (dev only)")
    args = parser.parse_args()

    app = create_federation_app()

    if args.no_tls or not args.mtls_cert:
        print(f"[Federation] Running WITHOUT TLS (dev mode) on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
        return

    # mTLS: server presents cert, and verifies client certs against CA.
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(args.mtls_cert, args.mtls_key)
    if args.mtls_ca:
        ssl_context.load_verify_locations(args.mtls_ca)
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        print(f"[Federation] mTLS enabled on {args.host}:{args.port}")
    else:
        print(f"[Federation] TLS (no client verification) on {args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, ssl=ssl_context)


if __name__ == "__main__":
    main()
