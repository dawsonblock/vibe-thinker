"""Local web UI for vibe-thinker.

A FastAPI app that wraps the HybridReasoningOrchestrator and exposes
a single-page web interface for submitting queries, viewing results,
inspecting verification traces, and managing the system.

Run with:  python3 run_ui.py [--vibe URL] [--generalist URL] [--port 8000]
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from hybrid_orchestrator import HybridReasoningOrchestrator
from sandbox.network_allowlist import NetworkAllowList
from serialization import serialize_for_json
from web_security import configure_security

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize(obj: Any) -> Any:
    """Recursively convert dataclasses / sets / bytes to JSON-safe values.

    Delegates to the shared ``serialization.serialize_for_json`` utility
    (Phase 5 dedup). Kept as a thin wrapper for backward compatibility.
    """
    return serialize_for_json(obj)


def _load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file, skipping malformed lines."""
    if not os.path.exists(path):
        return []
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


# ---------------------------------------------------------------------------
# Broadcast abstraction (v1.2 — HA web UI fan-out)
# ---------------------------------------------------------------------------

@runtime_checkable
class Broadcaster(Protocol):
    """Fan-out for WebSocket job-update messages.

    Two implementations:
      - :class:`LocalBroadcaster` — sends directly to this process's
        WebSocket clients (the pre-v1.2 single-server behavior).
      - :class:`RedisBroadcaster` — publishes to a Redis Pub/Sub channel;
        a subscriber task in each UI server re-broadcasts to its local
        clients. This keeps multiple UI servers behind a load balancer
        in sync: a job update published by server A reaches the clients
        connected to server B.

    Lifecycle: ``start()`` launches the subscriber (Redis mode);
    ``stop()`` tears it down. ``publish(msg)`` is called by
    ``AppState.broadcast``.
    """

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def publish(self, msg: Dict[str, Any]) -> None: ...


class LocalBroadcaster:
    """Send messages directly to in-process WebSocket clients.

    This is the default and reproduces the exact pre-v1.2 behavior.
    """

    def __init__(self, send_fn: Callable[[Dict[str, Any]], "asyncio.Future"]):
        # send_fn is AppState._send_to_local_clients.
        self._send_fn = send_fn

    async def start(self) -> None:
        pass  # nothing to do

    async def stop(self) -> None:
        pass

    async def publish(self, msg: Dict[str, Any]) -> None:
        await self._send_fn(msg)


class RedisBroadcaster:
    """Publish job updates to a Redis Pub/Sub channel for HA fan-out.

    Each UI server runs a subscriber on the same channel. When any
    server publishes an update, every server (including the publisher)
    receives it via the subscriber and forwards it to its local
    WebSocket clients. This is uniform — no special-casing of the
    publisher's own clients, and no double-send (the publisher does
    NOT also call the local send_fn directly; only the subscriber does).

    Requires the ``redis`` package (``pip install redis>=5``).
    """

    def __init__(
        self,
        redis_url: str,
        channel: str,
        send_fn: Callable[[Dict[str, Any]], "asyncio.Future"],
        redis_client: Any = None,
    ):
        self._channel = channel
        self._send_fn = send_fn
        self._sub_task: Optional[asyncio.Task] = None
        self._stopping = False
        if redis_client is not None:
            self._pub = redis_client
            self._sub_client = redis_client
            self._owns_clients = False
        else:
            try:
                import redis.asyncio as aioredis
            except ImportError as e:
                raise ImportError(
                    "RedisBroadcaster requires the 'redis' package: "
                    "pip install redis>=5"
                ) from e
            self._pub = aioredis.from_url(redis_url, decode_responses=False)
            self._sub_client = aioredis.from_url(redis_url, decode_responses=False)
            self._owns_clients = True

    async def start(self) -> None:
        """Launch the subscriber task that forwards messages to local clients."""
        if self._sub_task is not None:
            return
        self._stopping = False
        self._sub_task = asyncio.create_task(self._subscriber_loop())

    async def stop(self) -> None:
        """Tear down the subscriber and close Redis connections we own."""
        self._stopping = True
        if self._sub_task is not None:
            self._sub_task.cancel()
            try:
                await self._sub_task
            except (asyncio.CancelledError, Exception):
                pass
            self._sub_task = None
        if self._owns_clients:
            try:
                await self._pub.aclose()
            except Exception:
                pass
            try:
                await self._sub_client.aclose()
            except Exception:
                pass

    async def _subscriber_loop(self) -> None:
        """Subscribe to the channel and forward messages to local clients."""
        import redis.asyncio as aioredis  # for PubSubError types
        backoff = 0.5
        while not self._stopping:
            try:
                pubsub = self._sub_client.pubsub()
                await pubsub.subscribe(self._channel)
                backoff = 0.5  # reset backoff after a successful connect
                while not self._stopping:
                    msg = await pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=1.0)
                    if msg is None:
                        continue
                    data = msg.get("data")
                    if data is None:
                        continue
                    if isinstance(data, bytes):
                        data = data.decode("utf-8", errors="replace")
                    try:
                        parsed = json.loads(data)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    try:
                        await self._send_fn(parsed)
                    except Exception:
                        pass  # a dead client is cleaned up by send_fn
                try:
                    await pubsub.unsubscribe(self._channel)
                    await pubsub.aclose()
                except Exception:
                    pass
            except asyncio.CancelledError:
                break
            except Exception:
                # Reconnect with backoff. Redis outages degrade to
                # single-server broadcasts (publish will also fail and
                # the local send_fn is called as a fallback there).
                if self._stopping:
                    break
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10.0)

    async def publish(self, msg: Dict[str, Any]) -> None:
        """Publish to the Redis channel.

        If the publish fails (Redis down), fall back to sending directly
        to local clients so a Redis outage does not silently drop UI
        updates on the publishing server.
        """
        payload = json.dumps(_serialize(msg), default=str)
        try:
            await self._pub.publish(self._channel, payload)
        except Exception:
            # Fail-closed-fallback: Redis down -> at least this server's
            # own clients still see the update.
            try:
                await self._send_fn(msg)
            except Exception:
                pass


def make_broadcaster(
    redis_url: Optional[str],
    send_fn: Callable[[Dict[str, Any]], "asyncio.Future"],
    channel: str = "vt:web:updates",
    redis_client: Any = None,
) -> Broadcaster:
    """Factory: Redis Pub/Sub when redis_url is set, else local-only."""
    if redis_url and redis_url.strip():
        return RedisBroadcaster(
            redis_url=redis_url, channel=channel,
            send_fn=send_fn, redis_client=redis_client,
        )
    return LocalBroadcaster(send_fn=send_fn)


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

class AppState:
    """Holds the orchestrator and active jobs."""

    def __init__(
        self,
        orchestrator: HybridReasoningOrchestrator,
        broadcaster: Optional[Broadcaster] = None,
    ):
        self.orch = orchestrator
        self.jobs: Dict[str, Dict[str, Any]] = {}  # job_id -> job dict
        self.ws_clients: List[WebSocket] = []
        self._started = False
        # Broadcaster defaults to local-only (pre-v1.2 behavior).
        self.broadcaster: Broadcaster = broadcaster or LocalBroadcaster(
            self._send_to_local_clients)

    async def ensure_started(self):
        if not self._started:
            await self.orch.start()
            await self.broadcaster.start()
            self._started = True

    async def cleanup(self):
        if self._started:
            await self.broadcaster.stop()
            await self.orch.cleanup()
            self._started = False

    def add_job(self, job_id: str, query: str, force_route: Optional[str] = None):
        self.jobs[job_id] = {
            "job_id": job_id,
            "query": query,
            "force_route": force_route,
            "status": "pending",
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
            "duration_ms": None,
        }

    def update_job(self, job_id: str, **kwargs):
        if job_id in self.jobs:
            self.jobs[job_id].update(kwargs)

    async def _send_to_local_clients(self, msg: Dict[str, Any]):
        """Send a message to all connected local WebSocket clients.

        This is the actual wire send. It is called by LocalBroadcaster
        directly, and by RedisBroadcaster's subscriber loop (so a
        published message reaches this server's own clients).
        """
        text = json.dumps(_serialize(msg), default=str)
        dead = []
        for ws in self.ws_clients:
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.ws_clients.remove(ws)

    async def broadcast(self, msg: Dict[str, Any]):
        """Fan out a message to all WebSocket clients.

        Delegates to the broadcaster: local-only by default, or Redis
        Pub/Sub when a redis_url was supplied to ``create_app`` (so
        multiple UI servers stay in sync).
        """
        await self.broadcaster.publish(msg)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    orchestrator: HybridReasoningOrchestrator,
    redis_url: Optional[str] = None,
    redis_client: Any = None,
    api_key: Optional[str] = None,
    allowed_origins: Optional[List[str]] = None,
    rate_limit_per_minute: int = 0,
    max_request_body_bytes: int = 0,
) -> FastAPI:
    """Create the vibe-thinker web UI FastAPI app.

    Args:
        orchestrator: the HybridReasoningOrchestrator instance.
        redis_url: optional Redis URL for HA multi-server WebSocket
            fan-out. When set, job updates are published to a Redis
            Pub/Sub channel so multiple UI servers behind a load
            balancer stay in sync. When None (default), updates are
            sent only to this server's local clients (pre-v1.2 behavior).
        redis_client: optional pre-built async Redis client (for testing
            with fakeredis). When provided, redis_url is ignored.
        api_key: If set, all HTTP requests must include ``X-API-Key``.
            If None, checks ``VIBE_THINKER_API_KEY`` env var.
        allowed_origins: CORS allowed origins. None = localhost only.
        rate_limit_per_minute: Max requests per IP per minute. 0 = disabled.
        max_request_body_bytes: Max request body size. 0 = disabled.
    """
    app = FastAPI(title="vibe-thinker UI", docs_url="/api/docs")
    # Security middleware: API key auth, CORS, rate limiting, body size.
    configure_security(
        app,
        api_key=api_key,
        allowed_origins=allowed_origins,
        rate_limit_per_minute=rate_limit_per_minute,
        max_request_body_bytes=max_request_body_bytes,
        exempt_paths={"/", "/api/health"},
    )
    # AppState defaults to a LocalBroadcaster. When redis_url is set,
    # swap in a RedisBroadcaster after the state exists (it needs the
    # state's _send_to_local_clients as the subscriber callback).
    state = AppState(orchestrator)
    if redis_url or redis_client:
        state.broadcaster = make_broadcaster(
            redis_url=redis_url,
            send_fn=state._send_to_local_clients,
            redis_client=redis_client,
        )
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    async def index():
        return (static_dir / "index.html").read_text()

    # ------------------------------------------------------------------
    # REST API
    # ------------------------------------------------------------------
    @app.get("/api/status")
    async def api_status():
        """System status — endpoints, config, job counts."""
        orch = state.orch
        return {
            "specialist_endpoint": getattr(orch, "vibe_endpoint", None),
            "generalist_endpoint": getattr(orch, "generalist_endpoint", None),
            "code_specialist_endpoint": getattr(orch, "code_specialist_endpoint", None),
            "use_clr": getattr(orch, "use_clr", False),
            "use_clr_cache": getattr(orch, "use_clr_cache", False),
            "use_embedding_router": getattr(orch, "use_embedding_router", False),
            "use_trajectory_store": getattr(orch, "use_trajectory_store", False),
            "code_candidates": getattr(orch, "code_candidates", 0),
            "max_repair_attempts": getattr(orch, "max_repair_attempts", 0),
            "network_allowlist": (
                orch._network_allowlist.summary()
                if hasattr(orch, "_network_allowlist")
                   and orch._network_allowlist is not None
                else None
            ),
            "jobs": {
                "total": len(state.jobs),
                "pending": sum(1 for j in state.jobs.values() if j["status"] == "pending"),
                "running": sum(1 for j in state.jobs.values() if j["status"] == "running"),
                "done": sum(1 for j in state.jobs.values() if j["status"] == "done"),
                "error": sum(1 for j in state.jobs.values() if j["status"] == "error"),
            },
        }

    @app.post("/api/query")
    async def api_query(body: Dict[str, Any]):
        """Submit a query. Returns job_id immediately; result via WebSocket."""
        await state.ensure_started()
        query = body.get("query", "").strip()
        if not query:
            return JSONResponse({"error": "query is required"}, status_code=400)
        if len(query) > 10000:
            return JSONResponse(
                {"error": "query too long (max 10000 chars)"},
                status_code=400,
            )
        force_route = body.get("force_route") or None
        job_id = uuid.uuid4().hex[:12]
        state.add_job(job_id, query, force_route)

        # Launch the job in the background.
        asyncio.create_task(_run_job(state, job_id, query, force_route))
        return {"job_id": job_id, "status": "pending"}

    async def _run_job(state: AppState, job_id: str, query: str, force_route: Optional[str]):
        """Background task that runs the query and broadcasts updates."""
        state.update_job(job_id, status="running", started_at=time.time())
        await state.broadcast({"type": "job_update", "job": state.jobs[job_id]})
        t0 = time.time()
        try:
            result = await state.orch.run(query, force_route=force_route)
            duration_ms = int((time.time() - t0) * 1000)
            serialized = _serialize(result)
            state.update_job(
                job_id,
                status="done",
                finished_at=time.time(),
                result=serialized,
                duration_ms=duration_ms,
            )
            await state.broadcast({"type": "job_update", "job": state.jobs[job_id]})
        except Exception as e:
            duration_ms = int((time.time() - t0) * 1000)
            # Log full error server-side; send a generic message to clients
            # to avoid leaking internal details (file paths, library versions).
            print(f"[WebUI] Job {job_id} failed: {e}")
            state.update_job(
                job_id,
                status="error",
                finished_at=time.time(),
                error="internal error — see server logs for details",
                duration_ms=duration_ms,
            )
            await state.broadcast({"type": "job_update", "job": state.jobs[job_id]})

    @app.get("/api/jobs")
    async def api_jobs():
        """List all jobs."""
        return list(state.jobs.values())

    @app.get("/api/jobs/{job_id}")
    async def api_job(job_id: str):
        """Get a single job by ID."""
        job = state.jobs.get(job_id)
        if not job:
            return JSONResponse({"error": "not found"}, status_code=404)
        return job

    # ------------------------------------------------------------------ #
    # Federation coordinator endpoints (v0.4.1)
    # ------------------------------------------------------------------ #
    # These endpoints allow this UI server to act as a federation
    # coordinator. Worker nodes can claim pending jobs and report
    # results back. This integrates the federation directly into the
    # existing web UI — no separate federation_server.py needed when
    # the UI is already running.
    #
    # Worker flow:
    #   1. POST /api/jobs/claim {"worker_id": "laptop-2"} -> gets a job
    #   2. Worker runs the job locally
    #   3. POST /api/jobs/complete {"job_id": "...", "result": {...}}

    @app.post("/api/jobs/claim")
    async def api_jobs_claim(request: dict):
        """A worker claims a pending job for federated execution.

        Returns the highest-priority pending job, or {"job_id": null}
        if no jobs are pending. The job's status is set to "running"
        with claimed_by set to the worker_id.
        """
        worker_id = request.get("worker_id", "unknown")
        # Find the highest-priority pending job.
        pending = [
            (jid, j) for jid, j in state.jobs.items()
            if j["status"] == "pending"
        ]
        if not pending:
            return {"job_id": None, "status": "no_jobs"}
        # Sort by created_at (FIFO) — priority could be added here.
        pending.sort(key=lambda x: x[1]["created_at"])
        job_id, job = pending[0]
        state.update_job(
            job_id, status="running", started_at=time.time(),
            claimed_by=worker_id, heartbeat_at=time.time(),
        )
        await state.broadcast({"type": "job_update", "job": state.jobs[job_id]})
        return {
            "job_id": job_id,
            "query": job["query"],
            "force_route": job.get("force_route"),
            "status": "claimed",
        }

    @app.post("/api/jobs/heartbeat")
    async def api_jobs_heartbeat(request: dict):
        """Phase 4.2: A worker sends a heartbeat while processing a job.

        Updates the heartbeat_at timestamp so the reaper doesn't re-queue
        a job that's still being actively processed. Returns 404 if the
        job doesn't exist or isn't claimed by the given worker.
        """
        job_id = request.get("job_id", "")
        worker_id = request.get("worker_id", "unknown")
        if job_id not in state.jobs:
            return JSONResponse({"error": "job not found"}, status_code=404)
        job = state.jobs[job_id]
        if job["status"] != "running" or job.get("claimed_by") != worker_id:
            return JSONResponse(
                {"error": "job not claimed by this worker"},
                status_code=404,
            )
        state.update_job(job_id, heartbeat_at=time.time())
        return {"status": "ok"}

    @app.post("/api/jobs/complete")
    async def api_jobs_complete(request: dict):
        """A worker reports a completed job.

        Sets the job's result and status. Triggers WebSocket broadcast
        so the UI updates in real time.
        """
        job_id = request.get("job_id", "")
        if job_id not in state.jobs:
            return JSONResponse({"error": "job not found"}, status_code=404)
        result = request.get("result")
        error = request.get("error")
        if error:
            state.update_job(
                job_id, status="error", finished_at=time.time(),
                error=error, result=result,
            )
        else:
            state.update_job(
                job_id, status="done", finished_at=time.time(),
                result=result,
            )
        await state.broadcast({"type": "job_update", "job": state.jobs[job_id]})
        return {"status": "ok"}

    @app.get("/api/memory")
    async def api_memory(limit: int = 50):
        """View orchestrator memory vault (orchestrator_memory.jsonl)."""
        items = _load_jsonl("orchestrator_memory.jsonl")
        return items[-limit:]

    @app.get("/api/trajectories")
    async def api_trajectories(limit: int = 50):
        """View verified trajectory store."""
        path = "verified_trajectories.json"
        if not os.path.exists(path):
            return []
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data[-limit:]
        if isinstance(data, dict) and "trajectories" in data:
            return data["trajectories"][-limit:]
        return data

    @app.get("/api/audit-log")
    async def api_audit_log(limit: int = 100):
        """View bi-temporal audit log."""
        items = _load_jsonl("rfsn_jobs_bitemporal.jsonl")
        return items[-limit:]

    @app.post("/api/synthesize")
    async def api_synthesize(body: Dict[str, Any] = None):
        """Synthesize trajectories from memory."""
        await state.ensure_started()
        body = body or {}
        threshold = body.get("similarity_threshold", 0.85)
        max_traj = body.get("max_trajectories", 5)
        result = await state.orch.synthesize_trajectories(
            similarity_threshold=threshold,
            max_trajectories=max_traj,
        )
        return _serialize(result)

    # ------------------------------------------------------------------
    # WebSocket — live job updates
    # ------------------------------------------------------------------
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        state.ws_clients.append(ws)
        try:
            # Send current job list on connect.
            await ws.send_text(json.dumps({
                "type": "init",
                "jobs": list(state.jobs.values()),
            }, default=str))
            # Keep connection open; we broadcast updates from the app.
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            if ws in state.ws_clients:
                state.ws_clients.remove(ws)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    @app.on_event("shutdown")
    async def shutdown():
        await state.cleanup()

    return app
