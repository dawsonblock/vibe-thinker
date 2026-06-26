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
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from hybrid_orchestrator import HybridReasoningOrchestrator
from sandbox.network_allowlist import NetworkAllowList

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize(obj: Any) -> Any:
    """Recursively convert dataclasses / sets / bytes to JSON-safe values."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, set):
        return [_serialize(v) for v in obj]
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


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
# App state
# ---------------------------------------------------------------------------

class AppState:
    """Holds the orchestrator and active jobs."""

    def __init__(self, orchestrator: HybridReasoningOrchestrator):
        self.orch = orchestrator
        self.jobs: Dict[str, Dict[str, Any]] = {}  # job_id -> job dict
        self.ws_clients: List[WebSocket] = []
        self._started = False

    async def ensure_started(self):
        if not self._started:
            await self.orch.start()
            self._started = True

    async def cleanup(self):
        if self._started:
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

    async def broadcast(self, msg: Dict[str, Any]):
        """Send a message to all connected WebSocket clients."""
        text = json.dumps(_serialize(msg), default=str)
        dead = []
        for ws in self.ws_clients:
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.ws_clients.remove(ws)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    orchestrator: HybridReasoningOrchestrator,
) -> FastAPI:
    app = FastAPI(title="vibe-thinker UI", docs_url="/api/docs")
    state = AppState(orchestrator)
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
            state.update_job(
                job_id,
                status="error",
                finished_at=time.time(),
                error=str(e),
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
