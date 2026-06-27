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
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

# Input validation constants.
_MAX_QUERY_LEN = 10000
_MAX_JOB_ID_LEN = 128
_MAX_WORKER_ID_LEN = 128
_MAX_ERROR_LEN = 5000
_MAX_PRIORITY = 1000
_JOB_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,128}$")
_WORKER_ID_RE = re.compile(r"^[a-zA-Z0-9_\-\.]{1,128}$")

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from web_security import configure_security
from vt_config import config as vt_config


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
    # Phase 4.2: heartbeat-based zombie detection. Updated by the
    # /heartbeat endpoint while a worker is actively processing a job.
    # If heartbeat_at is older than claim_timeout, the reaper transitions
    # the job back to pending (zombie re-queue).
    heartbeat_at: Optional[float] = None


# ---------------------------------------------------------------------------
# Federation state
# ---------------------------------------------------------------------------

@runtime_checkable
class FederationStateProtocol(Protocol):
    """Pluggable federation state backend.

    Implementations:
      - :class:`InMemoryFederationState` — single-coordinator, asyncio.Lock.
        The default; identical to the pre-v1.2 behavior.
      - :class:`RedisFederationState` — multi-coordinator HA backed by Redis.
        Atomic claim via a Lua script over a sorted set + job hashes.

    All methods are async. ``claim`` MUST be atomic with respect to
    concurrent coordinators — a naive read-then-write in Python races
    across processes. The in-memory backend serializes via asyncio.Lock
    (sufficient within one process); the Redis backend uses a Lua script
    evaluated atomically inside Redis.
    """

    async def submit(self, job_id: str, query: str, priority: int = 0,
                     force_route: Optional[str] = None,
                     submitted_by: str = "") -> FederatedJob: ...

    async def claim(self, worker_id: str) -> Optional[FederatedJob]: ...

    async def complete(self, job_id: str, result: Optional[Dict] = None,
                       error: Optional[str] = None) -> bool: ...

    async def heartbeat(self, job_id: str, worker_id: str) -> bool: ...

    async def reap_stale_claims(self, timeout: float = 300.0) -> List[str]: ...

    async def list_jobs(self) -> List[Dict[str, Any]]: ...

    async def get_job(self, job_id: str) -> Optional[FederatedJob]: ...

    async def count(self) -> int: ...


class InMemoryFederationState:
    """In-memory federation state. Single-coordinator design.

    For multi-coordinator deployments, back this with Redis or a shared
    database (see :class:`RedisFederationState`). For the typical 2-10
    node swarm, a single coordinator with in-memory state is sufficient
    and far simpler.
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
            job.heartbeat_at = time.time()  # Phase 4.2: initial heartbeat
            return job

    async def complete(self, job_id: str, result: Optional[Dict] = None,
                       error: Optional[str] = None) -> bool:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            # State-transition guard: reject completions for jobs that are
            # already done or errored. This prevents a stale worker from
            # overwriting a completed job's result (e.g. after a reaper
            # re-queued and another worker already completed it).
            if job.status in ("done", "error"):
                return False
            job.status = "error" if error else "done"
            job.result = result
            job.error = error
            job.completed_at = time.time()
            return True

    async def heartbeat(self, job_id: str, worker_id: str) -> bool:
        """Update the heartbeat timestamp for a claimed job (Phase 4.2).

        Returns True if the job was found and is still claimed by the
        given worker, False otherwise. This prevents a stale worker from
        extending a claim that has already been re-queued to another
        worker.
        """
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status != "claimed":
                return False
            if job.claimed_by != worker_id:
                return False
            job.heartbeat_at = time.time()
            return True

    async def reap_stale_claims(self, timeout: float = 300.0) -> List[str]:
        """Re-queue claimed jobs whose heartbeat is older than timeout (Phase 4.2).

        Scans all claimed jobs; if heartbeat_at (or claimed_at when no
        heartbeat was ever sent) is older than `timeout` seconds, the job
        is transitioned back to pending and its claim fields are cleared.
        This handles zombie workers that crashed after claiming but before
        completing.

        Returns a list of re-queued job IDs (for logging/metrics).
        """
        reaped: List[str] = []
        now = time.time()
        async with self._lock:
            for job in self._jobs.values():
                if job.status != "claimed":
                    continue
                # Use heartbeat_at if available, fall back to claimed_at.
                last_contact = job.heartbeat_at or job.claimed_at
                if last_contact is None:
                    continue  # no timestamp — shouldn't happen for claimed
                if (now - last_contact) > timeout:
                    # Zombie — re-queue.
                    job.status = "pending"
                    job.claimed_by = None
                    job.claimed_at = None
                    job.heartbeat_at = None
                    reaped.append(job.job_id)
        return reaped

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

    async def count(self) -> int:
        async with self._lock:
            return len(self._jobs)


# Backward-compat alias. Existing tests and callers import
# ``FederationState`` and instantiate it as the in-memory backend.
# ``FederationStateProtocol`` is the new structural type for type hints.
FederationState = InMemoryFederationState


# ---------------------------------------------------------------------------
# Redis-backed HA federation state (v1.2)
# ---------------------------------------------------------------------------

# Redis key layout:
#   vt:fed:job:<job_id>        — Redis Hash with all job fields
#   vt:fed:pending             — Redis Sorted Set; score = priority (negated
#                                for desc order is done in Lua), tie-break by
#                                member = "<created_at>:<job_id>" so ZPOPMIN
#                                yields highest-priority, FIFO-within-priority.
#   vt:fed:ids                 — Redis Set of all job ids (for list_jobs)
#
# Claim atomicity: a single Lua script reads the best member from the
# sorted set, removes it, and marks the job hash as claimed — all inside
# one Redis EVAL. This is race-free across coordinators because Redis
# executes scripts atomically (single-threaded command loop).

_CLAIM_LUA = """
-- KEYS[1] = vt:fed:pending (sorted set)
-- ARGV[1] = worker_id
-- ARGV[2] = claimed_at (float seconds)
-- ARGV[3] = job hash prefix "vt:fed:job:"
-- Returns: job_id (string) or nil if no pending jobs.
local best = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
if not best or #best == 0 then
    return nil
end
local member = best[1]
-- member is "<created_at>:<job_id>"
local sep = string.find(member, ':')
local job_id = string.sub(member, sep + 1)
redis.call('ZREM', KEYS[1], member)
local key = ARGV[3] .. job_id
redis.call('HSET', key, 'status', 'claimed',
           'claimed_by', ARGV[1],
           'claimed_at', ARGV[2],
           'heartbeat_at', ARGV[2])
return job_id
"""


def _job_from_hash(job_id: str, h: Dict[bytes, bytes]) -> Optional[FederatedJob]:
    """Reconstruct a FederatedJob from a Redis hash (byte keys/values)."""
    if not h:
        return None

    def _g(key: str, default: str = "") -> str:
        v = h.get(key.encode()) or h.get(key)
        if v is None:
            return default
        return v.decode() if isinstance(v, bytes) else str(v)

    def _gf(key: str) -> Optional[float]:
        v = h.get(key.encode()) or h.get(key)
        if v is None or v == b"" or v == "":
            return None
        try:
            return float(v.decode() if isinstance(v, bytes) else v)
        except (ValueError, TypeError):
            return None

    def _gi(key: str, default: int = 0) -> int:
        v = h.get(key.encode()) or h.get(key)
        if v is None or v == b"" or v == "":
            return default
        try:
            return int(v.decode() if isinstance(v, bytes) else v)
        except (ValueError, TypeError):
            return default

    def _gd(key: str) -> Optional[Dict[str, Any]]:
        v = h.get(key.encode()) or h.get(key)
        if v is None or v == b"" or v == "":
            return None
        try:
            return json.loads(v.decode() if isinstance(v, bytes) else v)
        except (json.JSONDecodeError, ValueError):
            return None

    status = _g("status", "pending")
    return FederatedJob(
        job_id=job_id,
        query=_g("query"),
        priority=_gi("priority"),
        # force_route is a string in the dataclass; store as plain string.
        force_route=(_g("force_route") or None),
        submitted_by=_g("submitted_by"),
        status=status,
        claimed_by=_g("claimed_by") or None,
        result=_gd("result"),
        error=_g("error") or None,
        created_at=_gf("created_at") or time.time(),
        claimed_at=_gf("claimed_at"),
        completed_at=_gf("completed_at"),
        heartbeat_at=_gf("heartbeat_at"),
    )


class RedisFederationState:
    """Multi-coordinator HA federation state backed by Redis.

    Enables deploying multiple ``federation_server.py`` instances behind
    a load balancer, all sharing one Redis cluster. The claim operation
    is atomic across coordinators via a Lua script (Redis executes EVAL
    single-threaded, so two coordinators can never claim the same job).

    Jobs are stored as Redis Hashes (``vt:fed:job:<id>``); the pending
    queue is a Redis Sorted Set (``vt:fed:pending``) scored so that
    ``ZRANGE ... 0 0`` returns the highest-priority, FIFO-within-priority
    job. A Redis Set (``vt:fed:ids``) tracks all job ids for
    ``list_jobs``.

    Args:
        redis_url: Redis connection URL (e.g. ``redis://localhost:6379/0``).
        key_prefix: namespace prefix for all Redis keys (default ``vt:fed``).
        redis_client: optional pre-built async Redis client (for testing
            with ``fakeredis.aioredis.FakeRedis``). When provided,
            ``redis_url`` is ignored.

    Requires the ``redis`` package (``pip install redis>=5``). When Redis
    is unreachable, methods raise ``ConnectionError`` — the caller
    (``create_federation_app``) is expected to fail-closed or retry.
    """

    _CLAIM_RETRIES = 8  # WATCH/MULTI optimistic-locking retry bound.

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "vt:fed",
        redis_client: Any = None,
    ):
        self._prefix = key_prefix
        self._job_prefix = f"{key_prefix}:job:"
        self._pending_key = f"{key_prefix}:pending"
        self._ids_key = f"{key_prefix}:ids"
        self._claim_script_sha: Optional[str] = None
        if redis_client is not None:
            self._redis = redis_client
        else:
            try:
                import redis.asyncio as aioredis
            except ImportError as e:
                raise ImportError(
                    "RedisFederationState requires the 'redis' package: "
                    "pip install redis>=5"
                ) from e
            self._redis = aioredis.from_url(redis_url, decode_responses=False)

    async def _ensure_claim_script(self) -> None:
        """Load the claim Lua script once; cache its SHA for EVALSHA."""
        if self._claim_script_sha is not None:
            return
        self._claim_script_sha = await self._redis.script_load(_CLAIM_LUA)

    async def submit(self, job_id: str, query: str, priority: int = 0,
                     force_route: Optional[str] = None,
                     submitted_by: str = "") -> FederatedJob:
        job = FederatedJob(
            job_id=job_id or uuid.uuid4().hex[:12],
            query=query, priority=priority,
            force_route=force_route,
            submitted_by=submitted_by,
        )
        key = f"{self._job_prefix}{job.job_id}"
        # Sorted-set member encodes tie-break: "<created_at>:<job_id>".
        # Score is the priority; we negate in Lua by reading ZRANGE 0 0
        # only works if higher priority sorts first. Redis sorted sets
        # order by score ascending, so to get highest-priority-first we
        # use score = -priority. Within the same score, members sort
        # lexicographically — and "<created_at>:<job_id>" with a
        # fixed-precision created_at gives FIFO. We use the float
        # created_at directly; for sub-second FIFO correctness across
        # jobs submitted in the same second, the lexicographic compare
        # of the float string is monotonic for identical precision.
        member = f"{job.created_at:.6f}:{job.job_id}"
        score = float(-job.priority)
        pipe = self._redis.pipeline()
        pipe.hset(key, mapping={
            "job_id": job.job_id,
            "query": job.query,
            "priority": str(job.priority),
            "force_route": job.force_route or "",
            "submitted_by": job.submitted_by,
            "status": "pending",
            "claimed_by": "",
            "result": "",
            "error": "",
            "created_at": str(job.created_at),
            "claimed_at": "",
            "completed_at": "",
        })
        pipe.zadd(self._pending_key, {member: score})
        pipe.sadd(self._ids_key, job.job_id)
        await pipe.execute()
        return job

    async def claim(self, worker_id: str) -> Optional[FederatedJob]:
        """Atomically claim the highest-priority pending job.

        Two strategies, both race-free across coordinators:

          1. **Lua (preferred on real Redis).** A single EVALSHA script
             reads the best sorted-set member, removes it, and marks the
             job hash claimed — all inside one atomic Redis command.
          2. **WATCH/MULTI fallback.** Used when the Redis server (or
             ``fakeredis`` in tests) does not support scripting. WATCHes
             the pending sorted set; if another coordinator modifies it
             between WATCH and EXEC, EXEC returns nil and we retry
             (optimistic locking, bounded to ``_CLAIM_RETRIES``).

        Returns ``None`` if no jobs are pending.
        """
        # Try the Lua path first.
        try:
            await self._ensure_claim_script()
            job_id = await self._redis.evalsha(
                self._claim_script_sha, 1, self._pending_key,
                worker_id, str(time.time()), self._job_prefix,
            )
            if not job_id:
                return None
            job_id = job_id.decode() if isinstance(job_id, bytes) else job_id
            h = await self._redis.hgetall(f"{self._job_prefix}{job_id}")
            return _job_from_hash(job_id, h)
        except Exception as e:
            # Scripting unsupported (fakeredis) or NOSCRIPT after flush.
            # Fall through to the WATCH/MULTI path. Mark the cached SHA
            # invalid so subsequent claims skip the failed EVALSHA.
            self._claim_script_sha = None
            if "NOSCRIPT" not in str(e) and "unknown command" not in str(e) \
                    and "script" not in str(e).lower():
                # Unexpected error — don't silently swallow it.
                raise
        return await self._claim_via_watch(worker_id)

    async def _claim_via_watch(self, worker_id: str) -> Optional[FederatedJob]:
        """Optimistic-locking claim fallback (WATCH/MULTI/EXEC).

        Bounded retry: if EXEC is aborted because another coordinator
        touched the pending set, we retry up to ``_CLAIM_RETRIES`` times.
        Uses the pipeline's WATCH (not the client's) to avoid the
        redis-py deprecation warning.
        """
        for _ in range(self._CLAIM_RETRIES):
            pipe = self._redis.pipeline()
            try:
                await pipe.watch(self._pending_key)
                best = await pipe.zrange(
                    self._pending_key, 0, 0, withscores=True)
                if not best:
                    await pipe.unwatch()
                    return None
                member = best[0][0]
                if isinstance(member, bytes):
                    member_str = member.decode()
                else:
                    member_str = str(member)
                sep = member_str.find(":")
                job_id = member_str[sep + 1:]
                pipe.multi()
                pipe.zrem(self._pending_key, member)
                pipe.hset(f"{self._job_prefix}{job_id}", mapping={
                    "status": "claimed",
                    "claimed_by": worker_id,
                    "claimed_at": str(time.time()),
                    "heartbeat_at": str(time.time()),
                })
                result = await pipe.execute()
                # If EXEC was aborted (nil), result is None — retry.
                if result is None:
                    continue
                h = await self._redis.hgetall(f"{self._job_prefix}{job_id}")
                return _job_from_hash(job_id, h)
            except Exception:
                # WATCH interrupted — retry.
                continue
        return None

    async def complete(self, job_id: str, result: Optional[Dict] = None,
                       error: Optional[str] = None) -> bool:
        key = f"{self._job_prefix}{job_id}"
        exists = await self._redis.exists(key)
        if not exists:
            return False
        # State-transition guard: reject completions for jobs already
        # done or errored (prevents stale workers from overwriting results).
        current_status = await self._redis.hget(key, "status")
        if current_status:
            if isinstance(current_status, bytes):
                current_status = current_status.decode()
            if current_status in ("done", "error"):
                return False
        status = "error" if error else "done"
        mapping: Dict[str, str] = {
            "status": status,
            "completed_at": str(time.time()),
        }
        if result is not None:
            mapping["result"] = json.dumps(result)
        if error:
            mapping["error"] = error
        await self._redis.hset(key, mapping=mapping)
        return True

    async def heartbeat(self, job_id: str, worker_id: str) -> bool:
        """Update the heartbeat timestamp for a claimed job (Phase 4.2).

        Returns True if the job was found, is claimed, and belongs to the
        given worker. Atomically checks status + claimed_by before updating
        to prevent a stale worker from extending a re-queued claim.
        """
        key = f"{self._job_prefix}{job_id}"
        h = await self._redis.hgetall(key)
        if not h:
            return False
        job = _job_from_hash(job_id, h)
        if job is None or job.status != "claimed" or job.claimed_by != worker_id:
            return False
        await self._redis.hset(key, "heartbeat_at", str(time.time()))
        return True

    async def reap_stale_claims(self, timeout: float = 300.0) -> List[str]:
        """Re-queue claimed jobs whose heartbeat is older than timeout (Phase 4.2).

        Scans all job hashes for claimed status; if heartbeat_at (or
        claimed_at fallback) is older than `timeout`, transitions the job
        back to pending and re-adds it to the pending sorted set.
        """
        now = time.time()
        ids = await self._redis.smembers(self._ids_key)
        ids = [i.decode() if isinstance(i, bytes) else i for i in ids]
        reaped: List[str] = []
        for jid in ids:
            key = f"{self._job_prefix}{jid}"
            h = await self._redis.hgetall(key)
            if not h:
                continue
            job = _job_from_hash(jid, h)
            if job is None or job.status != "claimed":
                continue
            last_contact = job.heartbeat_at or job.claimed_at
            if last_contact is None:
                continue
            if (now - last_contact) > timeout:
                # Zombie — re-queue.
                pipe = self._redis.pipeline()
                pipe.hset(key, mapping={
                    "status": "pending",
                    "claimed_by": "",
                    "claimed_at": "",
                    "heartbeat_at": "",
                })
                # Re-add to pending sorted set. Recompute member + score.
                created_at = job.created_at or now
                priority = job.priority
                member = f"{created_at:.6f}:{jid}"
                score = float(-priority)
                pipe.zadd(self._pending_key, {member: score})
                await pipe.execute()
                reaped.append(jid)
        return reaped

    async def list_jobs(self) -> List[Dict[str, Any]]:
        ids = await self._redis.smembers(self._ids_key)
        ids = [i.decode() if isinstance(i, bytes) else i for i in ids]
        if not ids:
            return []
        pipe = self._redis.pipeline()
        for jid in ids:
            pipe.hgetall(f"{self._job_prefix}{jid}")
        hashes = await pipe.execute()
        out: List[Dict[str, Any]] = []
        for jid, h in zip(ids, hashes):
            if not h:
                continue
            job = _job_from_hash(jid, h)
            if job is None:
                continue
            out.append({
                "job_id": job.job_id, "query": job.query,
                "priority": job.priority, "status": job.status,
                "submitted_by": job.submitted_by,
                "claimed_by": job.claimed_by,
                "created_at": job.created_at,
                "completed_at": job.completed_at,
                "error": job.error,
            })
        return out

    async def get_job(self, job_id: str) -> Optional[FederatedJob]:
        h = await self._redis.hgetall(f"{self._job_prefix}{job_id}")
        if not h:
            return None
        return _job_from_hash(job_id, h)

    async def count(self) -> int:
        return int(await self._redis.scard(self._ids_key))


def make_federation_state(
    redis_url: Optional[str] = None,
    redis_client: Any = None,
) -> FederationStateProtocol:
    """Factory: select the federation state backend.

    Precedence:
      1. ``redis_client`` (testing) or non-empty ``redis_url`` ->
         :class:`RedisFederationState`.
      2. Otherwise -> :class:`InMemoryFederationState` (the default,
         unchanged single-coordinator behavior).
    """
    if redis_client is not None or (redis_url and redis_url.strip()):
        return RedisFederationState(
            redis_url=redis_url or "redis://localhost:6379/0",
            redis_client=redis_client,
        )
    return InMemoryFederationState()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_federation_app(
    state: Optional[FederationStateProtocol] = None,
    federation_secret: Optional[str] = None,
    api_key: Optional[str] = None,
    allowed_origins: Optional[List[str]] = None,
    rate_limit_per_minute: int = 0,
    max_request_body_bytes: int = 0,
) -> FastAPI:
    """Create the federation server FastAPI app.

    Args:
        state: an optional federation state backend. Defaults to
            :class:`InMemoryFederationState` (the pre-v1.2 single-
            coordinator behavior). Pass a :class:`RedisFederationState`
            for multi-coordinator HA deployments.
        federation_secret: v3.0 zero-trust encryption secret. When set,
            the server decrypts incoming payloads encrypted with the
            same secret by worker nodes. See FederatedJobQueue.
        api_key: If set, all requests must include ``X-API-Key`` header.
            If None, checks ``VIBE_THINKER_API_KEY`` env var. If neither
            is set, auth is skipped (dev mode only).
        allowed_origins: CORS allowed origins. None = localhost only.
        rate_limit_per_minute: Max requests per IP per minute. 0 = disabled.
        max_request_body_bytes: Max request body size. 0 = disabled.
    """
    app = FastAPI(title="vibe-thinker federation", docs_url="/docs")
    if state is None:
        state = InMemoryFederationState()

    # Security middleware: API key auth, CORS, rate limiting, body size.
    configure_security(
        app,
        api_key=api_key,
        allowed_origins=allowed_origins,
        rate_limit_per_minute=rate_limit_per_minute,
        max_request_body_bytes=max_request_body_bytes,
        exempt_paths={"/health"},
    )

    # v3.0: Set up decryption if a secret is provided.
    _fernet = None
    if federation_secret:
        try:
            from cryptography.fernet import Fernet
            import base64
            import hashlib
            key = base64.urlsafe_b64encode(
                hashlib.sha256(federation_secret.encode()).digest()
            )
            _fernet = Fernet(key)
        except ImportError:
            pass

    def _decrypt(body: dict) -> dict:
        """Decrypt an incoming payload if it's encrypted."""
        if "__encrypted__" not in body:
            return body  # Plaintext (no encryption configured by sender)
        if _fernet is None:
            raise ValueError("Received encrypted payload but no "
                             "federation_secret configured on server")
        import json as _json
        ciphertext = body["__encrypted__"].encode("ascii")
        plaintext = _fernet.decrypt(ciphertext).decode("utf-8")
        return _json.loads(plaintext)

    def _encrypt(data: dict) -> dict:
        """Encrypt a response payload for zero-trust federation (v3.0).

        If encryption is enabled, the payload is JSON-serialized and
        encrypted. If not enabled, returns the payload unchanged.
        """
        if _fernet is None:
            return data
        import json as _json
        plaintext = _json.dumps(data, default=str).encode("utf-8")
        ciphertext = _fernet.encrypt(plaintext).decode("ascii")
        return {"__encrypted__": ciphertext}

    @app.get("/health")
    async def health():
        return {"status": "ok", "jobs": await state.count()}

    @app.post("/submit")
    async def submit_job(req: Request):
        body = await req.json()
        body = _decrypt(body)
        query = body.get("query", "")
        if not query or not isinstance(query, str):
            return JSONResponse({"error": "query is required"}, status_code=400)
        if len(query) > _MAX_QUERY_LEN:
            return JSONResponse(
                {"error": f"query too long (max {_MAX_QUERY_LEN} chars)"},
                status_code=400,
            )
        job_id = body.get("job_id", "")
        if job_id and not _JOB_ID_RE.match(job_id):
            return JSONResponse(
                {"error": "job_id must be alphanumeric/underscore/hyphen, "
                 f"max {_MAX_JOB_ID_LEN} chars"},
                status_code=400,
            )
        priority = body.get("priority", 0)
        if not isinstance(priority, int) or abs(priority) > _MAX_PRIORITY:
            return JSONResponse(
                {"error": f"priority must be an integer in [{-_MAX_PRIORITY}, {_MAX_PRIORITY}]"},
                status_code=400,
            )
        job = await state.submit(
            job_id=job_id,
            query=query,
            priority=priority,
            force_route=body.get("force_route"),
            submitted_by=body.get("submitted_by", ""),
        )
        return {"job_id": job.job_id, "status": "pending"}

    @app.post("/claim")
    async def claim_job(req: Request):
        body = await req.json()
        worker_id = body.get("worker_id", "unknown")
        if not isinstance(worker_id, str) or not _WORKER_ID_RE.match(worker_id):
            return JSONResponse(
                {"error": "worker_id must be alphanumeric/underscore/hyphen/dot, "
                 f"max {_MAX_WORKER_ID_LEN} chars"},
                status_code=400,
            )
        job = await state.claim(worker_id)
        if job is None:
            return {"job_id": None, "status": "no_jobs"}
        # v3.0: Encrypt the claim response — the query contains user
        # data that must not leak in plaintext over the federation.
        return _encrypt({
            "job_id": job.job_id, "query": job.query,
            "priority": job.priority, "force_route": job.force_route,
            "status": "claimed",
        })

    @app.post("/complete")
    async def complete_job(req: Request):
        body = await req.json()
        body = _decrypt(body)
        job_id = body.get("job_id", "")
        if not job_id or not isinstance(job_id, str):
            return JSONResponse({"error": "job_id is required"}, status_code=400)
        result = body.get("result")
        error = body.get("error")
        if error and not isinstance(error, str):
            return JSONResponse({"error": "error must be a string"}, status_code=400)
        if error and len(error) > _MAX_ERROR_LEN:
            return JSONResponse(
                {"error": f"error message too long (max {_MAX_ERROR_LEN} chars)"},
                status_code=400,
            )
        ok = await state.complete(job_id, result=result, error=error)
        if not ok:
            return JSONResponse({"error": "job not found"}, status_code=404)
        return {"status": "ok"}

    # Phase 4.2: Heartbeat endpoint — workers call this periodically
    # while processing a claimed job to prove they're still alive. If
    # the heartbeat stops (worker crashed), the reaper will re-queue
    # the job for another worker to pick up.
    @app.post("/heartbeat")
    async def heartbeat_job(req: Request):
        body = await req.json()
        body = _decrypt(body)
        job_id = body.get("job_id", "")
        if not job_id or not isinstance(job_id, str):
            return JSONResponse({"error": "job_id is required"}, status_code=400)
        worker_id = body.get("worker_id", "unknown")
        if not isinstance(worker_id, str) or not _WORKER_ID_RE.match(worker_id):
            return JSONResponse(
                {"error": "invalid worker_id"},
                status_code=400,
            )
        ok = await state.heartbeat(job_id, worker_id)
        if not ok:
            return JSONResponse(
                {"error": "job not found or not claimed by this worker"},
                status_code=404,
            )
        return {"status": "ok"}

    # Phase 4.2: Background reaper task — periodically scans for stale
    # claims and re-queues them. Runs every claim_timeout / 2 seconds
    # (so a zombie is detected within ~1.5x the timeout).
    #
    # Timeout consistency (Phase 5 reliability fix):
    #   - reaper_claim_timeout = 180s (3x heartbeat interval of 60s)
    #   - This is well below the job execution timeout (600s in
    #     FederatedJobQueue), ensuring the reaper re-queues a zombie
    #     BEFORE the job execution timeout expires. Previously the
    #     reaper timeout (300s) was close to the job timeout (600s),
    #     which could cause duplicate work if a slow worker was
    #     mistakenly reaped.
    _reaper_claim_timeout = vt_config.reaper_claim_timeout  # 3x heartbeat
    _reaper_task: Optional[asyncio.Task] = None

    @app.on_event("startup")
    async def _start_reaper():
        async def _reaper_loop():
            while True:
                await asyncio.sleep(_reaper_claim_timeout / 2)
                try:
                    reaped = await state.reap_stale_claims(
                        timeout=_reaper_claim_timeout
                    )
                    if reaped:
                        print(
                            f"[Federation] Reaped {len(reaped)} zombie "
                            f"claim(s): {reaped} — re-queued as pending"
                        )
                except Exception as e:
                    print(f"[Federation] Reaper error: {e}")

        # Store the task so the shutdown handler can cancel it.
        nonlocal _reaper_task
        _reaper_task = asyncio.create_task(_reaper_loop())

    @app.on_event("shutdown")
    async def _stop_reaper():
        """Cancel the reaper task on shutdown to prevent resource leaks."""
        nonlocal _reaper_task
        if _reaper_task is not None:
            _reaper_task.cancel()
            try:
                await _reaper_task
            except asyncio.CancelledError:
                pass
            _reaper_task = None

    @app.get("/jobs")
    async def list_jobs():
        return _encrypt(await state.list_jobs())

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        job = await state.get_job(job_id)
        if not job:
            return JSONResponse({"error": "not found"}, status_code=404)
        return _encrypt({
            "job_id": job.job_id, "query": job.query,
            "status": job.status, "result": job.result,
            "error": job.error,
        })

    # v3.0: SONA gossip protocol — Distributed Brain endpoint.
    # Workers export their learned patterns (clustered trajectories,
    # MicroLoRA matrices) to the coordinator. The coordinator aggregates
    # them and broadcasts a global update back.
    #
    # Data flow:
    #   1. Worker POSTs to /api/sona/sync with its patterns + stats.
    #   2. Coordinator merges patterns into a global set (dedup by ID).
    #   3. Worker GETs /api/sona/sync to retrieve the global pattern set.
    #   4. Worker imports the global patterns into its local SONA engine.
    _sona_global_patterns: Dict[str, dict] = {}
    _sona_worker_stats: Dict[str, dict] = {}

    @app.post("/api/sona/sync")
    async def sona_sync(req: Request):
        """Receive SONA pattern export from a worker node.

        Body:
            worker_id: str — the exporting node's ID.
            patterns: list of dict — learned patterns (centroid, cluster_size,
                avg_quality, etc.).
            stats: dict — SONA stats (total_trajectories, updates, etc.).
        """
        body = await req.json()
        body = _decrypt(body)
        worker_id = body.get("worker_id", "unknown")
        patterns = body.get("patterns", [])
        if not isinstance(patterns, list):
            patterns = []
        stats = body.get("stats", {})
        if not isinstance(stats, dict):
            stats = {}

        # Merge patterns into the global set (dedup by pattern ID).
        for p in patterns:
            pid = str(p.get("id", ""))
            if pid:
                _sona_global_patterns[pid] = p

        _sona_worker_stats[worker_id] = stats

        return {
            "status": "ok",
            "global_pattern_count": len(_sona_global_patterns),
            "worker_count": len(_sona_worker_stats),
        }

    @app.get("/api/sona/sync")
    async def sona_get_global():
        """Retrieve the global aggregated SONA patterns.

        Returns the merged pattern set from all worker nodes. Workers
        call this periodically to update their local SONA engine with
        patterns learned by other nodes in the swarm.

        v3.0: Response is encrypted if federation_secret is configured.
        """
        return _encrypt({
            "patterns": list(_sona_global_patterns.values()),
            "worker_stats": _sona_worker_stats,
            "total_patterns": len(_sona_global_patterns),
        })

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
    parser.add_argument(
        "--redis-url", default="",
        help="Redis URL for HA multi-coordinator state (e.g. "
             "redis://localhost:6379/0). When set, the server uses "
             "RedisFederationState so multiple instances can share state. "
             "When empty (default), uses in-memory single-coordinator state.",
    )
    parser.add_argument(
        "--no-redis", action="store_true",
        help="Force in-memory state even if --redis-url is set (dev/testing).",
    )
    parser.add_argument(
        "--api-key", default="",
        help="API key for authenticating federation requests. If not set, "
             "checks VIBE_THINKER_API_KEY env var. If neither is set, "
             "auth is disabled (dev mode only — NOT for production).",
    )
    parser.add_argument(
        "--rate-limit", type=int, default=0,
        help="Max requests per IP per minute. 0 = disabled (default).",
    )
    parser.add_argument(
        "--max-body-bytes", type=int, default=0,
        help="Max request body size in bytes. 0 = disabled (default).",
    )
    args = parser.parse_args()

    redis_url = "" if args.no_redis else (args.redis_url or "").strip()
    state = make_federation_state(redis_url=redis_url)
    backend = "redis" if redis_url else "in-memory"
    app = create_federation_app(
        state=state,
        api_key=args.api_key or None,
        rate_limit_per_minute=args.rate_limit,
        max_request_body_bytes=args.max_body_bytes,
    )
    print(f"[Federation] State backend: {backend}")

    if args.no_tls or not args.mtls_cert:
        import warnings
        warnings.warn(
            "Federation server running WITHOUT TLS — this is insecure. "
            "Use --mtls-cert/--mtls-key/--mtls-ca for production.",
            stacklevel=2,
        )
        if not (args.api_key or os.environ.get("VIBE_THINKER_API_KEY")):
            warnings.warn(
                "No API key configured — federation endpoints are "
                "unauthenticated. Set --api-key or VIBE_THINKER_API_KEY.",
                stacklevel=2,
            )
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
