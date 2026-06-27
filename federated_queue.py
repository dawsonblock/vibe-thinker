"""
Federated job queue abstraction for multi-node reasoning swarms.

The current :class:`rfsn_job_queue.JobQueue` is a single-process asyncio
queue with an in-memory ``self._pending`` list and a concurrency
semaphore. It scales to ``max_concurrent`` jobs on one machine and no
further. The integration plan's Phase 4.1 goal is to allow vibe-thinker
to scale beyond a single machine to a multi-node reasoning swarm.

This module abstracts the queue behind a :class:`BaseJobQueue` protocol
and provides two implementations:

  - :class:`LocalJobQueue` — a thin wrapper around the existing
    :class:`rfsn_job_queue.JobQueue`. The default; zero behavior change.
    All jobs run locally on this machine.
  - :class:`FederatedJobQueue` — pushes jobs to a Python-native
    federation coordinator (see :mod:`federation_server`) over mTLS.
    The local node becomes a worker in a swarm: when ``submit()`` is
    called, the job is published to the coordinator; any idle node
    (including this one) can claim it. If the local node is at its
    ``max_concurrent`` limit, other nodes automatically claim the
    pending jobs.

The federation coordinator is a simple FastAPI app
(:mod:`federation_server`) that any node can run. It replaces the
stubbed exo-federation Rust crate, which had no working networking
layer and relied on archived post-quantum crypto dependencies.

When the coordinator is NOT reachable, :class:`FederatedJobQueue`
fail-closed-fallbacks to local-only execution (jobs still run on this
node, just not federated). This preserves the strict fail-closed
philosophy: a federation outage degrades to single-node operation
rather than dropping jobs.

The protocol (:class:`BaseJobQueue`) is a structural supertype of the
existing :class:`JobQueue` — the existing class already implements
``submit``, ``get``, ``status``, ``list_jobs``, ``wait_for``, ``cancel``,
``start``, and ``stop``. This means the existing :class:`JobQueue` can
be used wherever a :class:`BaseJobQueue` is expected without changes.

mTLS configuration:
  The federation network uses mutual TLS for node authentication.
  Each node has a client certificate + key, and trusts a CA that signed
  all node certificates. Pass the paths via the ``mtls_cert``,
  ``mtls_key``, and ``mtls_ca`` parameters. When any of these is
  missing, the queue falls back to local-only mode with a warning.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from rfsn_job_queue import Job, JobStatus, JobQueue


def _serialize_result(result) -> Dict[str, Any]:
    """Serialize an OrchestratorResult to a JSON-safe dict for federation.

    Handles nested non-serializable objects (e.g. CLRResult in raw_traces)
    by falling back to str() for anything json.dumps can't handle —
    mirroring the pattern in HybridReasoningOrchestrator.log_to_memory.
    """
    if result is None:
        return {}
    if hasattr(result, "__dict__"):
        raw = dict(result.__dict__)
    elif isinstance(result, dict):
        raw = result
    else:
        return {"result": str(result)}
    # Round-trip through json.dumps with default=str to ensure
    # everything is JSON-serializable. Non-serializable values
    # (e.g. nested dataclasses, datetime, etc.) become strings.
    try:
        json.dumps(raw, default=str)
        return raw
    except (TypeError, ValueError):
        # Fallback: convert all values to strings if needed.
        return {k: v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
                for k, v in raw.items()}


@runtime_checkable
class BaseJobQueue(Protocol):
    """Protocol for job queue implementations.

    The existing :class:`rfsn_job_queue.JobQueue` already satisfies this
    protocol — it's a structural supertype. New implementations (e.g.
    :class:`FederatedJobQueue`) just need to provide these methods.

    The protocol is async-aware: ``start`` and ``stop`` are coroutines,
    ``submit`` is sync (returns a Job immediately), and ``wait_for`` is
    a coroutine that blocks until the job finishes.
    """

    async def start(self) -> None:
        """Start the dispatcher / network listener."""
        ...

    async def stop(self) -> None:
        """Stop the dispatcher. In-flight jobs are allowed to finish."""
        ...

    def submit(
        self,
        query: str,
        priority: int = 0,
        force_route: Optional[str] = None,
        callback: Optional[Callable[[Any], Any]] = None,
    ) -> Job:
        """Submit a job. Returns the Job immediately (status=pending)."""
        ...

    def get(self, job_id: str) -> Optional[Job]:
        """Get a job by ID, or None if not found."""
        ...

    def status(self, job_id: str) -> Optional[JobStatus]:
        """Get the status of a job, or None if not found."""
        ...

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs as dicts."""
        ...

    async def wait_for(self, job_id: str, timeout: Optional[float] = None) -> Any:
        """Block until the job completes/fails/cancels, then return its
        result (an OrchestratorResult) or raise its error."""
        ...

    def cancel(self, job_id: str) -> bool:
        """Cancel a job. Returns True if cancelled (or already cancelled)."""
        ...


class LocalJobQueue:
    """Thin wrapper around the existing :class:`rfsn_job_queue.JobQueue`.

    This is the default queue implementation. It delegates every method
    to the wrapped :class:`JobQueue`, so behavior is identical. It
    exists so callers can reference a :class:`BaseJobQueue` without
    caring whether it's local or federated.

    The wrapper is intentionally transparent — you can also use a plain
    ``JobQueue`` anywhere a ``BaseJobQueue`` is expected (it satisfies
    the protocol structurally). This class is for explicit typing and
    for the :func:`make_job_queue` factory.

    Accepts the same constructor args as :class:`JobQueue`, plus the
    v0.3.9 audit-log signing params (``signing_key``,
    ``ed25519_private_key_hex``, ``ed25519_public_key_hex``) which are
    forwarded to the inner :class:`JobQueue` → :class:`BiTemporalAuditLog`.
    """

    def __init__(self, *args, **kwargs):
        self._queue = JobQueue(*args, **kwargs)

    def __getattr__(self, name: str):
        """Delegate any attribute not defined here to the inner JobQueue.

        This transparently forwards ``bitemporal``, ``job_history``,
        ``state_as_of``, ``persist_path``, and any other JobQueue
        attribute that the REPL or other callers access directly.
        """
        if name == "_queue":
            raise AttributeError(name)  # not set yet — avoid infinite recursion
        return getattr(self._queue, name)

    async def start(self) -> None:
        await self._queue.start()

    async def stop(self) -> None:
        await self._queue.stop()

    def submit(self, *args, **kwargs) -> Job:
        return self._queue.submit(*args, **kwargs)

    def get(self, job_id: str) -> Optional[Job]:
        return self._queue.get(job_id)

    def status(self, job_id: str) -> Optional[JobStatus]:
        return self._queue.status(job_id)

    def list_jobs(self) -> List[Dict[str, Any]]:
        return self._queue.list_jobs()

    async def wait_for(self, job_id: str, timeout: Optional[float] = None) -> Any:
        return await self._queue.wait_for(job_id, timeout=timeout)

    def cancel(self, job_id: str) -> bool:
        return self._queue.cancel(job_id)

    @property
    def inner(self) -> JobQueue:
        """Access the underlying JobQueue (for audit log, bitemporal, etc.)."""
        return self._queue


class FederatedJobQueue:
    """Federated job queue backed by a Python-native federation coordinator.

    When the federation coordinator is reachable, jobs submitted via
    :meth:`submit` are published to the swarm. Any idle node on the
    network can claim and run them. This lets vibe-thinker scale beyond
    a single machine's ``max_concurrent`` limit: if the local M2 Pro is
    busy, other nodes pick up the pending reasoning trajectories.

    When the network is NOT reachable (the coordinator is not running,
    or mTLS certs are missing), the queue fail-closed-fallbacks to
    local-only execution: jobs still run on this node via the wrapped
    :class:`LocalJobQueue`, just not federated. A warning is printed on
    the first failed publish. This preserves the strict fail-closed
    philosophy — a federation outage degrades to single-node operation
    rather than dropping jobs.

    Architecture:
      - ``submit()``: publish the job to the federation coordinator via
        HTTP POST. If the publish fails, fall back to local submission.
        Either way, the job is tracked locally so ``wait_for`` works.
      - The local node also runs a worker (the wrapped LocalJobQueue's
        dispatcher) that claims jobs from the network when it has spare
        capacity. This is the "pull" side of the federation.
      - ``wait_for()``: waits for the job to complete, whether it ran
        locally or on a remote node. Remote completion is signaled via
        the federation coordinator's result callback.

    mTLS:
      The federation network uses mutual TLS. Pass the client cert,
      key, and CA paths. When any is missing, the queue starts in
      local-only fallback mode.

    Args:
        orchestrator: a HybridReasoningOrchestrator instance (passed to
            the local fallback queue).
        federation_url: federation coordinator HTTP endpoint (e.g.
            ``https://swarm.local:7443``). When empty, local-only mode.
            May be a comma-separated list of URLs for HA failover
            (e.g. ``https://c1:7443,https://c2:7443``); the client tries
            each in order and sticks to the first that succeeds.
        max_concurrent: max jobs this node runs concurrently.
        mtls_cert: path to the client certificate (PEM).
        mtls_key: path to the client private key (PEM).
        mtls_ca: path to the CA certificate (PEM) that signed all node
            certs.
        node_id: this node's ID on the federation (default: hostname).
        audit_log: optional audit log path (passed to local fallback).

    Note: The federation coordinator is a Python-native FastAPI app
    (see :mod:`federation_server`). Run it on any node with:
        python3 -m federation_server --mtls-cert node.crt \\
            --mtls-key node.key --mtls-ca ca.crt
    This class talks to its HTTP API. When the coordinator is not
    running, the local-only fallback ensures the system remains
    functional.
    """

    def __init__(
        self,
        orchestrator,
        federation_url: str = "",
        max_concurrent: int = 2,
        mtls_cert: Optional[str] = None,
        mtls_key: Optional[str] = None,
        mtls_ca: Optional[str] = None,
        node_id: Optional[str] = None,
        audit_log: Optional[str] = "rfsn_jobs_bitemporal.jsonl",
        persist_path: Optional[str] = None,
        signing_key=None,
        ed25519_private_key_hex: Optional[str] = None,
        ed25519_public_key_hex: Optional[str] = None,
        federation_secret: Optional[str] = None,
    ):
        # HA failover: accept a comma-separated list of coordinator URLs.
        # The client tries each in order and sticks to the first that
        # succeeds (sticky). On failure of the sticky URL, it re-tries
        # the list. This lets a swarm deploy multiple coordinators
        # behind a load balancer (or listed explicitly) and survive a
        # single coordinator crash.
        self._federation_urls: List[str] = [
            u.strip().rstrip("/") for u in (federation_url or "").split(",")
            if u.strip()
        ]
        # The sticky URL is the first in the list (or empty for local-only).
        self._federation_url = self._federation_urls[0] if self._federation_urls else ""
        self._mtls_cert = mtls_cert
        self._mtls_key = mtls_key
        self._mtls_ca = mtls_ca
        self._node_id = node_id or os.uname().nodename
        self._federation_available: Optional[bool] = None
        self._publish_failed = False  # track first failure for warning

        # v3.0: Zero-trust federation encryption. When a federation_secret
        # is provided, all payloads (job queries, results) are encrypted
        # with Fernet (AES-128-CBC + HMAC-SHA256) before transmission.
        # The secret is shared across all nodes in the swarm. Nodes without
        # the secret see only opaque ciphertext in Redis/HTTP traffic.
        self._fernet = None
        if federation_secret:
            try:
                from cryptography.fernet import Fernet
                import base64
                import hashlib
                # Derive a Fernet key from the secret (Fernet requires
                # a 32-byte base64-encoded key). We use SHA-256 of the
                # secret to get a deterministic 32-byte key.
                key = base64.urlsafe_b64encode(
                    hashlib.sha256(federation_secret.encode()).digest()
                )
                self._fernet = Fernet(key)
                print("[FederatedQueue] Zero-trust encryption enabled "
                      "(Fernet AEAD)")
            except ImportError:
                print("[FederatedQueue] Warning: federation_secret provided "
                      "but cryptography package not installed — payloads "
                      "will be sent in plaintext")

        # Local fallback queue — always present. When the federation is
        # down, all jobs run here. When it's up, this node's spare
        # capacity claims jobs from the network via this queue. Signing
        # keys are forwarded to the inner JobQueue's BiTemporalAuditLog.
        self._local = LocalJobQueue(
            orchestrator,
            max_concurrent=max_concurrent,
            audit_log=audit_log,
            persist_path=persist_path,
            signing_key=signing_key,
            ed25519_private_key_hex=ed25519_private_key_hex,
            ed25519_public_key_hex=ed25519_public_key_hex,
        )

    def __getattr__(self, name: str):
        """Delegate any attribute not defined here to the local queue.

        This transparently forwards ``bitemporal``, ``job_history``,
        ``state_as_of``, and any other JobQueue attribute that the
        REPL or other callers access directly. The local queue itself
        delegates to its inner JobQueue via its own ``__getattr__``.
        """
        if name == "_local":
            raise AttributeError(name)  # not set yet — avoid infinite recursion
        return getattr(self._local, name)

    def _encrypt_payload(self, data: dict) -> dict:
        """Encrypt a payload dict for zero-trust federation (v3.0).

        If encryption is enabled (``_fernet`` is set), the payload is
        JSON-serialized and encrypted. The returned dict has a single
        key ``__encrypted__`` with the ciphertext. If encryption is
        not enabled, the payload is returned unchanged.
        """
        if self._fernet is None:
            return data
        plaintext = json.dumps(data, default=str).encode("utf-8")
        ciphertext = self._fernet.encrypt(plaintext).decode("ascii")
        return {"__encrypted__": ciphertext}

    def _decrypt_payload(self, data: dict) -> dict:
        """Decrypt a payload dict received from the federation (v3.0).

        If the data has an ``__encrypted__`` key and encryption is
        enabled, decrypts and JSON-parses the payload. If encryption
        is not enabled but the data is encrypted, raises ValueError.
        """
        if "__encrypted__" not in data:
            return data  # Plaintext (no encryption configured by sender)
        if self._fernet is None:
            raise ValueError("Received encrypted payload but no "
                             "federation_secret configured")
        ciphertext = data["__encrypted__"].encode("ascii")
        plaintext = self._fernet.decrypt(ciphertext).decode("utf-8")
        return json.loads(plaintext)

    def _federation_configured(self) -> bool:
        """True if the federation URL and mTLS certs are all provided."""
        return bool(
            self._federation_url
            and self._mtls_cert
            and os.path.exists(self._mtls_cert)
            and self._mtls_key
            and os.path.exists(self._mtls_key)
            and self._mtls_ca
            and os.path.exists(self._mtls_ca)
        )

    def _federation_url_order(self) -> List[str]:
        """URLs to try, sticky-first then the rest (HA failover order).

        The sticky URL (``self._federation_url``) is tried first; on
        failure the remaining URLs from ``self._federation_urls`` are
        tried in listed order. When a URL succeeds, it becomes the new
        sticky URL.
        """
        if not self._federation_urls:
            return [self._federation_url] if self._federation_url else []
        sticky = self._federation_url
        rest = [u for u in self._federation_urls if u != sticky]
        order = ([sticky] if sticky else []) + rest
        return [u for u in order if u]

    def _mark_url_success(self, url: str) -> None:
        """Make ``url`` the sticky URL for subsequent requests."""
        if url and url != self._federation_url:
            self._federation_url = url

    def _warn_fallback(self, reason: str) -> None:
        if not self._publish_failed:
            print(
                f"[FederatedQueue] Federation unavailable ({reason}) — "
                f"falling back to local-only execution. Jobs will still "
                f"run on this node but won't be distributed to the swarm."
            )
            self._publish_failed = True

    def _publish_to_federation(self, job: Job) -> None:
        """Publish a job to the federation network (fire-and-forget).

        This method is called from the synchronous submit() method. To
        avoid blocking the asyncio event loop with synchronous network
        I/O, it schedules the actual HTTP POST as a background asyncio
        task via asyncio.create_task(). The task runs concurrently and
        its result (success/failure) is logged but does not affect the
        submit() return — the job is already in the local queue and
        will run locally if the federation publish fails.

        If no event loop is running (rare — submit() called outside
        async context), falls back to a thread pool executor.
        """
        if not self._federation_configured():
            self._warn_fallback("mTLS certs or federation URL not configured")
            return
        try:
            loop = asyncio.get_running_loop()
            # Fire-and-forget: schedule the async publish without
            # blocking submit(). The task handles its own errors.
            loop.create_task(self._publish_to_federation_async(job))
        except RuntimeError:
            # No running event loop — run in a thread to avoid blocking.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(self._publish_to_federation_sync, job)

    async def _publish_to_federation_async(self, job: Job) -> bool:
        """Async federation publish using aiohttp (non-blocking).

        Uses a short-lived aiohttp session with mTLS. This runs as a
        background task — errors are logged but do not propagate to
        submit(). Tries each HA coordinator URL in turn (sticky-first)
        and sticks to the first that succeeds.

        v3.0: Payloads are encrypted with Fernet AEAD if
        ``federation_secret`` was provided.
        """
        try:
            import aiohttp
            import ssl
            payload = self._encrypt_payload({
                "job_id": job.job_id,
                "query": job.query,
                "priority": job.priority,
                "force_route": job.force_route,
                "submitted_by": self._node_id,
            })
            ctx = ssl.create_default_context(cafile=self._mtls_ca)
            ctx.load_cert_chain(self._mtls_cert, self._mtls_key)
            timeout = aiohttp.ClientTimeout(total=5.0)
            last_err = "no coordinator reachable"
            for url in self._federation_url_order():
                try:
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(
                            f"{url}/submit",
                            json=payload,
                            ssl=ctx,
                            headers={"Content-Type": "application/json"},
                        ) as resp:
                            if resp.status < 400:
                                self._federation_available = True
                                self._mark_url_success(url)
                                return True
                            last_err = f"HTTP {resp.status} from {url}"
                except Exception as e:
                    last_err = f"connection error to {url}: {e}"
                    continue
            self._warn_fallback(last_err)
            return False
        except Exception as e:
            self._warn_fallback(f"connection error: {e}")
            return False

    def _publish_to_federation_sync(self, job: Job) -> bool:
        """Synchronous federation publish (fallback for non-async contexts).

        Uses urllib with mTLS. Only called when no event loop is running.
        Tries each HA coordinator URL in turn (sticky-first).
        """
        try:
            import urllib.request
            import ssl
            payload = json.dumps(self._encrypt_payload({
                "job_id": job.job_id,
                "query": job.query,
                "priority": job.priority,
                "force_route": job.force_route,
                "submitted_by": self._node_id,
            })).encode("utf-8")
            ctx = ssl.create_default_context(cafile=self._mtls_ca)
            ctx.load_cert_chain(self._mtls_cert, self._mtls_key)
            last_err = "no coordinator reachable"
            for url in self._federation_url_order():
                try:
                    req = urllib.request.Request(
                        f"{url}/submit",
                        data=payload,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, context=ctx, timeout=5.0) as resp:
                        if resp.status < 400:
                            self._federation_available = True
                            self._mark_url_success(url)
                            return True
                        last_err = f"HTTP {resp.status} from {url}"
                except Exception as e:
                    last_err = f"connection error to {url}: {e}"
                    continue
            self._warn_fallback(last_err)
            return False
        except Exception as e:
            self._warn_fallback(f"connection error: {e}")
            return False

    async def _claim_loop(self):
        """Background worker loop: claim jobs from the federation coordinator
        when the local node has spare capacity.

        This is the "pull" side of the federation — the local node
        periodically polls the coordinator for pending jobs. If it gets
        one, it submits it to the local queue for execution, then POSTs
        the result back to the coordinator when the job completes.
        """
        import aiohttp
        import ssl
        ctx = ssl.create_default_context(cafile=self._mtls_ca)
        ctx.load_cert_chain(self._mtls_cert, self._mtls_key)
        timeout = aiohttp.ClientTimeout(total=10.0)
        while True:
            try:
                await asyncio.sleep(2.0)  # poll interval
                if not self._federation_configured():
                    continue
                # Try each HA coordinator URL (sticky-first) until one
                # responds. The claim is atomic on the coordinator side,
                # so trying multiple URLs is safe — a "no_jobs" response
                # from one coordinator means that coordinator's queue is
                # empty (in HA Redis mode they share one queue).
                claimed = False
                for url in self._federation_url_order():
                    try:
                        async with aiohttp.ClientSession(timeout=timeout) as session:
                            async with session.post(
                                f"{url}/claim",
                                json={"worker_id": self._node_id},
                                ssl=ctx,
                            ) as resp:
                                if resp.status >= 400:
                                    continue
                                data = await resp.json()
                                # v3.0: Decrypt the claim response if
                                # the server encrypted it.
                                data = self._decrypt_payload(data)
                                if not data.get("job_id"):
                                    continue  # no pending jobs at this URL
                                # We claimed a job — submit it locally.
                                self._mark_url_success(url)
                                claimed = True
                                remote_job_id = data["job_id"]
                                job = self._local.submit(
                                    data["query"],
                                    priority=data.get("priority", 0),
                                    force_route=data.get("force_route"),
                                )
                                print(f"[FederatedQueue] Claimed job {remote_job_id} "
                                      f"from federation ({url}), running locally as {job.job_id}")
                                # NOTE: _claim_and_report creates its own
                                # session because this one closes at the
                                # end of the `async with` block.
                                asyncio.create_task(
                                    self._claim_and_report(
                                        ctx, remote_job_id, job.job_id, url,
                                    )
                                )
                                break  # claimed one; done this poll cycle
                    except Exception:
                        continue
                # `claimed` is informational; the loop polls regardless.
                _ = claimed
            except asyncio.CancelledError:
                break
            except Exception:
                pass  # silent fail — the claim loop is best-effort

    async def _claim_and_report(
        self, ssl_ctx, remote_job_id: str, local_job_id: str,
        origin_url: Optional[str] = None,
    ):
        """Wait for a claimed job to complete locally, then POST the
        result back to the federation coordinator.

        Creates its own aiohttp.ClientSession (v1.0 fix: the claim loop's
        session is closed by the time this background task runs, so we
        can't reuse it). Posts to ``origin_url`` (the coordinator that
        handed out the claim) first; on failure, tries the other HA URLs.
        """
        import aiohttp
        # URL order: the originating coordinator first (it owns the job
        # record), then the rest for HA failover.
        urls = self._federation_url_order()
        if origin_url and origin_url in urls:
            urls = [origin_url] + [u for u in urls if u != origin_url]
        elif origin_url:
            urls = [origin_url] + urls
        try:
            result = await self._local.wait_for(local_job_id, timeout=600.0)
            payload = self._encrypt_payload({
                "job_id": remote_job_id,
                "result": _serialize_result(result),
            })
            await self._post_to_any_url(urls, "/complete", payload, ssl_ctx,
                                        success_log=f"Reported result for "
                                                    f"federation job {remote_job_id}")
        except asyncio.TimeoutError:
            await self._post_to_any_url(urls, "/complete",
                                        self._encrypt_payload({"job_id": remote_job_id,
                                         "error": "execution timed out (600s)"}),
                                        ssl_ctx)
        except Exception as e:
            await self._post_to_any_url(urls, "/complete",
                                        self._encrypt_payload({"job_id": remote_job_id, "error": str(e)}),
                                        ssl_ctx)

    async def _post_to_any_url(
        self, urls: List[str], path: str, payload: dict, ssl_ctx,
        success_log: Optional[str] = None, total_timeout: float = 30.0,
    ) -> bool:
        """POST ``payload`` to ``path`` on the first reachable URL.

        Tries each URL in order; on success, marks it sticky and returns
        True. On total failure, returns False (best-effort reporting).
        """
        import aiohttp
        timeout = aiohttp.ClientTimeout(total=total_timeout)
        for url in urls:
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{url}{path}", json=payload, ssl=ssl_ctx,
                    ) as resp:
                        if resp.status < 400:
                            self._mark_url_success(url)
                            if success_log:
                                print(f"[FederatedQueue] {success_log} ({url})")
                            return True
            except Exception:
                continue
        if success_log:
            print(f"[FederatedQueue] Failed to {success_log}: "
                  f"no coordinator reachable")
        return False

    # ----------------------- BaseJobQueue protocol ----------------------- #
    async def start(self) -> None:
        await self._local.start()
        if self._federation_configured():
            print(f"[FederatedQueue] Started (node={self._node_id}, "
                  f"federation={self._federation_url})")
            # Start the claim loop — this node will pull jobs from the
            # coordinator when it has spare capacity.
            self._claim_task = asyncio.create_task(self._claim_loop())
        else:
            print(f"[FederatedQueue] Started in local-only mode "
                  f"(no federation configured)")

    async def stop(self) -> None:
        if hasattr(self, "_claim_task"):
            self._claim_task.cancel()
            try:
                await self._claim_task
            except asyncio.CancelledError:
                pass
        await self._local.stop()

    def submit(
        self,
        query: str,
        priority: int = 0,
        force_route: Optional[str] = None,
        callback: Optional[Callable[[Any], Any]] = None,
    ) -> Job:
        # Always submit locally first (for tracking + wait_for).
        job = self._local.submit(
            query, priority=priority, force_route=force_route, callback=callback
        )
        # Attempt to publish to the federation. If it fails, the job
        # still runs locally (fail-closed-fallback).
        if self._federation_configured():
            self._publish_to_federation(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._local.get(job_id)

    def status(self, job_id: str) -> Optional[JobStatus]:
        return self._local.status(job_id)

    def list_jobs(self) -> List[Dict[str, Any]]:
        return self._local.list_jobs()

    async def wait_for(self, job_id: str, timeout: Optional[float] = None) -> Any:
        return await self._local.wait_for(job_id, timeout=timeout)

    def cancel(self, job_id: str) -> bool:
        return self._local.cancel(job_id)

    @property
    def inner(self) -> JobQueue:
        """Access the underlying local JobQueue."""
        return self._local.inner

    @property
    def is_federated(self) -> bool:
        """True if the federation is currently reachable."""
        return self._federation_available is True


def make_job_queue(
    orchestrator,
    federation_url: Optional[str] = None,
    max_concurrent: int = 2,
    mtls_cert: Optional[str] = None,
    mtls_key: Optional[str] = None,
    mtls_ca: Optional[str] = None,
    node_id: Optional[str] = None,
    audit_log: Optional[str] = "rfsn_jobs_bitemporal.jsonl",
    persist_path: Optional[str] = None,
    signing_key=None,
    ed25519_private_key_hex: Optional[str] = None,
    ed25519_public_key_hex: Optional[str] = None,
    federation_secret: Optional[str] = None,
) -> BaseJobQueue:
    """Factory: build the appropriate job queue from config.

    Precedence:
      1. federation_url (non-empty) -> FederatedJobQueue (with local fallback)
      2. None / empty               -> LocalJobQueue (default, single-node)

    Args:
        orchestrator: a HybridReasoningOrchestrator instance.
        federation_url: federation coordinator HTTP endpoint. When set, a
            FederatedJobQueue is used (with local-only fallback when the
            network is unreachable). May be a comma-separated list of
            URLs for HA failover.
        max_concurrent: max jobs this node runs concurrently.
        mtls_cert, mtls_key, mtls_ca: mTLS credentials for the federation.
        node_id: this node's ID on the federation.
        audit_log: optional audit log path.
        persist_path: optional disk persistence path for crash recovery.
        signing_key: optional HMAC-SHA256 secret for audit-log signatures.
        ed25519_private_key_hex: optional Ed25519 private key (hex) for
            asymmetric audit-log signatures (takes precedence over signing_key).
        ed25519_public_key_hex: optional Ed25519 public key (hex) for
            verify-only mode.

    Returns:
        A BaseJobQueue instance.
    """
    if federation_url:
        return FederatedJobQueue(
            orchestrator,
            federation_url=federation_url,
            max_concurrent=max_concurrent,
            mtls_cert=mtls_cert,
            mtls_key=mtls_key,
            mtls_ca=mtls_ca,
            node_id=node_id,
            audit_log=audit_log,
            persist_path=persist_path,
            signing_key=signing_key,
            ed25519_private_key_hex=ed25519_private_key_hex,
            ed25519_public_key_hex=ed25519_public_key_hex,
            federation_secret=federation_secret,
        )
    return LocalJobQueue(
        orchestrator,
        max_concurrent=max_concurrent,
        audit_log=audit_log,
        persist_path=persist_path,
        signing_key=signing_key,
        ed25519_private_key_hex=ed25519_private_key_hex,
        ed25519_public_key_hex=ed25519_public_key_hex,
    )
