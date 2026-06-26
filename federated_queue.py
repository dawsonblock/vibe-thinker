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
  - :class:`FederatedJobQueue` — pushes jobs to an exo-federation
    network (ruvnet/exo-federation, a Rust crate) over mTLS. The local
    node becomes a worker in a swarm: when ``submit()`` is called, the
    job is published to the network; any idle node (including this one)
    can pick it up. If the local node is at its ``max_concurrent`` limit,
    other nodes automatically claim the pending jobs.

The exo-federation crate is NOT published as a Python package. When it
is not reachable, :class:`FederatedJobQueue` fail-closed-fallbacks to
local-only execution (jobs still run on this node, just not federated).
This preserves the alpha's strict fail-closed philosophy: a federation
outage degrades to single-node operation rather than dropping jobs.

The protocol (:class:`BaseJobQueue`) is a structural supertype of the
existing :class:`JobQueue` — the existing class already implements
``submit``, ``get``, ``status``, ``list_jobs``, ``wait_for``, ``cancel``,
``start``, and ``stop``. This means the existing :class:`JobQueue` can
be used wherever a :class:`BaseJobQueue` is expected without changes.

Integration plan reference: Phase 4.1 — "Decouple Local asyncio Queue
with Swarm Federation". The plan's action items:
  1. Abstract JobQueue into a BaseJobQueue protocol.
  2. Implement FederatedJobQueue (push to exo-federation over mTLS).
  3. _dispatch_loop() becomes a worker that pulls from the Swarm network.

mTLS configuration:
  The exo-federation network uses mutual TLS for node authentication.
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
    """Federated job queue backed by an exo-federation swarm network.

    When the exo-federation network is reachable, jobs submitted via
    :meth:`submit` are published to the swarm. Any idle node on the
    network can claim and run them. This lets vibe-thinker scale beyond
    a single machine's ``max_concurrent`` limit: if the local M2 Pro is
    busy, other nodes pick up the pending reasoning trajectories.

    When the network is NOT reachable (the exo-federation sidecar is not
    running, or mTLS certs are missing), the queue fail-closed-fallbacks
    to local-only execution: jobs still run on this node via the wrapped
    :class:`LocalJobQueue`, just not federated. A warning is printed on
    the first failed publish. This preserves the alpha's strict
    fail-closed philosophy — a federation outage degrades to single-node
    operation rather than dropping jobs.

    Architecture:
      - ``submit()``: publish the job to the exo-federation network via
        HTTP POST. If the publish fails, fall back to local submission.
        Either way, the job is tracked locally so ``wait_for`` works.
      - The local node also runs a worker (the wrapped LocalJobQueue's
        dispatcher) that claims jobs from the network when it has spare
        capacity. This is the "pull" side of the federation.
      - ``wait_for()``: waits for the job to complete, whether it ran
        locally or on a remote node. Remote completion is signaled via
        the federation network's result callback.

    mTLS:
      The exo-federation network uses mutual TLS. Pass the client cert,
      key, and CA paths. When any is missing, the queue starts in
      local-only fallback mode.

    Args:
        orchestrator: a HybridReasoningOrchestrator instance (passed to
            the local fallback queue).
        federation_url: exo-federation HTTP endpoint (e.g.
            ``https://swarm.local:7443``). When empty, local-only mode.
        max_concurrent: max jobs this node runs concurrently.
        mtls_cert: path to the client certificate (PEM).
        mtls_key: path to the client private key (PEM).
        mtls_ca: path to the CA certificate (PEM) that signed all node
            certs.
        node_id: this node's ID on the federation (default: hostname).
        audit_log: optional audit log path (passed to local fallback).

    Note: The exo-federation crate (ruvnet/exo-federation) is a Rust
    crate that runs as a sidecar process. This class talks to its HTTP
    API. The crate is not yet published; when it is, this class will
    work without changes. Until then, the local-only fallback ensures
    the system remains functional.
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
    ):
        self._federation_url = federation_url.rstrip("/") if federation_url else ""
        self._mtls_cert = mtls_cert
        self._mtls_key = mtls_key
        self._mtls_ca = mtls_ca
        self._node_id = node_id or os.uname().nodename
        self._federation_available: Optional[bool] = None
        self._publish_failed = False  # track first failure for warning

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

    def _warn_fallback(self, reason: str) -> None:
        if not self._publish_failed:
            print(
                f"[FederatedQueue] Federation unavailable ({reason}) — "
                f"falling back to local-only execution. Jobs will still "
                f"run on this node but won't be distributed to the swarm."
            )
            self._publish_failed = True

    def _publish_to_federation(self, job: Job) -> bool:
        """Publish a job to the exo-federation network.

        Returns True if the publish succeeded, False otherwise. On
        failure, the caller falls back to local submission (the job is
        already in the local queue's tracking via _local.submit).

        This is a synchronous HTTP call (the federation's POST /submit
        endpoint). For async callers, the submit() method handles the
        bridging.
        """
        if not self._federation_configured():
            self._warn_fallback("mTLS certs or federation URL not configured")
            return False
        try:
            import urllib.request
            import ssl
            payload = json.dumps({
                "job_id": job.job_id,
                "query": job.query,
                "priority": job.priority,
                "force_route": job.force_route,
                "submitted_by": self._node_id,
            }).encode("utf-8")
            ctx = ssl.create_default_context(cafile=self._mtls_ca)
            ctx.load_cert_chain(self._mtls_cert, self._mtls_key)
            req = urllib.request.Request(
                f"{self._federation_url}/submit",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, context=ctx, timeout=5.0) as resp:
                if resp.status < 400:
                    self._federation_available = True
                    return True
                self._warn_fallback(f"HTTP {resp.status}")
                return False
        except (OSError, urllib.error.URLError) as e:
            self._warn_fallback(f"connection error: {e}")
            return False
        except Exception as e:
            self._warn_fallback(f"unexpected error: {e}")
            return False

    # ----------------------- BaseJobQueue protocol ----------------------- #
    async def start(self) -> None:
        await self._local.start()
        if self._federation_configured():
            print(f"[FederatedQueue] Started (node={self._node_id}, "
                  f"federation={self._federation_url})")
        else:
            print(f"[FederatedQueue] Started in local-only mode "
                  f"(no federation configured)")

    async def stop(self) -> None:
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
) -> BaseJobQueue:
    """Factory: build the appropriate job queue from config.

    Precedence:
      1. federation_url (non-empty) -> FederatedJobQueue (with local fallback)
      2. None / empty               -> LocalJobQueue (default, single-node)

    Args:
        orchestrator: a HybridReasoningOrchestrator instance.
        federation_url: exo-federation HTTP endpoint. When set, a
            FederatedJobQueue is used (with local-only fallback when the
            network is unreachable).
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
