"""
RFSN-style async job queue adapter for the Hybrid Reasoning Orchestrator.

A lightweight, in-process async job queue with:
  - Priority-ordered dispatch (higher priority jobs run first)
  - Concurrency limit (max concurrent jobs)
  - Per-job lifecycle: pending -> running -> completed | failed | cancelled
  - Status tracking + result retrieval by job_id
  - Optional completion callback per job
  - Optional JSONL audit log of every job state transition

This is intentionally dependency-free (stdlib only) and designed to plug
directly into the HybridReasoningOrchestrator. It is NOT a distributed
queue — for multi-process/multi-node, swap the dispatcher for something
like RQ/Celery/Dramatiq. The Job/JobStatus/JobQueue API stays the same.

Usage:
    from hybrid_orchestrator import HybridReasoningOrchestrator
    from rfsn_job_queue import JobQueue, Job

    queue = JobQueue(orchestrator, max_concurrent=2)
    await queue.start()

    job = await queue.submit("Solve a_1=2, a_{n+1}=a_n^2-a_n+1, find a_5",
                             priority=5)
    result = await queue.wait_for(job.id)   # awaits completion
    print(result.final_answer)

    await queue.stop()
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from bitemporal_log import BiTemporalAuditLog


# ====================================================================== #
# Job status enum
# ====================================================================== #
class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ====================================================================== #
# Job
# ====================================================================== #
@dataclass
class Job:
    query: str
    priority: int = 0
    force_route: Optional[str] = None
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: JobStatus = JobStatus.PENDING
    result: Optional[Any] = None  # OrchestratorResult when done
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    callback: Optional[Callable[[Any], Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "query": self.query,
            "priority": self.priority,
            "force_route": self.force_route,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "has_result": self.result is not None,
        }


# ====================================================================== #
# Job queue
# ====================================================================== #
class JobQueue:
    """
    Priority async job queue wrapping a HybridReasoningOrchestrator.

    Args:
        orchestrator: a HybridReasoningOrchestrator instance.
        max_concurrent: max jobs running at once.
        audit_log: optional path to a JSONL file recording every state change.
    """

    def __init__(
        self,
        orchestrator,
        max_concurrent: int = 2,
        audit_log: Optional[str] = "rfsn_jobs_bitemporal.jsonl",
    ):
        self.orchestrator = orchestrator
        self.max_concurrent = max_concurrent
        self.audit_log = audit_log
        # Bi-temporal log: records both valid_time (event time) and
        # transaction_time (record time) for every state transition.
        self.bitemporal: Optional[BiTemporalAuditLog] = (
            BiTemporalAuditLog(audit_log) if audit_log else None
        )

        self._pending: List[Job] = []  # kept sorted by priority desc
        self._jobs: Dict[str, Job] = {}
        self._events: Dict[str, asyncio.Event] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._running = False
        self._dispatcher_task: Optional[asyncio.Task] = None
        self._wakeup = asyncio.Event()
        self._job_tasks: set = set()  # tracks in-flight job tasks for clean shutdown
        self._job_task_map: Dict[str, asyncio.Task] = {}  # job_id -> task for cancellation

    # ----------------------- audit log ----------------------- #
    def _log(self, job: Job, event: str, extra: Optional[Dict] = None) -> None:
        """Record a bi-temporal state-transition entry for ``job``.

        valid_time is captured *here* (the real-world moment of the
        transition); transaction_time is captured inside the log writer at
        the moment of the write.
        """
        if self.bitemporal is None:
            return
        self.bitemporal.record(job, event=event, extra=extra)

    # ----------------------- public API ----------------------- #
    async def start(self) -> None:
        """Start the background dispatcher loop."""
        if self._running:
            return
        self._running = True
        self._dispatcher_task = asyncio.create_task(self._dispatch_loop())
        print(f"[JobQueue] Started (max_concurrent={self.max_concurrent})")

    async def stop(self) -> None:
        """Stop the dispatcher. In-flight jobs are allowed to finish."""
        self._running = False
        self._wakeup.set()
        if self._dispatcher_task:
            await self._dispatcher_task
        # Wait for any in-flight job tasks to finish so they aren't
        # silently cancelled when the event loop closes.
        if self._job_tasks:
            await asyncio.gather(*self._job_tasks, return_exceptions=True)
        print("[JobQueue] Stopped")

    def submit(
        self,
        query: str,
        priority: int = 0,
        force_route: Optional[str] = None,
        callback: Optional[Callable[[Any], Any]] = None,
    ) -> Job:
        """Submit a job. Returns the Job immediately (status=pending)."""
        job = Job(
            query=query,
            priority=priority,
            force_route=force_route,
            callback=callback,
        )
        self._jobs[job.job_id] = job
        self._events[job.job_id] = asyncio.Event()
        # Insert sorted by priority (desc); stable for equal priorities (FIFO)
        inserted = False
        for i, existing in enumerate(self._pending):
            if job.priority > existing.priority:
                self._pending.insert(i, job)
                inserted = True
                break
        if not inserted:
            self._pending.append(job)
        self._log(job, "submitted")
        self._wakeup.set()
        return job

    async def submit_async(self, *args, **kwargs) -> Job:
        """Async submit (for symmetry; submit() is sync)."""
        return self.submit(*args, **kwargs)

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def status(self, job_id: str) -> Optional[JobStatus]:
        job = self._jobs.get(job_id)
        return job.status if job else None

    def list_jobs(self) -> List[Dict[str, Any]]:
        return [j.to_dict() for j in self._jobs.values()]

    # ------------------- bi-temporal queries ------------------- #
    def job_history(self, job_id: str, axis: str = "valid") -> List[Dict[str, Any]]:
        """Bi-temporal history of a job (delegates to the audit log)."""
        if self.bitemporal is None:
            return []
        return self.bitemporal.history(job_id, axis=axis)

    def state_as_of(self, job_id: str, as_of: str, axis: str = "valid"):
        if self.bitemporal is None:
            return None
        return self.bitemporal.state_as_of(job_id, as_of, axis=axis)

    async def wait_for(self, job_id: str, timeout: Optional[float] = None) -> Any:
        """Block until the job completes/fails/cancels, then return its result
        (an OrchestratorResult) or raise its error."""
        ev = self._events.get(job_id)
        if ev is None:
            raise KeyError(f"Unknown job_id: {job_id}")
        try:
            await asyncio.wait_for(ev.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Job {job_id} did not finish within {timeout}s")
        job = self._jobs[job_id]
        if job.status == JobStatus.FAILED:
            raise RuntimeError(f"Job {job_id} failed: {job.error}")
        if job.status == JobStatus.CANCELLED:
            raise RuntimeError(f"Job {job_id} was cancelled")
        return job.result

    def cancel(self, job_id: str) -> bool:
        """Cancel a job. Pending jobs are removed from the queue immediately.
        Running jobs are cooperatively cancelled via asyncio task cancellation.
        Returns True if the job was cancelled (or already cancelled)."""
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job.status == JobStatus.CANCELLED:
            return True  # already cancelled
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            return False  # can't cancel a finished job

        if job.status == JobStatus.PENDING:
            if job in self._pending:
                self._pending.remove(job)
            job.status = JobStatus.CANCELLED
            job.finished_at = datetime.now(timezone.utc).isoformat()
            self._events[job.job_id].set()
            self._log(job, "cancelled")
            return True

        if job.status == JobStatus.RUNNING:
            # Cooperatively cancel the running task
            task = self._job_task_map.get(job_id)
            if task is not None and not task.done():
                task.cancel()
                return True
            return False

        return False

    # ----------------------- dispatcher ----------------------- #
    async def _dispatch_loop(self) -> None:
        """Main loop: pop highest-priority pending job, run it under the
        concurrency semaphore."""
        while self._running or self._pending:
            if not self._pending:
                self._wakeup.clear()
                await self._wakeup.wait()
                continue

            job = self._pending.pop(0)
            # Launch the job, respecting the concurrency limit
            task = asyncio.create_task(self._run_job(job))
            self._job_tasks.add(task)
            self._job_task_map[job.job_id] = task
            def _cleanup(t, jid=job.job_id):
                self._job_tasks.discard(t)
                self._job_task_map.pop(jid, None)
            task.add_done_callback(_cleanup)

    async def _run_job(self, job: Job) -> None:
        async with self._semaphore:
            if job.status == JobStatus.CANCELLED:
                return
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc).isoformat()
            self._log(job, "started")

            try:
                result = await self.orchestrator.run(
                    job.query, force_route=job.force_route
                )
                job.result = result
                job.status = JobStatus.COMPLETED
                self._log(
                    job,
                    "completed",
                    extra={
                        "route": result.route_taken,
                        "clr_score": result.clr_score,
                    },
                )
                if job.callback is not None:
                    try:
                        cb = job.callback(result)
                        if asyncio.iscoroutine(cb):
                            await cb
                    except Exception as cb_err:
                        print(f"[JobQueue] callback error for {job.job_id}: {cb_err}")
            except asyncio.CancelledError:
                # Cooperative cancellation from cancel()
                job.error = "cancelled by user"
                job.status = JobStatus.CANCELLED
                self._log(job, "cancelled", extra={"reason": "user_cancelled"})
                raise  # re-raise so the task is properly marked as cancelled
            except Exception as e:
                job.error = str(e)
                job.status = JobStatus.FAILED
                self._log(job, "failed", extra={"error": str(e)})
            finally:
                job.finished_at = datetime.now(timezone.utc).isoformat()
                self._events[job.job_id].set()


# ====================== EXAMPLE USAGE ======================

async def _example():
    # Lazy import to avoid hard dependency at module load
    from hybrid_orchestrator import HybridReasoningOrchestrator

    orchestrator = HybridReasoningOrchestrator(
        vibe_endpoint="http://127.0.0.1:8080",
        generalist_endpoint="http://127.0.0.1:8081",
        use_clr=True,
        clr_k=4,
    )
    queue = JobQueue(orchestrator, max_concurrent=2)
    await queue.start()

    j1 = queue.submit("Solve a_1=2, a_{n+1}=a_n^2-a_n+1, find a_5.", priority=5)
    j2 = queue.submit("Explain the Riemann Hypothesis briefly.", priority=1)

    r1 = await queue.wait_for(j1.job_id)
    r2 = await queue.wait_for(j2.job_id)
    print("Job1:", r1.final_answer[:120])
    print("Job2:", r2.final_answer[:120])

    print("\nAll jobs:", json.dumps(queue.list_jobs(), indent=2))
    await queue.stop()


if __name__ == "__main__":
    asyncio.run(_example())
