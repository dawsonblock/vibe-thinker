"""
Bi-temporal audit log for the RFSN job queue.

A bi-temporal log records two independent time axes for every event:

  - valid_time     : when the event *actually happened* in the real world
                     (e.g. the moment a job transitioned to "running").
  - transaction_time: when the event was *recorded* into the log (the system
                     / record time). This is the moment the write hit disk.

Keeping the two separate lets you answer questions the old single-timestamp
log could not:

  - "What did we *know* about job X at time T?"      -> filter by transaction_time <= T
  - "What was the *true* state of job X at time T?"  -> filter by valid_time <= T
  - "Was an event recorded late (after the fact)?"  -> transaction_time > valid_time

The log is append-only and immutable. Corrections are never in-place edits;
they are new entries that reference the entry they supersede via
``correction_of`` (the superseded entry's ``record_id``). This preserves a
full, replayable history.

Entry schema (one JSON object per line, JSONL):

    {
      "record_id":      "r_0f1a...",      # stable id for this record
      "valid_time":     "2026-06-24T...", # real-world event time
      "transaction_time":"2026-06-24T...",# when written to the log
      "job_id":         "e165b4a57f65",
      "event":          "submitted",      # submitted|started|completed|failed|cancelled
      "status":         "pending",        # resulting status
      "query":          "...",
      "priority":       5,
      "force_route":    null,
      "extra":          { ... },          # route, clr_score, error, etc.
      "correction_of":  null              # record_id this entry corrects (if any)
    }

Dependency-free (stdlib only), matching the rest of the project.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BiTemporalAuditLog:
    """Append-only bi-temporal JSONL audit log.

    Args:
        path: path to the JSONL file. Created on first write.
        clock: optional callable returning a (valid_time, transaction_time)
            pair of ISO strings. Defaults to "now" for both. Useful for
            injecting clocks in tests or for back-filling migrated data.
    """

    def __init__(self, path: str, clock=None):
        self.path = path
        self._clock = clock

    # ----------------------- writing ----------------------- #
    def record(
        self,
        job,
        event: str,
        extra: Optional[Dict[str, Any]] = None,
        valid_time: Optional[str] = None,
        transaction_time: Optional[str] = None,
        correction_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Append a bi-temporal entry for a state transition of ``job``.

        ``job`` is a Job-like object exposing ``job_id``, ``status`` (enum or
        str), ``query``, ``priority``, ``force_route``.
        Returns the written entry (with its assigned ``record_id``).
        """
        if self._clock is not None and valid_time is None and transaction_time is None:
            valid_time, transaction_time = self._clock()
        if valid_time is None:
            valid_time = _now_iso()
        if transaction_time is None:
            transaction_time = _now_iso()

        status = job.status.value if hasattr(job.status, "value") else str(job.status)

        entry: Dict[str, Any] = {
            "record_id": "r_" + uuid.uuid4().hex[:16],
            "valid_time": valid_time,
            "transaction_time": transaction_time,
            "job_id": job.job_id,
            "event": event,
            "status": status,
            "query": job.query,
            "priority": job.priority,
            "force_route": job.force_route,
            "extra": extra or {},
            "correction_of": correction_of,
        }
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except (OSError, TypeError) as e:
            print(f"[BiTemporalLog] write failed: {e}")
        return entry

    # ----------------------- reading ----------------------- #
    def read_all(self) -> List[Dict[str, Any]]:
        """Read every entry, in file order."""
        if not os.path.exists(self.path):
            return []
        entries = []
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def history(self, job_id: str, axis: str = "valid") -> List[Dict[str, Any]]:
        """All events for ``job_id`` ordered along the chosen time axis.

        axis="valid"       -> order by valid_time     (true-world order)
        axis="transaction" -> order by transaction_time (what we knew, in order)
        """
        rows = [e for e in self.read_all() if e["job_id"] == job_id]
        key = "valid_time" if axis == "valid" else "transaction_time"
        return sorted(rows, key=lambda e: e[key])

    def state_as_of(
        self, job_id: str, as_of: str, axis: str = "valid"
    ) -> Optional[Dict[str, Any]]:
        """Reconstruct the latest known event for ``job_id`` as of ``as_of``.

        axis="valid"       : the true state of the job at that real-world time.
        axis="transaction" : what the system knew at that recording time.
        Returns the most recent event entry <= as_of, or None.
        """
        key = "valid_time" if axis == "valid" else "transaction_time"
        rows = [e for e in self.read_all() if e["job_id"] == job_id and e[key] <= as_of]
        if not rows:
            return None
        return max(rows, key=lambda e: e[key])

    def current_state(self, axis: str = "valid") -> Dict[str, Dict[str, Any]]:
        """Reconstruct the current state of every job from the log.

        Returns {job_id: latest_event_entry}.
        """
        key = "valid_time" if axis == "valid" else "transaction_time"
        latest: Dict[str, Dict[str, Any]] = {}
        for e in self.read_all():
            cur = latest.get(e["job_id"])
            if cur is None or e[key] >= cur[key]:
                latest[e["job_id"]] = e
        return latest

    def jobs(self) -> List[str]:
        """Distinct job_ids present in the log."""
        return sorted({e["job_id"] for e in self.read_all()})


# ====================================================================== #
# Migration: convert the legacy single-timestamp JSONL into bi-temporal.
# ====================================================================== #
def migrate_legacy_log(
    legacy_path: str,
    out_path: str,
    valid_time_field: str = "timestamp",
    overwrite: bool = False,
) -> int:
    """Convert a legacy flat audit log (one ``timestamp`` per row) into the
    bi-temporal format.

    For migrated rows, ``valid_time`` is taken from the legacy ``timestamp``
    and ``transaction_time`` is set to the migration run time (since we can't
    know when the original write occurred beyond the timestamp itself). Each
    migrated row is marked with ``extra.migrated = True``.

    By default this refuses to overwrite an existing ``out_path`` to avoid
    silently duplicating entries on a re-run; pass ``overwrite=True`` (or the
    CLI ``--force`` flag) to truncate it first.

    Returns the number of entries written.
    """
    if not os.path.exists(legacy_path):
        return 0
    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists; re-running would duplicate entries. "
            f"Pass overwrite=True (or --force) to truncate it first."
        )

    migration_time = _now_iso()
    written = 0
    reserved = {"timestamp", "job_id", "event", "status", "query", "priority",
                "force_route"}

    mode = "w" if overwrite else "a"
    with open(out_path, mode) as out_f:
        with open(legacy_path, "r") as in_f:
            for line in in_f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                valid_time = row.get(valid_time_field, migration_time)
                extra = {
                    k: v for k, v in row.items() if k not in reserved
                }
                extra["migrated"] = True
                entry = {
                    "record_id": "r_" + uuid.uuid4().hex[:16],
                    "valid_time": valid_time,
                    "transaction_time": migration_time,
                    "job_id": row["job_id"],
                    "event": row.get("event", "unknown"),
                    "status": row.get("status", "unknown"),
                    "query": row.get("query", ""),
                    "priority": row.get("priority", 0),
                    "force_route": row.get("force_route"),
                    "extra": extra,
                    "correction_of": None,
                }
                out_f.write(json.dumps(entry) + "\n")
                written += 1
    return written


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Bi-temporal audit log utilities.")
    sub = p.add_subparsers(dest="cmd")

    m = sub.add_parser("migrate", help="Migrate a legacy flat JSONL log to bi-temporal.")
    m.add_argument("legacy", help="Path to legacy JSONL log.")
    m.add_argument("out", help="Path for the new bi-temporal JSONL log.")
    m.add_argument("--force", action="store_true",
                   help="Truncate out_path if it already exists.")

    q = sub.add_parser("history", help="Print the history of a job.")
    q.add_argument("path", help="Bi-temporal log path.")
    q.add_argument("job_id", help="Job id to inspect.")
    q.add_argument(
        "--axis", choices=["valid", "transaction"], default="valid",
    )

    s = sub.add_parser("state", help="Reconstruct current state of all jobs.")
    s.add_argument("path", help="Bi-temporal log path.")
    s.add_argument(
        "--axis", choices=["valid", "transaction"], default="valid",
    )

    args = p.parse_args()
    if args.cmd == "migrate":
        n = migrate_legacy_log(args.legacy, args.out, overwrite=args.force)
        print(f"Migrated {n} entries -> {args.out}")
    elif args.cmd == "history":
        log = BiTemporalAuditLog(args.path)
        for e in log.history(args.job_id, axis=args.axis):
            print(json.dumps(e, indent=2))
    elif args.cmd == "state":
        log = BiTemporalAuditLog(args.path)
        for jid, e in log.current_state(axis=args.axis).items():
            print(f"{jid}\t{e['status']}\t{e['event']}\t{e['valid_time']}")
    else:
        p.print_help()
