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
      "schema_version":  2,                  # log format version
      "sequence_number": 42,                 # monotonic per-file sequence
      "record_id":      "r_0f1a...",         # stable id for this record
      "previous_hash":  "sha256:abc123...",  # hash of the previous entry
      "record_hash":    "sha256:def456...",  # hash of this entry's content
      "valid_time":     "2026-06-24T...",    # real-world event time
      "transaction_time":"2026-06-24T...",   # when written to the log
      "job_id":         "e165b4a57f65",
      "event":          "submitted",         # submitted|started|completed|failed|cancelled
      "status":         "pending",           # resulting status
      "query":          "...",
      "priority":       5,
      "force_route":    null,
      "extra":          { ... },             # route, clr_score, error, etc.
      "correction_of":  null                 # record_id this entry corrects (if any)
    }

Integrity: each entry includes a SHA-256 hash of its content and the hash of
the previous entry, forming a tamper-evident chain. The ``verify_chain()``
method checks that the chain is intact.

Tamper-proofing (optional): when a ``signing_key`` is provided to the
constructor, each entry is additionally signed with HMAC-SHA256. Unlike the
plain hash chain (which an attacker can recompute after tampering), the HMAC
signature cannot be forged without the key. ``verify_chain(strict=True)``
with a key configured mathematically proves the log was written by a process
holding the key. When no key is configured, behavior is unchanged
(tamper-evident only).

Asymmetric provenance (v0.3.9, optional): for SLSA L2 compliant Ed25519
signatures, pass ``ed25519_private_key_hex`` (sign+verify) or
``ed25519_public_key_hex`` (verify-only) to the constructor. Ed25519 is
asymmetric — the public key can verify but cannot forge — so publishing the
public key proves authorship without sharing the signing capability. This
requires the optional ``cryptography`` package. When neither Ed25519 nor an
HMAC key is configured, behavior is unchanged (tamper-evident only). The
``signer=`` parameter accepts any object implementing the :class:`signers.Signer`
protocol for custom schemes.

Dependency-free (stdlib only) by default; Ed25519 needs ``cryptography``.
"""

import hashlib
import hmac
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from signers import Signer, HmacSigner, Ed25519Signer, make_signer


SCHEMA_VERSION = 2


class AuditCorruptionError(Exception):
    """Raised when the audit log fails strict integrity verification.

    This is a hard error: the chain is broken or a record is malformed.
    Callers should treat this as evidence of tampering or corruption, not
    a transient read glitch.
    """


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_entry(entry: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of an entry's content (excluding record_hash and signature)."""
    # Create a copy without record_hash and signature for hashing
    content = {k: v for k, v in entry.items() if k not in ("record_hash", "signature")}
    raw = json.dumps(content, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()


def _sign_entry(entry: Dict[str, Any], key: bytes) -> str:
    """Compute HMAC-SHA256 signature of an entry's content.

    The signature covers the same content as :func:`_hash_entry` (excluding
    ``record_hash`` and ``signature`` itself). Unlike the plain hash, an
    attacker cannot recompute this without the secret key.
    """
    content = {k: v for k, v in entry.items() if k not in ("record_hash", "signature")}
    raw = json.dumps(content, sort_keys=True, default=str)
    return "hmac-sha256:" + hmac.new(key, raw.encode(), hashlib.sha256).hexdigest()


class BiTemporalAuditLog:
    """Append-only bi-temporal JSONL audit log with hash-chain integrity.

    Args:
        path: path to the JSONL file. Created on first write.
        clock: optional callable returning a (valid_time, transaction_time)
            pair of ISO strings. Defaults to "now" for both. Useful for
            injecting clocks in tests or for back-filling migrated data.
        signing_key: optional secret key (bytes or str) for HMAC-SHA256
            signatures. When set, each entry gets a ``signature`` field
            that ``verify_chain`` checks. This upgrades the chain from
            tamper-evident to tamper-proof (an attacker cannot forge
            signatures without the key). When None (default), behavior
            is unchanged — the hash chain is still tamper-evident.
        ed25519_private_key_hex: optional hex-encoded Ed25519 private key
            for asymmetric signatures (v0.3.9). Stronger than HMAC: the
            public key can verify but cannot forge. Requires the optional
            ``cryptography`` package. Takes precedence over signing_key.
        ed25519_public_key_hex: optional hex-encoded Ed25519 public key
            for verify-only mode (nodes that read but don't write the log).
            Takes precedence over signing_key for verification.
        signer: optional pre-constructed :class:`signers.Signer` instance.
            Takes precedence over all key-based parameters — use this for
            custom signer schemes. When provided, the key parameters are
            ignored.
    """

    def __init__(
        self,
        path: str,
        clock=None,
        signing_key=None,
        ed25519_private_key_hex: Optional[str] = None,
        ed25519_public_key_hex: Optional[str] = None,
        signer: Optional[Signer] = None,
    ):
        self.path = path
        self._clock = clock
        if signer is not None:
            # Caller provided a fully-constructed signer — use it directly.
            self._signer: Optional[Signer] = signer
            self._signing_key = None  # legacy field, kept for compat
        else:
            # Build from key parameters via the factory. Precedence:
            # Ed25519 private > Ed25519 public > HMAC key > none.
            self._signer = make_signer(
                signing_key=signing_key,
                ed25519_private_key_hex=ed25519_private_key_hex,
                ed25519_public_key_hex=ed25519_public_key_hex,
            )
            # Keep the legacy _signing_key field for backward compat with
            # code that inspects it (e.g. tests checking "is signing on?").
            if signing_key is not None and not isinstance(signing_key, bytes):
                signing_key = signing_key.encode("utf-8")
            self._signing_key: Optional[bytes] = signing_key

    def _last_entry(self) -> Optional[Dict[str, Any]]:
        """Read the last entry from the log file (for chain continuation)."""
        if not os.path.exists(self.path):
            return None
        last = None
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        last = json.loads(line)
                    except json.JSONDecodeError:
                        continue
        return last

    def _next_sequence_number(self) -> int:
        """Get the next sequence number (1-based, monotonic)."""
        last = self._last_entry()
        if last is not None and "sequence_number" in last:
            return int(last["sequence_number"]) + 1
        return 1

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

        # Get previous hash for chain integrity
        last = self._last_entry()
        previous_hash = last.get("record_hash") if last else None
        seq = self._next_sequence_number()

        entry: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "sequence_number": seq,
            "record_id": "r_" + uuid.uuid4().hex[:16],
            "previous_hash": previous_hash,
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
        # Compute and add this entry's hash
        entry["record_hash"] = _hash_entry(entry)
        # If a signer is configured, add a cryptographic signature.
        # This makes the chain tamper-proof (not just tamper-evident):
        # an attacker who modifies content cannot recompute the signature
        # without the signing key (HMAC) or private key (Ed25519).
        if self._signer is not None:
            entry["signature"] = self._signer.sign(entry)

        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except (OSError, TypeError) as e:
            print(f"[BiTemporalLog] write failed: {e}")
        return entry

    # ----------------------- chain verification ----------------------- #
    def iter_raw_records_strict(self) -> List[Dict[str, Any]]:
        """Read every line as JSON, raising AuditCorruptionError on the first
        malformed line. Unlike :meth:`read_all` (which skips malformed lines
        for dashboard use), this is the strict reader used by verification.
        """
        if not os.path.exists(self.path):
            return []
        records: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise AuditCorruptionError(
                        f"Malformed JSONL at line {line_no}: {e}"
                    ) from e
                records.append(record)
        return records

    def verify_chain(self, strict: bool = True) -> Tuple[bool, List[str]]:
        """Verify the integrity of the hash chain.

        Args:
            strict: When True (default), malformed JSON, missing required
                fields, duplicate/out-of-order sequence numbers, invalid
                hashes, and invalid schema versions all count as failures.
                When False, falls back to the lenient behavior that skips
                malformed lines (useful only for quick dashboard reads).

        Returns ``(is_valid, list_of_errors)``. Each error describes a
        broken link. In strict mode, a malformed line raises
        :class:`AuditCorruptionError` — callers should catch it if they
        want the error list instead of an exception.
        """
        if strict:
            try:
                entries = self.iter_raw_records_strict()
            except AuditCorruptionError as e:
                return False, [str(e)]
        else:
            entries = self.read_all()

        errors: List[str] = []
        prev_hash: Optional[str] = None
        expected_seq = 1
        seen_seqs: set = set()

        for i, entry in enumerate(entries):
            line_label = f"line {i + 1}"

            # --- Required field checks (strict only) ---
            if strict:
                if "record_hash" not in entry:
                    errors.append(f"{line_label}: missing record_hash")
                    # Can't continue hash verification for this entry
                    prev_hash = entry.get("record_hash")
                    expected_seq += 1
                    continue
                if "previous_hash" not in entry:
                    errors.append(f"{line_label}: missing previous_hash")
                if "sequence_number" not in entry:
                    errors.append(f"{line_label}: missing sequence_number")
                    expected_seq += 1
                    continue
                if "schema_version" not in entry:
                    errors.append(f"{line_label}: missing schema_version")
                elif entry["schema_version"] != SCHEMA_VERSION:
                    errors.append(
                        f"{line_label}: invalid schema_version="
                        f"{entry['schema_version']}, expected {SCHEMA_VERSION}"
                    )

            # --- Sequence number checks ---
            seq = entry.get("sequence_number")
            if seq is not None:
                if strict and seq in seen_seqs:
                    errors.append(f"{line_label}: duplicate sequence_number={seq}")
                seen_seqs.add(seq)
                if seq != expected_seq:
                    errors.append(
                        f"{line_label}: sequence_number={seq}, expected={expected_seq}"
                    )
            expected_seq += 1

            # --- previous_hash linkage ---
            if entry.get("previous_hash") != prev_hash:
                errors.append(
                    f"{line_label}: previous_hash mismatch "
                    f"(got {entry.get('previous_hash')}, expected {prev_hash})"
                )

            # --- record_hash verification ---
            stored_hash = entry.get("record_hash")
            if stored_hash is not None:
                computed_hash = _hash_entry(entry)
                if stored_hash != computed_hash:
                    errors.append(f"{line_label}: record_hash mismatch (tampered content)")

            # --- Cryptographic signature verification (tamper-proofing) ---
            # Only checked when a signer is configured. An entry without a
            # signature when a signer is set, or a signature that doesn't
            # match, indicates tampering or an unsigned injection. The signer
            # may be HMAC-SHA256 (symmetric, stdlib) or Ed25519 (asymmetric,
            # optional cryptography package).
            if self._signer is not None:
                stored_sig = entry.get("signature")
                if stored_sig is None:
                    errors.append(f"{line_label}: missing signature (signer configured "
                                  "but entry is unsigned)")
                else:
                    if not self._signer.verify(entry, stored_sig):
                        errors.append(f"{line_label}: signature mismatch (forged or "
                                      "tampered content)")

            prev_hash = stored_hash

        return (len(errors) == 0, errors)

    # ----------------------- reading ----------------------- #
    @staticmethod
    def _validate_axis(axis: str) -> str:
        if axis not in ("valid", "transaction"):
            raise ValueError(f"axis must be 'valid' or 'transaction', got: {axis!r}")
        return axis

    def read_all(self) -> List[Dict[str, Any]]:
        """Read every entry, in file order. Malformed lines are skipped
        with a warning rather than crashing the entire read."""
        if not os.path.exists(self.path):
            return []
        entries = []
        with open(self.path, "r") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[BiTemporalLog] skipping malformed line {lineno}: {e}")
        return entries

    def _superseded_record_ids(self) -> set:
        """Return the set of record_ids that have been superseded by a
        correction entry (via correction_of). These should be ignored
        when reconstructing current state."""
        return {
            e["correction_of"]
            for e in self.read_all()
            if e.get("correction_of") is not None
        }

    def history(self, job_id: str, axis: str = "valid") -> List[Dict[str, Any]]:
        """All events for ``job_id`` ordered along the chosen time axis.

        axis="valid"       -> order by valid_time     (true-world order)
        axis="transaction" -> order by transaction_time (what we knew, in order)

        Superseded entries (referenced by a correction_of) are excluded.
        """
        self._validate_axis(axis)
        superseded = self._superseded_record_ids()
        rows = [
            e for e in self.read_all()
            if e["job_id"] == job_id and e.get("record_id") not in superseded
        ]
        key = "valid_time" if axis == "valid" else "transaction_time"
        return sorted(rows, key=lambda e: e[key])

    def state_as_of(
        self, job_id: str, as_of: str, axis: str = "valid"
    ) -> Optional[Dict[str, Any]]:
        """Reconstruct the latest known event for ``job_id`` as of ``as_of``.

        axis="valid"       : the true state of the job at that real-world time.
        axis="transaction" : what the system knew at that recording time.
        Returns the most recent event entry <= as_of, or None.

        Superseded entries (referenced by a correction_of) are excluded.
        """
        self._validate_axis(axis)
        key = "valid_time" if axis == "valid" else "transaction_time"
        superseded = self._superseded_record_ids()
        rows = [
            e for e in self.read_all()
            if e["job_id"] == job_id
            and e[key] <= as_of
            and e.get("record_id") not in superseded
        ]
        if not rows:
            return None
        return max(rows, key=lambda e: e[key])

    def current_state(self, axis: str = "valid") -> Dict[str, Dict[str, Any]]:
        """Reconstruct the current state of every job from the log.

        Returns {job_id: latest_event_entry}.

        Superseded entries (referenced by a correction_of) are excluded.
        """
        self._validate_axis(axis)
        key = "valid_time" if axis == "valid" else "transaction_time"
        superseded = self._superseded_record_ids()
        latest: Dict[str, Dict[str, Any]] = {}
        for e in self.read_all():
            if e.get("record_id") in superseded:
                continue
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
    previous_hash = None
    seq = 1
    # If appending, continue from the last entry's hash/seq
    if mode == "a" and os.path.exists(out_path):
        last = BiTemporalAuditLog(out_path)._last_entry()
        if last:
            previous_hash = last.get("record_hash")
            seq = int(last.get("sequence_number", 0)) + 1

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
                    "schema_version": SCHEMA_VERSION,
                    "sequence_number": seq,
                    "record_id": "r_" + uuid.uuid4().hex[:16],
                    "previous_hash": previous_hash,
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
                entry["record_hash"] = _hash_entry(entry)
                out_f.write(json.dumps(entry) + "\n")
                previous_hash = entry["record_hash"]
                seq += 1
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

    v = sub.add_parser("verify", help="Verify the hash-chain integrity of the log.")
    v.add_argument("path", help="Bi-temporal log path.")

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
    elif args.cmd == "verify":
        log = BiTemporalAuditLog(args.path)
        ok, errors = log.verify_chain(strict=True)
        if ok:
            print("Chain integrity: OK")
        else:
            print(f"Chain integrity: BROKEN ({len(errors)} errors)")
            for err in errors:
                print(f"  {err}")
    else:
        p.print_help()
