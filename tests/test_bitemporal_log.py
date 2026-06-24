"""Pytest tests for the bi-temporal audit log."""

import json
import os
import tempfile

import pytest

from bitemporal_log import BiTemporalAuditLog, migrate_legacy_log


class FakeJob:
    """Minimal job-like object for log.record()."""
    def __init__(self, job_id="j1", status="pending", query="q", priority=5, force_route=None):
        self.job_id = job_id
        self.status = type("S", (), {"value": status})()
        self.query = query
        self.priority = priority
        self.force_route = force_route


@pytest.fixture
def log_path():
    path = tempfile.mktemp(suffix=".jsonl")
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def log(log_path):
    return BiTemporalAuditLog(log_path)


class TestWriteAndRead:
    def test_write_creates_file(self, log, log_path):
        log.record(FakeJob(), "submitted")
        assert os.path.exists(log_path)

    def test_entry_has_all_schema_keys(self, log):
        log.record(FakeJob(), "submitted",
                   valid_time="2026-01-01T00:00:00+00:00",
                   transaction_time="2026-01-01T00:00:01+00:00")
        entry = log.read_all()[0]
        for key in ("record_id", "valid_time", "transaction_time", "job_id",
                    "event", "status", "query", "priority", "force_route",
                    "extra", "correction_of"):
            assert key in entry

    def test_extra_fields_preserved(self, log):
        log.record(FakeJob(), "completed", extra={"route": "specialist_clr", "clr_score": 1.0},
                   valid_time="2026-01-01T00:00:00+00:00",
                   transaction_time="2026-01-01T00:00:01+00:00")
        entry = log.read_all()[0]
        assert entry["extra"]["route"] == "specialist_clr"
        assert entry["extra"]["clr_score"] == 1.0

    def test_valid_time_le_transaction_time(self, log):
        log.record(FakeJob(), "submitted",
                   valid_time="2026-01-01T00:00:00+00:00",
                   transaction_time="2026-01-01T00:00:01+00:00")
        entry = log.read_all()[0]
        assert entry["valid_time"] <= entry["transaction_time"]


class TestHistory:
    def test_history_valid_axis_ordered(self, log):
        for i, evt in enumerate(["submitted", "started", "completed"]):
            log.record(FakeJob(), evt,
                       valid_time=f"2026-01-01T00:00:0{i}+00:00",
                       transaction_time=f"2026-01-01T00:00:0{i}+00:00")
        hist = log.history("j1", axis="valid")
        assert [e["event"] for e in hist] == ["submitted", "started", "completed"]

    def test_history_transaction_axis_ordered(self, log):
        for i, evt in enumerate(["submitted", "started", "completed"]):
            log.record(FakeJob(), evt,
                       valid_time=f"2026-01-01T00:00:0{i}+00:00",
                       transaction_time=f"2026-01-01T00:00:{i+10}+00:00")
        hist = log.history("j1", axis="transaction")
        assert [e["event"] for e in hist] == ["submitted", "started", "completed"]

    def test_invalid_axis_raises(self, log):
        with pytest.raises(ValueError, match="axis must be"):
            log.history("j1", axis="bogus")


class TestStateAsOf:
    def test_as_of_before_any_event_returns_none(self, log):
        log.record(FakeJob(), "submitted",
                   valid_time="2026-01-01T10:00:00+00:00",
                   transaction_time="2026-01-01T10:00:01+00:00")
        assert log.state_as_of("j1", "2026-01-01T09:00:00+00:00") is None

    def test_as_of_valid_midpoint(self, log):
        for i, evt in enumerate(["submitted", "started", "completed"]):
            log.record(FakeJob(), evt,
                       valid_time=f"2026-01-01T10:00:0{i}+00:00",
                       transaction_time=f"2026-01-01T10:00:0{i}+00:00")
        # At 10:00:00, only 'submitted' has happened
        assert log.state_as_of("j1", "2026-01-01T10:00:00+00:00")["event"] == "submitted"
        # At 10:00:01, 'started' is the latest (valid_time <= 10:00:01)
        assert log.state_as_of("j1", "2026-01-01T10:00:01+00:00")["event"] == "started"
        # At 10:00:02, 'completed' is the latest
        assert log.state_as_of("j1", "2026-01-01T10:00:02+00:00")["event"] == "completed"

    def test_as_of_transaction_lag(self, log):
        log.record(FakeJob(), "submitted",
                   valid_time="2026-01-01T10:00:00+00:00",
                   transaction_time="2026-01-01T10:00:01+00:00")
        log.record(FakeJob(), "started",
                   valid_time="2026-01-01T10:00:05+00:00",
                   transaction_time="2026-01-01T10:00:06+00:00")
        # At 10:00:05 transaction time, 'started' hasn't been recorded yet
        assert log.state_as_of("j1", "2026-01-01T10:00:05+00:00", axis="transaction")["event"] == "submitted"


class TestCorrections:
    def test_correction_excludes_superseded(self, log):
        log.record(FakeJob(), "completed", extra={"route": "wrong"},
                   valid_time="2026-01-01T10:00:00+00:00",
                   transaction_time="2026-01-01T10:00:01+00:00")
        original = log.read_all()[0]
        # Correction
        log.record(FakeJob(), "completed", extra={"route": "corrected"},
                   valid_time="2026-01-01T10:00:00+00:00",
                   transaction_time="2026-01-01T11:00:00+00:00",
                   correction_of=original["record_id"])
        # History should exclude the superseded entry
        hist = log.history("j1")
        assert len(hist) == 1
        assert hist[0]["extra"]["route"] == "corrected"

    def test_current_state_excludes_superseded(self, log):
        log.record(FakeJob(), "completed", extra={"route": "wrong"},
                   valid_time="2026-01-01T10:00:00+00:00",
                   transaction_time="2026-01-01T10:00:01+00:00")
        original = log.read_all()[0]
        log.record(FakeJob(), "completed", extra={"route": "corrected"},
                   valid_time="2026-01-01T10:00:00+00:00",
                   transaction_time="2026-01-01T11:00:00+00:00",
                   correction_of=original["record_id"])
        state = log.current_state()
        assert state["j1"]["extra"]["route"] == "corrected"


class TestMalformedLines:
    def test_malformed_line_skipped(self, log_path):
        with open(log_path, "w") as f:
            f.write(json.dumps({"record_id": "r1", "valid_time": "t1",
                                "transaction_time": "t2", "job_id": "j1",
                                "event": "submitted", "status": "pending",
                                "query": "q", "priority": 1, "force_route": None,
                                "extra": {}, "correction_of": None}) + "\n")
            f.write("THIS IS NOT JSON\n")
            f.write(json.dumps({"record_id": "r2", "valid_time": "t3",
                                "transaction_time": "t4", "job_id": "j1",
                                "event": "completed", "status": "completed",
                                "query": "q", "priority": 1, "force_route": None,
                                "extra": {}, "correction_of": None}) + "\n")
        log = BiTemporalAuditLog(log_path)
        entries = log.read_all()
        assert len(entries) == 2  # malformed line skipped
        assert entries[0]["event"] == "submitted"
        assert entries[1]["event"] == "completed"


class TestMigration:
    def test_migration_roundtrip(self, log_path):
        legacy = tempfile.mktemp(suffix=".jsonl")
        try:
            with open(legacy, "w") as f:
                f.write(json.dumps({"timestamp": "2026-01-01T09:00:00", "job_id": "x",
                                    "event": "submitted", "status": "pending",
                                    "query": "hi", "priority": 2, "route": "generalist"}) + "\n")
            n = migrate_legacy_log(legacy, log_path, overwrite=True)
            assert n == 1
            entries = BiTemporalAuditLog(log_path).read_all()
            assert entries[0]["valid_time"] == "2026-01-01T09:00:00"
            assert entries[0]["extra"]["migrated"] is True
            assert entries[0]["extra"]["route"] == "generalist"
        finally:
            if os.path.exists(legacy):
                os.unlink(legacy)

    def test_migration_duplicate_guard(self, log_path):
        legacy = tempfile.mktemp(suffix=".jsonl")
        try:
            with open(legacy, "w") as f:
                f.write(json.dumps({"timestamp": "t", "job_id": "x", "event": "e",
                                    "status": "s", "query": "q", "priority": 1}) + "\n")
            migrate_legacy_log(legacy, log_path, overwrite=True)
            with pytest.raises(FileExistsError):
                migrate_legacy_log(legacy, log_path)
        finally:
            if os.path.exists(legacy):
                os.unlink(legacy)
