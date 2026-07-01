"""Tests for the AgentDB migration script and finalize-migration command.

Tests the backfill, recall verification, and fail-closed paths using
LocalVectorStore as a stand-in for AgentDB (so no live sidecar is needed).
"""

import importlib
import json
import os
import sys
import pytest
from unittest.mock import patch, MagicMock


def _has_module(name: str) -> bool:
    """Check if a module is importable without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ImportError):
        return False


_NUMPY_AVAILABLE = _has_module("numpy")

pytestmark = [
    pytest.mark.embeddings,
    pytest.mark.skipif(
        not _NUMPY_AVAILABLE,
        reason="requires numpy for migration tests",
    ),
]

# numpy is an embeddings-extra dependency. Guard the import so collection
# succeeds without numpy; tests are skipped via skipif above.
if _NUMPY_AVAILABLE:
    import numpy as np  # noqa: E402
else:
    np = None  # type: ignore[assignment]

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import agentdb_migration as migrate_mod
from vector_store import LocalVectorStore, AgentDBVectorStore


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _make_cache_file(path, entries, store_type="clr"):
    """Write a cache file in the on-disk JSON format."""
    data = {
        "model_name": "test-model",
        "schema_version": 3,
        "entries": entries,
    }
    with open(path, "w") as f:
        json.dump(data, f)


def _make_clr_entry(i, embedding=None):
    return {
        "problem": f"problem {i}",
        "embedding": embedding or [float(i) / 10, 0.5, 0.5, 0.5],
        "best_answer": f"answer {i}",
        "best_score": 0.9,
        "verified": True,
        "verification_method": "math_verifier",
        "task_type": "math",
        "schema_version": 3,
    }


def _make_traj_entry(i, embedding=None):
    return {
        "query": f"query {i}",
        "embedding": embedding or [float(i) / 10, 0.5, 0.5, 0.5],
        "answer": f"answer {i}",
        "score": 0.85,
        "verification_method": "math_verifier",
        "task_type": "math",
        "verified": True,
        "schema_version": 3,
        "best_answer": f"answer {i}",
        "best_score": 0.85,
        "claim_count": 10,
    }


class FakeAgentDB:
    """A fake AgentDB that stores entries in a dict, for testing backfill
    without a real sidecar. Mimics the AgentDBVectorStore interface."""
    def __init__(self):
        self.entries = {}
    def upsert(self, vector_id, embedding, metadata=None):
        self.entries[vector_id] = (embedding, metadata or {})
    def search(self, query_embedding, top_k=10, filters=None):
        # Simple cosine similarity.
        results = []
        for vid, (emb, meta) in self.entries.items():
            sim = sum(a*b for a, b in zip(query_embedding, emb))
            results.append((vid, sim, meta))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]
    def delete(self, vector_id):
        return self.entries.pop(vector_id, None) is not None
    def count(self):
        return len(self.entries)


# ---------------------------------------------------------------------- #
# Backfill tests
# ---------------------------------------------------------------------- #
class TestBackfill:
    def test_backfill_clr_cache(self, tmp_path):
        clr_path = str(tmp_path / "clr.json")
        _make_cache_file(clr_path, [_make_clr_entry(i) for i in range(5)])
        agentdb = FakeAgentDB()
        summary = migrate_mod.backfill(agentdb, clr_path, None)
        assert summary["clr_count"] == 5
        assert summary["traj_count"] == 0
        assert summary["total"] == 5
        assert summary["failures"] == 0
        assert agentdb.count() == 5

    def test_backfill_trajectory_store(self, tmp_path):
        traj_path = str(tmp_path / "traj.json")
        _make_cache_file(traj_path, [_make_traj_entry(i) for i in range(3)])
        agentdb = FakeAgentDB()
        summary = migrate_mod.backfill(agentdb, None, traj_path)
        assert summary["clr_count"] == 0
        assert summary["traj_count"] == 3
        assert summary["total"] == 3
        assert agentdb.count() == 3

    def test_backfill_both_stores(self, tmp_path):
        clr_path = str(tmp_path / "clr.json")
        traj_path = str(tmp_path / "traj.json")
        _make_cache_file(clr_path, [_make_clr_entry(i) for i in range(4)])
        _make_cache_file(traj_path, [_make_traj_entry(i) for i in range(6)])
        agentdb = FakeAgentDB()
        summary = migrate_mod.backfill(agentdb, clr_path, traj_path)
        assert summary["clr_count"] == 4
        assert summary["traj_count"] == 6
        assert summary["total"] == 10
        assert agentdb.count() == 10

    def test_backfill_dry_run_does_not_write(self, tmp_path):
        clr_path = str(tmp_path / "clr.json")
        _make_cache_file(clr_path, [_make_clr_entry(i) for i in range(3)])
        agentdb = FakeAgentDB()
        summary = migrate_mod.backfill(agentdb, clr_path, None, dry_run=True)
        assert summary["dry_run"] is True
        assert summary["clr_count"] == 3  # counted but not written
        assert agentdb.count() == 0  # nothing actually written

    def test_backfill_missing_file_skipped(self, tmp_path):
        agentdb = FakeAgentDB()
        summary = migrate_mod.backfill(
            agentdb, str(tmp_path / "nonexistent.json"), None,
        )
        assert summary["clr_count"] == 0
        assert summary["failures"] == 0

    def test_backfill_empty_file_skipped(self, tmp_path):
        clr_path = str(tmp_path / "clr.json")
        _make_cache_file(clr_path, [])
        agentdb = FakeAgentDB()
        summary = migrate_mod.backfill(agentdb, clr_path, None)
        assert summary["clr_count"] == 0
        assert agentdb.count() == 0

    def test_backfill_entries_without_embeddings_skipped(self, tmp_path):
        clr_path = str(tmp_path / "clr.json")
        entries = [_make_clr_entry(0), {"problem": "no embedding"}, _make_clr_entry(1)]
        _make_cache_file(clr_path, entries)
        agentdb = FakeAgentDB()
        summary = migrate_mod.backfill(agentdb, clr_path, None)
        assert summary["clr_count"] == 2  # only 2 have embeddings
        assert agentdb.count() == 2

    def test_backfill_metadata_extracted_correctly(self, tmp_path):
        clr_path = str(tmp_path / "clr.json")
        _make_cache_file(clr_path, [_make_clr_entry(0)])
        agentdb = FakeAgentDB()
        migrate_mod.backfill(agentdb, clr_path, None)
        _, meta = agentdb.entries["clr_0"]
        assert meta["problem"] == "problem 0"
        assert meta["best_answer"] == "answer 0"
        assert meta["verified"] is True
        assert meta["verification_method"] == "math_verifier"


# ---------------------------------------------------------------------- #
# Recall verification tests
# ---------------------------------------------------------------------- #
class TestVerifyRecall:
    def test_perfect_recall_when_agentdb_has_all_entries(self, tmp_path):
        """When AgentDB has the same entries as local, recall should be high."""
        clr_path = str(tmp_path / "clr.json")
        # Use orthogonal embeddings so ranking is unambiguous across
        # different similarity metrics (dot product vs cosine).
        basis = [
            [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
            [0.9, 0.1, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0], [0.0, 0.1, 0.9, 0.0],
            [0.8, 0.0, 0.0, 0.2], [0.2, 0.0, 0.0, 0.8],
        ]
        entries = [_make_clr_entry(i, basis[i]) for i in range(10)]
        _make_cache_file(clr_path, entries)
        # Build a FakeAgentDB with the same entries.
        agentdb = FakeAgentDB()
        migrate_mod.backfill(agentdb, clr_path, None)
        result = migrate_mod.verify_recall(agentdb, clr_path, None)
        assert result["overall_recall"] >= 0.8
        assert result["passed"] is True

    def test_zero_recall_when_agentdb_empty(self, tmp_path):
        """When AgentDB has no entries, recall should be 0%."""
        clr_path = str(tmp_path / "clr.json")
        entries = [_make_clr_entry(i, [float(i), 0.0, 0.0, 0.0]) for i in range(5)]
        _make_cache_file(clr_path, entries)
        agentdb = FakeAgentDB()  # empty
        result = migrate_mod.verify_recall(agentdb, clr_path, None)
        assert result["overall_recall"] < 0.5
        assert result["passed"] is False

    def test_missing_file_skipped(self, tmp_path):
        agentdb = FakeAgentDB()
        result = migrate_mod.verify_recall(
            agentdb, str(tmp_path / "nonexistent.json"), None,
        )
        # No stores to verify -> overall_recall 0, but no crash.
        assert "stores" in result
        assert len(result["stores"]) == 0

    def test_threshold_configurable(self, tmp_path):
        clr_path = str(tmp_path / "clr.json")
        entries = [_make_clr_entry(i, [float(i), 0.0, 0.0, 0.0]) for i in range(5)]
        _make_cache_file(clr_path, entries)
        agentdb = FakeAgentDB()
        # Only backfill half the entries -> partial recall.
        for i in range(3):
            agentdb.upsert(f"clr_{i}", entries[i]["embedding"],
                           migrate_mod._extract_clr_metadata(entries[i]))
        result = migrate_mod.verify_recall(
            agentdb, clr_path, None, recall_threshold=0.1,
        )
        assert result["threshold"] == 0.1
        # With a low threshold, even partial recall might pass.
        # Just check it doesn't crash.
        assert "overall_recall" in result


# ---------------------------------------------------------------------- #
# Main entry point tests
# ---------------------------------------------------------------------- #
class TestMainEntryPoint:
    def test_no_paths_given_returns_error(self):
        """When neither --clr-cache-path nor --trajectory-store-path is set,
        the script should exit with code 1."""
        old_argv = sys.argv
        sys.argv = ["migrate", "--agentdb-url", "http://127.0.0.1:1"]
        try:
            rc = migrate_mod.main()
            assert rc == 1
        finally:
            sys.argv = old_argv

    def test_unreachable_agentdb_returns_error(self, tmp_path):
        """When AgentDB is unreachable, the script should exit with code 1
        (fail-closed — no backfill attempted)."""
        clr_path = str(tmp_path / "clr.json")
        _make_cache_file(clr_path, [_make_clr_entry(0)])
        old_argv = sys.argv
        sys.argv = [
            "migrate", "--agentdb-url", "http://127.0.0.1:1",
            "--clr-cache-path", clr_path,
        ]
        try:
            rc = migrate_mod.main()
            assert rc == 1
        finally:
            sys.argv = old_argv

    def test_dry_run_succeeds_without_agentdb(self, tmp_path):
        """Dry run should succeed (exit 0) even if AgentDB is unreachable,
        because it doesn't actually write — it just reports what would
        be migrated."""
        clr_path = str(tmp_path / "clr.json")
        _make_cache_file(clr_path, [_make_clr_entry(0)])
        old_argv = sys.argv
        sys.argv = [
            "migrate", "--agentdb-url", "http://127.0.0.1:1",
            "--clr-cache-path", clr_path, "--dry-run",
        ]
        try:
            rc = migrate_mod.main()
            assert rc == 0  # dry run succeeds without AgentDB
        finally:
            sys.argv = old_argv


# ---------------------------------------------------------------------- #
# Finalize-migration command tests
# ---------------------------------------------------------------------- #
class TestFinalizeMigration:
    def test_finalize_unreachable_agentdb_refuses(self, tmp_path):
        """finalize-migration should refuse when AgentDB is unreachable."""
        clr_path = str(tmp_path / "clr.json")
        _make_cache_file(clr_path, [_make_clr_entry(i) for i in range(5)])
        old_argv = sys.argv
        # Note: "finalize-migration" is stripped by main() before calling
        # _run_finalize_migration, so we strip it here too.
        sys.argv = [
            "rfsn_cli.py",
            "--agentdb-url", "http://127.0.0.1:1",
            "--clr-cache-path", clr_path,
        ]
        try:
            sys.path.insert(0, _PROJECT_ROOT)
            from rfsn_cli import _run_finalize_migration
            rc = _run_finalize_migration()
            assert rc == 1  # AgentDB unreachable -> refuse
        finally:
            sys.argv = old_argv
            if _PROJECT_ROOT in sys.path:
                sys.path.remove(_PROJECT_ROOT)

    def test_finalize_no_paths_returns_error(self):
        old_argv = sys.argv
        sys.argv = [
            "rfsn_cli.py",
            "--agentdb-url", "http://127.0.0.1:1",
        ]
        try:
            sys.path.insert(0, _PROJECT_ROOT)
            from rfsn_cli import _run_finalize_migration
            rc = _run_finalize_migration()
            assert rc == 1
        finally:
            sys.argv = old_argv
            if _PROJECT_ROOT in sys.path:
                sys.path.remove(_PROJECT_ROOT)

    def test_finalize_low_recall_refuses_and_does_not_archive(self, tmp_path):
        """When recall is below threshold, finalize should refuse (exit 2)
        and NOT archive the local files (no data loss)."""
        clr_path = str(tmp_path / "clr.json")
        _make_cache_file(clr_path, [_make_clr_entry(i) for i in range(5)])
        old_argv = sys.argv
        sys.argv = [
            "rfsn_cli.py",
            "--agentdb-url", "http://127.0.0.1:1",
            "--clr-cache-path", clr_path,
            "--recall-threshold", "0.99",
        ]
        try:
            sys.path.insert(0, _PROJECT_ROOT)
            from rfsn_cli import _run_finalize_migration
            # Patch AgentDBVectorStore to use FakeAgentDB (empty -> low recall).
            # Patch at the source module since rfsn_cli imports it inside the function.
            with patch("vector_store.AgentDBVectorStore", side_effect=lambda *a, **kw: FakeAgentDB()):
                rc = _run_finalize_migration()
            assert rc == 2  # recall failed
            # Local file should NOT be archived.
            assert os.path.exists(clr_path)
            assert not os.path.exists(clr_path + ".bak")
        finally:
            sys.argv = old_argv
            if _PROJECT_ROOT in sys.path:
                sys.path.remove(_PROJECT_ROOT)
