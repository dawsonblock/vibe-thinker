"""Tests for the SONA gossip protocol (v3.0 — Distributed Brain).

Tests the federation server's /api/sona/sync endpoints and the
orchestrator's SONA export/import methods.
"""

import pytest

from federation_server import create_federation_app


class TestSonaSyncEndpoint:
    """Tests for the /api/sona/sync federation server endpoints."""

    def test_post_patterns_merges_into_global(self):
        """POSTing patterns merges them into the global set."""
        app = create_federation_app()
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            # POST some patterns from worker A.
            resp = client.post("/api/sona/sync", json={
                "worker_id": "node-A",
                "patterns": [
                    {"id": 1, "centroid": [1.0, 0.0], "cluster_size": 5,
                     "avg_quality": 0.85},
                    {"id": 2, "centroid": [0.0, 1.0], "cluster_size": 3,
                     "avg_quality": 0.72},
                ],
                "stats": {"total_trajectories": 10},
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["global_pattern_count"] == 2
            assert data["worker_count"] == 1

            # POST from worker B — pattern 2 is updated, pattern 3 is new.
            resp = client.post("/api/sona/sync", json={
                "worker_id": "node-B",
                "patterns": [
                    {"id": 2, "centroid": [0.1, 0.9], "cluster_size": 7,
                     "avg_quality": 0.80},
                    {"id": 3, "centroid": [0.5, 0.5], "cluster_size": 2,
                     "avg_quality": 0.60},
                ],
                "stats": {"total_trajectories": 5},
            })
            assert resp.status_code == 200
            assert resp.json()["global_pattern_count"] == 3
            assert resp.json()["worker_count"] == 2

    def test_get_global_returns_all_patterns(self):
        """GET /api/sona/sync returns the merged global pattern set."""
        app = create_federation_app()
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            # Post some patterns.
            client.post("/api/sona/sync", json={
                "worker_id": "node-A",
                "patterns": [{"id": 1, "centroid": [1.0], "cluster_size": 3,
                              "avg_quality": 0.9}],
                "stats": {"total_trajectories": 3},
            })

            # GET the global set.
            resp = client.get("/api/sona/sync")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_patterns"] == 1
            assert len(data["patterns"]) == 1
            assert data["patterns"][0]["id"] == 1
            assert "node-A" in data["worker_stats"]

    def test_get_global_empty_when_no_posts(self):
        """GET /api/sona/sync returns empty when no patterns have been posted."""
        app = create_federation_app()
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            resp = client.get("/api/sona/sync")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_patterns"] == 0
            assert data["patterns"] == []

    def test_pattern_dedup_by_id(self):
        """Posting the same pattern ID from different workers updates it."""
        app = create_federation_app()
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            # Worker A posts pattern 1.
            client.post("/api/sona/sync", json={
                "worker_id": "A",
                "patterns": [{"id": 1, "centroid": [1.0], "cluster_size": 3,
                              "avg_quality": 0.7}],
                "stats": {},
            })
            # Worker B posts the same pattern ID with updated data.
            client.post("/api/sona/sync", json={
                "worker_id": "B",
                "patterns": [{"id": 1, "centroid": [1.1], "cluster_size": 10,
                              "avg_quality": 0.9}],
                "stats": {},
            })

            resp = client.get("/api/sona/sync")
            data = resp.json()
            assert data["total_patterns"] == 1  # Still 1, not 2
            # The latest version (from B) should be the one stored.
            pattern = data["patterns"][0]
            assert pattern["cluster_size"] == 10
            assert pattern["avg_quality"] == 0.9


class TestSonaOrchestratorSync:
    """Tests for the orchestrator's SONA export/import methods."""

    def test_export_patterns_returns_empty_when_no_recorder(self):
        """When no SONA recorder is configured, export returns []."""
        from hybrid_orchestrator import HybridReasoningOrchestrator
        orch = object.__new__(HybridReasoningOrchestrator)
        orch._sona_recorder = None
        assert orch._sona_export_patterns() == []

    def test_import_patterns_returns_zero_when_no_recorder(self):
        """When no SONA recorder is configured, import returns 0."""
        from hybrid_orchestrator import HybridReasoningOrchestrator
        orch = object.__new__(HybridReasoningOrchestrator)
        orch._sona_recorder = None
        assert orch._sona_import_patterns([{"id": 1, "centroid": [1.0]}]) == 0

    def test_import_patterns_returns_zero_for_empty_list(self):
        """Importing an empty pattern list returns 0."""
        from hybrid_orchestrator import HybridReasoningOrchestrator
        orch = object.__new__(HybridReasoningOrchestrator)
        # Even with a recorder, empty list = 0 imports.
        orch._sona_recorder = object()  # dummy
        assert orch._sona_import_patterns([]) == 0

    @pytest.mark.asyncio
    async def test_sona_sync_once_returns_disabled_when_no_url(self):
        """sona_sync_once returns disabled when no sync URL is configured."""
        from hybrid_orchestrator import HybridReasoningOrchestrator
        orch = object.__new__(HybridReasoningOrchestrator)
        orch._sona_sync_url = None
        orch._sona_recorder = None
        result = await orch.sona_sync_once()
        assert result["status"] == "disabled"
