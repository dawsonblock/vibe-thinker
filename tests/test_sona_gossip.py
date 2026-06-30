"""Tests for the SONA gossip protocol (v3.0 — Distributed Brain).

Tests the federation server's /api/sona/sync endpoints and the
orchestrator's SONA export/import methods.
"""

import importlib.util

import pytest


def _has_module(name: str) -> bool:
    """Check if a module is importable without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ImportError):
        return False


_FASTAPI_AVAILABLE = _has_module("fastapi")
_CRYPTOGRAPHY_AVAILABLE = _has_module("cryptography")

pytestmark = [
    pytest.mark.web,
    pytest.mark.federation,
    pytest.mark.skipif(
        not _FASTAPI_AVAILABLE,
        reason="requires fastapi web extra (pip install -e '.[web]')",
    ),
]

# Guard the federation_server import: it imports fastapi at module load
# time. When deps are absent the name is set to None; it is never
# referenced at runtime because all tests are skipped via skipif above.
if _FASTAPI_AVAILABLE:
    from federation_server import create_federation_app
else:
    create_federation_app = None  # type: ignore[assignment]


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


class TestEncryptionFailClosed:
    """v0.4.6a5: federation_secret + no cryptography => startup failure.

    The server must NOT silently downgrade to plaintext when a secret
    was configured. This is a security-critical fail-closed contract:
    if an operator sets federation_secret, they expect encryption. A
    silent plaintext fallback would leak query data over the federation
    without any error. The fix: create_federation_app raises RuntimeError
    at app creation time when federation_secret is set but Fernet is
    unavailable.

    This test runs in ALL environments (with or without cryptography)
    because it mocks the ImportError path via sys.modules manipulation.
    """

    def test_secret_without_cryptography_raises_runtimeerror(self, monkeypatch):
        """create_federation_app(federation_secret=...) must raise
        RuntimeError when cryptography/Fernet is unavailable."""
        import sys
        # Simulate cryptography being absent by blocking the import.
        # Store and restore the original module so we don't break other
        # tests that need real cryptography.
        original_crypto = sys.modules.get("cryptography", None)
        original_fernet = sys.modules.get("cryptography.fernet", None)
        monkeypatch.setitem(sys.modules, "cryptography", None)
        monkeypatch.setitem(sys.modules, "cryptography.fernet", None)
        try:
            with pytest.raises(RuntimeError, match="federation_secret"):
                create_federation_app(federation_secret="test_secret")
        finally:
            # Restore real cryptography if it was available.
            if original_crypto is not None:
                monkeypatch.setitem(sys.modules, "cryptography", original_crypto)
            else:
                monkeypatch.delitem(sys.modules, "cryptography", raising=False)
            if original_fernet is not None:
                monkeypatch.setitem(sys.modules, "cryptography.fernet", original_fernet)
            else:
                monkeypatch.delitem(sys.modules, "cryptography.fernet", raising=False)

    def test_no_secret_without_cryptography_does_not_raise(self, monkeypatch):
        """Without federation_secret, missing cryptography is fine —
        plaintext is intentional when no secret was configured."""
        import sys
        original_crypto = sys.modules.get("cryptography", None)
        original_fernet = sys.modules.get("cryptography.fernet", None)
        monkeypatch.setitem(sys.modules, "cryptography", None)
        monkeypatch.setitem(sys.modules, "cryptography.fernet", None)
        try:
            # Should NOT raise — no secret means plaintext is intentional.
            app = create_federation_app()
            assert app is not None
        finally:
            if original_crypto is not None:
                monkeypatch.setitem(sys.modules, "cryptography", original_crypto)
            else:
                monkeypatch.delitem(sys.modules, "cryptography", raising=False)
            if original_fernet is not None:
                monkeypatch.setitem(sys.modules, "cryptography.fernet", original_fernet)
            else:
                monkeypatch.delitem(sys.modules, "cryptography.fernet", raising=False)


class TestEncryptedClaimResponse:
    """v3.0 fix: /claim endpoint encrypts the response when secret is set."""

    pytestmark = [
        pytest.mark.skipif(
            not _CRYPTOGRAPHY_AVAILABLE,
            reason="requires cryptography (pip install -e '.[federation]')",
        ),
    ]

    def test_claim_response_encrypted_with_secret(self):
        """When federation_secret is set, /claim response is encrypted."""
        app = create_federation_app(federation_secret="test_secret")
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            # Submit a job first.
            client.post("/submit", json={
                "job_id": "j1", "query": "secret query",
                "priority": 0, "submitted_by": "test",
            })
            # Claim it.
            resp = client.post("/claim", json={"worker_id": "w1"})
            assert resp.status_code == 200
            data = resp.json()
            # The response should be encrypted.
            assert "__encrypted__" in data
            # The plaintext query should NOT appear in the response.
            assert "secret query" not in str(data)

    def test_claim_response_plaintext_without_secret(self):
        """Without federation_secret, /claim response is plaintext."""
        app = create_federation_app()
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            client.post("/submit", json={
                "job_id": "j1", "query": "hello",
                "priority": 0, "submitted_by": "test",
            })
            resp = client.post("/claim", json={"worker_id": "w1"})
            data = resp.json()
            assert "query" in data
            assert data["query"] == "hello"

    def test_sona_get_response_encrypted_with_secret(self):
        """GET /api/sona/sync response is encrypted when secret is set."""
        app = create_federation_app(federation_secret="test_secret")
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            # Post a pattern.
            client.post("/api/sona/sync", json={
                "worker_id": "w1",
                "patterns": [{"id": 1, "centroid": [1.0], "cluster_size": 3,
                              "avg_quality": 0.8}],
                "stats": {},
            })
            # GET the global patterns.
            resp = client.get("/api/sona/sync")
            data = resp.json()
            assert "__encrypted__" in data


class TestEncryptedJobsEndpoints:
    """v3.0 fix: /jobs endpoints encrypt responses when secret is set."""

    pytestmark = [
        pytest.mark.skipif(
            not _CRYPTOGRAPHY_AVAILABLE,
            reason="requires cryptography (pip install -e '.[federation]')",
        ),
    ]

    def test_jobs_list_encrypted_with_secret(self):
        """GET /jobs encrypts the response when federation_secret is set."""
        app = create_federation_app(federation_secret="test_secret")
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            client.post("/submit", json={
                "job_id": "j1", "query": "secret",
                "priority": 0, "submitted_by": "test",
            })
            resp = client.get("/jobs")
            data = resp.json()
            assert "__encrypted__" in data

    def test_job_detail_encrypted_with_secret(self):
        """GET /jobs/{id} encrypts the response when secret is set."""
        app = create_federation_app(federation_secret="test_secret")
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            client.post("/submit", json={
                "job_id": "j1", "query": "secret",
                "priority": 0, "submitted_by": "test",
            })
            resp = client.get("/jobs/j1")
            data = resp.json()
            assert "__encrypted__" in data

    def test_jobs_list_plaintext_without_secret(self):
        """GET /jobs returns plaintext without secret."""
        app = create_federation_app()
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            resp = client.get("/jobs")
            data = resp.json()
            assert "__encrypted__" not in data


class TestSonaSyncValidation:
    """v3.0 fix: /api/sona/sync POST validates input types."""

    def test_post_with_non_list_patterns_doesnt_crash(self):
        """POSTing non-list patterns doesn't crash (handled gracefully)."""
        app = create_federation_app()
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            resp = client.post("/api/sona/sync", json={
                "worker_id": "w1",
                "patterns": "not a list",
                "stats": {},
            })
            assert resp.status_code == 200
            assert resp.json()["global_pattern_count"] == 0

    def test_post_with_non_dict_stats_doesnt_crash(self):
        """POSTing non-dict stats doesn't crash (handled gracefully)."""
        app = create_federation_app()
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            resp = client.post("/api/sona/sync", json={
                "worker_id": "w1",
                "patterns": [],
                "stats": "not a dict",
            })
            assert resp.status_code == 200
