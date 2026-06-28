"""Tests for trajectory synthesis / memory pruning (Phase 4.1)."""

import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

pytestmark = pytest.mark.embeddings

import numpy as np

from persistent_cache import VerifiedTrajectoryStore, _clear_embedding_model_cache


@pytest.fixture(autouse=True)
def _clear_model_cache():
    """Clear the shared embedding model cache before each test so mocks
    are used instead of cached real models (Phase 5 singleton fix)."""
    _clear_embedding_model_cache()
    yield
    _clear_embedding_model_cache()


def _mock_encode(embedding):
    """Build a mock encode that returns numpy arrays (matching the real
    SentenceTransformer.encode return type)."""
    def encode(texts, **kwargs):
        return np.array([embedding for _ in texts])
    return encode


# ---------------------------------------------------------------------- #
# VerifiedTrajectoryStore.find_clusters
# ---------------------------------------------------------------------- #
class TestFindClusters:
    def _make_store(self, tmp_path, entries):
        """Build a store with pre-populated entries (bypassing the model
        by patching SentenceTransformer to return fixed embeddings)."""
        path = str(tmp_path / "trajectories.json")
        # Pre-write the entries to disk, then load.
        data = {
            "model_name": "test",
            "schema_version": 3,
            "entries": entries,
        }
        with open(path, "w") as f:
            json.dump(data, f)

        # Patch SentenceTransformer to avoid needing the real model.
        mock_model = MagicMock()
        # Return identity-like embeddings so similarity is predictable.
        def _encode(texts, **kw):
            return np.array([[float(hash(t) % 100) / 100] * 4 for t in texts])
        mock_model.encode = _encode
        with patch("persistent_cache.SentenceTransformer", return_value=mock_model):
            store = VerifiedTrajectoryStore(path=path, autosave=True)
        return store

    def _entry(self, query, answer, task_type="math", embedding=None):
        return {
            "query": query, "answer": answer, "score": 0.9,
            "verification_method": "math_verifier", "task_type": task_type,
            "verified": True, "embedding": embedding or [0.1, 0.2, 0.3, 0.4],
            "schema_version": 3, "best_answer": answer, "best_score": 0.9,
            "claim_count": 10,
        }

    def test_no_entries_returns_empty(self, tmp_path):
        store = self._make_store(tmp_path, [])
        assert store.find_clusters() == []

    def test_fewer_than_min_cluster_size(self, tmp_path):
        entries = [self._entry(f"q{i}", f"a{i}") for i in range(2)]
        store = self._make_store(tmp_path, entries)
        assert store.find_clusters(min_cluster_size=3) == []

    def test_clusters_found_with_identical_embeddings(self, tmp_path):
        """Entries with identical embeddings should cluster together."""
        emb = [0.5, 0.5, 0.5, 0.5]
        entries = [
            self._entry(f"q{i}", f"a{i}", embedding=emb) for i in range(5)
        ]
        store = self._make_store(tmp_path, entries)
        clusters = store.find_clusters(similarity_threshold=0.9, min_cluster_size=3)
        assert len(clusters) >= 1
        # All 5 should be in one cluster (identical embeddings -> sim=1.0).
        total_in_clusters = sum(len(c) for c in clusters)
        assert total_in_clusters == 5

    def test_task_type_filter(self, tmp_path):
        emb = [0.5, 0.5, 0.5, 0.5]
        entries = [
            self._entry(f"math_q{i}", f"math_a{i}", "math", embedding=emb)
            for i in range(3)
        ] + [
            self._entry(f"code_q{i}", f"code_a{i}", "code", embedding=emb)
            for i in range(3)
        ]
        store = self._make_store(tmp_path, entries)
        math_clusters = store.find_clusters(
            similarity_threshold=0.9, min_cluster_size=3, task_type="math",
        )
        assert len(math_clusters) == 1
        assert len(math_clusters[0]) == 3
        code_clusters = store.find_clusters(
            similarity_threshold=0.9, min_cluster_size=3, task_type="code",
        )
        assert len(code_clusters) == 1
        assert len(code_clusters[0]) == 3

    def test_dissimilar_entries_dont_cluster(self, tmp_path):
        """Entries with orthogonal embeddings should NOT cluster."""
        entries = [
            self._entry("q0", "a0", embedding=[1.0, 0.0, 0.0, 0.0]),
            self._entry("q1", "a1", embedding=[0.0, 1.0, 0.0, 0.0]),
            self._entry("q2", "a2", embedding=[0.0, 0.0, 1.0, 0.0]),
        ]
        store = self._make_store(tmp_path, entries)
        clusters = store.find_clusters(similarity_threshold=0.5, min_cluster_size=2)
        assert clusters == []


# ---------------------------------------------------------------------- #
# VerifiedTrajectoryStore.remove_entries
# ---------------------------------------------------------------------- #
class TestRemoveEntries:
    def test_remove_reduces_count(self, tmp_path):
        path = str(tmp_path / "trajectories.json")
        entries = []
        for i in range(5):
            entries.append({
                "query": f"q{i}", "answer": f"a{i}", "score": 0.9,
                "verification_method": "math_verifier", "verified": True,
                "embedding": [0.1 * i, 0.2, 0.3, 0.4], "task_type": "math",
                "schema_version": 3, "best_answer": f"a{i}", "best_score": 0.9,
                "claim_count": 10,
            })
        with open(path, "w") as f:
            json.dump({"model_name": "test", "schema_version": 3, "entries": entries}, f)
        mock_model = MagicMock()
        mock_model.encode = _mock_encode([0.1, 0.2, 0.3, 0.4])
        with patch("persistent_cache.SentenceTransformer", return_value=mock_model):
            store = VerifiedTrajectoryStore(path=path, autosave=True)
        assert len(store) == 5
        store.remove_entries([0, 2, 4])
        assert len(store) == 2
        # Remaining entries should be q1 and q3.
        queries = [e["query"] for e in store.entries]
        assert "q1" in queries
        assert "q3" in queries
        assert "q0" not in queries

    def test_remove_empty_list_noop(self, tmp_path):
        path = str(tmp_path / "trajectories.json")
        entries = [{
            "query": "q", "answer": "a", "score": 0.9,
            "verification_method": "math_verifier", "verified": True,
            "embedding": [0.1, 0.2, 0.3, 0.4], "task_type": "math",
            "schema_version": 3, "best_answer": "a", "best_score": 0.9,
            "claim_count": 10,
        }]
        with open(path, "w") as f:
            json.dump({"model_name": "test", "schema_version": 3, "entries": entries}, f)
        mock_model = MagicMock()
        mock_model.encode = _mock_encode([0.1, 0.2, 0.3, 0.4])
        with patch("persistent_cache.SentenceTransformer", return_value=mock_model):
            store = VerifiedTrajectoryStore(path=path, autosave=True)
        store.remove_entries([])
        assert len(store) == 1


# ---------------------------------------------------------------------- #
# VerifiedTrajectoryStore.store_synthesized
# ---------------------------------------------------------------------- #
class TestStoreSynthesized:
    def test_synthesized_entry_stored_with_correct_metadata(self, tmp_path):
        path = str(tmp_path / "trajectories.json")
        with open(path, "w") as f:
            json.dump({"model_name": "test", "schema_version": 3, "entries": []}, f)
        mock_model = MagicMock()
        mock_model.encode = _mock_encode([0.1, 0.2, 0.3, 0.4])
        with patch("persistent_cache.SentenceTransformer", return_value=mock_model):
            store = VerifiedTrajectoryStore(path=path, autosave=True)
        store.store_synthesized(
            query="general math pattern",
            answer="To solve recurrence relations, use characteristic equations.",
            task_type="math",
            source_count=5,
            source_queries=["q1", "q2", "q3", "q4", "q5"],
        )
        assert len(store) == 1
        entry = store.entries[0]
        assert entry["verification_method"] == "synthesized"
        assert entry["verified"] is False
        assert entry["synthesized"] is True
        assert entry["source_count"] == 5
        assert entry["source_queries"] == ["q1", "q2", "q3", "q4", "q5"]
        assert entry["score"] == 0.0  # no independent verification score

    def test_synthesized_entries_excluded_from_retrieval(self, tmp_path):
        """Synthesized masters must NOT be served as few-shot verified
        examples — that would be epistemic contamination."""
        path = str(tmp_path / "trajectories.json")
        # One verified + one synthesized entry with identical embeddings.
        emb = [0.5, 0.5, 0.5, 0.5]
        entries = [
            {
                "query": "verified q", "answer": "verified a", "score": 0.9,
                "verification_method": "math_verifier", "verified": True,
                "embedding": emb, "task_type": "math", "schema_version": 3,
                "best_answer": "verified a", "best_score": 0.9, "claim_count": 10,
            },
            {
                "query": "synth q", "answer": "synth a", "score": 0.0,
                "verification_method": "synthesized", "verified": False,
                "synthesized": True, "embedding": emb, "task_type": "math",
                "schema_version": 3, "best_answer": "synth a",
                "best_score": 0.0, "claim_count": 0,
            },
        ]
        with open(path, "w") as f:
            json.dump({"model_name": "test", "schema_version": 3, "entries": entries}, f)
        mock_model = MagicMock()
        mock_model.encode = _mock_encode(emb)
        with patch("persistent_cache.SentenceTransformer", return_value=mock_model):
            store = VerifiedTrajectoryStore(path=path, autosave=True)
        results = store.retrieve("verified q", task_type="math")
        # Only the verified entry should be returned — synthesized excluded.
        assert len(results) == 1
        assert results[0]["verification_method"] == "math_verifier"

    def test_empty_answer_not_stored(self, tmp_path):
        path = str(tmp_path / "trajectories.json")
        with open(path, "w") as f:
            json.dump({"model_name": "test", "schema_version": 3, "entries": []}, f)
        mock_model = MagicMock()
        mock_model.encode = _mock_encode([0.1, 0.2, 0.3, 0.4])
        with patch("persistent_cache.SentenceTransformer", return_value=mock_model):
            store = VerifiedTrajectoryStore(path=path, autosave=True)
        store.store_synthesized("q", "", "math", 3, ["q1", "q2", "q3"])
        assert len(store) == 0


# ---------------------------------------------------------------------- #
# VerifiedTrajectoryStore.store_synthesized_verified (v3.1)
# ---------------------------------------------------------------------- #
class TestStoreSynthesizedVerified:
    def test_reverified_entry_stored_with_correct_metadata(self, tmp_path):
        """v3.1: A re-verified synthesized master is stored with
        verified=True and verification_method='synthesized_and_proven'."""
        path = str(tmp_path / "trajectories.json")
        with open(path, "w") as f:
            json.dump({"model_name": "test", "schema_version": 3, "entries": []}, f)
        mock_model = MagicMock()
        mock_model.encode = _mock_encode([0.1, 0.2, 0.3, 0.4])
        with patch("persistent_cache.SentenceTransformer", return_value=mock_model):
            store = VerifiedTrajectoryStore(path=path, autosave=True)
        store.store_synthesized_verified(
            query="general math pattern",
            answer="To solve recurrence relations, use characteristic equations.",
            task_type="math",
            source_count=5,
            source_queries=["q1", "q2", "q3", "q4", "q5"],
        )
        assert len(store) == 1
        entry = store.entries[0]
        assert entry["verification_method"] == "synthesized_and_proven"
        assert entry["verified"] is True
        assert entry["synthesized"] is True
        assert entry["source_count"] == 5
        assert entry["source_queries"] == ["q1", "q2", "q3", "q4", "q5"]
        assert entry["score"] == 0.65  # default re-verified score

    def test_reverified_entry_is_retrievable(self, tmp_path):
        """v3.1: A re-verified synthesized master IS retrievable as
        few-shot context (unlike the unverified synthesized master)."""
        path = str(tmp_path / "trajectories.json")
        emb = [0.5, 0.5, 0.5, 0.5]
        entries = [
            {
                "query": "synth q", "answer": "synth a", "score": 0.65,
                "verification_method": "synthesized_and_proven",
                "verified": True, "synthesized": True,
                "embedding": emb, "task_type": "math", "schema_version": 3,
                "best_answer": "synth a", "best_score": 0.65, "claim_count": 10,
                "source_count": 3, "source_queries": ["q1", "q2", "q3"],
            },
        ]
        with open(path, "w") as f:
            json.dump({"model_name": "test", "schema_version": 3, "entries": entries}, f)
        mock_model = MagicMock()
        mock_model.encode = _mock_encode(emb)
        with patch("persistent_cache.SentenceTransformer", return_value=mock_model):
            store = VerifiedTrajectoryStore(path=path, autosave=True)
        results = store.retrieve("synth q", task_type="math")
        # The re-verified synthesized master IS retrievable.
        assert len(results) == 1
        assert results[0]["verification_method"] == "synthesized_and_proven"

    def test_empty_answer_not_stored(self, tmp_path):
        """Empty answer is not stored even for re-verified masters."""
        path = str(tmp_path / "trajectories.json")
        with open(path, "w") as f:
            json.dump({"model_name": "test", "schema_version": 3, "entries": []}, f)
        mock_model = MagicMock()
        mock_model.encode = _mock_encode([0.1, 0.2, 0.3, 0.4])
        with patch("persistent_cache.SentenceTransformer", return_value=mock_model):
            store = VerifiedTrajectoryStore(path=path, autosave=True)
        store.store_synthesized_verified("q", "", "math", 3, ["q1", "q2", "q3"])
        assert len(store) == 0


# ---------------------------------------------------------------------- #
# Orchestrator.synthesize_trajectories
# ---------------------------------------------------------------------- #
class TestOrchestratorSynthesize:
    @pytest.mark.asyncio
    async def test_no_store_returns_error(self):
        from hybrid_orchestrator import HybridReasoningOrchestrator
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            use_clr=False, use_embedding_router=False,
            use_clr_cache=False, use_trajectory_store=False,
        )
        result = await o.synthesize_trajectories()
        assert "error" in result

    @pytest.mark.asyncio
    async def test_no_clusters_found(self, tmp_path):
        from hybrid_orchestrator import HybridReasoningOrchestrator
        # Empty store -> no clusters.
        path = str(tmp_path / "trajectories.json")
        with open(path, "w") as f:
            json.dump({"model_name": "test", "schema_version": 3, "entries": []}, f)
        mock_model = MagicMock()
        mock_model.encode = _mock_encode([0.1, 0.2, 0.3, 0.4])
        with patch("persistent_cache.SentenceTransformer", return_value=mock_model):
            o = HybridReasoningOrchestrator(
                vibe_endpoint="http://localhost:0",
                generalist_endpoint="http://localhost:0",
                use_clr=False, use_embedding_router=False,
                use_clr_cache=False, use_trajectory_store=True,
                trajectory_store_path=path,
            )
        result = await o.synthesize_trajectories()
        assert result["clusters_found"] == 0
        assert result["masters_stored"] == 0

    @pytest.mark.asyncio
    async def test_full_synthesis_flow(self, tmp_path):
        """End-to-end: 5 similar verified trajectories -> 1 cluster ->
        generalist synthesizes -> master stored, raw entries removed.

        v3.1: Without verification_context, the master is stored as
        unverified (the v0.4.0 behavior). With verification_context and
        a passing verifier, it's stored as verified=True.
        """
        from hybrid_orchestrator import HybridReasoningOrchestrator
        path = str(tmp_path / "trajectories.json")
        emb = [0.5, 0.5, 0.5, 0.5]
        entries = [
            {
                "query": f"Solve a_{i+1} = a_i + 2, a_1 = 1",
                "answer": f"a_n = 2n - 1 (variant {i})",
                "score": 0.9,
                "verification_method": "math_verifier",
                "verified": True, "embedding": emb, "task_type": "math",
                "schema_version": 3, "best_answer": f"a_n = 2n - 1 (variant {i})",
                "best_score": 0.9, "claim_count": 10,
            }
            for i in range(5)
        ]
        with open(path, "w") as f:
            json.dump({"model_name": "test", "schema_version": 3, "entries": entries}, f)
        mock_model = MagicMock()
        mock_model.encode = _mock_encode(emb)
        with patch("persistent_cache.SentenceTransformer", return_value=mock_model):
            o = HybridReasoningOrchestrator(
                vibe_endpoint="http://localhost:0",
                generalist_endpoint="http://localhost:0",
                use_clr=False, use_embedding_router=False,
                use_clr_cache=False, use_trajectory_store=True,
                trajectory_store_path=path,
            )
        # Mock the generalist to produce a synthesis.
        o._call_generalist = AsyncMock(
            return_value="For linear recurrences a_{n+1} = a_n + d, "
                         "the solution is a_n = a_1 + (n-1)*d."
        )
        result = await o.synthesize_trajectories(
            similarity_threshold=0.9, min_cluster_size=3,
        )
        assert result["clusters_found"] >= 1
        assert result["masters_stored"] == 1
        assert result["entries_removed"] == 5
        # The store should now have 1 synthesized entry (5 raw removed).
        assert len(o.trajectory_store) == 1
        assert o.trajectory_store.entries[0]["synthesized"] is True
        # No verification_context -> not re-verified -> verified=False.
        assert o.trajectory_store.entries[0]["verified"] is False
        assert o.trajectory_store.entries[0]["source_count"] == 5

    @pytest.mark.asyncio
    async def test_synthesis_reverified_when_context_available(self, tmp_path):
        """v3.1: When children have verification_context and the master
        passes re-verification, it's stored as verified=True with
        verification_method='synthesized_and_proven'."""
        from hybrid_orchestrator import HybridReasoningOrchestrator
        from verifiers.base import VerificationResult
        path = str(tmp_path / "trajectories.json")
        emb = [0.5, 0.5, 0.5, 0.5]
        entries = [
            {
                "query": f"Solve a_{i+1} = a_i + 2, a_1 = 1",
                "answer": f"a_n = 2n - 1 (variant {i})",
                "score": 0.9,
                "verification_method": "math_verifier",
                "verified": True, "embedding": emb, "task_type": "math",
                "schema_version": 3, "best_answer": f"a_n = 2n - 1 (variant {i})",
                "best_score": 0.9, "claim_count": 10,
                "verification_context": {"expected_answer": str(2 * i + 1)},
            }
            for i in range(5)
        ]
        with open(path, "w") as f:
            json.dump({"model_name": "test", "schema_version": 3, "entries": entries}, f)
        mock_model = MagicMock()
        mock_model.encode = _mock_encode(emb)
        with patch("persistent_cache.SentenceTransformer", return_value=mock_model):
            o = HybridReasoningOrchestrator(
                vibe_endpoint="http://localhost:0",
                generalist_endpoint="http://localhost:0",
                use_clr=False, use_embedding_router=False,
                use_clr_cache=False, use_trajectory_store=True,
                trajectory_store_path=path,
            )
        o._call_generalist = AsyncMock(
            return_value="For linear recurrences a_{n+1} = a_n + d, "
                         "the solution is a_n = a_1 + (n-1)*d."
        )
        # Mock the math verifier to pass re-verification.
        mock_verifier = MagicMock()
        mock_verifier.verify = AsyncMock(return_value=VerificationResult(
            verified=True, score=1.0, method="numeric_comparison",
            evidence={},
        ))
        with patch("hybrid_orchestrator.select_verifier", return_value=mock_verifier):
            result = await o.synthesize_trajectories(
                similarity_threshold=0.9, min_cluster_size=3,
            )
        assert result["masters_stored"] == 1
        assert result["masters_reverified"] == 1
        assert len(o.trajectory_store) == 1
        entry = o.trajectory_store.entries[0]
        assert entry["synthesized"] is True
        assert entry["verified"] is True
        assert entry["verification_method"] == "synthesized_and_proven"

    @pytest.mark.asyncio
    async def test_synthesis_not_reverified_when_verifier_fails(self, tmp_path):
        """v3.1: When the master fails re-verification, it's stored as
        unverified (the v0.4.0 fallback behavior)."""
        from hybrid_orchestrator import HybridReasoningOrchestrator
        from verifiers.base import VerificationResult
        path = str(tmp_path / "trajectories.json")
        emb = [0.5, 0.5, 0.5, 0.5]
        entries = [
            {
                "query": f"q{i}", "answer": f"a{i}", "score": 0.9,
                "verification_method": "math_verifier", "verified": True,
                "embedding": emb, "task_type": "math", "schema_version": 3,
                "best_answer": f"a{i}", "best_score": 0.9, "claim_count": 10,
                "verification_context": {"expected_answer": str(i)},
            }
            for i in range(4)
        ]
        with open(path, "w") as f:
            json.dump({"model_name": "test", "schema_version": 3, "entries": entries}, f)
        mock_model = MagicMock()
        mock_model.encode = _mock_encode(emb)
        with patch("persistent_cache.SentenceTransformer", return_value=mock_model):
            o = HybridReasoningOrchestrator(
                vibe_endpoint="http://localhost:0",
                generalist_endpoint="http://localhost:0",
                use_clr=False, use_embedding_router=False,
                use_clr_cache=False, use_trajectory_store=True,
                trajectory_store_path=path,
            )
        o._call_generalist = AsyncMock(return_value="master solution")
        # Mock the verifier to FAIL re-verification.
        mock_verifier = MagicMock()
        mock_verifier.verify = AsyncMock(return_value=VerificationResult(
            verified=False, score=0.0, method="numeric_comparison",
            evidence={}, error="answer mismatch",
        ))
        with patch("hybrid_orchestrator.select_verifier", return_value=mock_verifier):
            result = await o.synthesize_trajectories(
                similarity_threshold=0.9, min_cluster_size=3,
            )
        assert result["masters_stored"] == 1
        assert result["masters_reverified"] == 0
        entry = o.trajectory_store.entries[0]
        assert entry["verified"] is False
        assert entry["verification_method"] == "synthesized"

    @pytest.mark.asyncio
    async def test_generalist_failure_keeps_raw_entries(self, tmp_path):
        """If the generalist fails, raw entries are NOT removed (no data loss)."""
        from hybrid_orchestrator import HybridReasoningOrchestrator
        path = str(tmp_path / "trajectories.json")
        emb = [0.5, 0.5, 0.5, 0.5]
        entries = [
            {
                "query": f"q{i}", "answer": f"a{i}", "score": 0.9,
                "verification_method": "math_verifier", "verified": True,
                "embedding": emb, "task_type": "math", "schema_version": 3,
                "best_answer": f"a{i}", "best_score": 0.9, "claim_count": 10,
            }
            for i in range(4)
        ]
        with open(path, "w") as f:
            json.dump({"model_name": "test", "schema_version": 3, "entries": entries}, f)
        mock_model = MagicMock()
        mock_model.encode = _mock_encode(emb)
        with patch("persistent_cache.SentenceTransformer", return_value=mock_model):
            o = HybridReasoningOrchestrator(
                vibe_endpoint="http://localhost:0",
                generalist_endpoint="http://localhost:0",
                use_clr=False, use_embedding_router=False,
                use_clr_cache=False, use_trajectory_store=True,
                trajectory_store_path=path,
            )
        o._call_generalist = AsyncMock(side_effect=RuntimeError("model down"))
        result = await o.synthesize_trajectories(
            similarity_threshold=0.9, min_cluster_size=3,
        )
        assert result["masters_stored"] == 0
        assert result["entries_removed"] == 0
        # All 4 raw entries preserved.
        assert len(o.trajectory_store) == 4
        assert len(result["errors"]) >= 1
