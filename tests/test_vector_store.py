"""Tests for the vector store abstraction (vector_store.py).

Covers:
  - LocalVectorStore (in-memory numpy + sklearn, default)
  - AgentDBVectorStore (HTTP sidecar, fail-closed when not running)
  - ShadowVectorStore (dual-write, primary-read-with-fallback)
  - make_vector_store factory
  - Integration with CLRResultCache and VerifiedTrajectoryStore
"""

import os
import tempfile

import pytest

from vector_store import (
    LocalVectorStore,
    AgentDBVectorStore,
    ShadowVectorStore,
    make_vector_store,
    VectorStore,
)


# Skip tests that need numpy/sklearn if not installed.
embeddings_available = True
try:
    import numpy  # noqa: F401
    from sklearn.metrics.pairwise import cosine_similarity  # noqa: F401
except ImportError:
    embeddings_available = False

skip_no_embeddings = pytest.mark.skipif(
    not embeddings_available,
    reason="numpy/sklearn not installed — LocalVectorStore tests skipped",
)


@skip_no_embeddings
class TestLocalVectorStore:
    """Tests for the in-memory vector store."""

    def test_upsert_and_count(self):
        s = LocalVectorStore()
        s.upsert("a", [1.0, 0.0, 0.0])
        s.upsert("b", [0.0, 1.0, 0.0])
        assert s.count() == 2

    def test_upsert_replaces_existing(self):
        s = LocalVectorStore()
        s.upsert("a", [1.0, 0.0], {"v": 1})
        s.upsert("a", [0.0, 1.0], {"v": 2})  # replace
        assert s.count() == 1
        res = s.search([0.0, 1.0], top_k=1)
        assert res[0][0] == "a"
        assert res[0][2]["v"] == 2

    def test_search_returns_sorted_by_similarity(self):
        s = LocalVectorStore()
        s.upsert("a", [1.0, 0.0, 0.0])
        s.upsert("b", [0.0, 1.0, 0.0])
        s.upsert("c", [0.9, 0.1, 0.0])
        res = s.search([1.0, 0.0, 0.0], top_k=3)
        # 'a' is exact match (sim=1.0), 'c' is close, 'b' is orthogonal
        assert res[0][0] == "a"
        assert res[0][1] == pytest.approx(1.0)
        assert res[1][0] == "c"
        assert res[2][0] == "b"

    def test_search_respects_top_k(self):
        s = LocalVectorStore()
        for i in range(5):
            s.upsert(f"id{i}", [float(i), 0.0])
        res = s.search([0.0, 0.0], top_k=2)
        assert len(res) == 2

    def test_search_with_filters(self):
        s = LocalVectorStore()
        s.upsert("a", [1.0, 0.0], {"task_type": "math"})
        s.upsert("b", [0.0, 1.0], {"task_type": "code"})
        s.upsert("c", [0.9, 0.1], {"task_type": "math"})
        res = s.search([1.0, 0.0], top_k=5, filters={"task_type": "math"})
        ids = [r[0] for r in res]
        assert "a" in ids and "c" in ids
        assert "b" not in ids

    def test_search_empty_store(self):
        s = LocalVectorStore()
        assert s.search([1.0], top_k=5) == []

    def test_delete(self):
        s = LocalVectorStore()
        s.upsert("a", [1.0, 0.0])
        assert s.delete("a")
        assert s.count() == 0
        assert not s.delete("a")  # already deleted
        assert not s.delete("nonexistent")

    def test_protocol_conformance(self):
        s = LocalVectorStore()
        assert isinstance(s, VectorStore)


class TestAgentDBVectorStore:
    """Tests for the AgentDB HTTP sidecar (fail-closed when not running)."""

    def test_fail_closed_search_returns_empty(self):
        # Point at an invalid port — no sidecar running
        s = AgentDBVectorStore("http://127.0.0.1:1", "test")
        assert s.search([1.0], top_k=5) == []

    def test_fail_closed_count_returns_zero(self):
        s = AgentDBVectorStore("http://127.0.0.1:1", "test")
        assert s.count() == 0

    def test_fail_closed_delete_returns_false(self):
        s = AgentDBVectorStore("http://127.0.0.1:1", "test")
        assert s.delete("x") is False

    def test_fail_closed_upsert_does_not_raise(self):
        s = AgentDBVectorStore("http://127.0.0.1:1", "test")
        # Should warn but not raise
        s.upsert("x", [1.0], {"k": "v"})

    def test_protocol_conformance(self):
        s = AgentDBVectorStore("http://127.0.0.1:1", "test")
        assert isinstance(s, VectorStore)


@skip_no_embeddings
class TestShadowVectorStore:
    """Tests for the dual-write shadow store."""

    def test_dual_write_primary_serves_reads(self):
        local = LocalVectorStore()
        agentdb = AgentDBVectorStore("http://127.0.0.1:1", "test")  # down
        shadow = ShadowVectorStore(local, agentdb)
        shadow.upsert("a", [1.0, 0.0], {"k": "v"})
        # Read should come from primary (local)
        res = shadow.search([1.0, 0.0], top_k=1)
        assert len(res) == 1
        assert res[0][0] == "a"

    def test_fallback_to_secondary_when_primary_empty(self):
        local = LocalVectorStore()  # empty
        # Use a local store as the "secondary" too (simulating a
        # reachable AgentDB with data)
        secondary = LocalVectorStore()
        secondary.upsert("remote", [1.0, 0.0], {"src": "remote"})
        shadow = ShadowVectorStore(local, secondary)
        res = shadow.search([1.0, 0.0], top_k=1)
        assert len(res) == 1
        assert res[0][0] == "remote"

    def test_delete_propagates_to_both(self):
        local = LocalVectorStore()
        secondary = LocalVectorStore()
        shadow = ShadowVectorStore(local, secondary)
        shadow.upsert("a", [1.0, 0.0])
        assert shadow.delete("a")
        assert local.count() == 0
        assert secondary.count() == 0

    def test_count_uses_primary(self):
        local = LocalVectorStore()
        secondary = LocalVectorStore()
        secondary.upsert("x", [1.0])  # only in secondary
        shadow = ShadowVectorStore(local, secondary)
        assert shadow.count() == 0  # primary is empty

    def test_protocol_conformance(self):
        local = LocalVectorStore()
        secondary = LocalVectorStore()
        shadow = ShadowVectorStore(local, secondary)
        assert isinstance(shadow, VectorStore)


class TestMakeVectorStoreFactory:
    """Tests for the make_vector_store factory."""

    @skip_no_embeddings
    def test_no_url_returns_local(self):
        s = make_vector_store()
        assert isinstance(s, LocalVectorStore)

    def test_url_returns_agentdb(self):
        s = make_vector_store(agentdb_url="http://127.0.0.1:1")
        assert isinstance(s, AgentDBVectorStore)

    @skip_no_embeddings
    def test_url_with_shadow_primary_returns_shadow(self):
        s = make_vector_store(
            agentdb_url="http://127.0.0.1:1",
            shadow_primary=LocalVectorStore(),
        )
        assert isinstance(s, ShadowVectorStore)


@skip_no_embeddings
class TestVectorStoreClustering:
    """Tests for the cluster() method (v1.0).

    Tests LocalVectorStore.cluster() with real vectors, AgentDBVectorStore
    fail-closed behavior, and ShadowVectorStore delegation.
    """

    def test_local_cluster_finds_two_clusters(self):
        """5 near-identical + 5 near-identical (different direction) = 2 clusters."""
        import numpy as np
        s = LocalVectorStore()
        np.random.seed(42)
        for i in range(5):
            s.upsert(f"a_{i}", [1.0 + np.random.randn() * 0.01, 0.01, 0.01, 0.01])
        for i in range(5):
            s.upsert(f"b_{i}", [0.01, 1.0 + np.random.randn() * 0.01, 0.01, 0.01])
        clusters = s.cluster(similarity_threshold=0.85, min_cluster_size=3)
        assert len(clusters) == 2
        assert len(clusters[0]) == 5
        assert len(clusters[1]) == 5

    def test_local_cluster_single_cluster(self):
        """All similar vectors form one cluster."""
        import numpy as np
        s = LocalVectorStore()
        np.random.seed(42)
        for i in range(6):
            s.upsert(f"v_{i}", [1.0 + np.random.randn() * 0.01, 0.01, 0.01])
        clusters = s.cluster(similarity_threshold=0.85, min_cluster_size=3)
        assert len(clusters) == 1
        assert len(clusters[0]) == 6

    def test_local_cluster_dissimilar_no_clusters(self):
        """Orthogonal vectors don't cluster."""
        s = LocalVectorStore()
        s.upsert("a", [1.0, 0.0, 0.0])
        s.upsert("b", [0.0, 1.0, 0.0])
        s.upsert("c", [0.0, 0.0, 1.0])
        clusters = s.cluster(similarity_threshold=0.5, min_cluster_size=2)
        assert clusters == []

    def test_local_cluster_min_cluster_size_filter(self):
        """Clusters smaller than min_cluster_size are excluded."""
        import numpy as np
        s = LocalVectorStore()
        np.random.seed(42)
        # 2 similar (below min_cluster_size=3)
        s.upsert("a1", [1.0, 0.01, 0.01])
        s.upsert("a2", [1.0, 0.01, 0.01])
        # 5 similar (above min_cluster_size=3)
        for i in range(5):
            s.upsert(f"b_{i}", [0.01, 1.0 + np.random.randn() * 0.01, 0.01])
        clusters = s.cluster(similarity_threshold=0.85, min_cluster_size=3)
        assert len(clusters) == 1
        assert len(clusters[0]) == 5

    def test_local_cluster_with_filters(self):
        """Filters narrow the clustering candidates."""
        import numpy as np
        s = LocalVectorStore()
        np.random.seed(42)
        for i in range(5):
            s.upsert(f"math_{i}", [1.0 + np.random.randn() * 0.01, 0.01, 0.01, 0.01],
                     {"task_type": "math"})
        for i in range(5):
            s.upsert(f"code_{i}", [0.01, 1.0 + np.random.randn() * 0.01, 0.01, 0.01],
                     {"task_type": "code"})
        # Filter to only math
        math_clusters = s.cluster(
            similarity_threshold=0.85, min_cluster_size=3,
            filters={"task_type": "math"},
        )
        assert len(math_clusters) == 1
        assert all("math_" in vid for vid in math_clusters[0])

    def test_local_cluster_empty_store(self):
        """Empty store returns no clusters."""
        s = LocalVectorStore()
        assert s.cluster() == []

    def test_local_cluster_fewer_than_min(self):
        """Store with fewer entries than min_cluster_size returns no clusters."""
        s = LocalVectorStore()
        s.upsert("a", [1.0, 0.0])
        s.upsert("b", [1.0, 0.0])
        assert s.cluster(min_cluster_size=3) == []

    def test_agentdb_cluster_fail_closed_when_down(self):
        """AgentDBVectorStore.cluster() returns [] when sidecar is down."""
        s = AgentDBVectorStore(base_url="http://127.0.0.1:9999", collection="test")
        assert s.cluster() == []

    def test_shadow_cluster_delegates_to_primary(self):
        """ShadowVectorStore.cluster() delegates to the primary store."""
        import numpy as np
        primary = LocalVectorStore()
        secondary = AgentDBVectorStore(
            base_url="http://127.0.0.1:9999", collection="shadow"
        )
        shadow = ShadowVectorStore(primary, secondary)
        np.random.seed(42)
        for i in range(5):
            primary.upsert(f"a_{i}", [1.0 + np.random.randn() * 0.01, 0.01, 0.01])
        clusters = shadow.cluster(similarity_threshold=0.85, min_cluster_size=3)
        assert len(clusters) == 1
        assert len(clusters[0]) == 5


@skip_no_embeddings
class TestCacheIntegration:
    """Tests that CLRResultCache and VerifiedTrajectoryStore accept
    the new vector_store / agentdb_url parameters without breaking."""

    @pytest.fixture
    def cache_path(self):
        path = tempfile.mktemp(suffix=".json")
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_clr_cache_accepts_vector_store_param(self, cache_path):
        from persistent_cache import CLRResultCache
        vs = LocalVectorStore()
        cache = CLRResultCache(cache_path, vector_store=vs)
        assert cache._vector_store is vs

    def test_clr_cache_accepts_agentdb_url_param(self, cache_path):
        from persistent_cache import CLRResultCache
        cache = CLRResultCache(cache_path, agentdb_url="http://127.0.0.1:1")
        assert cache._vector_store is not None
        assert isinstance(cache._vector_store, ShadowVectorStore)

    def test_clr_cache_default_no_vector_store(self, cache_path):
        from persistent_cache import CLRResultCache
        cache = CLRResultCache(cache_path)
        assert cache._vector_store is None  # unchanged default behavior

    def test_trajectory_store_accepts_vector_store_param(self, cache_path):
        from persistent_cache import VerifiedTrajectoryStore
        vs = LocalVectorStore()
        store = VerifiedTrajectoryStore(cache_path, vector_store=vs)
        assert store._vector_store is vs

    def test_trajectory_store_accepts_agentdb_url_param(self, cache_path):
        from persistent_cache import VerifiedTrajectoryStore
        store = VerifiedTrajectoryStore(cache_path, agentdb_url="http://127.0.0.1:1")
        assert store._vector_store is not None
        assert isinstance(store._vector_store, ShadowVectorStore)

    def test_trajectory_store_default_no_vector_store(self, cache_path):
        from persistent_cache import VerifiedTrajectoryStore
        store = VerifiedTrajectoryStore(cache_path)
        assert store._vector_store is None

    def test_clr_cache_dual_writes_to_vector_store(self, cache_path):
        """Insert into CLRResultCache should also upsert to the vector store
        (shadow-mode dual-write)."""
        from persistent_cache import CLRResultCache
        vs = LocalVectorStore()
        cache = CLRResultCache(cache_path, vector_store=vs)
        cache.insert(
            problem="what is 2+2",
            best_answer="4",
            best_score=0.95,
            k=8,
            trajectory_count=8,
            verified=True,
            verification_method="python_eval",
        )
        # The vector store should have received the upsert
        assert vs.count() == 1
        results = vs.search([0.0] * 384, top_k=1)  # dim doesn't matter for count
        assert len(results) == 1

    def test_trajectory_store_dual_writes_to_vector_store(self, cache_path):
        """Store into VerifiedTrajectoryStore should also upsert to the vector
        store (shadow-mode dual-write)."""
        from persistent_cache import VerifiedTrajectoryStore
        vs = LocalVectorStore()
        store = VerifiedTrajectoryStore(cache_path, vector_store=vs)
        store.store(
            query="solve fibonacci",
            answer="0, 1, 1, 2, 3, 5, 8",
            score=0.92,
            verification_method="python_eval",
            task_type="math",
        )
        # The vector store should have received the upsert
        assert vs.count() == 1

    def test_clr_cache_dual_write_failure_is_non_fatal(self, cache_path):
        """If the vector store raises, the insert should still succeed."""
        from persistent_cache import CLRResultCache

        class BrokenVectorStore:
            def upsert(self, *args, **kwargs):
                raise RuntimeError("AgentDB is down")
            def search(self, *args, **kwargs):
                return []
            def count(self):
                return 0
            def delete(self, *args):
                return False

        cache = CLRResultCache(cache_path, vector_store=BrokenVectorStore())
        # Should not raise despite the broken vector store
        cache.insert(
            problem="what is 2+2",
            best_answer="4",
            best_score=0.95,
            k=8,
            trajectory_count=8,
            verified=True,
            verification_method="python_eval",
        )
        # The local cache should still have the entry
        assert len(cache) == 1


class TestRuvLLMVectorStore:
    """Tests for the in-process HNSW vector store (v2.0).

    These tests only run when the ruvllm_py binding is installed.
    """

    @pytest.fixture
    def store(self):
        """Create a RuvLLMVectorStore with dim=4."""
        from vector_store import RuvLLMVectorStore, is_ruvllm_vector_store_available
        if not is_ruvllm_vector_store_available():
            pytest.skip("ruvllm_py HNSW binding not installed")
        return RuvLLMVectorStore(dim=4, m=8, ef_construction=50, ef_search=16)

    def test_upsert_and_search(self, store):
        """Upserting vectors and searching returns the closest match."""
        store.upsert("v1", [1.0, 0.0, 0.0, 0.0], {"source": "test"})
        store.upsert("v2", [0.0, 1.0, 0.0, 0.0], {"source": "test"})
        results = store.search([1.0, 0.1, 0.0, 0.0], top_k=2)
        # HNSW may return fewer than top_k results for very small indexes.
        assert len(results) >= 1
        # v1 should be closer to the query than v2 (if both returned).
        assert results[0][0] == "v1"

    def test_count(self, store):
        """Count returns the number of stored vectors."""
        store.upsert("a", [1.0, 0.0, 0.0, 0.0])
        store.upsert("b", [0.0, 1.0, 0.0, 0.0])
        assert store.count() == 2

    def test_delete_returns_false(self, store):
        """HNSW doesn't support deletion — returns False."""
        store.upsert("x", [1.0, 0.0, 0.0, 0.0])
        assert store.delete("x") is False

    def test_cluster_returns_empty(self, store):
        """HNSW doesn't expose clustering — returns []."""
        store.upsert("x", [1.0, 0.0, 0.0, 0.0])
        assert store.cluster() == []

    def test_search_with_filters(self, store):
        """Filters exclude entries whose metadata doesn't match."""
        store.upsert("v1", [1.0, 0.0, 0.0, 0.0], {"type": "math"})
        store.upsert("v2", [0.0, 1.0, 0.0, 0.0], {"type": "code"})
        results = store.search([0.5, 0.5, 0.0, 0.0], top_k=10, filters={"type": "math"})
        assert len(results) == 1
        assert results[0][0] == "v1"

    def test_is_ruvllm_vector_store_available(self):
        """The availability check should return a bool."""
        from vector_store import is_ruvllm_vector_store_available
        assert isinstance(is_ruvllm_vector_store_available(), bool)
