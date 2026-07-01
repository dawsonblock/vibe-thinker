"""Tests for AgentDB-only mode in the semantic caches.

Verifies that when ``agentdb_only=True`` and the local JSON cache is empty
or archived, the caches still query the AgentDB vector store.
"""

from pathlib import Path

import numpy as np
import pytest

from persistent_cache import CLRResultCache, VerifiedTrajectoryStore


pytestmark = [
    pytest.mark.embeddings,
]


class FakeVectorStore:
    """In-memory vector store that reproduces the AgentDBVectorStore interface."""

    def __init__(self):
        self.entries = {}

    def upsert(self, vector_id, embedding, metadata=None):
        self.entries[vector_id] = (embedding, metadata or {})

    def search(self, query_embedding, top_k=10, filters=None):
        # Flatten a 2D query embedding (model.encode returns a batch).
        if hasattr(query_embedding, "shape") and len(query_embedding.shape) > 1:
            query_embedding = query_embedding[0].tolist()
        elif isinstance(query_embedding, list) and query_embedding and isinstance(query_embedding[0], list):
            query_embedding = query_embedding[0]
        query_embedding = [float(x) for x in query_embedding]
        results = []
        for vid, (emb, meta) in self.entries.items():
            emb = [float(x) for x in emb]
            sim = sum(a * b for a, b in zip(query_embedding, emb))
            results.append((vid, sim, meta))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def delete(self, vector_id):
        return self.entries.pop(vector_id, None) is not None

    def count(self):
        return len(self.entries)


class FakeEmbedder:
    """Deterministic embedder for tests.

    Encodes the first token as a one-hot vector of length 4. Two queries are
    similar when they share the same first token.
    """

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            vec = [0.0, 0.0, 0.0, 0.0]
            token = t.split()[0] if t else ""
            # Map a few common tokens to dimensions; unknown -> 0.
            mapping = {
                "problem": 0, "Compute": 0,
                "query": 1, "Write": 1,
                "Solve": 2, "answer": 2,
                "List": 3,
            }
            idx = mapping.get(token, 0)
            vec[idx] = 1.0
            out.append(np.array(vec, dtype=float))
        return np.array(out, dtype=float)


def _make_clr_cache(tmp_path, vector_store, empty=True):
    path = str(tmp_path / "clr.json")
    cache = CLRResultCache(
        path=path,
        vector_store=vector_store,
        similarity_mode=None,
        agentdb_only=True,
    )
    # Inject a deterministic embedder so tests run without sentence-transformers.
    cache.model = FakeEmbedder()
    if not empty:
        cache.insert(
            problem="problem alpha",
            best_answer="answer alpha",
            best_score=0.9,
            verified=True,
            verification_method="math_verifier",
            k=3,
            trajectory_count=5,
        )
    return cache


def _make_traj_store(tmp_path, vector_store, empty=True):
    path = str(tmp_path / "trajectories.json")
    store = VerifiedTrajectoryStore(
        path=path,
        vector_store=vector_store,
        embedding_model=FakeEmbedder(),
        similarity_mode=None,
        agentdb_only=True,
    )
    if not empty:
        store.store(
            query="query alpha",
            answer="answer alpha",
            score=0.85,
            verification_method="math_verifier",
            task_type="math",
        )
    return store


def _populate_agentdb(vector_store, kind="clr"):
    """Seed the fake vector store with a trustworthy entry."""
    embedder = FakeEmbedder()
    if kind == "clr":
        emb = embedder.encode(["problem beta"])[0]
        vector_store.upsert(
            "clr_beta",
            emb,
            {
                "problem": "problem beta",
                "best_answer": "answer beta",
                "best_score": 0.95,
                "k": 4,
                "timestamp": "2024-01-01T00:00:00",
                "trajectory_count": 6,
                "verified": True,
                "verification_method": "math_verifier",
                "task_type": "math",
                "schema_version": 3,
                "claim_count": 10,
            },
        )
    else:
        emb = embedder.encode(["query beta"])[0]
        vector_store.upsert(
            "traj_beta",
            emb,
            {
                "query": "query beta",
                "answer": "answer beta",
                "score": 0.88,
                "best_answer": "answer beta",
                "best_score": 0.88,
                "verification_method": "math_verifier",
                "task_type": "math",
                "verified": True,
                "schema_version": 3,
                "claim_count": 10,
            },
        )


class TestAgentDBOnlyCLRResultCache:
    """CLRResultCache returns AgentDB results even when local JSON is empty."""

    def test_lookup_with_empty_local_and_agentdb_result(self, tmp_path):
        vs = FakeVectorStore()
        _populate_agentdb(vs, kind="clr")
        cache = _make_clr_cache(tmp_path, vs, empty=True)
        assert cache.entries == []
        result = cache.lookup("problem beta")
        assert result is not None
        assert result["best_answer"] == "answer beta"
        assert result["best_score"] == 0.95
        assert result["matched_problem"] == "problem beta"
        assert result["cached"] is True

    def test_lookup_with_empty_agentdb_returns_none(self, tmp_path):
        vs = FakeVectorStore()
        cache = _make_clr_cache(tmp_path, vs, empty=True)
        assert cache.lookup("problem beta") is None


class TestAgentDBOnlyTrajectoryStore:
    """VerifiedTrajectoryStore returns AgentDB results with empty local JSON."""

    def test_retrieve_with_empty_local_and_agentdb_result(self, tmp_path):
        vs = FakeVectorStore()
        _populate_agentdb(vs, kind="traj")
        store = _make_traj_store(tmp_path, vs, empty=True)
        assert store.entries == []
        results = store.retrieve("query beta")
        assert len(results) == 1
        assert results[0]["answer"] == "answer beta"
        assert results[0]["score"] == 0.88

    def test_retrieve_with_empty_agentdb_returns_empty(self, tmp_path):
        vs = FakeVectorStore()
        store = _make_traj_store(tmp_path, vs, empty=True)
        assert store.retrieve("query beta") == []
