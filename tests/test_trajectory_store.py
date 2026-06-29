"""Tests for the VerifiedTrajectoryStore (self-improving few-shot memory).

These tests verify:
  - Only verified results with deterministic verifiers are stored
  - self_claims_only and unverified results are rejected
  - Semantic retrieval finds similar verified trajectories
  - Few-shot context is built correctly
  - task_type filtering works (math examples don't pollute code queries)
  - Trust re-checking on retrieval rejects stale/corrupt entries

These are UNIT tests: they inject a fake embedding model so they exercise
the real semantic store/retrieve logic WITHOUT requiring sentence-transformers.
They only need numpy (embedding matrix) + scikit-learn (cosine similarity),
both of which are lightweight. They skip cleanly when either is absent.

Real sentence-transformers integration lives behind an explicit
@pytest.mark.embeddings integration profile (see the embeddings gate).
"""

import importlib.util
import os
import tempfile

import pytest


def _has_module(name: str) -> bool:
    """Check if a module is importable without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ImportError):
        return False


_NUMPY_AVAILABLE = _has_module("numpy")
_SKLEARN_AVAILABLE = _has_module("sklearn")

# numpy + sklearn are required for the semantic path (np.array embedding
# matrix + cosine_similarity). sentence-transformers is NOT required: a
# fake embedding model is injected via the embedding_model= constructor
# parameter, so these unit tests run in any environment that has the two
# lightweight numeric deps.
pytestmark = [
    pytest.mark.embeddings,
    pytest.mark.skipif(
        not (_NUMPY_AVAILABLE and _SKLEARN_AVAILABLE),
        reason="requires numpy + scikit-learn for semantic retrieval "
        "(NOT sentence-transformers — a fake embedding model is injected)",
    ),
]

# numpy is needed for the embedding matrix operations. The import is
# guarded so collection succeeds without numpy; tests are skipped via
# skipif above.
from persistent_cache import VerifiedTrajectoryStore


class _FakeEmbeddingModel:
    """Deterministic fake embedding model for unit tests.

    Returns fixed-dim numpy bag-of-words vectors so texts that share words
    have high cosine similarity (e.g. "sort a list" vs "filter a list" share
    most words -> sim well above the retrieval threshold), while unrelated
    texts score lower. No sentence-transformers / network needed. The real
    SentenceTransformer.encode returns a numpy array, so this mimics that
    contract (including ``.tolist()`` and ``.reshape``).

    Determinism: uses a stable per-word hash (not Python's randomized
    ``hash``), so embeddings do not change across runs (PYTHONHASHSEED).
    """

    def __init__(self, dim: int = 64):
        import numpy as np
        self._np = np
        self._dim = dim

    @staticmethod
    def _stable_hash(word: str) -> int:
        # Stable across processes (unlike hash()). Simple polynomial hash.
        h = 0
        for ch in word:
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
        return h

    def encode(self, texts, **kwargs):
        np = self._np
        if isinstance(texts, str):
            texts = [texts]
        vecs = []
        for t in texts:
            v = [0.0] * self._dim
            for word in t.lower().split():
                v[self._stable_hash(word) % self._dim] += 1.0
            if not any(v):  # non-empty guard so the norm is never 0
                v[0] = 1.0
            vecs.append(v)
        return np.array(vecs)


@pytest.fixture
def fake_embedding_model():
    return _FakeEmbeddingModel()


@pytest.fixture
def store(tmp_path, fake_embedding_model):
    """Fresh trajectory store in a temp directory with a fake embedding model.

    Injecting embedding_model= exercises the real EMBEDDINGS-mode store +
    semantic-retrieve path without requiring sentence-transformers.
    """
    path = str(tmp_path / "trajectories.json")
    return VerifiedTrajectoryStore(
        path=path,
        retrieval_threshold=0.50,  # lower for test reliability
        max_few_shot=3,
        embedding_model=fake_embedding_model,
    )


class TestTrajectoryStoreInsert:
    def test_store_verified_result(self, store):
        store.store(
            query="Compute the sum of 1 + 2 + 3 + 4 + 5",
            answer="15",
            score=0.895,
            verification_method="math_verifier",
            task_type="math",
        )
        assert len(store) == 1

    def test_refuses_self_claims_only(self, store):
        store.store(
            query="Some query",
            answer="Some answer",
            score=0.65,
            verification_method="self_claims_only",
            task_type="math",
        )
        assert len(store) == 0

    def test_refuses_empty_answer(self, store):
        store.store(
            query="Some query",
            answer="",
            score=1.0,
            verification_method="math_verifier",
            task_type="math",
        )
        assert len(store) == 0

    def test_refuses_bad_answer(self, store):
        store.store(
            query="Some query",
            answer="no clear answer found",
            score=1.0,
            verification_method="math_verifier",
            task_type="math",
        )
        assert len(store) == 0

    def test_persists_to_disk(self, tmp_path, fake_embedding_model):
        path = str(tmp_path / "trajectories.json")
        s1 = VerifiedTrajectoryStore(
            path=path, retrieval_threshold=0.50,
            embedding_model=fake_embedding_model,
        )
        s1.store(
            query="What is 2+2?",
            answer="4",
            score=0.9,
            verification_method="math_verifier",
            task_type="math",
        )
        # Create a new store from the same file — should load the entry
        s2 = VerifiedTrajectoryStore(
            path=path, retrieval_threshold=0.50,
            embedding_model=fake_embedding_model,
        )
        assert len(s2) == 1


class TestTrajectoryStoreRetrieval:
    def test_retrieve_similar(self, store):
        store.store(
            query="Compute the sum of 1 + 2 + 3 + 4 + 5",
            answer="15",
            score=0.895,
            verification_method="math_verifier",
            task_type="math",
        )
        # A similar query should find it
        results = store.retrieve("Compute the sum of 2 + 3 + 4 + 5 + 6")
        assert len(results) >= 1
        assert results[0]["answer"] == "15"
        assert results[0]["verification_method"] == "math_verifier"

    def test_retrieve_filters_by_task_type(self, store):
        store.store(
            query="Write a Python function to sort a list",
            answer="def sort(l): return sorted(l)",
            score=1.0,
            verification_method="unit_tests",
            task_type="code",
        )
        # A math query should NOT get code examples
        math_results = store.retrieve("Solve the equation 2x + 3 = 7", task_type="math")
        assert len(math_results) == 0
        # A code query SHOULD get code examples
        code_results = store.retrieve("Write a Python function to filter a list", task_type="code")
        assert len(code_results) >= 1

    def test_retrieve_returns_empty_when_no_entries(self, store):
        assert store.retrieve("any query") == []

    def test_build_few_shot_prefix(self, store):
        store.store(
            query="Compute the sum of 1 + 2 + 3 + 4 + 5",
            answer="15",
            score=0.895,
            verification_method="math_verifier",
            task_type="math",
        )
        prefix = store.build_few_shot_prefix("Compute the sum of 2 + 3 + 4 + 5 + 6", task_type="math")
        assert "verified" in prefix.lower()
        assert "15" in prefix
        assert "math_verifier" in prefix

    def test_build_few_shot_prefix_empty_when_no_match(self, store):
        prefix = store.build_few_shot_prefix("completely unrelated query about gardening")
        assert prefix == ""

    def test_max_few_shot_limit(self, store):
        # Store 5 verified math results
        for i in range(5):
            store.store(
                query=f"Compute the sum of {i} + {i+1} + {i+2} + {i+3} + {i+4}",
                answer=str(5*i + 10),
                score=0.9,
                verification_method="math_verifier",
                task_type="math",
            )
        results = store.retrieve("Compute the sum of 1 + 2 + 3 + 4 + 5", task_type="math")
        assert len(results) <= store.max_few_shot


class TestTrajectoryStoreTrust:
    def test_corrupt_entry_rejected_on_load(self, tmp_path, fake_embedding_model):
        """An entry with verified=False should be filtered out on load."""
        import json
        path = str(tmp_path / "trajectories.json")
        # Write a corrupt file with an unverified entry
        with open(path, "w") as f:
            json.dump({
                "model_name": "all-MiniLM-L6-v2",
                "schema_version": 3,
                "entries": [{
                    "query": "test",
                    "embedding": [0.0] * 384,
                    "answer": "test answer",
                    "score": 0.9,
                    "verification_method": "math_verifier",
                    "task_type": "math",
                    "verified": False,  # NOT verified — should be rejected
                    "schema_version": 3,
                    "best_answer": "test answer",
                    "best_score": 0.9,
                    "claim_count": 10,
                }],
            }, f)
        store = VerifiedTrajectoryStore(
            path=path, retrieval_threshold=0.0,
            embedding_model=fake_embedding_model,
        )
        assert len(store) == 0  # corrupt entry filtered out

    def test_self_claims_entry_rejected_on_load(self, tmp_path, fake_embedding_model):
        """An entry with self_claims_only should be filtered out on load."""
        import json
        path = str(tmp_path / "trajectories.json")
        with open(path, "w") as f:
            json.dump({
                "model_name": "all-MiniLM-L6-v2",
                "schema_version": 3,
                "entries": [{
                    "query": "test",
                    "embedding": [0.0] * 384,
                    "answer": "test answer",
                    "score": 0.9,
                    "verification_method": "self_claims_only",  # rejected
                    "task_type": "math",
                    "verified": True,
                    "schema_version": 3,
                    "best_answer": "test answer",
                    "best_score": 0.9,
                    "claim_count": 10,
                }],
            }, f)
        store = VerifiedTrajectoryStore(
            path=path, retrieval_threshold=0.0,
            embedding_model=fake_embedding_model,
        )
        assert len(store) == 0
