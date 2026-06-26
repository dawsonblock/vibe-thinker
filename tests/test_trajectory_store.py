"""Tests for the VerifiedTrajectoryStore (self-improving few-shot memory).

These tests verify:
  - Only verified results with deterministic verifiers are stored
  - self_claims_only and unverified results are rejected
  - Semantic retrieval finds similar verified trajectories
  - Few-shot context is built correctly
  - task_type filtering works (math examples don't pollute code queries)
  - Trust re-checking on retrieval rejects stale/corrupt entries
"""

import os
import tempfile

import pytest

from persistent_cache import VerifiedTrajectoryStore


@pytest.fixture
def store(tmp_path):
    """Fresh trajectory store in a temp directory."""
    path = str(tmp_path / "trajectories.json")
    return VerifiedTrajectoryStore(
        path=path,
        retrieval_threshold=0.50,  # lower for test reliability
        max_few_shot=3,
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

    def test_persists_to_disk(self, tmp_path):
        path = str(tmp_path / "trajectories.json")
        s1 = VerifiedTrajectoryStore(path=path, retrieval_threshold=0.50)
        s1.store(
            query="What is 2+2?",
            answer="4",
            score=0.9,
            verification_method="math_verifier",
            task_type="math",
        )
        # Create a new store from the same file — should load the entry
        s2 = VerifiedTrajectoryStore(path=path, retrieval_threshold=0.50)
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
    def test_corrupt_entry_rejected_on_load(self, tmp_path):
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
        store = VerifiedTrajectoryStore(path=path, retrieval_threshold=0.0)
        assert len(store) == 0  # corrupt entry filtered out

    def test_self_claims_entry_rejected_on_load(self, tmp_path):
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
        store = VerifiedTrajectoryStore(path=path, retrieval_threshold=0.0)
        assert len(store) == 0
