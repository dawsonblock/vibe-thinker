"""Pytest tests for the cache LRU logic (no embedding deps needed for route cache)."""

import tempfile
import os

import pytest

from persistent_cache import PersistentRouteCache, should_cache


@pytest.fixture
def cache_path():
    path = tempfile.mktemp(suffix=".json")
    yield path
    if os.path.exists(path):
        os.unlink(path)


class TestLRUUpdateBug:
    """Tests for the LRU update bug: updating an existing key at capacity
    should NOT evict another entry."""

    def test_update_does_not_evict(self, cache_path):
        cache = PersistentRouteCache(path=cache_path, cache_size=2, autosave=False)
        cache.put_route("a", "specialist", 0.9)
        cache.put_route("b", "generalist", 0.8)
        assert len(cache.route_cache) == 2
        # Update 'a' — should NOT evict 'b'
        cache.put_route("a", "hybrid", 0.7)
        assert len(cache.route_cache) == 2
        assert "a" in cache.route_cache
        assert "b" in cache.route_cache
        assert cache.route_cache["a"] == ("hybrid", 0.7)

    def test_update_embedding_does_not_evict(self, cache_path):
        cache = PersistentRouteCache(path=cache_path, cache_size=2, autosave=False)
        cache.put_embedding("a", [1.0, 2.0])
        cache.put_embedding("b", [3.0, 4.0])
        assert len(cache.embedding_cache) == 2
        # Update 'a' — should NOT evict 'b'
        cache.put_embedding("a", [5.0, 6.0])
        assert len(cache.embedding_cache) == 2
        assert "a" in cache.embedding_cache
        assert "b" in cache.embedding_cache
        assert cache.embedding_cache["a"] == [5.0, 6.0]

    def test_new_key_at_capacity_evicts_oldest(self, cache_path):
        cache = PersistentRouteCache(path=cache_path, cache_size=2, autosave=False)
        cache.put_route("a", "specialist", 0.9)
        cache.put_route("b", "generalist", 0.8)
        # Insert 'c' — should evict 'a' (oldest)
        cache.put_route("c", "hybrid", 0.7)
        assert len(cache.route_cache) == 2
        assert "a" not in cache.route_cache
        assert "b" in cache.route_cache
        assert "c" in cache.route_cache

    def test_get_moves_to_end(self, cache_path):
        cache = PersistentRouteCache(path=cache_path, cache_size=2, autosave=False)
        cache.put_route("a", "specialist", 0.9)
        cache.put_route("b", "generalist", 0.8)
        # Access 'a' -> moves to end (most recently used)
        cache.get_route("a")
        # Insert 'c' -> should evict 'b' (now oldest), not 'a'
        cache.put_route("c", "hybrid", 0.7)
        assert "a" in cache.route_cache
        assert "b" not in cache.route_cache
        assert "c" in cache.route_cache


class TestShouldCache:
    """Tests for the strict cache promotion policy.

    Weak self-verification must NOT enter the cache by default.
    Self-agreement is not proof of correctness.
    """

    def _good_result(self, **overrides):
        base = {
            "answer": "42",
            "score": 0.91,
            "answer_present": True,
            "claim_count": 8,
            "verification_method": "deterministic_check",
            "failure": None,
            "transport_failures": 0,
        }
        base.update(overrides)
        return base

    def test_does_not_cache_no_clear_answer(self):
        result = self._good_result(answer="No clear answer found", answer_present=True)
        assert should_cache(result) is False

    def test_does_not_cache_self_claims_only_by_default(self):
        result = self._good_result(verification_method="self_claims_only")
        assert should_cache(result) is False

    def test_allows_self_claims_only_with_explicit_flag(self):
        result = self._good_result(verification_method="self_claims_only")
        assert should_cache(result, allow_weak_cache=True) is True

    def test_does_not_cache_transport_failure(self):
        result = self._good_result(transport_failures=2)
        assert should_cache(result) is False

    def test_does_not_cache_failure_result(self):
        result = self._good_result(failure="all trajectories failed")
        assert should_cache(result) is False

    def test_does_not_cache_low_claim_count(self):
        result = self._good_result(claim_count=3)
        assert should_cache(result) is False

    def test_does_not_cache_low_score(self):
        result = self._good_result(score=0.5)
        assert should_cache(result) is False

    def test_does_not_cache_no_answer_present(self):
        result = self._good_result(answer_present=False)
        assert should_cache(result) is False

    def test_does_not_cache_empty_answer(self):
        result = self._good_result(answer="")
        assert should_cache(result) is False

    def test_cache_accepts_deterministically_verified_answer(self):
        result = self._good_result(verification_method="python_eval")
        assert should_cache(result) is True

    def test_cache_accepts_unit_test_verified_answer(self):
        result = self._good_result(verification_method="unit_tests")
        assert should_cache(result) is True

    def test_cache_accepts_deterministic_check_verified_answer(self):
        result = self._good_result(verification_method="deterministic_check")
        assert should_cache(result) is True

    def test_cache_rejects_null_answer(self):
        result = self._good_result(answer="null")
        assert should_cache(result) is False

    def test_cache_rejects_none_answer(self):
        result = self._good_result(answer="none")
        assert should_cache(result) is False

