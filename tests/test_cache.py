"""Pytest tests for the cache LRU logic (no embedding deps needed for route cache)."""

import tempfile
import os

import pytest

from persistent_cache import PersistentRouteCache


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
