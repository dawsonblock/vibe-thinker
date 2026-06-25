"""Pytest tests for the synchronous CLR wrapper.

Verifies that the sync wrapper delegates to the async scoring rules and
does not reintroduce the old broken ``mean ** 5`` scoring or silent
failure swallowing.
"""

from unittest.mock import AsyncMock, patch

import pytest

from vibe_clr import VibeThinkerCLR
from vibe_clr_async import CLRResult


@pytest.fixture
def clr():
    return VibeThinkerCLR(server_url="http://localhost:0", k=1)


class TestSyncUsesAsyncScoring:
    def test_sync_clr_uses_async_scoring_rules(self, clr):
        """The sync wrapper must use the same scoring as the async engine."""
        claims = ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20]
        # All verified, meaningful, answer present -> 1.0 (async rule)
        score = clr._calculate_reliability(
            [1, 1, 1, 1, 1], claims=claims, answer_present=True
        )
        assert score == 1.0

    def test_sync_clr_rejects_single_garbage_claim(self, clr):
        """Single garbage claim must score 0, not 1.0 (the old bug)."""
        score = clr._calculate_reliability(
            [1], claims=["by step reasoning."], answer_present=True
        )
        assert score == 0.0

    def test_sync_clr_rejects_fewer_than_min_claims(self, clr):
        """Fewer than 5 meaningful claims -> 0.0."""
        claims = ["a" * 20, "b" * 20]
        score = clr._calculate_reliability([1, 1], claims=claims, answer_present=True)
        assert score == 0.0

    def test_sync_clr_no_answer_returns_zero(self, clr):
        claims = ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20]
        score = clr._calculate_reliability(
            [1, 1, 1, 1, 1], claims=claims, answer_present=False
        )
        assert score == 0.0


class TestSyncDeadEndpoint:
    def test_sync_clr_dead_endpoint_raises(self, clr):
        """A dead endpoint must raise RuntimeError, not return an empty answer."""
        async def boom(*args, **kwargs):
            raise RuntimeError("Connection refused")
        with patch.object(clr._async, "_generate_one_trajectory",
                          new=AsyncMock(side_effect=boom)):
            with pytest.raises(RuntimeError, match="All CLR trajectories failed"):
                clr.run("test problem")

    def test_sync_clr_returns_result_on_success(self, clr):
        """Successful run returns a CLRResult with the best answer."""
        good_traj = {
            "score": 1.0,
            "answer": "42",
            "claims": ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20],
            "verdicts": [1, 1, 1, 1, 1],
            "raw_trace": "reasoning \\boxed{42}",
            "answer_present": True,
        }
        with patch.object(clr._async, "_generate_one_trajectory",
                          new=AsyncMock(return_value=good_traj)):
            result = clr.run("test problem")
        assert isinstance(result, CLRResult)
        assert result.best_answer == "42"
        assert result.best_score == 1.0
