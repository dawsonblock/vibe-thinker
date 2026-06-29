"""Regression tests for the orchestrator runtime spine.

These guard against the class of bug where ``HybridReasoningOrchestrator.run``
calls ``self._run_clr_with_cache`` but the method is missing from the class.
That exact breakage shipped once because the method body was orphaned as dead
code (a bare string literal) after a ``return`` inside another method, so the
class compiled but ``run()`` crashed at runtime with AttributeError.

The original breakage was hidden by tests that monkeypatched
``_run_clr_with_cache`` directly onto instances — so these tests deliberately
do NOT patch that method. Instead they:

  * assert the method really exists on the *class* (not just on an instance via
    monkeypatching), and
  * exercise the real ``_run_clr_with_cache`` end-to-end by faking only the
    layer BELOW it (``self.reasoner.run``), so a missing method would raise
    AttributeError instead of being masked.

No real LLM, network, Docker, or embeddings are required.
"""

import pytest
from unittest.mock import AsyncMock

from hybrid_orchestrator import HybridReasoningOrchestrator
from vibe_clr_async import CLRResult


def _model_less_orchestrator() -> HybridReasoningOrchestrator:
    """Build an orchestrator with no embedding/cache/trajectory deps.

    Endpoints point at localhost:0 so any accidental real model call fails
    fast instead of hanging the test.
    """
    return HybridReasoningOrchestrator(
        vibe_endpoint="http://localhost:0",
        generalist_endpoint="http://localhost:0",
        use_clr=True,
        use_embedding_router=False,
        use_clr_cache=False,
        use_trajectory_store=False,
        code_verifier=None,
        retrieval_backend=None,
    )


@pytest.mark.asyncio
async def test_orchestrator_run_has_real_clr_cache_method():
    """The method must exist on the class itself, not be supplied by a test."""
    orchestrator = _model_less_orchestrator()
    assert hasattr(orchestrator, "_run_clr_with_cache")
    # Stronger than hasattr: this checks the class dict, so a test that
    # monkeypatched the method onto a bare instance would NOT satisfy it.
    assert "_run_clr_with_cache" in vars(HybridReasoningOrchestrator)
    assert callable(orchestrator._run_clr_with_cache)


@pytest.mark.asyncio
async def test_run_reaches_clr_cache_method_via_real_path():
    """run() must flow through the REAL _run_clr_with_cache.

    We fake only the layer below it (self.reasoner.run) so that a missing
    _run_clr_with_cache raises AttributeError instead of being masked. If the
    method is absent this test fails with the exact regression signature
    ("'HybridReasoningOrchestrator' object has no attribute
    '_run_clr_with_cache'").
    """
    orchestrator = _model_less_orchestrator()

    fake_clr = CLRResult(
        best_answer="x = 2",
        best_score=0.9,
        best_raw_trace="",
        verified=True,
        verification_method="math_verifier",
    )
    # Fake the layer BELOW _run_clr_with_cache — never the method itself.
    orchestrator.reasoner.run = AsyncMock(return_value=fake_clr)

    result = await orchestrator.run("Solve the equation 2x + 3 = 7")

    # The real _run_clr_with_cache called through to the fake reasoner, which
    # proves the method executed rather than being monkeypatched in.
    orchestrator.reasoner.run.assert_awaited_once()
    assert result.final_answer == "x = 2"
    assert result.route_taken == "specialist_clr"
    assert result.clr_score == 0.9
