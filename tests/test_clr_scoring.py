"""Pytest tests for the CLR scoring logic (no model servers needed)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vibe_clr_async import CLRResult, VibeThinkerCLRAsync


@pytest.fixture
def clr():
    """VibeThinkerCLRAsync without needing a real server."""
    return VibeThinkerCLRAsync(server_url="http://localhost:0", k=1)


class TestReliabilityScoring:
    def test_empty_verdicts_returns_zero(self, clr):
        assert clr._calculate_reliability([]) == 0.0

    def test_no_answer_returns_zero(self, clr):
        # No answer_present flag -> score 0, even with 5 verified claims
        claims = ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20]
        assert clr._calculate_reliability([1, 1, 1, 1, 1], claims=claims, answer_present=False) == 0.0

    def test_fewer_than_min_claims_returns_zero(self, clr):
        # Only 2 meaningful claims — below MIN_CLAIMS_FOR_SCORING=5
        claims = ["a" * 20, "b" * 20]
        assert clr._calculate_reliability([1, 1], claims=claims, answer_present=True) == 0.0

    def test_single_claim_returns_zero(self, clr):
        # The smoking gun from the audit: 1 verified claim -> 1.0
        # Now it must return 0.0
        assert clr._calculate_reliability([1], claims=["a meaningful claim here"], answer_present=True) == 0.0

    def test_garbage_claims_rejected(self, clr):
        # The exact garbage from the audit: "by step reasoning."
        claims = ["by step reasoning.", "by step.", "by step reasoning. So we can elaborate."]
        # All are garbage -> filtered out -> 0 meaningful -> score 0
        assert clr._calculate_reliability([1, 1, 1], claims=claims, answer_present=True) == 0.0

    def test_short_claims_rejected(self, clr):
        # Claims shorter than MIN_CLAIM_LENGTH (15 chars) are too trivial
        claims = ["short", "tiny", "x"]
        assert clr._calculate_reliability([1, 1, 1], claims=claims, answer_present=True) == 0.0

    def test_any_failed_verdict_capped(self, clr):
        # One wrong claim out of 5 -> score capped at 0.3
        claims = ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20]
        score = clr._calculate_reliability([1, 1, 1, 1, 0], claims=claims, answer_present=True)
        assert score <= 0.3
        assert score > 0.0  # not zero, but heavily penalized

    def test_self_claims_only_is_capped_at_065(self, clr):
        """The most important test: 5 self-verified claims must NOT reach 1.0.
        Self-verification alone is capped at 0.65 — model self-agreement is
        not proof of correctness."""
        claims = [
            "This is a meaningful claim with enough detail one.",
            "This is a meaningful claim with enough detail two.",
            "This is a meaningful claim with enough detail three.",
            "This is a meaningful claim with enough detail four.",
            "This is a meaningful claim with enough detail five.",
        ]
        score = clr._calculate_reliability(
            [1, 1, 1, 1, 1], claims=claims, answer_present=True,
            consistency_check=None,
        )
        assert score <= 0.65, f"Self-claims-only score {score} exceeds 0.65 cap"

    def test_all_verified_meaningful_claims_capped_without_verifier(self, clr):
        """Without a deterministic verifier, even perfect self-verification
        cannot exceed 0.65."""
        claims = ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20]
        score = clr._calculate_reliability([1, 1, 1, 1, 1], claims=claims, answer_present=True)
        assert score <= 0.65

    def test_consistency_check_does_not_exceed_065(self, clr):
        """Cross-trajectory consistency (model agreeing with itself) must NOT
        allow score above 0.65. Consensus is not proof — only external
        verifiers can exceed the cap."""
        claims = ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20]
        score = clr._calculate_reliability(
            [1, 1, 1, 1, 1], claims=claims, answer_present=True,
            consistency_check=True,
        )
        assert score <= 0.65, f"Consistency-boosted score {score} exceeds 0.65 cap"

    def test_consistency_check_refutation_penalizes(self, clr):
        """If trajectories contradict, the score is penalized (but not zeroed
        — that's only for external verifier refutation)."""
        claims = ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20]
        # Use mixed verdicts so base score is below the cap, allowing
        # us to see the consistency boost vs contradiction penalty.
        score_consistent = clr._calculate_reliability(
            [1, 1, 1, 1, 0], claims=claims, answer_present=True,
            consistency_check=True,
        )
        score_contradicted = clr._calculate_reliability(
            [1, 1, 1, 1, 0], claims=claims, answer_present=True,
            consistency_check=False,
        )
        assert score_contradicted < score_consistent
        assert score_contradicted <= 0.65
        assert score_consistent <= 0.65

    def test_mixed_garbage_and_real_claims_capped(self, clr):
        # 2 garbage + 5 real, all verified -> only 5 count, but capped at 0.65
        claims = ["by step.", "short",
                  "real claim one here", "real claim two here",
                  "real claim three here", "real claim four here",
                  "real claim five here"]
        score = clr._calculate_reliability([1, 1, 1, 1, 1, 1, 1], claims=claims, answer_present=True)
        assert score <= 0.65


class TestIsMeaningfulClaim:
    @pytest.mark.parametrize("claim,expected", [
        ("by step reasoning.", False),
        ("by step.", False),
        ("by step reasoning. So we can elaborate.", False),
        ("step by step.", False),
        ("none", False),
        ("null", False),
        ("n/a", False),
        ("short", False),
        ("ab", False),
        ("...", False),
        ("123", False),
        ("The recurrence relation produces values 2, 3, 7, 43, 1807", True),
        ("We compute a_2 = 2^2 - 2 + 1 = 3", True),
        ("The geometric series converges to 3/2", True),
    ])
    def test_meaningful_claim_filter(self, clr, claim, expected):
        assert clr._is_meaningful_claim(claim) == expected


class TestFailClosedRun:
    """Tests for the fail-closed behavior of VibeThinkerCLRAsync.run().

    A dead model server is infrastructure failure, not a low-confidence answer.
    """

    @pytest.mark.asyncio
    async def test_all_trajectories_transport_fail_raises(self, clr):
        """All trajectories fail with transport exceptions -> RuntimeError."""
        async def boom(*args, **kwargs):
            raise RuntimeError("Connection refused")
        with patch.object(clr, "_generate_one_trajectory", new=AsyncMock(side_effect=boom)):
            with pytest.raises(RuntimeError, match="All CLR trajectories failed"):
                await clr.run("test problem")

    @pytest.mark.asyncio
    async def test_partial_trajectory_failure_still_returns_with_metadata(self):
        """Some trajectories fail, some succeed -> continue with warning metadata."""
        clr = VibeThinkerCLRAsync(server_url="http://localhost:0", k=4)
        good_traj = {
            "score": 1.0,
            "answer": "42",
            "claims": ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20],
            "verdicts": [1, 1, 1, 1, 1],
            "raw_trace": "reasoning \\boxed{42}",
            "answer_present": True,
        }

        call_count = 0
        async def mixed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise RuntimeError("Connection refused")
            return good_traj

        with patch.object(clr, "_generate_one_trajectory", new=AsyncMock(side_effect=mixed)):
            result = await clr.run("test problem")
        assert result.partial_failure is True
        assert result.transport_failures > 0
        assert result.best_answer == "42"

    @pytest.mark.asyncio
    async def test_successful_empty_answer_returns_zero_score_completed(self, clr):
        """Trajectories succeed but none produce a final answer -> score 0, completed."""
        empty_traj = {
            "score": 0.0,
            "answer": None,
            "claims": [],
            "verdicts": [],
            "raw_trace": "reasoning with no boxed answer",
            "answer_present": False,
        }
        with patch.object(clr, "_generate_one_trajectory",
                          new=AsyncMock(return_value=empty_traj)):
            result = await clr.run("test problem")
        assert result.best_score == 0.0
        assert result.best_answer == "No clear answer found"
        assert result.failure_reason is None  # not an infrastructure failure


class TestVerifierIntegration:
    """Tests for the verifier integration in CLR run().

    A deterministic verifier is the ONLY path that allows the final score
    to exceed the self-claims-only cap of 0.65.
    """

    @pytest.mark.asyncio
    async def test_no_verifier_caps_at_065(self, clr):
        """Without a verifier, score is capped at 0.65 even with perfect claims."""
        good_traj = {
            "score": 0.65,
            "answer": "42",
            "claims": ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20],
            "verdicts": [1, 1, 1, 1, 1],
            "raw_trace": "reasoning \\boxed{42}",
            "answer_present": True,
        }
        with patch.object(clr, "_generate_one_trajectory",
                          new=AsyncMock(return_value=good_traj)):
            result = await clr.run("test problem", verifier=None)
        assert result.verification_method == "self_claims_only"
        assert result.verified is False
        assert result.best_score <= 0.65

    @pytest.mark.asyncio
    async def test_math_verifier_allows_above_065(self, clr):
        """With a passing math verifier, score CAN exceed 0.65."""
        from verifiers import MathVerifier
        from verifiers.base import VerificationResult

        good_traj = {
            "score": 0.65,
            "answer": "4",
            "claims": ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20],
            "verdicts": [1, 1, 1, 1, 1],
            "raw_trace": "reasoning \\boxed{4}",
            "answer_present": True,
        }
        # Mock the math verifier to return verified=True
        verifier = MathVerifier()
        async def mock_verify(query, answer, context):
            return VerificationResult(
                verified=True, score=1.0, method="numeric_comparison",
                evidence={"candidate": 4.0, "expected": 4.0},
            )
        verifier.verify = mock_verify

        # Adaptive mode uses lightweight trajectory when verifier is present.
        # Mock both methods to cover both paths.
        with patch.object(clr, "_generate_one_trajectory",
                          new=AsyncMock(return_value=good_traj)):
            with patch.object(clr, "_generate_lightweight_trajectory",
                              new=AsyncMock(return_value=good_traj)):
                result = await clr.run("What is 2+2?", verifier=verifier, task_type="math")
        assert result.verification_method == "math_verifier"
        assert result.verified is True
        assert result.best_score > 0.65

    @pytest.mark.asyncio
    async def test_verifier_refutation_scores_zero(self, clr):
        """If a verifier refutes the answer, score must be 0."""
        from verifiers import MathVerifier
        from verifiers.base import VerificationResult

        good_traj = {
            "score": 0.65,
            "answer": "5",
            "claims": ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20],
            "verdicts": [1, 1, 1, 1, 1],
            "raw_trace": "reasoning \\boxed{5}",
            "answer_present": True,
        }
        verifier = MathVerifier()
        async def mock_verify(query, answer, context):
            return VerificationResult(
                verified=False, score=0.0, method="numeric_comparison",
                evidence={"candidate": 5.0, "expected": 4.0},
                error="5.0 != expected 4.0",
            )
        verifier.verify = mock_verify

        with patch.object(clr, "_generate_one_trajectory",
                          new=AsyncMock(return_value=good_traj)):
            with patch.object(clr, "_generate_lightweight_trajectory",
                              new=AsyncMock(return_value=good_traj)):
                result = await clr.run("What is 2+2?", verifier=verifier, task_type="math")
        assert result.verified is False
        assert result.best_score == 0.0

    @pytest.mark.asyncio
    async def test_verifier_error_falls_back_to_self_claims(self, clr):
        """If a verifier raises an exception, fall back to self-claims-only."""
        from verifiers import MathVerifier

        good_traj = {
            "score": 0.65,
            "answer": "4",
            "claims": ["a" * 20, "b" * 20, "c" * 20, "d" * 20, "e" * 20],
            "verdicts": [1, 1, 1, 1, 1],
            "raw_trace": "reasoning \\boxed{4}",
            "answer_present": True,
        }
        verifier = MathVerifier()
        async def boom_verify(query, answer, context):
            raise RuntimeError("verifier crashed")
        verifier.verify = boom_verify

        with patch.object(clr, "_generate_one_trajectory",
                          new=AsyncMock(return_value=good_traj)):
            with patch.object(clr, "_generate_lightweight_trajectory",
                              new=AsyncMock(return_value=good_traj)):
                result = await clr.run("What is 2+2?", verifier=verifier, task_type="math")
        assert result.verification_method == "self_claims_only"
        assert result.verified is False
        assert result.best_score <= 0.65


class TestGrammarEnforcement:
    """Tests for GBNF grammar enforcement in claim extraction.

    The grammar parameter is passed to llama-server's /completion endpoint
    to constrain the model's output to valid JSON, preventing small models
    from producing malformed JSON that causes trajectory scoring to fail.
    """

    def test_grammar_constant_exists(self):
        from vibe_clr_async import _CLAIMS_JSON_GRAMMAR
        assert "root ::=" in _CLAIMS_JSON_GRAMMAR
        assert "claims" in _CLAIMS_JSON_GRAMMAR
        assert "final_answer" in _CLAIMS_JSON_GRAMMAR

    def test_grammar_requires_claims_array_and_answer(self):
        from vibe_clr_async import _CLAIMS_JSON_GRAMMAR
        # The grammar must enforce both "claims" (array) and "final_answer"
        # The GBNF grammar uses escaped quotes: \"claims\" and \"final_answer\"
        assert "claims" in _CLAIMS_JSON_GRAMMAR
        assert "final_answer" in _CLAIMS_JSON_GRAMMAR
        assert "string" in _CLAIMS_JSON_GRAMMAR  # strings in the array

    @pytest.mark.asyncio
    async def test_call_model_passes_grammar_in_payload(self, clr):
        """When grammar is provided, it's included in the POST payload."""
        import aiohttp
        from unittest.mock import patch, AsyncMock

        captured_payload = {}

        class FakeResp:
            def raise_for_status(self): pass
            async def json(self):
                return {"content": '{"claims": [], "final_answer": null}'}
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass

        class FakeSession:
            def post(self, url, json=None, **kw):
                captured_payload.update(json or {})
                return FakeResp()
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass

        # We need to mock the semaphore context manager
        with patch.object(clr, 'semaphore'):
            result = await clr._call_model(
                FakeSession(), "test prompt", max_tokens=100,
                grammar='root ::= "test"',
            )
        assert captured_payload.get("grammar") == 'root ::= "test"'

    @pytest.mark.asyncio
    async def test_call_model_omits_grammar_when_not_provided(self, clr):
        """When grammar is None, it's NOT included in the POST payload."""
        from unittest.mock import patch

        captured_payload = {}

        class FakeResp:
            def raise_for_status(self): pass
            async def json(self):
                return {"content": "some response"}
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass

        class FakeSession:
            def post(self, url, json=None, **kw):
                captured_payload.update(json or {})
                return FakeResp()
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass

        with patch.object(clr, 'semaphore'):
            await clr._call_model(FakeSession(), "test prompt", max_tokens=100)
        assert "grammar" not in captured_payload


class TestFastSpecialistPolicy:
    """Tests for the --fast-specialist adaptive profile (up to 3/5/15, capped at k).

    The fast-specialist profile is for ultra-tiny models (e.g. 0.5B) where
    shotgun-sampling many trajectories is cheap. It must NOT replace the
    default 1/2/6 policy used for 3B+ specialists on constrained hardware.
    """

    def test_fast_specialist_policy_values(self):
        from vibe_clr_async import make_fast_specialist_policy
        p = make_fast_specialist_policy()
        assert p.initial_k_with_verifier == 3
        assert p.initial_k_without_verifier == 5
        assert p.max_k == 15
        # The self-claim cap is unchanged — a fast model agreeing with itself
        # more often is NOT independent verification.
        assert p.self_claim_cap == 0.65

    def test_fast_specialist_policy_capped_at_k(self):
        """The fast policy respects the user's k setting, like the default."""
        from vibe_clr_async import make_fast_specialist_policy
        p = make_fast_specialist_policy(k=8)
        assert p.initial_k_with_verifier == 3
        assert p.initial_k_without_verifier == 5
        assert p.max_k == 8  # capped at k=8, not 15
        # Very small k
        p4 = make_fast_specialist_policy(k=4)
        assert p4.initial_k_with_verifier == 3
        assert p4.initial_k_without_verifier == 4  # min(5, 4)
        assert p4.max_k == 4  # min(15, 4)

    def test_fast_specialist_flag_constructs_policy(self):
        clr_fast = VibeThinkerCLRAsync(
            server_url="http://localhost:0", k=8, fast_specialist=True
        )
        assert clr_fast.policy.initial_k_with_verifier == 3
        assert clr_fast.policy.initial_k_without_verifier == 5
        assert clr_fast.policy.max_k == 8  # capped at k=8
        assert clr_fast.fast_specialist is True

    def test_fast_specialist_with_large_k_gets_15(self):
        """With k>=15, the fast policy gets the full 15."""
        clr_fast = VibeThinkerCLRAsync(
            server_url="http://localhost:0", k=20, fast_specialist=True
        )
        assert clr_fast.policy.max_k == 15

    def test_default_keeps_standard_policy(self):
        """Without fast_specialist, the default 1/2/6 policy is used."""
        clr_default = VibeThinkerCLRAsync(server_url="http://localhost:0", k=8)
        assert clr_default.policy.initial_k_with_verifier == 1
        assert clr_default.policy.initial_k_without_verifier == 2
        assert clr_default.policy.max_k == 6
        assert clr_default.fast_specialist is False

    def test_explicit_policy_overrides_fast_specialist(self):
        """An explicit policy wins over the fast_specialist flag."""
        from vibe_clr_async import AdaptivePolicy
        custom = AdaptivePolicy(initial_k_with_verifier=2, max_k=10)
        clr = VibeThinkerCLRAsync(
            server_url="http://localhost:0", k=8,
            fast_specialist=True, policy=custom,
        )
        assert clr.policy is custom
        assert clr.policy.max_k == 10

    def test_fast_specialist_queue_load_adjusts_relative_to_cap(self):
        """Queue-load adjustment caps relative to the fast max_k (capped at k)."""
        clr_fast = VibeThinkerCLRAsync(
            server_url="http://localhost:0", k=20, fast_specialist=True
        )
        assert clr_fast._original_max_k == 15
        # High load -> min(2, 15) = 2
        clr_fast.adjust_max_k_for_queue_load(0.9)
        assert clr_fast.policy.max_k == 2
        # Restore at low load
        clr_fast.adjust_max_k_for_queue_load(0.3)
        assert clr_fast.policy.max_k == 15


class TestInProcessBackend:
    """Tests for the in-process specialist backend (llama-cpp-python).

    The in-process backend loads a GGUF directly into the Python process and
    calls it via a thread executor, bypassing HTTP. These tests mock llama_cpp
    so they run without the dependency or a real model.
    """

    def test_default_backend_is_http(self, clr):
        assert clr.backend == "http"
        assert clr._local_llm is None
        assert clr._local_grammar is None

    def test_init_falls_back_when_llama_cpp_missing(self):
        """If llama-cpp-python is not installed, warn and fall back to HTTP."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "llama_cpp" or name.startswith("llama_cpp."):
                raise ImportError("no llama_cpp")
            return real_import(name, *a, **kw)

        with patch("builtins.__import__", side_effect=fake_import):
            clr = VibeThinkerCLRAsync(
                server_url="http://localhost:0", k=1,
                local_model="/tmp/nonexistent.gguf",
            )
        assert clr.backend == "http"
        assert clr._local_llm is None

    def test_init_falls_back_on_load_failure(self):
        """If Llama()/from_pretrained raises, fall back to HTTP, not crash."""
        with patch.dict("sys.modules", {
            "llama_cpp": MagicMock(),
            "llama_cpp.llama_grammar": MagicMock(),
        }):
            import sys
            llama_mod = sys.modules["llama_cpp"]
            llama_mod.Llama = MagicMock(side_effect=RuntimeError("oom"))
            llama_mod.Llama.from_pretrained = MagicMock(side_effect=RuntimeError("oom"))
            llama_mod.LlamaGrammar = MagicMock()
            clr = VibeThinkerCLRAsync(
                server_url="http://localhost:0", k=1,
                local_model="/tmp/nonexistent.gguf",
            )
        assert clr.backend == "http"
        assert clr._local_llm is None

    def test_init_loads_inprocess_from_path(self, tmp_path):
        """A .gguf path that exists on disk loads via Llama(model_path=...)."""
        gguf = tmp_path / "tiny.gguf"
        gguf.write_bytes(b"fake")
        with patch.dict("sys.modules", {
            "llama_cpp": MagicMock(),
            "llama_cpp.llama_grammar": MagicMock(),
        }):
            import sys
            llama_mod = sys.modules["llama_cpp"]
            fake_llm = MagicMock(name="loaded_llm")
            llama_mod.Llama = MagicMock(return_value=fake_llm)
            grammar_cls = MagicMock()
            llama_mod.LlamaGrammar = grammar_cls
            clr = VibeThinkerCLRAsync(
                server_url="http://localhost:0", k=1,
                local_model=str(gguf),
                local_n_ctx=2048, local_n_threads=4,
            )
        assert clr.backend == "in-process"
        assert clr._local_llm is fake_llm
        llama_mod.Llama.assert_called_once()
        _, kwargs = llama_mod.Llama.call_args
        assert kwargs["model_path"] == str(gguf)
        assert kwargs["n_ctx"] == 2048
        assert kwargs["n_threads"] == 4
        # Grammar pre-compiled once at init
        grammar_cls.from_string.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_model_uses_inprocess_when_local_llm_set(self, clr):
        """When _local_llm is set, _call_model bypasses HTTP (ignores session)."""
        clr._local_llm = MagicMock()
        clr._local_llm.return_value = {"choices": [{"text": "in-process reply"}]}
        clr.backend = "in-process"
        # A session that would raise if used — proves the HTTP path is skipped
        bomb_session = MagicMock()
        bomb_session.post = MagicMock(side_effect=AssertionError("HTTP used!"))

        result = await clr._call_model(bomb_session, "prompt", max_tokens=50)
        assert result == "in-process reply"
        clr._local_llm.assert_called_once()
        _, kwargs = clr._local_llm.call_args
        assert kwargs["max_tokens"] == 50
        assert kwargs["stop"] == ["<|im_end|>"]
        # No grammar requested -> grammar_obj is None
        assert kwargs["grammar"] is None

    @pytest.mark.asyncio
    async def test_call_model_inprocess_reuses_claims_grammar(self, clr):
        """The pre-compiled claims grammar is reused, not recompiled."""
        from vibe_clr_async import _CLAIMS_JSON_GRAMMAR
        clr._local_llm = MagicMock()
        clr._local_llm.return_value = {"choices": [{"text": '{"claims":[]}'}]}
        clr._local_grammar = MagicMock(name="precompiled_grammar")
        clr.backend = "in-process"

        await clr._call_model(
            MagicMock(), "prompt", max_tokens=10,
            grammar=_CLAIMS_JSON_GRAMMAR,
        )
        _, kwargs = clr._local_llm.call_args
        # The pre-compiled object is passed directly, not the raw string
        assert kwargs["grammar"] is clr._local_grammar

    @pytest.mark.asyncio
    async def test_call_model_inprocess_raises_on_empty(self, clr):
        """Empty model output raises RuntimeError (fail-closed, not silent)."""
        clr._local_llm = MagicMock()
        clr._local_llm.return_value = {"choices": [{"text": ""}]}
        clr.backend = "in-process"
        with pytest.raises(RuntimeError, match="empty content"):
            await clr._call_model(MagicMock(), "prompt", max_tokens=10)

    @pytest.mark.asyncio
    async def test_call_model_inprocess_raises_on_empty_choices_list(self, clr):
        """An empty choices list raises RuntimeError, not IndexError."""
        clr._local_llm = MagicMock()
        clr._local_llm.return_value = {"choices": []}
        clr.backend = "in-process"
        with pytest.raises(RuntimeError, match="empty content"):
            await clr._call_model(MagicMock(), "prompt", max_tokens=10)

    @pytest.mark.asyncio
    async def test_call_model_inprocess_raises_on_none_response(self, clr):
        """A None response raises RuntimeError, not AttributeError."""
        clr._local_llm = MagicMock(return_value=None)
        clr.backend = "in-process"
        with pytest.raises(RuntimeError, match="empty content"):
            await clr._call_model(MagicMock(), "prompt", max_tokens=10)

    @pytest.mark.asyncio
    async def test_call_model_inprocess_wraps_unexpected_errors(self, clr):
        """Non-RuntimeError exceptions from the LLM are wrapped as RuntimeError."""
        clr._local_llm = MagicMock(side_effect=ValueError("bad state"))
        clr.backend = "in-process"
        with pytest.raises(RuntimeError, match="In-process specialist call failed"):
            await clr._call_model(MagicMock(), "prompt", max_tokens=10)

    @pytest.mark.asyncio
    async def test_call_model_inprocess_serializes_with_lock(self, clr):
        """The threading.Lock is held during inference (single Llama instance
        is not safe to call concurrently)."""
        clr._local_llm = MagicMock()
        clr._local_llm.return_value = {"choices": [{"text": "ok"}]}
        clr.backend = "in-process"
        with patch.object(clr, "_local_lock") as lock:
            lock.__enter__ = MagicMock(return_value=None)
            lock.__exit__ = MagicMock(return_value=False)
            await clr._call_model(MagicMock(), "prompt", max_tokens=10)
            lock.__enter__.assert_called_once()
            lock.__exit__.assert_called_once()


class TestInProcessPool:
    """Tests for the in-process Llama instance pool (true parallelism).

    When local_pool_size > 1, N separate Llama instances are loaded into a
    queue.Queue. Each call checks out one instance, runs it in a thread
    executor, and returns it to the pool. This enables true parallel inference
    (a single Llama instance is not thread-safe).
    """

    def test_pool_size_defaults_to_1(self, clr):
        """Default is single-instance mode (no pool)."""
        assert clr._local_pool_size == 1
        assert clr._local_llm_pool is None

    def test_init_pool_mode_loads_n_instances(self, tmp_path):
        """pool_size > 1 loads N instances into a queue.Queue."""
        import queue as queue_mod
        gguf = tmp_path / "tiny.gguf"
        gguf.write_bytes(b"fake")
        with patch.dict("sys.modules", {
            "llama_cpp": MagicMock(),
            "llama_cpp.llama_grammar": MagicMock(),
        }):
            import sys
            llama_mod = sys.modules["llama_cpp"]
            # Each call to Llama() returns a distinct mock instance
            instances = [MagicMock(name=f"llm_{i}") for i in range(4)]
            llama_mod.Llama = MagicMock(side_effect=instances)
            llama_mod.LlamaGrammar = MagicMock()

            clr = VibeThinkerCLRAsync(
                server_url="http://localhost:0", k=8,
                local_model=str(gguf),
                local_pool_size=4,
                local_n_threads=8,
            )
        assert clr.backend == "in-process-pool"
        assert clr._local_llm is None  # single-instance not used in pool mode
        assert clr._local_llm_pool is not None
        assert clr._local_llm_pool.qsize() == 4
        assert clr._local_pool_size == 4
        # Llama() was called 4 times (once per instance)
        assert llama_mod.Llama.call_count == 4

    def test_pool_partial_load_failure_keeps_loaded_instances(self, tmp_path):
        """If some instances fail to load, the pool keeps the ones that worked."""
        gguf = tmp_path / "tiny.gguf"
        gguf.write_bytes(b"fake")
        with patch.dict("sys.modules", {
            "llama_cpp": MagicMock(),
            "llama_cpp.llama_grammar": MagicMock(),
        }):
            import sys
            llama_mod = sys.modules["llama_cpp"]
            # First 2 succeed, 3rd fails, 4th succeeds
            good = MagicMock(name="good")
            llama_mod.Llama = MagicMock(side_effect=[
                good, good, RuntimeError("oom"), good,
            ])
            llama_mod.LlamaGrammar = MagicMock()

            clr = VibeThinkerCLRAsync(
                server_url="http://localhost:0", k=8,
                local_model=str(gguf),
                local_pool_size=4,
            )
        assert clr.backend == "in-process-pool"
        assert clr._local_pool_size == 3  # only 3 loaded successfully
        assert clr._local_llm_pool.qsize() == 3

    def test_pool_all_load_failures_fall_back_to_http(self, tmp_path):
        """If ALL pool instances fail to load, fall back to HTTP."""
        gguf = tmp_path / "tiny.gguf"
        gguf.write_bytes(b"fake")
        with patch.dict("sys.modules", {
            "llama_cpp": MagicMock(),
            "llama_cpp.llama_grammar": MagicMock(),
        }):
            import sys
            llama_mod = sys.modules["llama_cpp"]
            llama_mod.Llama = MagicMock(side_effect=RuntimeError("oom"))
            llama_mod.LlamaGrammar = MagicMock()

            clr = VibeThinkerCLRAsync(
                server_url="http://localhost:0", k=8,
                local_model=str(gguf),
                local_pool_size=3,
            )
        assert clr.backend == "http"
        assert clr._local_llm_pool is None
        assert clr._local_llm is None

    @pytest.mark.asyncio
    async def test_pool_call_checks_out_and_returns_instance(self, clr):
        """Pool mode: _call_model checks out an instance and returns it."""
        import queue as queue_mod
        fake_llm = MagicMock(name="pool_llm")
        fake_llm.return_value = {"choices": [{"text": "pool reply"}]}
        clr._local_llm_pool = queue_mod.Queue()
        clr._local_llm_pool.put(fake_llm)
        clr._local_pool_size = 1
        clr._local_llm = None
        clr.backend = "in-process-pool"

        result = await clr._call_model(MagicMock(), "prompt", max_tokens=50)
        assert result == "pool reply"
        fake_llm.assert_called_once()
        # The instance was returned to the pool
        assert clr._local_llm_pool.qsize() == 1

    @pytest.mark.asyncio
    async def test_pool_concurrent_calls_use_different_instances(self, clr):
        """Two concurrent calls check out two different instances from the pool."""
        import asyncio as aio
        import queue as queue_mod
        llm1 = MagicMock(name="llm1")
        llm1.return_value = {"choices": [{"text": "reply1"}]}
        llm2 = MagicMock(name="llm2")
        llm2.return_value = {"choices": [{"text": "reply2"}]}
        clr._local_llm_pool = queue_mod.Queue()
        clr._local_llm_pool.put(llm1)
        clr._local_llm_pool.put(llm2)
        clr._local_pool_size = 2
        clr._local_llm = None
        clr.backend = "in-process-pool"

        # Fire both concurrently — they should each get a different instance
        results = await aio.gather(
            clr._call_model(MagicMock(), "p1", max_tokens=10),
            clr._call_model(MagicMock(), "p2", max_tokens=10),
        )
        assert set(results) == {"reply1", "reply2"}
        llm1.assert_called_once()
        llm2.assert_called_once()
        # Both returned to pool
        assert clr._local_llm_pool.qsize() == 2

    @pytest.mark.asyncio
    async def test_pool_empty_choices_raises_runtime_error(self, clr):
        """Pool mode: empty choices list raises RuntimeError, not IndexError."""
        import queue as queue_mod
        fake_llm = MagicMock(return_value={"choices": []})
        clr._local_llm_pool = queue_mod.Queue()
        clr._local_llm_pool.put(fake_llm)
        clr.backend = "in-process-pool"
        with pytest.raises(RuntimeError, match="empty content"):
            await clr._call_model(MagicMock(), "prompt", max_tokens=10)


class TestParallelVerifyClaims:
    """Tests for parallel claim verification (asyncio.gather).

    _verify_claims was previously sequential (for claim in claims: await).
    Now it uses asyncio.gather to verify all claims concurrently.
    """

    @pytest.mark.asyncio
    async def test_verify_claims_runs_concurrently(self, clr):
        """All claims are verified in parallel, not sequentially."""
        import asyncio as aio
        import time

        call_times = []

        async def slow_call_model(session, prompt, **kwargs):
            call_times.append(time.monotonic())
            await aio.sleep(0.05)  # Simulate 50ms LLM call
            return "1"  # Verdict: correct

        clr._call_model = slow_call_model
        clr.backend = "http"

        claims = ["claim 1 is true", "claim 2 is true", "claim 3 is true"]
        start = time.monotonic()
        verdicts = await clr._verify_claims(MagicMock(), claims)
        elapsed = time.monotonic() - start

        assert verdicts == [1, 1, 1]
        # If sequential: 3 x 50ms = 150ms. With gather: ~50ms.
        assert elapsed < 0.12, f"Took {elapsed:.3f}s — expected parallel (<0.12s)"

    @pytest.mark.asyncio
    async def test_verify_claims_short_claim_returns_zero(self, clr):
        """Claims shorter than 5 chars get verdict 0 without an LLM call."""
        clr._call_model = AsyncMock(return_value="1")
        clr.backend = "http"

        verdicts = await clr._verify_claims(MagicMock(), ["ok", "", "  "])
        assert verdicts == [0, 0, 0]
        clr._call_model.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_verify_claims_one_failure_doesnt_kill_all(self, clr):
        """If one claim verification raises, it returns 0 (not crash)."""
        call_count = 0

        async def flaky_call_model(session, prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("LLM crashed")
            return "1"

        clr._call_model = flaky_call_model
        clr.backend = "http"

        claims = ["claim 1 is true", "claim 2 is true", "claim 3 is true"]
        verdicts = await clr._verify_claims(MagicMock(), claims)
        # Second claim failed → verdict 0, but others still got verified
        assert verdicts == [1, 0, 1]


class TestInProcessSessionSkip:
    """Tests for skipping aiohttp session creation in in-process mode."""

    @pytest.mark.asyncio
    async def test_run_static_skips_session_in_inprocess_mode(self, clr):
        """In-process mode: _run_static uses nullcontext, not aiohttp session."""
        clr.backend = "in-process"
        clr._local_llm = MagicMock(return_value={"choices": [{"text": "answer"}]})
        clr._local_grammar = None
        clr.k = 2

        # Mock _generate_one_trajectory to verify session is None
        received_sessions = []

        async def mock_gen(session, problem, max_tokens):
            received_sessions.append(session)
            return {"answer": "42", "claims": [], "verdicts": [], "raw": "",
                    "raw_trace": ""}

        clr._generate_one_trajectory = mock_gen
        clr._score_trajectories = MagicMock(return_value=[
            {"answer": "42", "score": 0.9, "claims": [], "verdicts": [],
             "raw": "", "raw_trace": ""}
        ])

        await clr._run_static("test problem", 512, None, "math", None)
        # In in-process mode, session should be None (nullcontext)
        assert all(s is None for s in received_sessions)


