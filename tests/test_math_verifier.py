"""Pytest tests for the math verifier."""

import pytest

from verifiers.math_verifier import MathVerifier


@pytest.fixture
def verifier():
    return MathVerifier()


class TestMathVerifier:
    @pytest.mark.asyncio
    async def test_accepts_equivalent_fraction(self, verifier):
        result = await verifier.verify(
            "What is 1/2 + 1/2?",
            "The answer is \\boxed{1/1}",
            context={"expected_answer": "1"},
        )
        assert result.verified is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_accepts_decimal_tolerance(self, verifier):
        result = await verifier.verify(
            "What is 1/3?",
            "The answer is 0.333333",
            context={"expected_answer": "1/3", "tolerance": 1e-4},
        )
        assert result.verified is True

    @pytest.mark.asyncio
    async def test_rejects_wrong_sum(self, verifier):
        result = await verifier.verify(
            "What is 2+2?",
            "The answer is 5",
            context={"expected_answer": "4"},
        )
        assert result.verified is False
        assert "5.0 != expected 4.0" in result.error

    @pytest.mark.asyncio
    async def test_returns_unverified_when_unparseable(self, verifier):
        result = await verifier.verify(
            "Explain quantum mechanics",
            "It is a complex theory about subatomic particles.",
            context={"expected_answer": "42"},
        )
        assert result.verified is False
        assert "could not extract" in result.error

    @pytest.mark.asyncio
    async def test_returns_unverified_when_no_expected(self, verifier):
        result = await verifier.verify(
            "What is 2+2?",
            "The answer is 4",
            context={},
        )
        assert result.verified is False
        assert "no expected_answer" in result.error

    @pytest.mark.asyncio
    async def test_handles_boxed_fraction(self, verifier):
        result = await verifier.verify(
            "What is 7/2?",
            "\\boxed{\\frac{7}{2}}",
            context={"expected_answer": "3.5"},
        )
        assert result.verified is True

    @pytest.mark.asyncio
    async def test_handles_negative_numbers(self, verifier):
        result = await verifier.verify(
            "What is 3-5?",
            "The answer is -2",
            context={"expected_answer": "-2"},
        )
        assert result.verified is True

    @pytest.mark.asyncio
    async def test_handles_comma_formatted_numbers(self, verifier):
        result = await verifier.verify(
            "What is 1000*1000?",
            "The answer is 1,000,000",
            context={"expected_answer": "1000000"},
        )
        assert result.verified is True

    @pytest.mark.asyncio
    async def test_geometric_series(self, verifier):
        result = await verifier.verify(
            "Sum of 1 + 1/2 + 1/4 + ...",
            "The sum is 2",
            context={"expected_answer": "2"},
        )
        assert result.verified is True

    @pytest.mark.asyncio
    async def test_recurrence_output(self, verifier):
        result = await verifier.verify(
            "Find a_5 where a_1=2, a_{n+1}=a_n^2-a_n+1",
            "The answer is \\boxed{1807}",
            context={"expected_answer": "1807"},
        )
        assert result.verified is True

    # --- Space-variant \boxed{} regex tests (v0.4.1 fix) ---

    @pytest.mark.asyncio
    async def test_boxed_with_space_before_brace(self, verifier):
        """\\boxed {42} (space before brace) should be extracted correctly."""
        result = await verifier.verify(
            "What is 6*7?",
            "The answer is \\boxed {42}",
            context={"expected_answer": "42"},
        )
        assert result.verified is True

    @pytest.mark.asyncio
    async def test_boxed_with_multiple_spaces_before_brace(self, verifier):
        """\\boxed  {42} (multiple spaces) should also work."""
        result = await verifier.verify(
            "What is 6*7?",
            "The answer is \\boxed  {42}",
            context={"expected_answer": "42"},
        )
        assert result.verified is True

    @pytest.mark.asyncio
    async def test_boxed_with_space_and_nested_braces(self, verifier):
        """\\boxed {\\frac{7}{2}} (space + nested braces) should work."""
        result = await verifier.verify(
            "What is 7/2?",
            "The answer is \\boxed {\\frac{7}{2}}",
            context={"expected_answer": "3.5"},
        )
        assert result.verified is True

    def test_extract_boxed_no_space(self):
        """Direct test of _extract_boxed with no space."""
        from verifiers.math_verifier import MathVerifier
        assert MathVerifier._extract_boxed("Answer: \\boxed{42}") == "42"

    def test_extract_boxed_single_space(self):
        """Direct test of _extract_boxed with single space."""
        from verifiers.math_verifier import MathVerifier
        assert MathVerifier._extract_boxed("Answer: \\boxed {42}") == "42"

    def test_extract_boxed_multiple_spaces(self):
        """Direct test of _extract_boxed with multiple spaces."""
        from verifiers.math_verifier import MathVerifier
        assert MathVerifier._extract_boxed("Answer: \\boxed   {42}") == "42"

    def test_extract_boxed_nested_braces_with_space(self):
        """Direct test of _extract_boxed with nested braces and space."""
        from verifiers.math_verifier import MathVerifier
        result = MathVerifier._extract_boxed("\\boxed {\\frac{7}{2}}")
        assert result == "\\frac{7}{2}"

    def test_extract_boxed_no_boxed_returns_none(self):
        """No \\boxed{} in text returns None."""
        from verifiers.math_verifier import MathVerifier
        assert MathVerifier._extract_boxed("The answer is 42") is None

    def test_extract_boxed_takes_last_match(self):
        """Multiple \\boxed{} — the last one is returned."""
        from verifiers.math_verifier import MathVerifier
        text = "First \\boxed{1} then \\boxed{2}"
        assert MathVerifier._extract_boxed(text) == "2"
