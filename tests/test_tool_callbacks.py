"""Tests for the inter-turn tool-callback scaffold (v3.2 Step 4.2)."""

import pytest
from unittest.mock import AsyncMock

from tool_callbacks import (
    CallbackRegistry,
    ToolCallRequest,
    parse_tool_calls,
    strip_tool_call_markers,
    execute_tool_call,
    run_with_callbacks,
)


class TestParseToolCalls:
    def test_finds_single_call(self):
        text = 'Some reasoning.\n<tool_call name="rag_lookup">capital of France</tool_call>\nMore.'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "rag_lookup"
        assert calls[0].query == "capital of France"

    def test_finds_multiple_calls_in_order(self):
        text = ('<tool_call name="a">q1</tool_call>'
                '<tool_call name="b">q2</tool_call>')
        calls = parse_tool_calls(text)
        assert len(calls) == 2
        assert calls[0].name == "a"
        assert calls[1].name == "b"

    def test_no_calls_returns_empty(self):
        assert parse_tool_calls("plain text, no markers") == []

    def test_multiline_query(self):
        text = ('<tool_call name="rag">\n'
                'a multi-line\nquery\n</tool_call>')
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert "multi-line" in calls[0].query

    def test_strip_markers_removes_all(self):
        text = ('before<tool_call name="x">q</tool_call>after'
                '<tool_call name="y">q2</tool_call>end')
        stripped = strip_tool_call_markers(text)
        assert "tool_call" not in stripped
        assert "before" in stripped
        assert "after" in stripped
        assert "end" in stripped


class TestExecuteToolCall:
    @pytest.mark.asyncio
    async def test_known_tool_executes(self):
        reg = CallbackRegistry()
        reg.register("echo", AsyncMock(return_value="echoed"))
        req = ToolCallRequest(name="echo", query="hi", start=0, end=0)
        result = await execute_tool_call(reg, req)
        assert result == "echoed"

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_note(self):
        reg = CallbackRegistry()
        req = ToolCallRequest(name="nope", query="hi", start=0, end=0)
        result = await execute_tool_call(reg, req)
        assert "unknown tool" in result
        assert "nope" in result

    @pytest.mark.asyncio
    async def test_tool_raising_feeds_exception_back(self):
        """A tool that raises must NOT crash the loop — the exception
        text is fed back so the specialist can adapt."""
        reg = CallbackRegistry()

        async def bad_tool(q):
            raise ValueError("kaboom")

        reg.register("bad", bad_tool)
        req = ToolCallRequest(name="bad", query="hi", start=0, end=0)
        result = await execute_tool_call(reg, req)
        assert "tool error" in result
        assert "ValueError" in result
        assert "kaboom" in result


class TestRunWithCallbacks:
    @pytest.mark.asyncio
    async def test_no_registry_is_single_turn(self):
        """Without a registry, the loop runs once (backward-compat)."""
        gen = AsyncMock(return_value="just an answer")
        result = await run_with_callbacks(gen, "prompt", registry=None)
        assert result == "just an answer"
        assert gen.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_registry_is_single_turn(self):
        gen = AsyncMock(return_value="just an answer")
        reg = CallbackRegistry()  # empty
        result = await run_with_callbacks(gen, "prompt", registry=reg)
        assert result == "just an answer"
        assert gen.call_count == 1

    @pytest.mark.asyncio
    async def test_tool_call_then_continue(self):
        """Specialist requests a tool, gets the result, continues, done."""
        reg = CallbackRegistry()
        reg.register("rag", AsyncMock(return_value="Paris"))

        # Round 1: emits a tool call. Round 2: final answer (no call).
        gen = AsyncMock(side_effect=[
            'I need a fact.\n<tool_call name="rag">capital of France</tool_call>',
            "The capital is Paris. The answer is \\boxed{Paris}.",
        ])
        result = await run_with_callbacks(gen, "What is the capital of France?", registry=reg)
        assert "Paris" in result
        assert "tool_call" not in result  # markers stripped from final
        assert gen.call_count == 2

    @pytest.mark.asyncio
    async def test_max_rounds_cuts_off_infinite_loop(self):
        """A specialist stuck requesting tools every turn is cut off
        after max_rounds. The last generation is returned (markers stripped)."""
        reg = CallbackRegistry()
        reg.register("rag", AsyncMock(return_value="result"))

        # Always emits a tool call -> infinite loop -> cut off at max_rounds.
        gen = AsyncMock(return_value='<tool_call name="rag">q</tool_call>')
        result = await run_with_callbacks(gen, "p", registry=reg, max_rounds=2)
        assert gen.call_count == 2  # exactly max_rounds, not infinite
        assert "tool_call" not in result  # markers stripped

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_one_turn(self):
        """A single turn can request multiple tools; all are executed
        and fed back together."""
        reg = CallbackRegistry()
        reg.register("a", AsyncMock(return_value="A-result"))
        reg.register("b", AsyncMock(return_value="B-result"))

        gen = AsyncMock(side_effect=[
            '<tool_call name="a">q1</tool_call>'
            '<tool_call name="b">q2</tool_call>',
            "Final answer using A-result and B-result.",
        ])
        result = await run_with_callbacks(gen, "p", registry=reg)
        assert "Final answer" in result
        # Both tools were called.
        reg.tools["a"].assert_awaited_with("q1")
        reg.tools["b"].assert_awaited_with("q2")

    @pytest.mark.asyncio
    async def test_tool_results_visible_to_next_turn(self):
        """The tool result must appear in the prompt for round 2 so the
        model can use it. Verify by capturing the round-2 prompt."""
        reg = CallbackRegistry()
        reg.register("rag", AsyncMock(return_value="42"))

        captured_prompts = []

        async def capturing_gen(prompt):
            captured_prompts.append(prompt)
            if len(captured_prompts) == 1:
                return '<tool_call name="rag">meaning of life</tool_call>'
            return "The answer is 42."

        result = await run_with_callbacks(capturing_gen, "What is the meaning of life?", registry=reg)
        assert "42" in result
        # The round-2 prompt must contain the tool result.
        assert "42" in captured_prompts[1]
        assert "tool_result" in captured_prompts[1]
