"""Tests for the FormatEnforcer abstraction and chat transports (v1.1).

Covers:
  - Each enforcer renders the contract to its native format correctly.
  - LlamaCppEnforcer GBNF is byte-identical to the historical grammars.
  - The shared parser handles valid/invalid/markdown-wrapped/claims JSON.
  - The repair loop recovers malformed JSON and respects the cap.
  - The OpenAI and Anthropic chat transports apply native enforcement
    (response_format / tool_choice) and parse the response.
  - ChatML stripping and JSON-instruction augmentation helpers.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from format_enforcer import (
    CLAIMS_JSON_GRAMMAR,
    STRUCTURED_OUTPUT_GRAMMAR,
    AnthropicEnforcer,
    FormatEnforcer,
    LlamaCppEnforcer,
    OpenAIEnforcer,
    SchemaKind,
    make_enforcer,
    parse_error_detail,
    parse_structured_output,
)
from vibe_clr_async import VibeThinkerCLRAsync, _CLAIMS_JSON_GRAMMAR, _STRUCTURED_OUTPUT_GRAMMAR


# --------------------------------------------------------------------------- #
# Enforcer rendering
# --------------------------------------------------------------------------- #

class TestEnforcerRendering:
    def test_llama_cpp_grammar_byte_identical_to_historical(self):
        """The GBNF strings must be the exact same objects imported back into
        vibe_clr_async — identity is used for `grammar == _CLAIMS_JSON_GRAMMAR`."""
        assert CLAIMS_JSON_GRAMMAR is _CLAIMS_JSON_GRAMMAR
        assert STRUCTURED_OUTPUT_GRAMMAR is _STRUCTURED_OUTPUT_GRAMMAR

    def test_llama_cpp_enforcer_returns_canonical_grammars(self):
        e_struct = make_enforcer(SchemaKind.STRUCTURED_OUTPUT, "completion")
        e_claims = make_enforcer(SchemaKind.CLAIMS, "completion")
        assert isinstance(e_struct, LlamaCppEnforcer)
        assert e_struct.to_llama_cpp_grammar() == STRUCTURED_OUTPUT_GRAMMAR
        assert e_claims.to_llama_cpp_grammar() == CLAIMS_JSON_GRAMMAR

    def test_openai_enforcer_response_format_strict(self):
        e = make_enforcer(SchemaKind.STRUCTURED_OUTPUT, "openai_chat")
        assert isinstance(e, OpenAIEnforcer)
        rf = e.to_openai_response_format()
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["strict"] is True
        schema = rf["json_schema"]["schema"]
        assert schema["type"] == "object"
        assert "reasoning_steps" in schema["properties"]
        assert "boxed_answer" in schema["properties"]
        assert "code_solution" in schema["properties"]
        assert set(schema["required"]) == {"reasoning_steps", "boxed_answer", "code_solution"}
        assert schema["additionalProperties"] is False

    def test_openai_enforcer_claims_schema(self):
        e = make_enforcer(SchemaKind.CLAIMS, "openai_chat")
        rf = e.to_openai_response_format()
        schema = rf["json_schema"]["schema"]
        assert set(schema["required"]) == {"claims", "final_answer"}
        assert schema["properties"]["claims"]["type"] == "array"

    def test_openai_enforcer_non_strict_falls_back_to_json_object(self):
        e = OpenAIEnforcer.__new__(OpenAIEnforcer)
        # Build with strict=False via factory path
        from format_enforcer import _structured_output_schema
        e.__init__(_structured_output_schema(), strict=False)
        assert e.to_openai_response_format() == {"type": "json_object"}

    def test_anthropic_enforcer_tool_definition(self):
        e = make_enforcer(SchemaKind.STRUCTURED_OUTPUT, "anthropic")
        assert isinstance(e, AnthropicEnforcer)
        tool = e.to_anthropic_tool()
        assert tool["name"] == "structured_output"
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"
        assert "reasoning_steps" in tool["input_schema"]["properties"]

    def test_enforcer_satisfies_protocol(self):
        """Each enforcer should satisfy the FormatEnforcer runtime Protocol."""
        for kind in (SchemaKind.STRUCTURED_OUTPUT, SchemaKind.CLAIMS):
            for transport in ("completion", "openai_chat", "anthropic"):
                e = make_enforcer(kind, transport)
                assert isinstance(e, FormatEnforcer), (
                    f"{type(e).__name__} for {kind}/{transport} does not satisfy "
                    "FormatEnforcer protocol"
                )


# --------------------------------------------------------------------------- #
# Shared parser
# --------------------------------------------------------------------------- #

class TestSharedParser:
    def test_parse_valid_structured(self):
        out = json.dumps({
            "reasoning_steps": ["s1", "s2"],
            "boxed_answer": "42",
            "code_solution": None,
        })
        result = parse_structured_output(out)
        assert result is not None
        assert result["reasoning_steps"] == ["s1", "s2"]
        assert result["boxed_answer"] == "42"
        assert result["code_solution"] is None

    def test_parse_valid_claims(self):
        out = json.dumps({"claims": ["c1", "c2"], "final_answer": "yes"})
        result = parse_structured_output(out)
        assert result is not None
        assert result["claims"] == ["c1", "c2"]
        assert result["final_answer"] == "yes"

    def test_parse_nullish_string_normalized(self):
        out = json.dumps({
            "reasoning_steps": [],
            "boxed_answer": "null",
            "code_solution": "none",
        })
        result = parse_structured_output(out)
        assert result["boxed_answer"] is None
        assert result["code_solution"] is None

    def test_parse_markdown_fences(self):
        out = '```json\n{"reasoning_steps": ["s"], "boxed_answer": "5", "code_solution": null}\n```'
        result = parse_structured_output(out)
        assert result is not None
        assert result["boxed_answer"] == "5"

    def test_parse_invalid_returns_none(self):
        assert parse_structured_output("not json") is None
        assert parse_structured_output("{broken") is None
        assert parse_structured_output("") is None
        assert parse_structured_output('["not", "an", "object"]') is None
        assert parse_structured_output('{"unrelated": "keys"}') is None

    def test_delegated_method_matches_shared(self):
        """VibeThinkerCLRAsync.parse_structured_output delegates to the shared
        parser — both must return identical results."""
        out = json.dumps({
            "reasoning_steps": ["s"],
            "boxed_answer": "42",
            "code_solution": None,
        })
        assert VibeThinkerCLRAsync.parse_structured_output(out) == parse_structured_output(out)
        assert VibeThinkerCLRAsync.parse_structured_output("garbage") is None


# --------------------------------------------------------------------------- #
# Repair prompt + error detail
# --------------------------------------------------------------------------- #

class TestRepairPrompt:
    def test_error_detail_for_empty(self):
        assert "empty" in (parse_error_detail("") or "")

    def test_error_detail_for_no_brace(self):
        assert "no JSON object" in (parse_error_detail("just text") or "")

    def test_error_detail_for_invalid_json(self):
        detail = parse_error_detail('{"reasoning_steps": [broken}')
        assert detail is not None
        assert "Invalid JSON" in detail

    def test_error_detail_for_missing_keys(self):
        detail = parse_error_detail('{"unrelated": "x"}')
        assert detail is not None
        assert "required keys" in detail

    def test_repair_prompt_includes_error_and_schema(self):
        e = make_enforcer(SchemaKind.STRUCTURED_OUTPUT, "openai_chat")
        prompt = e.repair_prompt('{"reasoning_steps": [broken}', "Invalid JSON: boom")
        assert "Invalid JSON: boom" in prompt
        assert "reasoning_steps" in prompt
        assert "ONLY" in prompt

    def test_repair_prompt_truncates_long_bad_text(self):
        e = make_enforcer(SchemaKind.CLAIMS, "openai_chat")
        long_bad = "x" * 5000
        prompt = e.repair_prompt(long_bad)
        assert "[truncated]" in prompt
        assert len(prompt) < len(long_bad) + 1000


# --------------------------------------------------------------------------- #
# Repair loop (via _call_model_with_repair)
# --------------------------------------------------------------------------- #

class TestParseRepairLoop:
    @pytest.mark.asyncio
    async def test_repair_recovers_malformed_json(self):
        """First call returns malformed JSON; the repair call returns valid
        JSON. The loop should return the repaired text + parsed dict."""
        clr = VibeThinkerCLRAsync(
            server_url="http://localhost:0", k=1,
            use_structured_output=True, max_parse_repairs=2,
            specialist_transport="openai_chat",
        )
        valid = json.dumps({
            "reasoning_steps": ["s1"],
            "boxed_answer": "42",
            "code_solution": None,
        })
        call_count = {"n": 0}

        async def fake_call(session, prompt, max_tokens=8192, temperature=1.0,
                            stop=None, grammar=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return '{"reasoning_steps": [broken'
            return valid

        with patch.object(clr, "_call_model", side_effect=fake_call):
            raw, parsed = await clr._call_model_with_repair(
                None, "prompt", 100, 1.0, _STRUCTURED_OUTPUT_GRAMMAR,
            )
        assert call_count["n"] == 2  # initial + 1 repair
        assert parsed is not None
        assert parsed["boxed_answer"] == "42"

    @pytest.mark.asyncio
    async def test_repair_respects_cap(self):
        """When all repair attempts fail, returns (raw, None) — fail-closed."""
        clr = VibeThinkerCLRAsync(
            server_url="http://localhost:0", k=1,
            use_structured_output=True, max_parse_repairs=2,
        )

        async def always_bad(session, prompt, max_tokens=8192, temperature=1.0,
                             stop=None, grammar=None):
            return "not json at all"

        with patch.object(clr, "_call_model", side_effect=always_bad):
            raw, parsed = await clr._call_model_with_repair(
                None, "prompt", 100, 1.0, _STRUCTURED_OUTPUT_GRAMMAR,
            )
        assert parsed is None
        assert raw == "not json at all"

    @pytest.mark.asyncio
    async def test_repair_disabled_when_max_zero(self):
        """max_parse_repairs=0 means no repair — single call, caller parses."""
        clr = VibeThinkerCLRAsync(
            server_url="http://localhost:0", k=1,
            use_structured_output=True, max_parse_repairs=0,
        )
        call_count = {"n": 0}

        async def fake_call(session, prompt, max_tokens=8192, temperature=1.0,
                            stop=None, grammar=None):
            call_count["n"] += 1
            return "not json"

        with patch.object(clr, "_call_model", side_effect=fake_call):
            raw, parsed = await clr._call_model_with_repair(
                None, "prompt", 100, 1.0, _STRUCTURED_OUTPUT_GRAMMAR,
            )
        assert call_count["n"] == 1
        assert parsed is None

    @pytest.mark.asyncio
    async def test_repair_skipped_when_grammar_none(self):
        """Unstructured generation (grammar=None) skips repair entirely."""
        clr = VibeThinkerCLRAsync(
            server_url="http://localhost:0", k=1, max_parse_repairs=5,
        )

        async def fake_call(session, prompt, max_tokens=8192, temperature=1.0,
                            stop=None, grammar=None):
            return "freeform text"

        with patch.object(clr, "_call_model", side_effect=fake_call):
            raw, parsed = await clr._call_model_with_repair(
                None, "prompt", 100, 1.0, None,
            )
        assert raw == "freeform text"
        assert parsed is None


# --------------------------------------------------------------------------- #
# Chat transports (OpenAI + Anthropic) with native enforcement
# --------------------------------------------------------------------------- #

class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass


class _CapturingSession:
    def __init__(self, payload):
        self._payload = payload
        self.captured = {}

    def post(self, url, json=None, headers=None, **kw):
        self.captured["url"] = url
        self.captured["payload"] = json or {}
        self.captured["headers"] = headers or {}
        return _FakeResp(self._payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass


class TestOpenAIChatTransport:
    @pytest.mark.asyncio
    async def test_openai_chat_applies_response_format(self):
        clr = VibeThinkerCLRAsync(
            server_url="http://localhost:0", k=1,
            specialist_transport="openai_chat",
            specialist_model_name="gpt-4o-mini",
            specialist_api_key="sk-test",
        )
        valid = json.dumps({
            "reasoning_steps": ["s"],
            "boxed_answer": "42",
            "code_solution": None,
        })
        session = _CapturingSession({
            "choices": [{"message": {"content": valid}}],
        })
        with patch.object(clr, "semaphore"):
            result = await clr._call_model(
                session,
                "<|im_start|>user\nSolve 2+2<|im_end|>\n<|im_start|>assistant\n",
                max_tokens=100, grammar=_STRUCTURED_OUTPUT_GRAMMAR,
            )
        assert "42" in result
        assert session.captured["url"].endswith("/v1/chat/completions")
        assert session.captured["payload"]["model"] == "gpt-4o-mini"
        rf = session.captured["payload"]["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["strict"] is True
        # Auth header present, key not logged elsewhere.
        assert session.captured["headers"]["Authorization"] == "Bearer sk-test"

    @pytest.mark.asyncio
    async def test_openai_chat_strips_chatml(self):
        clr = VibeThinkerCLRAsync(
            server_url="http://localhost:0", k=1,
            specialist_transport="openai_chat",
        )
        session = _CapturingSession({
            "choices": [{"message": {"content": "ok"}}],
        })
        with patch.object(clr, "semaphore"):
            await clr._call_model(
                session,
                "<|im_start|>user\nHello world<|im_end|>\n<|im_start|>assistant\n",
                max_tokens=10,
            )
        msg = session.captured["payload"]["messages"][0]
        assert msg["role"] == "user"
        assert "Hello world" in msg["content"]
        assert "<|im_start|>" not in msg["content"]


class TestAnthropicTransport:
    @pytest.mark.asyncio
    async def test_anthropic_applies_tool_choice(self):
        clr = VibeThinkerCLRAsync(
            server_url="http://localhost:0", k=1,
            specialist_transport="anthropic",
            specialist_model_name="claude-3-5-sonnet-20241022",
            specialist_api_key="sk-ant-test",
        )
        # Anthropic returns a tool_use block with the structured input.
        session = _CapturingSession({
            "content": [
                {"type": "tool_use", "name": "structured_output",
                 "input": {"reasoning_steps": ["s"], "boxed_answer": "42",
                           "code_solution": None}},
            ],
        })
        with patch.object(clr, "semaphore"):
            result = await clr._call_model(
                session,
                "<|im_start|>user\nSolve 2+2<|im_end|>\n<|im_start|>assistant\n",
                max_tokens=100, grammar=_STRUCTURED_OUTPUT_GRAMMAR,
            )
        # The tool_use input is returned as a JSON string for the shared parser.
        parsed = json.loads(result)
        assert parsed["boxed_answer"] == "42"
        payload = session.captured["payload"]
        assert payload["model"] == "claude-3-5-sonnet-20241022"
        assert payload["tool_choice"]["type"] == "tool"
        assert payload["tool_choice"]["name"] == "structured_output"
        assert payload["tools"][0]["name"] == "structured_output"
        assert session.captured["headers"]["x-api-key"] == "sk-ant-test"
        assert session.captured["headers"]["anthropic-version"] == "2023-06-01"

    @pytest.mark.asyncio
    async def test_anthropic_text_fallback_when_no_tool_use(self):
        """If the model emits text despite tool_choice, concatenate text blocks."""
        clr = VibeThinkerCLRAsync(
            server_url="http://localhost:0", k=1,
            specialist_transport="anthropic",
        )
        session = _CapturingSession({
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
            ],
        })
        with patch.object(clr, "semaphore"):
            result = await clr._call_model(
                session,
                "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
                max_tokens=10, grammar=_STRUCTURED_OUTPUT_GRAMMAR,
            )
        assert result == "Hello world"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class TestHelpers:
    def test_strip_chatml_extracts_user_and_prefill(self):
        user, prefill = VibeThinkerCLRAsync._strip_chatml(
            "<|im_start|>user\nSolve it<|im_end|>\n<|im_start|>assistant\nSure, "
        )
        assert "Solve it" in user
        assert "<|im_start|>" not in user
        assert prefill == "Sure, "

    def test_strip_chatml_no_markers_returns_raw(self):
        user, prefill = VibeThinkerCLRAsync._strip_chatml("just a string")
        assert user == "just a string"
        assert prefill is None

    def test_augment_with_structured_grammar(self):
        out = VibeThinkerCLRAsync._augment_with_json_instruction(
            "solve", _STRUCTURED_OUTPUT_GRAMMAR,
        )
        assert "reasoning_steps" in out
        assert "boxed_answer" in out

    def test_augment_with_claims_grammar(self):
        out = VibeThinkerCLRAsync._augment_with_json_instruction(
            "extract", _CLAIMS_JSON_GRAMMAR,
        )
        assert "claims" in out
        assert "final_answer" in out

    def test_augment_unknown_grammar_passthrough(self):
        out = VibeThinkerCLRAsync._augment_with_json_instruction(
            "do thing", "root ::= 'test'",
        )
        assert out == "do thing"
