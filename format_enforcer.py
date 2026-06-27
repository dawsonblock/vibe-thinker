"""Provider-agnostic structured-output enforcement (v1.1).

The specialist's structured-output contract (``reasoning_steps``,
``boxed_answer``, ``code_solution``) and the claim-extraction contract
(``claims``, ``final_answer``) were originally expressed only as llama.cpp
GBNF grammar strings. OpenAI and Anthropic do not accept the ``grammar``
payload field and have no GBNF support, so sending those grammars to their
APIs silently dropped the constraint and produced malformed JSON.

This module abstracts the *contract* from the *enforcement mechanism*. A
:class:`FormatEnforcer` knows how to:

  - render the contract as a llama.cpp GBNF string (``to_llama_cpp_grammar``),
  - render it as an OpenAI ``response_format`` JSON-schema dict
    (``to_openai_response_format``),
  - render it as an Anthropic tool definition (``to_anthropic_tool``),
  - parse a model response back into the structured dict (``parse``),
  - build a repair prompt feeding the parse error back to the model
    (``repair_prompt``).

The GBNF output of :class:`LlamaCppEnforcer` is byte-identical to the
historical ``_STRUCTURED_OUTPUT_GRAMMAR`` / ``_CLAIMS_JSON_GRAMMAR`` strings
in ``vibe_clr_async.py`` — those strings are re-exported here as the single
source of truth and ``vibe_clr_async`` imports them back from this module so
behavior is unchanged for the llama.cpp / RuvLLM path.

The parse logic is shared across enforcers (the JSON shape is the same
regardless of how it was enforced). ``VibeThinkerCLRAsync.parse_structured_output``
delegates to :func:`parse_structured_output` here so the existing tests keep
passing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


# --------------------------------------------------------------------------- #
# The two structured-output contracts, expressed as GBNF (llama.cpp source of
# truth). These strings are imported back into vibe_clr_async.py so the
# in-process and /completion paths keep using exactly the same grammars.
# --------------------------------------------------------------------------- #

CLAIMS_JSON_GRAMMAR = r"""root ::= "{" ws "\"claims\"" ws ":" ws "[" ws string ("," ws string)* ws "]" ws "," ws "\"final_answer\"" ws ":" ws (string | "null") ws "}"
string ::= "\"" ([^"\\] | "\\" .)* "\""
ws ::= [ \t\n]*
"""

STRUCTURED_OUTPUT_GRAMMAR = r"""root ::= "{" ws "\"reasoning_steps\"" ws ":" ws "[" ws string ("," ws string)* ws "]" ws "," ws "\"boxed_answer\"" ws ":" ws (string | "null") ws "," ws "\"code_solution\"" ws ":" ws (string | "null") ws "}"
string ::= "\"" ([^"\\] | "\\" .)* "\""
ws ::= [ \t\n]*
"""


class SchemaKind(Enum):
    """Which structured-output contract an enforcer enforces."""

    STRUCTURED_OUTPUT = "structured_output"
    CLAIMS = "claims"


@dataclass
class FormatSchema:
    """A structured-output contract: the keys and their types.

    This is the transport-agnostic description from which each enforcer
    derives its native representation (GBNF, JSON-schema, tool input-schema).
    """

    kind: SchemaKind
    # Ordered list of (key, type, required) tuples. type is one of:
    # "string", "string_or_null", "string_array".
    fields: List[tuple] = field(default_factory=list)

    @property
    def required_keys(self) -> List[str]:
        return [k for (k, _t, required) in self.fields if required]

    @property
    def all_keys(self) -> List[str]:
        return [k for (k, _t, _r) in self.fields]


def _structured_output_schema() -> FormatSchema:
    return FormatSchema(
        kind=SchemaKind.STRUCTURED_OUTPUT,
        fields=[
            ("reasoning_steps", "string_array", True),
            ("boxed_answer", "string_or_null", True),
            ("code_solution", "string_or_null", True),
        ],
    )


def _claims_schema() -> FormatSchema:
    return FormatSchema(
        kind=SchemaKind.CLAIMS,
        fields=[
            ("claims", "string_array", True),
            ("final_answer", "string_or_null", True),
        ],
    )


# --------------------------------------------------------------------------- #
# JSON-schema subset used for the OpenAI response_format mapping.
# --------------------------------------------------------------------------- #

def _field_to_json_schema(ftype: str) -> Dict[str, Any]:
    if ftype == "string":
        return {"type": "string"}
    if ftype == "string_or_null":
        return {"type": ["string", "null"]}
    if ftype == "string_array":
        return {"type": "array", "items": {"type": "string"}}
    return {"type": "string"}


def _schema_to_json_schema(schema: FormatSchema) -> Dict[str, Any]:
    """Render a FormatSchema as a JSON-schema dict (OpenAI response_format)."""
    properties: Dict[str, Any] = {}
    for key, ftype, _required in schema.fields:
        properties[key] = _field_to_json_schema(ftype)
    return {
        "type": "object",
        "properties": properties,
        "required": schema.required_keys,
        "additionalProperties": False,
    }


# --------------------------------------------------------------------------- #
# Parse + repair (shared across enforcers — the JSON shape is the same).
# --------------------------------------------------------------------------- #

_NULLISH = frozenset({"null", "none", "", "n/a"})


def _normalize_nullable(value: Any) -> Any:
    """Normalize a string 'null'/'none'/'' to None (mirrors the v0.4.1 logic)."""
    if isinstance(value, str) and value.strip().lower() in _NULLISH:
        return None
    return value


def parse_structured_output(text: str) -> Optional[Dict[str, Any]]:
    """Parse a structured specialist output (v0.4.1, moved here in v1.1).

    Accepts the JSON object produced by either the structured-output grammar
    or the claims grammar. Strips markdown code fences, finds the JSON object
    span, validates the required keys, and normalizes 'null' strings to None.

    Returns None if the text is not valid structured output (caller falls
    back to regex-based extraction). This is the single parse implementation
    shared by all enforcers and by ``VibeThinkerCLRAsync.parse_structured_output``.
    """
    try:
        text = text.strip()
        # Strip markdown code fences if present.
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            )
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        data = json.loads(text[start:end + 1])
        if not isinstance(data, dict):
            return None
        # Accept either schema. Validate by presence of a required key from
        # each: structured output has 'reasoning_steps', claims has 'claims'.
        if "reasoning_steps" in data:
            result: Dict[str, Any] = {
                "reasoning_steps": data.get("reasoning_steps", []),
                "boxed_answer": _normalize_nullable(data.get("boxed_answer")),
                "code_solution": _normalize_nullable(data.get("code_solution")),
            }
            return result
        if "claims" in data:
            return {
                "claims": data.get("claims", []) or [],
                "final_answer": _normalize_nullable(data.get("final_answer")),
            }
        return None
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def parse_error_detail(text: str) -> Optional[str]:
    """Return a short human-readable reason why ``text`` failed to parse.

    Used to feed the repair loop — the model gets a concrete error rather
    than a generic 'fix your JSON'. Returns None if we can't produce a
    useful message (caller falls back to a generic prompt).
    """
    s = text.strip()
    if not s:
        return "The output was empty."
    if "{" not in s:
        return "The output contains no JSON object (no '{' character). Output ONLY a JSON object."
    # Try to locate a JSON span and decode it for a precise error.
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return "The output has an unbalanced JSON object (missing '{' or '}')."
    span = s[start:end + 1]
    try:
        data = json.loads(span)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e.msg} at line {e.lineno} column {e.colno}. Fix the JSON syntax and output ONLY the JSON object."
    if not isinstance(data, dict):
        return "The JSON parsed but is not an object (it is a list or scalar). Output a JSON object."
    if "reasoning_steps" not in data and "claims" not in data:
        return (
            "The JSON object is missing required keys. Include "
            "'reasoning_steps', 'boxed_answer', and 'code_solution' "
            "(or 'claims' and 'final_answer')."
        )
    return None


# --------------------------------------------------------------------------- #
# Enforcer protocol + implementations.
# --------------------------------------------------------------------------- #

@runtime_checkable
class FormatEnforcer(Protocol):
    """Maps a structured-output contract to a provider's native enforcement.

    Every enforcer can also ``parse`` the model's text response back into the
    structured dict (the JSON shape is provider-independent) and build a
    ``repair_prompt`` for the parse-repair loop.
    """

    @property
    def kind(self) -> SchemaKind: ...

    def to_llama_cpp_grammar(self) -> str: ...
    def to_openai_response_format(self) -> Dict[str, Any]: ...
    def to_anthropic_tool(self) -> Dict[str, Any]: ...

    def parse(self, text: str) -> Optional[Dict[str, Any]]: ...
    def repair_prompt(self, bad_text: str, error: Optional[str] = None) -> str: ...


class _BaseEnforcer:
    """Shared parse/repair logic. Subclasses provide the native renderings."""

    def __init__(self, schema: FormatSchema):
        self._schema = schema

    @property
    def kind(self) -> SchemaKind:
        return self._schema.kind

    def parse(self, text: str) -> Optional[Dict[str, Any]]:
        return parse_structured_output(text)

    def repair_prompt(self, bad_text: str, error: Optional[str] = None) -> str:
        """Build a one-shot repair prompt feeding the parse error back.

        The caller sends this as a fresh user turn. The model is told its
        previous output was invalid and given the specific error, then asked
        to output ONLY the corrected JSON object.
        """
        detail = error or parse_error_detail(bad_text) or (
            "The output was not valid JSON."
        )
        # Truncate the bad text to avoid blowing context on a runaway model.
        snippet = bad_text.strip()
        if len(snippet) > 2000:
            snippet = snippet[:2000] + "\n...[truncated]"
        if self._schema.kind == SchemaKind.STRUCTURED_OUTPUT:
            schema_desc = (
                'a JSON object with keys "reasoning_steps" (array of strings), '
                '"boxed_answer" (string or null), and "code_solution" '
                '(string or null)'
            )
        else:
            schema_desc = (
                'a JSON object with keys "claims" (array of strings) and '
                '"final_answer" (string or null)'
            )
        return (
            "Your previous output was invalid. Here is the error:\n"
            f"{detail}\n\n"
            "Your previous output was:\n"
            f"{snippet}\n\n"
            f"Output ONLY {schema_desc}. No markdown, no commentary, "
            "just the JSON object."
        )


class LlamaCppEnforcer(_BaseEnforcer):
    """llama.cpp / RuvLLM: enforce via compiled GBNF grammar string.

    The GBNF output is byte-identical to the historical grammars in
    ``vibe_clr_async.py`` (those strings are now imported from this module).
    """

    def to_llama_cpp_grammar(self) -> str:
        if self._schema.kind == SchemaKind.STRUCTURED_OUTPUT:
            return STRUCTURED_OUTPUT_GRAMMAR
        return CLAIMS_JSON_GRAMMAR

    def to_openai_response_format(self) -> Dict[str, Any]:
        # llama-server's /completion endpoint doesn't use response_format,
        # but if someone points the openai_chat transport at a llama-server
        # /v1/chat/completions endpoint, the JSON-schema form is the best we
        # can do. Most llama-server builds accept response_format json_object.
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self._schema.kind.value,
                "strict": True,
                "schema": _schema_to_json_schema(self._schema),
            },
        }

    def to_anthropic_tool(self) -> Dict[str, Any]:
        # Anthropic tool-use enforcement. The model is forced to call this
        # tool with the structured payload.
        return {
            "name": self._schema.kind.value,
            "description": "Structured reasoning output.",
            "input_schema": _schema_to_json_schema(self._schema),
        }


class OpenAIEnforcer(_BaseEnforcer):
    """OpenAI-compatible /v1/chat/completions: enforce via response_format.

    Uses ``response_format={"type":"json_schema","json_schema":{...}}`` with
    ``strict=True`` so the model is constrained to the schema. Falls back to
    ``{"type":"json_object"}`` if the caller requests a looser mode (the
    parse-repair loop covers the resulting malformed-JSON cases).
    """

    def __init__(self, schema: FormatSchema, strict: bool = True):
        super().__init__(schema)
        self._strict = strict

    def to_llama_cpp_grammar(self) -> str:
        # An OpenAI-targeted enforcer can still render GBNF for fallback to a
        # llama-server /completion endpoint. Reuse the canonical grammars.
        if self._schema.kind == SchemaKind.STRUCTURED_OUTPUT:
            return STRUCTURED_OUTPUT_GRAMMAR
        return CLAIMS_JSON_GRAMMAR

    def to_openai_response_format(self) -> Dict[str, Any]:
        if self._strict:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": self._schema.kind.value,
                    "strict": True,
                    "schema": _schema_to_json_schema(self._schema),
                },
            }
        return {"type": "json_object"}

    def to_anthropic_tool(self) -> Dict[str, Any]:
        return {
            "name": self._schema.kind.value,
            "description": "Structured reasoning output.",
            "input_schema": _schema_to_json_schema(self._schema),
        }


class AnthropicEnforcer(_BaseEnforcer):
    """Anthropic /v1/messages: enforce via tool_choice forcing a tool call.

    Anthropic does not support response_format. Structured output is achieved
    by defining a tool with the schema as its ``input_schema`` and setting
    ``tool_choice={"type":"tool","name":<kind>}`` to force the model to emit
    the payload as a tool_use block. The orchestrator then reads the tool's
    ``input`` field as the structured dict.
    """

    def to_llama_cpp_grammar(self) -> str:
        if self._schema.kind == SchemaKind.STRUCTURED_OUTPUT:
            return STRUCTURED_OUTPUT_GRAMMAR
        return CLAIMS_JSON_GRAMMAR

    def to_openai_response_format(self) -> Dict[str, Any]:
        # If an Anthropic-targeted enforcer is asked for an OpenAI response
        # format (e.g. misconfigured transport), produce the JSON-schema form.
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self._schema.kind.value,
                "strict": True,
                "schema": _schema_to_json_schema(self._schema),
            },
        }

    def to_anthropic_tool(self) -> Dict[str, Any]:
        return {
            "name": self._schema.kind.value,
            "description": "Structured reasoning output.",
            "input_schema": _schema_to_json_schema(self._schema),
        }


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

def make_enforcer(
    kind: SchemaKind,
    transport: str = "completion",
    *,
    strict: bool = True,
) -> FormatEnforcer:
    """Build a FormatEnforcer for the given schema kind and transport.

    Args:
        kind: which structured-output contract to enforce.
        transport: "completion" (llama-server/RuvLLM) -> LlamaCppEnforcer,
            "openai_chat" -> OpenAIEnforcer, "anthropic" -> AnthropicEnforcer.
        strict: for OpenAI, whether to use strict json_schema mode.
    """
    schema = (
        _structured_output_schema()
        if kind == SchemaKind.STRUCTURED_OUTPUT
        else _claims_schema()
    )
    if transport == "openai_chat":
        return OpenAIEnforcer(schema, strict=strict)
    if transport == "anthropic":
        return AnthropicEnforcer(schema)
    return LlamaCppEnforcer(schema)
