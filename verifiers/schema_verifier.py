"""Schema verifier — deterministic structural validation.

Verifies that an answer conforms to a specified structure:
  - JSON schema (subset of Draft 2020-12: type, properties, required,
    items, enum, minimum, maximum, minLength, maxLength, pattern)
  - YAML structure (parsed and checked against the same schema dict)
  - Regex pattern (fullmatch)

This is a deterministic verifier — it does not call any model. The schema
or pattern is provided in the verifier context by the caller (e.g. the
orchestrator's ``_build_verifier_context`` or a test spec). When no schema
or pattern is provided, returns ``verified=False`` with an honest "no
schema provided" error — never fakes structural validation.

Trust model (fail-closed):
  - No schema/pattern in context -> verified=False (honest, not a failure)
  - Parse error -> verified=False with the parse error as evidence
  - Schema mismatch -> verified=False with the specific violation
  - Match -> verified=True, score=1.0 (structural conformance is binary)

Requires PyYAML for YAML validation; JSON uses the stdlib json module.
When PyYAML is absent, YAML validation fail-closes (returns "yaml module
not available") rather than silently skipping.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Union

from verifiers.base import VerificationResult


# Optional YAML support — fail-closed when absent.
try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


class SchemaVerifier:
    """Deterministic verifier for structural/schema conformance.

    Supported context keys:
      - ``schema``: a JSON-schema dict (subset). Validated against the
        parsed answer (JSON or YAML).
      - ``format``: ``"json"`` (default), ``"yaml"``, or ``"text"``.
        Controls how the answer is parsed before schema validation.
        ``"text"`` skips parsing and only applies regex validation.
      - ``pattern``: a regex string. The answer (stripped) must fullmatch.
      - ``expected_keys``: a list of keys that must be present at the top
        level of the parsed JSON/YAML dict. A convenience shortcut for
        simple ``{"type": "object", "required": [...]}`` schemas.

    Any combination of schema/pattern/expected_keys can be provided; ALL
    provided checks must pass. If none are provided, returns
    ``verified=False`` (honest — no criteria to check).
    """

    name = "schema_verifier"

    async def verify(
        self, query: str, answer: str, context: Dict[str, Any]
    ) -> VerificationResult:
        schema: Optional[Dict[str, Any]] = context.get("schema")
        pattern: Optional[str] = context.get("pattern")
        expected_keys: Optional[List[str]] = context.get("expected_keys")
        fmt: str = context.get("format", "json")

        if schema is None and pattern is None and expected_keys is None:
            return VerificationResult(
                verified=False,
                score=0.0,
                method="schema_validation",
                evidence={"answer": answer[:200]},
                error="no schema, pattern, or expected_keys provided; "
                      "cannot verify structural conformance",
            )

        evidence: Dict[str, Any] = {"format": fmt}
        checks_passed = True
        errors: List[str] = []

        # 1. Regex pattern check (applies to the raw answer text).
        if pattern is not None:
            try:
                regex = re.compile(pattern)
            except re.error as e:
                return VerificationResult(
                    verified=False, score=0.0, method="schema_validation",
                    evidence=evidence,
                    error=f"invalid regex pattern: {e}",
                )
            if not regex.fullmatch(answer.strip()):
                checks_passed = False
                errors.append(
                    f"answer does not match pattern {pattern!r}"
                )
            else:
                evidence["pattern_matched"] = pattern

        # 2. Schema / expected_keys checks require parsing the answer.
        if schema is not None or expected_keys is not None:
            parsed, parse_err = self._parse(answer, fmt)
            if parse_err is not None:
                return VerificationResult(
                    verified=False, score=0.0, method="schema_validation",
                    evidence={**evidence, "answer": answer[:200]},
                    error=parse_err,
                )
            evidence["parsed_type"] = type(parsed).__name__

            # 2a. expected_keys shortcut.
            if expected_keys is not None:
                if not isinstance(parsed, dict):
                    checks_passed = False
                    errors.append(
                        f"expected a dict for expected_keys check, "
                        f"got {type(parsed).__name__}"
                    )
                else:
                    missing = [k for k in expected_keys if k not in parsed]
                    if missing:
                        checks_passed = False
                        errors.append(f"missing required keys: {missing}")
                    else:
                        evidence["expected_keys_present"] = expected_keys

            # 2b. Full schema validation.
            if schema is not None:
                schema_errors = self._validate_schema(parsed, schema, "$")
                if schema_errors:
                    checks_passed = False
                    errors.extend(schema_errors)
                else:
                    evidence["schema_valid"] = True

        if checks_passed:
            return VerificationResult(
                verified=True,
                score=1.0,
                method="schema_validation",
                evidence=evidence,
            )
        return VerificationResult(
            verified=False,
            score=0.0,
            method="schema_validation",
            evidence=evidence,
            error="; ".join(errors),
        )

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse(
        answer: str, fmt: str
    ) -> tuple[Optional[Any], Optional[str]]:
        """Parse the answer string into a Python object.

        Returns (parsed, None) on success or (None, error_msg) on failure.
        """
        text = answer.strip()
        if not text:
            return None, "empty answer; cannot parse"
        if fmt == "text":
            return text, None
        if fmt == "json":
            try:
                return json.loads(text), None
            except json.JSONDecodeError as e:
                return None, f"JSON parse error: {e}"
        if fmt == "yaml":
            if not _YAML_AVAILABLE:
                return None, ("yaml module not available; "
                              "pip install pyyaml")
            try:
                return yaml.safe_load(text), None
            except yaml.YAMLError as e:
                return None, f"YAML parse error: {e}"
        return None, f"unknown format: {fmt!r}"

    # ------------------------------------------------------------------ #
    # Schema validation (subset of JSON Schema Draft 2020-12)
    # ------------------------------------------------------------------ #
    def _validate_schema(
        self, value: Any, schema: Dict[str, Any], path: str
    ) -> List[str]:
        """Validate ``value`` against ``schema``. Returns a list of error
        strings (empty = valid). Supports a useful subset of JSON Schema."""
        errors: List[str] = []
        expected_type = schema.get("type")

        # type
        if expected_type is not None:
            if not self._check_type(value, expected_type):
                errors.append(
                    f"{path}: expected type {expected_type!r}, "
                    f"got {type(value).__name__}"
                )
                return errors  # further checks are meaningless if type wrong

        # enum
        if "enum" in schema:
            if value not in schema["enum"]:
                errors.append(
                    f"{path}: value {value!r} not in enum {schema['enum']!r}"
                )

        # string constraints
        if isinstance(value, str):
            if "minLength" in schema and len(value) < schema["minLength"]:
                errors.append(
                    f"{path}: string length {len(value)} < minLength "
                    f"{schema['minLength']}"
                )
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                errors.append(
                    f"{path}: string length {len(value)} > maxLength "
                    f"{schema['maxLength']}"
                )
            if "pattern" in schema:
                try:
                    if not re.search(schema["pattern"], value):
                        errors.append(
                            f"{path}: string does not match pattern "
                            f"{schema['pattern']!r}"
                        )
                except re.error as e:
                    errors.append(f"{path}: invalid schema pattern: {e}")

        # numeric constraints
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in schema and value < schema["minimum"]:
                errors.append(
                    f"{path}: {value} < minimum {schema['minimum']}"
                )
            if "maximum" in schema and value > schema["maximum"]:
                errors.append(
                    f"{path}: {value} > maximum {schema['maximum']}"
                )

        # object constraints
        if isinstance(value, dict):
            required = schema.get("required", [])
            missing = [k for k in required if k not in value]
            if missing:
                errors.append(f"{path}: missing required keys: {missing}")
            properties = schema.get("properties", {})
            for key, subschema in properties.items():
                if key in value:
                    errors.extend(
                        self._validate_schema(
                            value[key], subschema, f"{path}.{key}"
                        )
                    )
            # additionalProperties: false -> reject unknown keys
            if schema.get("additionalProperties") is False:
                extra = [k for k in value if k not in properties]
                if extra:
                    errors.append(
                        f"{path}: additional properties not allowed: {extra}"
                    )

        # array constraints
        if isinstance(value, list):
            if "minItems" in schema and len(value) < schema["minItems"]:
                errors.append(
                    f"{path}: array length {len(value)} < minItems "
                    f"{schema['minItems']}"
                )
            if "maxItems" in schema and len(value) > schema["maxItems"]:
                errors.append(
                    f"{path}: array length {len(value)} > maxItems "
                    f"{schema['maxItems']}"
                )
            items = schema.get("items")
            if items is not None:
                for i, item in enumerate(value):
                    errors.extend(
                        self._validate_schema(item, items, f"{path}[{i}]")
                    )

        return errors

    @staticmethod
    def _check_type(value: Any, expected: str) -> bool:
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }
        # bool is a subclass of int in Python — exclude it from integer/number.
        if expected in ("integer", "number") and isinstance(value, bool):
            return False
        if expected == "integer" and isinstance(value, float):
            return value.is_integer()
        py_type = type_map.get(expected)
        if py_type is None:
            return False
        return isinstance(value, py_type)
