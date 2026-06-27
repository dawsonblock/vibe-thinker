"""Shared serialization utilities for vibe-thinker.

Provides ``serialize_for_json()`` — a recursive converter that turns
arbitrary Python objects (dataclasses, sets, bytes, nested dicts/lists)
into JSON-safe values. Used by both ``web/app.py`` and
``federated_queue.py`` to avoid duplicating serialization logic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict


def serialize_for_json(obj: Any) -> Any:
    """Recursively convert dataclasses / sets / bytes to JSON-safe values.

    Handles:
      - dataclasses (via asdict)
      - dicts, lists, tuples, sets
      - bytes (decoded to str)
      - primitives (int, float, str, bool, None)
      - anything else → str(obj)
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: serialize_for_json(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_for_json(v) for v in obj]
    if isinstance(obj, set):
        return [serialize_for_json(v) for v in obj]
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


def serialize_result_dict(raw: Any) -> Dict[str, Any]:
    """Serialize an OrchestratorResult to a JSON-safe dict for federation.

    Handles nested non-serializable objects (e.g. CLRResult in raw_traces)
    by falling back to str() for anything json.dumps can't handle.
    Used by FederatedJobQueue.
    """
    if raw is None:
        return {}
    if hasattr(raw, "__dict__"):
        raw_dict = dict(raw.__dict__)
    elif isinstance(raw, dict):
        raw_dict = raw
    else:
        return {"result": str(raw)}
    # Use serialize_for_json for robust recursive conversion.
    return serialize_for_json(raw_dict)
