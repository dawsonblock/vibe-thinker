"""
Persistent caches for the Hybrid Reasoning Orchestrator.

Two caches:
  1. PersistentRouteCache  — disk-backed save/load of route decisions and
     query embeddings (JSON). Survives restarts so you don't re-encode
     repeated queries.
  2. CLRResultCache        — semantic cache for CLR results. Finds similar
     past problems (via embedding similarity) and returns a cached result
     if it was solved with a high reliability score, skipping a re-run.

Install (optional, for semantic CLR cache):
  pip install sentence-transformers scikit-learn numpy

Design notes:
  - Both caches are JSON-serializable on disk (embeddings stored as lists).
  - The route cache is exact-match (normalized key) + LRU in memory.
  - The CLR result cache is semantic: it compares the query embedding to
    cached problem embeddings and returns the closest high-score result
    above a similarity threshold.
  - All disk writes are atomic (write to temp file, then rename) to avoid
    corruption if the process is killed mid-write.
"""

import json
import os
import tempfile
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Optional deps for semantic CLR cache
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


# ====================================================================== #
# Atomic JSON write helper
# ====================================================================== #
# Answers that must NEVER be cached, regardless of score.
BAD_ANSWERS = frozenset({
    "no clear answer found",
    "all trajectories failed",
    "",
    "none",
    "null",
    "n/a",
})


def should_cache(result: Dict[str, Any], allow_weak_cache: bool = False) -> bool:
    """Decide whether a CLR result dict is safe to promote into the cache.

    Default policy is strict (fail-closed): weak self-verification must NOT
    enter the cache. The cache is a trust accelerator — caching uncertainty
    as truth is the core epistemic hazard this system guards against.

    Required keys in ``result``:
      - answer: the final answer string
      - score: reliability score (0.0–1.0)
      - answer_present: whether a final answer was produced
      - claim_count: number of meaningful claims that were scored
      - verification_method: how the answer was verified
      - failure: None if successful, error string if failed

    Args:
        result: the result dict to evaluate.
        allow_weak_cache: if True, allow ``self_claims_only`` verification
            to be cached. Default is False — self-agreement is NOT proof.

    Returns:
        True if the result is safe to cache, False otherwise.
    """
    if not result.get("answer_present"):
        return False
    if result.get("failure"):
        return False
    answer = (result.get("answer") or "").strip().lower()
    if answer in BAD_ANSWERS:
        return False
    if result.get("claim_count", 0) < 5:
        return False
    if result.get("score", 0.0) < 0.75:
        return False
    method = result.get("verification_method", "self_claims_only")
    if method == "self_claims_only" and not allow_weak_cache:
        return False
    if result.get("transport_failures", 0) > 0:
        return False
    return True


def _atomic_write_json(path: str, data: Any) -> None:
    """Write JSON to a temp file in the same dir, then atomically rename."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_json(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[PersistentCache] Failed to load {path}: {e}")
        return None


# ====================================================================== #
# Persistent route + embedding cache
# ====================================================================== #
class PersistentRouteCache:
    """
    Disk-backed cache for route decisions and query embeddings.

    On-disk format (JSON):
      {
        "model_name": "all-MiniLM-L6-v2",
        "embedding_cache": { "<normalized_query>": [float, ...], ... },
        "route_cache":    { "<normalized_query>": ["specialist", 0.82], ... }
      }
    """

    def __init__(
        self,
        path: str = "route_cache.json",
        cache_size: int = 512,
        autosave: bool = True,
    ):
        self.path = path
        self.cache_size = cache_size
        self.autosave = autosave

        self.embedding_cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self.route_cache: "OrderedDict[str, Tuple[str, float]]" = OrderedDict()

        self._load()

    # ----------------------- persistence ----------------------- #
    def _load(self) -> None:
        data = _load_json(self.path)
        if not data:
            return
        emb = data.get("embedding_cache", {}) or {}
        rts = data.get("route_cache", {}) or {}
        # Restore in insertion order (JSON objects preserve order in py3.7+)
        for k, v in emb.items():
            self.embedding_cache[k] = list(v)
            if len(self.embedding_cache) > self.cache_size:
                self.embedding_cache.popitem(last=False)
        for k, v in rts.items():
            try:
                self.route_cache[k] = (str(v[0]), float(v[1]))
            except (IndexError, TypeError, ValueError):
                continue
            if len(self.route_cache) > self.cache_size:
                self.route_cache.popitem(last=False)
        print(
            f"[PersistentRouteCache] Loaded {len(self.embedding_cache)} embeddings, "
            f"{len(self.route_cache)} route decisions from {self.path}"
        )

    def save(self) -> None:
        data = {
            "model_name": getattr(self, "model_name", None),
            "embedding_cache": dict(self.embedding_cache),
            "route_cache": {k: list(v) for k, v in self.route_cache.items()},
        }
        _atomic_write_json(self.path, data)

    # ----------------------- accessors ----------------------- #
    @staticmethod
    def _normalize(text: str) -> str:
        return text.lower().strip()

    def get_embedding(self, key: str) -> Optional[List[float]]:
        k = self._normalize(key)
        if k in self.embedding_cache:
            self.embedding_cache.move_to_end(k)
            return self.embedding_cache[k]
        return None

    def put_embedding(self, key: str, embedding: Any) -> None:
        k = self._normalize(key)
        emb_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        if k in self.embedding_cache:
            # Update existing key — do NOT evict (updating doesn't grow occupancy)
            self.embedding_cache.move_to_end(k)
        else:
            # New key — evict oldest if at capacity
            if len(self.embedding_cache) >= self.cache_size:
                self.embedding_cache.popitem(last=False)
        self.embedding_cache[k] = emb_list
        if self.autosave:
            self.save()

    def get_route(self, key: str) -> Optional[Tuple[str, float]]:
        k = self._normalize(key)
        if k in self.route_cache:
            self.route_cache.move_to_end(k)
            return self.route_cache[k]
        return None

    def put_route(self, key: str, route: str, confidence: float) -> None:
        k = self._normalize(key)
        if k in self.route_cache:
            # Update existing key — do NOT evict
            self.route_cache.move_to_end(k)
        else:
            # New key — evict oldest if at capacity
            if len(self.route_cache) >= self.cache_size:
                self.route_cache.popitem(last=False)
        self.route_cache[k] = (route, float(confidence))
        if self.autosave:
            self.save()

    def clear(self) -> None:
        self.embedding_cache.clear()
        self.route_cache.clear()
        if os.path.exists(self.path):
            os.unlink(self.path)
        print("[PersistentRouteCache] Cleared")


# ====================================================================== #
# Semantic CLR result cache
# ====================================================================== #
class CLRResultCache:
    """
    Semantic cache for CLR results.

    Stores completed CLR runs keyed by problem embedding. On lookup, computes
    the query embedding and returns the closest cached result IF:
      - cosine similarity >= similarity_threshold, AND
      - the cached result's best_score >= min_score

    This lets you skip re-running expensive CLR for repeated or near-identical
    high-confidence problems.

    On-disk format (JSON):
      {
        "model_name": "all-MiniLM-L6-v2",
        "entries": [
          {
            "problem": "...",
            "embedding": [float, ...],
            "best_answer": "...",
            "best_score": 0.98,
            "k": 8,
            "timestamp": "...",
            "trajectory_count": 8
          }, ...
        ]
      }
    """

    def __init__(
        self,
        path: str = "clr_result_cache.json",
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.92,
        min_score: float = 0.7,
        max_entries: int = 256,
        autosave: bool = True,
    ):
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "CLRResultCache needs: pip install sentence-transformers scikit-learn numpy"
            )
        self.path = path
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.min_score = min_score
        self.max_entries = max_entries
        self.autosave = autosave

        self.model = SentenceTransformer(model_name)
        self.entries: List[Dict[str, Any]] = []
        self._embeddings_matrix: Optional[Any] = None  # np.ndarray, rebuilt on load

        self._load()

    # ----------------------- persistence ----------------------- #
    def _load(self) -> None:
        data = _load_json(self.path)
        if not data:
            return
        self.entries = data.get("entries", []) or []
        self._rebuild_embeddings_matrix()
        print(f"[CLRResultCache] Loaded {len(self.entries)} entries from {self.path}")

    def _rebuild_embeddings_matrix(self) -> None:
        if not self.entries:
            self._embeddings_matrix = None
            return
        self._embeddings_matrix = np.array(
            [e["embedding"] for e in self.entries]
        )

    def save(self) -> None:
        data = {
            "model_name": self.model_name,
            "entries": self.entries,
        }
        _atomic_write_json(self.path, data)

    # ----------------------- lookup ----------------------- #
    def lookup(self, problem: str) -> Optional[Dict[str, Any]]:
        """Return a cached result if a similar high-score problem exists."""
        if not self.entries or self._embeddings_matrix is None:
            return None
        q_emb = self.model.encode([problem])[0].reshape(1, -1)
        sims = cosine_similarity(q_emb, self._embeddings_matrix)[0]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        entry = self.entries[best_idx]
        if best_sim >= self.similarity_threshold and entry.get("best_score", 0) >= self.min_score:
            return {
                "best_answer": entry["best_answer"],
                "best_score": entry["best_score"],
                "k": entry.get("k"),
                "timestamp": entry.get("timestamp"),
                "trajectory_count": entry.get("trajectory_count"),
                "cached": True,
                "similarity": best_sim,
                "matched_problem": entry["problem"],
            }
        return None

    # ----------------------- insert ----------------------- #
    def insert(
        self,
        problem: str,
        best_answer: str,
        best_score: float,
        k: int,
        trajectory_count: int,
        verified: bool = False,
        verification_method: str = "self_claims_only",
        claim_count: int = 0,
        answer_present: bool = True,
        deterministic_check: Optional[bool] = None,
        failure: Optional[str] = None,
        transport_failures: int = 0,
        model_failures: int = 0,
    ) -> None:
        """Insert a CLR result into the semantic cache.

        Enriched entry format (per audit requirements):
          - verified: whether the answer was independently verified
          - verification_method: how it was verified (python_eval, unit_tests,
            retrieval, or self_claims_only)
          - claim_count: number of meaningful claims that were scored
          - answer_present: whether a final answer was produced
          - deterministic_check: result of cross-trajectory deterministic check
          - failure: None if successful, error description if failed
          - transport_failures: number of trajectories that failed at transport level
          - model_failures: number of trajectories that failed at model level
          - schema_version: cache entry format version (2)
        """
        embedding = self.model.encode([problem])[0].tolist()
        entry = {
            "problem": problem,
            "embedding": embedding,
            "best_answer": best_answer,
            "best_score": float(best_score),
            "k": int(k),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trajectory_count": int(trajectory_count),
            "verified": verified,
            "verification_method": verification_method,
            "claim_count": int(claim_count),
            "answer_present": answer_present,
            "deterministic_check": deterministic_check,
            "failure": failure,
            "transport_failures": int(transport_failures),
            "model_failures": int(model_failures),
            "schema_version": 2,
        }
        self.entries.append(entry)
        # Evict oldest low-score entries if over capacity (keep best scores)
        if len(self.entries) > self.max_entries:
            self.entries.sort(key=lambda e: e.get("best_score", 0), reverse=True)
            self.entries = self.entries[: self.max_entries]
        self._rebuild_embeddings_matrix()
        if self.autosave:
            self.save()

    def clear(self) -> None:
        self.entries = []
        self._embeddings_matrix = None
        if os.path.exists(self.path):
            os.unlink(self.path)
        print("[CLRResultCache] Cleared")

    def __len__(self) -> int:
        return len(self.entries)
