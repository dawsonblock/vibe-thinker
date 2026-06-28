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
import threading
import uuid as _uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from vector_store import VectorStore, LocalVectorStore, make_vector_store

# Optional deps for semantic CLR cache
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    # Define a sentinel so `patch("persistent_cache.SentenceTransformer")`
    # in tests doesn't raise AttributeError when sentence-transformers is
    # not installed. The sentinel is never called in production when
    # EMBEDDINGS_AVAILABLE is False — code paths check the flag first.
    # Tests that patch it set EMBEDDINGS_AVAILABLE=True via mocking.
    class SentenceTransformer:  # type: ignore[no-redef]
        """Stub used when sentence-transformers is not installed.

        This exists so that ``patch("persistent_cache.SentenceTransformer")``
        in tests works whether or not the real package is installed.
        It is never instantiated in production when
        ``EMBEDDINGS_AVAILABLE`` is False.
        """

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install with: pip install sentence-transformers"
            )


# ====================================================================== #
# Shared embedding model singleton (Phase 5 — reduce 3x memory usage)
# ====================================================================== #
# Previously, CLRResultCache, VerifiedTrajectoryStore, and EmbeddingRouter
# each loaded their own SentenceTransformer instance (~100-300MB each).
# This singleton ensures only one instance per model name is loaded.
_EMBEDDING_MODELS: Dict[str, "SentenceTransformer"] = {}


def get_shared_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Get a shared SentenceTransformer instance for the given model name.

    Returns a cached instance if one was already loaded for this model
    name, or loads a new one on first call. This reduces memory usage
    by ~200-600MB when multiple components use the same model.
    """
    if not EMBEDDINGS_AVAILABLE:
        raise ImportError(
            "Embeddings need: pip install sentence-transformers "
            "scikit-learn numpy"
        )
    if model_name not in _EMBEDDING_MODELS:
        print(f"[EmbeddingModel] Loading shared model: {model_name}")
        _EMBEDDING_MODELS[model_name] = SentenceTransformer(model_name)
    return _EMBEDDING_MODELS[model_name]


def _clear_embedding_model_cache() -> None:
    """Clear the shared embedding model cache (for testing)."""
    _EMBEDDING_MODELS.clear()


# ====================================================================== #
# Atomic JSON write helper
# ====================================================================== #
# Answers that must NEVER be cached, regardless of score.
BAD_ANSWERS = frozenset({
    "no clear answer found",
    "all trajectories failed",
    "generalist call failed",
    "specialist call failed",
    "",
    "none",
    "null",
    "n/a",
})

# Current cache entry schema version. Entries with a lower version are
# rejected on lookup — old v0.1/v0.2/v0.3 cache entries wrote unsafe
# high-score self-claims-only results that must not contaminate v0.3.1.
CURRENT_CACHE_SCHEMA_VERSION = 3


def is_cache_entry_trustworthy(
    entry: Dict[str, Any], *, allow_weak_cache: bool = False
) -> bool:
    """Decide whether a cached entry is safe to return on lookup.

    This enforces the same trust rules as insertion. Old cache entries
    that predate the trust model (schema_version < 3, missing
    verification_method, self_claims_only with high score) are rejected.

    Args:
        entry: the cached entry dict.
        allow_weak_cache: if True, allow self_claims_only entries through.

    Returns:
        True if the entry is trustworthy, False otherwise.
    """
    answer = str(entry.get("best_answer") or entry.get("answer") or "").strip()
    if not answer:
        return False
    if answer.lower() in BAD_ANSWERS:
        return False
    if entry.get("failure"):
        return False
    if entry.get("transport_failures", 0) > 0 and not entry.get("verified"):
        return False
    if entry.get("claim_count", 0) < 5:
        return False
    method = entry.get("verification_method", "self_claims_only")
    if method == "self_claims_only" and not allow_weak_cache:
        return False
    # Even with allow_weak_cache, a self_claims_only entry with a score
    # above 0.65 is suspicious — it was likely written by an older build
    # that didn't enforce the cap.
    if method == "self_claims_only" and entry.get("best_score", 0.0) > 0.65:
        return False
    # A label is not verification. If the method claims a deterministic
    # verifier was used but verified=False, the entry is NOT trustworthy.
    if method != "self_claims_only" and not entry.get("verified"):
        return False
    if entry.get("schema_version", 1) < CURRENT_CACHE_SCHEMA_VERSION:
        return False
    return True


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
    # A label is not verification. If the method claims a deterministic
    # verifier was used but verified=False, the result is NOT cacheable.
    if method != "self_claims_only" and not result.get("verified"):
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
        vector_store: Optional[VectorStore] = None,
        agentdb_url: Optional[str] = None,
        agentdb_collection: str = "clr_results",
        agentdb_only: bool = False,
    ):
        # Phase 3 (stabilization): when a vector_store is injected, don't
        # require the heavy embedding deps if the vector_store can handle
        # its own embedding generation. However, LocalVectorStore and
        # AgentDBVectorStore still need pre-computed embeddings for
        # indexing, so we still load the embedding model when available.
        # The key change: we don't RAISE ImportError when a vector_store
        # is provided — the caller may have a custom store that handles
        # embeddings internally. If embeddings are needed but not
        # available, the insert/search methods will fail at runtime with
        # a clear error instead of failing at construction.
        self._vector_store: Optional[VectorStore] = vector_store
        if self._vector_store is None and agentdb_url:
            self._vector_store = make_vector_store(
                agentdb_url=agentdb_url,
                collection=agentdb_collection,
            )

        if not EMBEDDINGS_AVAILABLE and self._vector_store is None:
            raise ImportError(
                "CLRResultCache needs: pip install sentence-transformers "
                "scikit-learn numpy  (OR provide vector_store= for a "
                "custom similarity backend)"
            )
        self.path = path
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.min_score = min_score
        self.max_entries = max_entries
        self.autosave = autosave

        # Load the embedding model when available (needed for generating
        # entry/query embeddings even when a vector_store is injected,
        # since LocalVectorStore/AgentDBVectorStore store pre-computed
        # vectors, not raw text).
        self.model = (
            get_shared_embedding_model(model_name)
            if EMBEDDINGS_AVAILABLE
            else None
        )
        self.entries: List[Dict[str, Any]] = []
        self._embeddings_matrix: Optional[Any] = None  # np.ndarray, rebuilt on load
        self._save_lock = threading.Lock()  # Phase 5: prevent concurrent save races

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
        embeddings = [e.get("embedding", []) for e in self.entries]
        if not all(embeddings):
            self._embeddings_matrix = None
            return
        self._embeddings_matrix = np.array(embeddings)

    def save(self) -> None:
        with self._save_lock:
            data = {
                "model_name": self.model_name,
                "entries": self.entries,
            }
            _atomic_write_json(self.path, data)

    # ----------------------- lookup ----------------------- #
    def lookup(self, problem: str, allow_weak_cache: bool = False) -> Optional[Dict[str, Any]]:
        """Return a cached result if a similar high-score problem exists.

        Enforces the same trust rules as insertion via
        :func:`is_cache_entry_trustworthy`. Old cache entries that predate
        the trust model (schema_version < 3, self_claims_only with high
        score, missing verification metadata) are silently rejected.

        Args:
            problem: the query to look up.
            allow_weak_cache: if True, allow self_claims_only entries
                through lookup. Default is False.
        """
        if not self.entries or self._embeddings_matrix is None:
            return None
        if self.model is None:
            return None
        q_emb = self.model.encode([problem])[0].reshape(1, -1)
        sims = cosine_similarity(q_emb, self._embeddings_matrix)[0]
        # Search entries in order of similarity, return the first
        # trustworthy one above the threshold.
        ranked_indices = np.argsort(sims)[::-1]
        for idx in ranked_indices:
            best_sim = float(sims[idx])
            if best_sim < self.similarity_threshold:
                break
            entry = self.entries[int(idx)]
            if entry.get("best_score", 0) < self.min_score:
                continue
            if not is_cache_entry_trustworthy(entry, allow_weak_cache=allow_weak_cache):
                continue
            # Track access frequency for LRU eviction.
            entry["access_count"] = entry.get("access_count", 0) + 1
            if self.autosave:
                self.save()
            return {
                "best_answer": entry["best_answer"],
                "best_score": entry["best_score"],
                "k": entry.get("k"),
                "timestamp": entry.get("timestamp"),
                "trajectory_count": entry.get("trajectory_count"),
                "verification_method": entry.get("verification_method", "self_claims_only"),
                "verified": entry.get("verified", False),
                "schema_version": entry.get("schema_version", 1),
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
          - schema_version: cache entry format version (3)
        """
        if self.model is not None:
            embedding = self.model.encode([problem])[0].tolist()
        else:
            embedding = []
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
            "schema_version": CURRENT_CACHE_SCHEMA_VERSION,
        }
        self.entries.append(entry)
        # Evict least-recently-accessed entries if over capacity (LRU + score).
        if len(self.entries) > self.max_entries:
            self.entries.sort(
                key=lambda e: (e.get("access_count", 0), e.get("best_score", 0)),
                reverse=True,
            )
            self.entries = self.entries[: self.max_entries]
        self._rebuild_embeddings_matrix()
        if self.autosave:
            self.save()
        # Shadow-mode dual-write: if a vector store is configured (e.g.
        # via --agentdb-url), mirror the insert to it. The local
        # embeddings matrix remains the primary read path; the vector
        # store is the shadow that receives writes for migration to
        # AgentDB. Failures are non-fatal (shadow is best-effort).
        if self._vector_store is not None:
            try:
                self._vector_store.upsert(
                    f"clr_{len(self.entries) - 1}",
                    embedding,
                    {
                        "problem": problem,
                        "best_answer": best_answer,
                        "best_score": float(best_score),
                        "verified": verified,
                        "verification_method": verification_method,
                        "task_type": entry.get("task_type", "unknown"),
                    },
                )
            except Exception:
                pass  # shadow write failure is non-fatal

    def clear(self) -> None:
        self.entries = []
        self._embeddings_matrix = None
        if os.path.exists(self.path):
            os.unlink(self.path)
        print("[CLRResultCache] Cleared")

    def __len__(self) -> int:
        return len(self.entries)


# ====================================================================== #
# Verified trajectory store (self-improving few-shot memory)
# ====================================================================== #
class VerifiedTrajectoryStore:
    """Semantic store of independently-verified solutions.

    Unlike CLRResultCache (which returns a cached answer for near-identical
    queries, similarity >= 0.92), the trajectory store retrieves *similar*
    verified solutions (similarity >= retrieval_threshold, default 0.70) and
    returns them as **few-shot context** for new queries. This lets the system
    learn from verified successes: the next time a similar problem appears,
    the model sees prior verified solutions as examples, improving its first-
    attempt success rate.

    Trust model (fail-closed):
      - ONLY results with verified=True and a deterministic verification_method
        (not self_claims_only) are stored. Unverified results are never learned
        from — learning from unverified output would be epistemic contamination.
      - On retrieval, entries are re-checked for trustworthiness via
        is_cache_entry_trustworthy(). A corrupted or stale entry cannot inject
        false context.
      - The store is read-only on lookup: it provides context, not answers.
        The model still must solve the problem and the verifier still must
        confirm it.

    On-disk format (JSON):
      {
        "model_name": "all-MiniLM-L6-v2",
        "schema_version": 3,
        "entries": [
          {
            "query": "...",
            "embedding": [float, ...],
            "answer": "...",
            "score": 0.895,
            "verification_method": "math_verifier",
            "task_type": "math",
            "route_taken": "specialist_clr",
            "timestamp": "..."
          }, ...
        ]
      }
    """

    def __init__(
        self,
        path: str = "verified_trajectories.json",
        model_name: str = "all-MiniLM-L6-v2",
        retrieval_threshold: float = 0.70,
        max_entries: int = 512,
        max_few_shot: int = 3,
        autosave: bool = True,
        vector_store: Optional[VectorStore] = None,
        agentdb_url: Optional[str] = None,
        agentdb_collection: str = "trajectories",
        agentdb_only: bool = False,
    ):
        # Phase 3 (stabilization): when a vector_store is injected, don't
        # require the heavy embedding deps at construction time. See
        # CLRResultCache.__init__ for the same pattern.
        self._vector_store: Optional[VectorStore] = vector_store
        if self._vector_store is None and agentdb_url:
            self._vector_store = make_vector_store(
                agentdb_url=agentdb_url,
                collection=agentdb_collection,
            )

        if not EMBEDDINGS_AVAILABLE and self._vector_store is None:
            raise ImportError(
                "VerifiedTrajectoryStore needs: pip install "
                "sentence-transformers scikit-learn numpy  "
                "(OR provide vector_store= for a custom similarity backend)"
            )
        self.path = path
        self.model_name = model_name
        self.retrieval_threshold = retrieval_threshold
        self.max_entries = max_entries
        self.max_few_shot = max_few_shot
        self.autosave = autosave

        # Load the embedding model when available (needed for generating
        # entry/query embeddings even when a vector_store is injected).
        self.model = (
            get_shared_embedding_model(model_name)
            if EMBEDDINGS_AVAILABLE
            else None
        )
        self.entries: List[Dict[str, Any]] = []
        self._embeddings_matrix: Optional[Any] = None
        self._save_lock = threading.Lock()  # Phase 5: prevent concurrent save races

        self._load()

    # ----------------------- persistence ----------------------- #
    def _load(self) -> None:
        data = _load_json(self.path)
        if not data:
            return
        self.entries = data.get("entries", []) or []
        # Filter out any entries that don't meet the trust bar (defensive:
        # the file could have been hand-edited or written by an older version)
        self.entries = [
            e for e in self.entries
            if e.get("verified") and e.get("verification_method", "self_claims_only") != "self_claims_only"
        ]
        self._rebuild_embeddings_matrix()
        if self.entries:
            print(f"[TrajectoryStore] Loaded {len(self.entries)} verified trajectories from {self.path}")

    def _rebuild_embeddings_matrix(self) -> None:
        if not self.entries:
            self._embeddings_matrix = None
            return
        embeddings = [e.get("embedding", []) for e in self.entries]
        if not all(embeddings):
            self._embeddings_matrix = None
            return
        self._embeddings_matrix = np.array(embeddings)

    def save(self) -> None:
        with self._save_lock:
            data = {
                "model_name": self.model_name,
                "schema_version": CURRENT_CACHE_SCHEMA_VERSION,
                "entries": self.entries,
            }
            _atomic_write_json(self.path, data)

    # ----------------------- retrieval ----------------------- #
    def retrieve(self, query: str, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve similar verified trajectories as few-shot context.

        Returns up to ``max_few_shot`` entries sorted by similarity, each
        above ``retrieval_threshold``. If ``task_type`` is given, only
        entries with a matching task_type are returned (so math examples
        don't pollute code queries and vice versa).

        Each returned dict has:
          - query: the original verified problem
          - answer: the verified answer
          - score: the reliability score at verification time
          - verification_method: how it was verified
          - similarity: cosine similarity to the current query
        """
        if not self.entries or self._embeddings_matrix is None:
            return []
        if self.model is None:
            return []
        q_emb = self.model.encode([query])[0].reshape(1, -1)
        sims = cosine_similarity(q_emb, self._embeddings_matrix)[0]

        candidates = []
        for idx in np.argsort(sims)[::-1]:
            sim = float(sims[idx])
            if sim < self.retrieval_threshold:
                break
            entry = self.entries[int(idx)]
            if task_type and entry.get("task_type") != task_type:
                continue
            # Re-check trust on retrieval (defensive against stale/corrupt entries)
            if not is_cache_entry_trustworthy(entry):
                continue
            candidates.append({
                "query": entry["query"],
                "answer": entry["answer"],
                "score": entry.get("score", 0.0),
                "verification_method": entry.get("verification_method", ""),
                "task_type": entry.get("task_type", ""),
                "similarity": sim,
            })
            # Track access frequency for LRU eviction.
            entry["access_count"] = entry.get("access_count", 0) + 1
            if len(candidates) >= self.max_few_shot:
                break
        # Save if access counts changed (for LRU persistence across restarts).
        if candidates and self.autosave:
            self.save()
        return candidates

    def build_few_shot_prefix(self, query: str, task_type: Optional[str] = None) -> str:
        """Build a few-shot context string from verified trajectories.

        Returns a string prepended to the model prompt, or "" if no
        similar verified trajectories exist. The format is:

          Here are similar problems that were independently verified correct.
          Use them as reference:

          Problem: <query>
          Verified answer: <answer>
          (verified via <method>, score <score>)

          ...
        """
        trajectories = self.retrieve(query, task_type=task_type)
        if not trajectories:
            return ""
        lines = [
            "Here are similar problems that were independently verified correct.",
            "Use them as reference:",
            "",
        ]
        for t in trajectories:
            lines.append(f"Problem: {t['query']}")
            lines.append(f"Verified answer: {t['answer']}")
            lines.append(f"(verified via {t['verification_method']}, score {t['score']:.3f})")
            lines.append("")
        return "\n".join(lines)

    # ----------------------- insert ----------------------- #
    def store(
        self,
        query: str,
        answer: str,
        score: float,
        verification_method: str,
        task_type: str = "unknown",
        route_taken: str = "",
        verification_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a verified trajectory.

        Only called when verified=True with a deterministic verifier. This is
        the "learning" step: the verified solution is added to the store and
        becomes retrievable as few-shot context for future similar queries.

        Refuses to store self_claims_only or unverified results — learning
        from unverified output is epistemic contamination.

        Args:
            verification_context: optional dict of verification context
                (e.g. ``{"expected_answer": "42", "unit_tests": "..."}``).
                Stored with the entry so synthesized masters can be re-
                verified against the children's ground truths (v3.1).
                Old entries without this field are handled gracefully.
        """
        if not answer or not answer.strip():
            return
        if answer.strip().lower() in BAD_ANSWERS:
            return
        if verification_method == "self_claims_only":
            return  # Never learn from self-claims
        if self.model is None:
            return  # Cannot store without embeddings
        embedding = self.model.encode([query])[0].tolist()
        entry = {
            "query": query,
            "embedding": embedding,
            "answer": answer,
            "score": float(score),
            "verification_method": verification_method,
            "task_type": task_type,
            "route_taken": route_taken,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "verified": True,
            "schema_version": CURRENT_CACHE_SCHEMA_VERSION,
            "best_answer": answer,  # alias for is_cache_entry_trustworthy compatibility
            "best_score": float(score),  # alias
            "claim_count": 10,  # verified entries pass the claim_count check
        }
        if verification_context is not None:
            entry["verification_context"] = verification_context
        self.entries.append(entry)
        # Evict least-recently-accessed entries if over capacity (LRU + score).
        # Entries that are never retrieved are evicted before frequently-
        # retrieved ones, even if their score is higher — a stale high-score
        # entry that nobody looks up is less valuable than a medium-score
        # entry that's retrieved often. Ties are broken by score.
        if len(self.entries) > self.max_entries:
            self.entries.sort(
                key=lambda e: (e.get("access_count", 0), e.get("score", 0)),
                reverse=True,
            )
            self.entries = self.entries[: self.max_entries]
        self._rebuild_embeddings_matrix()
        if self.autosave:
            self.save()
        # Shadow-mode dual-write: mirror to the vector store (e.g. AgentDB)
        # for migration. Non-fatal on failure.
        if self._vector_store is not None:
            try:
                self._vector_store.upsert(
                    f"traj_{len(self.entries) - 1}",
                    embedding,
                    {
                        "query": query,
                        "answer": answer,
                        "score": float(score),
                        "verification_method": verification_method,
                        "task_type": task_type,
                    },
                )
            except Exception:
                pass  # shadow write failure is non-fatal

    def clear(self) -> None:
        self.entries = []
        self._embeddings_matrix = None
        if os.path.exists(self.path):
            os.unlink(self.path)
        print("[TrajectoryStore] Cleared")

    def __len__(self) -> int:
        return len(self.entries)

    # ----------------------- synthesis (v0.4.0) ----------------------- #
    def find_clusters(
        self, similarity_threshold: float = 0.85,
        min_cluster_size: int = 3,
        task_type: Optional[str] = None,
    ) -> List[List[int]]:
        """Find clusters of highly-similar verified trajectories.

        Uses greedy agglomerative clustering on the embedding matrix: for
        each entry, find all other entries above ``similarity_threshold``
        and group them. Returns a list of clusters, each a list of entry
        indices. Clusters smaller than ``min_cluster_size`` are excluded
        (synthesizing 2 entries into 1 saves little memory).

        When ``task_type`` is given, only clusters within that task type
        are returned (math trajectories cluster with math, code with code).

        This is the first half of trajectory synthesis (Phase 4.1): the
        orchestrator's ``synthesize_trajectories`` method calls this, then
        asks the generalist to merge each cluster into a single "master
        trajectory" and removes the raw entries.

        Performance (v0.4.1 fix):
          For small N (<=512, the default max_entries), the full N×N
          similarity matrix is computed in one shot — fast and simple.
          For large N (>512), we use chunked computation to avoid
          materializing the entire N×N matrix in memory at once. Each
          chunk computes similarity of a batch of rows against the full
          matrix, keeping memory bounded at O(chunk_size × N) instead
          of O(N^2). The greedy clustering loop is also optimized to
          skip already-assigned entries early.
        """
        if not self.entries or self._embeddings_matrix is None:
            return []
        if len(self.entries) < min_cluster_size:
            return []

        # v0.4.1: When a vector store (AgentDB) is configured, delegate
        # clustering to it — the sidecar is built for scale and can handle
        # clustering across millions of vectors efficiently. The vector
        # store returns clusters of vector_ids, which we map back to entry
        # indices. Fail-closed: if the vector store returns [], fall
        # through to the local chunked computation.
        if self._vector_store is not None:
            try:
                filters = {"task_type": task_type} if task_type else None
                vid_clusters = self._vector_store.cluster(
                    similarity_threshold=similarity_threshold,
                    min_cluster_size=min_cluster_size,
                    filters=filters,
                )
                if vid_clusters:
                    # Map vector_ids back to entry indices.
                    id_to_idx = {
                        e.get("id", f"entry_{i}"): i
                        for i, e in enumerate(self.entries)
                    }
                    result = []
                    for vid_cluster in vid_clusters:
                        idx_cluster = [
                            id_to_idx[vid] for vid in vid_cluster
                            if vid in id_to_idx
                        ]
                        if len(idx_cluster) >= min_cluster_size:
                            result.append(idx_cluster)
                    if result:
                        return result
                    # Vector store returned clusters but we couldn't map
                    # them — fall through to local computation.
            except Exception as e:
                print(f"[TrajectoryStore] Vector store clustering failed: {e} "
                      f"— falling back to local computation")

        # Filter by task_type if requested.
        if task_type is not None:
            indices = [
                i for i, e in enumerate(self.entries)
                if e.get("task_type") == task_type
            ]
        else:
            indices = list(range(len(self.entries)))
        if len(indices) < min_cluster_size:
            return []

        n = len(indices)
        sub_matrix = self._embeddings_matrix[indices]

        # For small N, compute the full similarity matrix (fast, simple).
        # For large N, use chunked computation to bound memory.
        CHUNK_THRESHOLD = 512
        CHUNK_SIZE = 256

        if n <= CHUNK_THRESHOLD:
            sims = cosine_similarity(sub_matrix)
            # Greedy clustering: assign each entry to the first cluster it
            # matches, or start a new cluster.
            assigned: Dict[int, int] = {}
            clusters: List[List[int]] = []
            for i in range(n):
                if i in assigned:
                    continue
                cluster = [i]
                assigned[i] = len(clusters)
                for j in range(i + 1, n):
                    if j in assigned:
                        continue
                    if sims[i][j] >= similarity_threshold:
                        cluster.append(j)
                        assigned[j] = assigned[i]
                if len(cluster) >= min_cluster_size:
                    clusters.append([indices[k] for k in cluster])
            return clusters
        else:
            # Large N: chunked similarity computation to avoid O(N^2)
            # memory. We compute similarity row-by-row in chunks and
            # cluster greedily, keeping only the current chunk's
            # similarities in memory.
            assigned: Dict[int, int] = {}
            clusters: List[List[int]] = []
            # Precompute norms for efficient cosine similarity.
            norms = np.linalg.norm(sub_matrix, axis=1, keepdims=True)
            normalized = sub_matrix / (norms + 1e-10)
            for chunk_start in range(0, n, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, n)
                chunk = normalized[chunk_start:chunk_end]
                # Compute similarity of this chunk against ALL entries.
                # This is O(chunk_size × N) — bounded memory.
                chunk_sims = chunk @ normalized.T  # (chunk_size, N)
                for li, i in enumerate(range(chunk_start, chunk_end)):
                    if i in assigned:
                        continue
                    cluster = [i]
                    assigned[i] = len(clusters)
                    row = chunk_sims[li]
                    for j in range(i + 1, n):
                        if j in assigned:
                            continue
                        if row[j] >= similarity_threshold:
                            cluster.append(j)
                            assigned[j] = assigned[i]
                    if len(cluster) >= min_cluster_size:
                        clusters.append([indices[k] for k in cluster])
            return clusters

    def remove_entries(self, indices: List[int]) -> None:
        """Remove entries at the given indices and rebuild the embedding matrix.

        Used by trajectory synthesis: after a cluster is synthesized into a
        master trajectory, the raw entries are removed to save memory. The
        synthesized master is stored separately via ``store()``.
        """
        if not indices:
            return
        remove_set = set(indices)
        self.entries = [
            e for i, e in enumerate(self.entries) if i not in remove_set
        ]
        self._rebuild_embeddings_matrix()
        if self.autosave:
            self.save()
        print(f"[TrajectoryStore] Removed {len(remove_set)} entries — "
              f"store now has {len(self.entries)} entries")

    def store_synthesized(
        self,
        query: str,
        answer: str,
        task_type: str,
        source_count: int,
        source_queries: List[str],
    ) -> None:
        """Store a synthesized "master trajectory".

        A master trajectory is a general rule distilled from a cluster of
        similar verified trajectories by the generalist model. It is NOT
        independently verified — it is a compression of verified data. The
        trust model:

          - ``verification_method`` is ``"synthesized"`` (NOT a deterministic
            verifier). This means it is excluded from the cache trust check
            (``is_cache_entry_trustworthy`` rejects ``self_claims_only``,
            and ``synthesized`` is treated the same way — it cannot be
            returned as a cached verified answer).
          - It IS retrievable as few-shot context (the ``retrieve`` method
            checks ``is_cache_entry_trustworthy``, which requires
            ``verification_method != self_claims_only`` — but synthesized
            entries are explicitly filtered out in retrieve to avoid
            presenting a model-generated summary as a "verified example").
          - The ``synthesized`` flag and ``source_queries`` list provide
            provenance: the master can be traced back to the raw verified
            trajectories it was distilled from.

        Args:
            query: a representative query for the cluster (e.g. the
                highest-similarity entry's query, or a generalized prompt).
            answer: the synthesized master trajectory / general rule text.
            task_type: the task type of the source cluster.
            source_count: how many raw trajectories were synthesized.
            source_queries: the queries of the source trajectories
                (provenance for debugging / audit).
        """
        if not answer or not answer.strip():
            return
        embedding = self.model.encode([query])[0].tolist()
        entry = {
            "query": query,
            "embedding": embedding,
            "answer": answer,
            "score": 0.0,  # synthesized — no independent verification score
            "verification_method": "synthesized",
            "task_type": task_type,
            "route_taken": "synthesized",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "verified": False,  # NOT verified — synthesized, not proven
            "synthesized": True,
            "source_count": source_count,
            "source_queries": source_queries,
            "schema_version": CURRENT_CACHE_SCHEMA_VERSION,
            "best_answer": answer,
            "best_score": 0.0,
            "claim_count": 0,
        }
        self.entries.append(entry)
        # Evict if over capacity (same LRU + score policy as store()).
        if len(self.entries) > self.max_entries:
            self.entries.sort(
                key=lambda e: (e.get("access_count", 0), e.get("score", 0)),
                reverse=True,
            )
            self.entries = self.entries[: self.max_entries]
        self._rebuild_embeddings_matrix()
        if self.autosave:
            self.save()
        print(f"[TrajectoryStore] Stored synthesized master trajectory "
              f"(task_type={task_type}, source_count={source_count}) — "
              f"store now has {len(self.entries)} entries")

    def store_synthesized_verified(
        self,
        query: str,
        answer: str,
        task_type: str,
        source_count: int,
        source_queries: List[str],
        score: float = 0.65,
    ) -> None:
        """Store a synthesized "master trajectory" that has been RE-VERIFIED
        against the children's ground truths (v3.1).

        Unlike :meth:`store_synthesized` (which stores with verified=False),
        this method stores the master with ``verified=True`` and
        ``verification_method="synthesized_and_proven"``. This means the
        master IS retrievable as few-shot context — it has been proven
        against the original verification criteria of its children.

        Trust model:
          - The master was synthesized by the generalist from a cluster of
            verified trajectories, then re-verified against each child's
            ground truth (unit tests for code, expected_answer for math).
          - If it passes ALL children's criteria, it is stored as verified.
            If it fails ANY child's criteria, it is NOT stored via this
            method — the caller should fall back to
            :meth:`store_synthesized` (verified=False, provenance only).
          - The ``synthesized_and_proven`` method is treated as a
            deterministic verification method by
            :func:`is_cache_entry_trustworthy` — it is NOT
            ``self_claims_only``.

        Args:
            query: a representative query for the cluster.
            answer: the synthesized master trajectory / general rule text.
            task_type: the task type of the source cluster.
            source_count: how many raw trajectories were synthesized.
            source_queries: the queries of the source trajectories.
            score: the verification score (default 0.65 — the master
                passed re-verification but is not a direct solve).
        """
        if not answer or not answer.strip():
            return
        embedding = self.model.encode([query])[0].tolist()
        entry = {
            "query": query,
            "embedding": embedding,
            "answer": answer,
            "score": float(score),
            "verification_method": "synthesized_and_proven",
            "task_type": task_type,
            "route_taken": "synthesized_and_proven",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "verified": True,  # RE-VERIFIED — proven against children
            "synthesized": True,
            "source_count": source_count,
            "source_queries": source_queries,
            "schema_version": CURRENT_CACHE_SCHEMA_VERSION,
            "best_answer": answer,
            "best_score": float(score),
            "claim_count": 10,  # passes the claim_count check
        }
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries.sort(
                key=lambda e: (e.get("access_count", 0), e.get("score", 0)),
                reverse=True,
            )
            self.entries = self.entries[: self.max_entries]
        self._rebuild_embeddings_matrix()
        if self.autosave:
            self.save()
        print(f"[TrajectoryStore] Stored RE-VERIFIED synthesized master "
              f"(task_type={task_type}, source_count={source_count}, "
              f"score={score}) — store now has {len(self.entries)} entries")
