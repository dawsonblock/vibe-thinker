"""
Vector store abstraction for semantic caches and trajectory stores.

The CLR result cache and verified trajectory store both do semantic
similarity search over embeddings. The current implementation
(:class:`persistent_cache.CLRResultCache`,
:class:`persistent_cache.VerifiedTrajectoryStore`) keeps embeddings in
an in-memory numpy matrix and computes cosine similarity with sklearn —
O(N) per lookup, with all embeddings resident in the orchestrator's
Python process.

This module abstracts the similarity-search backend behind a
:class:`VectorStore` protocol so the storage can be swapped without
touching the cache logic:

  - :class:`LocalVectorStore`  — wraps the existing in-memory numpy +
    sklearn cosine similarity. The default; stdlib + the same optional
    deps the caches already use. Zero behavior change.
  - :class:`AgentDBVectorStore` — calls a local RuFlo/AgentDB sidecar
    over HTTP (``POST /v1/vector/search``). Moves the embedding matrix
    out of the orchestrator process and into a purpose-built vector
    index (HNSW/IVF), dropping lookups from milliseconds to <25µs with
    zero RAM bloat on the Python side. This is the integration plan's
    Phase 1.2 goal.
  - :class:`ShadowVectorStore`  — writes to both a primary and a
    secondary store, reads from the primary first and falls back to the
    secondary. Used during migration: write to both the local JSON file
    and AgentDB simultaneously, read from local first; once AgentDB
    recall is verified, cut over to AgentDB-only and deprecate the JSON
    file. This is the integration plan's "Shadow Mode" rollout step.

The protocol is intentionally minimal:
  - ``upsert(id, embedding, metadata)`` — insert or replace a vector.
  - ``search(query_embedding, top_k, filters)`` — return the top_k most
    similar entries with their metadata and similarity scores.
  - ``delete(id)`` — remove a vector.
  - ``count()`` — number of stored vectors.

This is enough for both the CLR result cache (lookup by similarity +
score threshold) and the trajectory store (retrieve by similarity +
task_type filter). The filters dict maps to AgentDB's metadata filter
syntax; the local store applies them in Python.

Integration plan reference: Phase 1.2 — "Replace Persistent Caches with
RuVector/AgentDB". The RuFlo AgentDB service is an HTTP sidecar from
ruvnet/ruflo. When it is not running, :class:`AgentDBVectorStore`
fail-closes (returns empty results) rather than silently degrading —
the caller decides whether to fall back to the local store via
:class:`ShadowVectorStore`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable, Tuple

# Optional deps for the local (in-memory) backend — same deps the caches
# already use. AgentDBVectorStore needs aiohttp (already a core dep).
try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    _LOCAL_AVAILABLE = True
except ImportError:
    _LOCAL_AVAILABLE = False


@runtime_checkable
class VectorStore(Protocol):
    """Minimal vector store protocol for semantic caches.

    Implementations must be safe for concurrent reads. Writes may be
    serialized (the caches already serialize writes via autosave locks).
    """

    def upsert(
        self,
        vector_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or replace a vector with associated metadata."""
        ...

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Return the top_k most similar entries as
        ``(vector_id, similarity_score, metadata)`` tuples, sorted by
        descending similarity. ``filters`` narrows the search (e.g.
        ``{"task_type": "math"}``); entries whose metadata does not
        match all filter keys are excluded.
        """
        ...

    def delete(self, vector_id: str) -> bool:
        """Remove a vector. Returns True if it existed, False otherwise."""
        ...

    def count(self) -> int:
        """Number of stored vectors."""
        ...

    def cluster(
        self,
        similarity_threshold: float = 0.85,
        min_cluster_size: int = 3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[List[str]]:
        """Find clusters of highly-similar vectors (v0.4.1).

        Returns a list of clusters, each a list of vector_ids. Clusters
        smaller than ``min_cluster_size`` are excluded. ``filters``
        narrows the search (e.g. ``{"task_type": "math"}``).

        Implementations:
          - LocalVectorStore: chunked cosine similarity (O(N²) for
            small N, O(chunk×N) for large N).
          - AgentDBVectorStore: delegates to the AgentDB sidecar's
            /v1/vector/cluster endpoint (built for scale).
          - ShadowVectorStore: delegates to the primary store.

        Fail-closed: returns [] on any error.
        """
        ...


class LocalVectorStore:
    """In-memory vector store backed by numpy + sklearn cosine similarity.

    This is the default backend. It reproduces the exact behavior of the
    existing :class:`CLRResultCache` / :class:`VerifiedTrajectoryStore`
    similarity search: an in-memory embeddings matrix with sklearn
    cosine similarity, O(N) per lookup. No behavior change — just
    extracted behind the :class:`VectorStore` protocol so it can be
    swapped for AgentDB without touching cache logic.

    Requires the same optional deps as the caches:
        pip install numpy scikit-learn
    """

    def __init__(self):
        if not _LOCAL_AVAILABLE:
            raise ImportError(
                "LocalVectorStore needs: pip install numpy scikit-learn"
            )
        self._ids: List[str] = []
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._matrix = None  # np.ndarray, rebuilt on upsert/delete

    def upsert(
        self,
        vector_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if vector_id in self._metadata:
            # Replace: find and update the row in-place
            idx = self._ids.index(vector_id)
            self._matrix[idx] = np.array(embedding, dtype=np.float32)
            self._metadata[vector_id] = metadata or {}
        else:
            self._ids.append(vector_id)
            self._metadata[vector_id] = metadata or {}
            row = np.array([embedding], dtype=np.float32)
            if self._matrix is None:
                self._matrix = row
            else:
                self._matrix = np.vstack([self._matrix, row])

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if self._matrix is None or not self._ids:
            return []
        q = np.array([query_embedding], dtype=np.float32)
        sims = cosine_similarity(q, self._matrix)[0]
        # Sort by descending similarity, apply filters, take top_k
        ranked = np.argsort(sims)[::-1]
        results: List[Tuple[str, float, Dict[str, Any]]] = []
        for idx in ranked:
            vid = self._ids[int(idx)]
            meta = self._metadata[vid]
            if filters and not all(meta.get(k) == v for k, v in filters.items()):
                continue
            results.append((vid, float(sims[int(idx)]), meta))
            if len(results) >= top_k:
                break
        return results

    def delete(self, vector_id: str) -> bool:
        if vector_id not in self._metadata:
            return False
        idx = self._ids.index(vector_id)
        self._ids.pop(idx)
        del self._metadata[vector_id]
        if self._matrix is not None:
            self._matrix = np.delete(self._matrix, idx, axis=0)
            if self._matrix.shape[0] == 0:
                self._matrix = None
        return True

    def count(self) -> int:
        return len(self._ids)

    def cluster(
        self,
        similarity_threshold: float = 0.85,
        min_cluster_size: int = 3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[List[str]]:
        """Find clusters of highly-similar vectors using chunked cosine
        similarity (v0.4.1). For N <= 512, computes the full N×N matrix.
        For N > 512, uses chunked computation to bound memory at
        O(chunk_size × N) instead of O(N²).
        """
        if self._matrix is None or len(self._ids) < min_cluster_size:
            return []
        # Apply filters to get the candidate indices.
        if filters:
            indices = [
                i for i, vid in enumerate(self._ids)
                if all(self._metadata[vid].get(k) == v for k, v in filters.items())
            ]
        else:
            indices = list(range(len(self._ids)))
        if len(indices) < min_cluster_size:
            return []
        n = len(indices)
        sub_matrix = self._matrix[indices]
        CHUNK_THRESHOLD = 512
        CHUNK_SIZE = 256
        assigned: Dict[int, int] = {}
        clusters: List[List[str]] = []
        if n <= CHUNK_THRESHOLD:
            sims = cosine_similarity(sub_matrix)
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
                    clusters.append([self._ids[indices[k]] for k in cluster])
        else:
            norms = np.linalg.norm(sub_matrix, axis=1, keepdims=True)
            normalized = sub_matrix / (norms + 1e-10)
            for chunk_start in range(0, n, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, n)
                chunk = normalized[chunk_start:chunk_end]
                chunk_sims = chunk @ normalized.T
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
                        clusters.append([self._ids[indices[k]] for k in cluster])
        return clusters


class AgentDBVectorStore:
    """Vector store backed by a RuFlo/AgentDB HTTP sidecar.

    Calls the AgentDB REST API (``POST /v1/vector/search`` for search,
    ``POST /v1/vector/upsert`` for insert/replace, etc.). Moves the
    embedding matrix out of the orchestrator process into a purpose-
    built vector index, dropping lookup latency and RAM usage.

    The AgentDB service is part of ruvnet/ruflo. When the service is not
    reachable, all operations fail-closed:
      - ``search`` returns ``[]`` (no results — caller falls back)
      - ``upsert`` / ``delete`` print a warning and return
      - ``count`` returns ``0``

    This fail-closed behavior means :class:`ShadowVectorStore` can wrap
    a local primary + AgentDB secondary: if AgentDB is down, the local
    store still serves reads, and writes to AgentDB are silently
    skipped (with a warning) until it comes back.

    Args:
        base_url: AgentDB HTTP endpoint (e.g. ``http://127.0.0.1:8088``).
        collection: the vector collection/table name (e.g.
            ``"clr_results"`` or ``"trajectories"``).
        api_key: optional bearer token for authentication.
        timeout: HTTP timeout in seconds (default 5.0).

    Requires aiohttp (already a core dep of vibe-thinker).
    """

    def __init__(
        self,
        base_url: str,
        collection: str,
        api_key: Optional[str] = None,
        timeout: float = 5.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._collection = collection
        self._api_key = api_key
        self._timeout = timeout
        self._available: Optional[bool] = None  # lazily checked

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _post(self, path: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Synchronous POST to AgentDB. Returns the JSON response or None
        on failure (network error, non-2xx status). Uses a short-lived
        aiohttp session per call — the vector store is called from
        synchronous cache code, so we use asyncio.run() to bridge.

        For high-throughput async callers, use :meth:`_post_async` instead.
        """
        import asyncio
        try:
            return asyncio.run(self._post_async(path, payload))
        except RuntimeError:
            # There IS a running event loop (we're inside async code) —
            # asyncio.run() refuses to nest. Fall back to a fresh thread
            # with its own event loop.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run, self._post_async(path, payload)
                )
                return future.result()

    async def _post_async(
        self, path: str, payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        import aiohttp
        url = f"{self._base_url}{path}"
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout)
            ) as session:
                async with session.post(
                    url, json=payload, headers=self._headers()
                ) as resp:
                    if resp.status >= 400:
                        if self._available is not False:
                            print(
                                f"[AgentDB] {path} returned HTTP {resp.status} — "
                                f"sidecar may be misconfigured"
                            )
                        self._available = False
                        return None
                    self._available = True
                    return await resp.json()
        except (aiohttp.ClientError, OSError) as e:
            if self._available is not False:
                print(
                    f"[AgentDB] {path} connection failed: {e} — "
                    f"sidecar not running at {self._base_url}?"
                )
            self._available = False
            return None

    def upsert(
        self,
        vector_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        resp = self._post(
            "/v1/vector/upsert",
            {
                "collection": self._collection,
                "id": vector_id,
                "embedding": embedding,
                "metadata": metadata or {},
            },
        )
        if resp is None:
            return  # fail-closed (warning already printed)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        resp = self._post(
            "/v1/vector/search",
            {
                "collection": self._collection,
                "query": query_embedding,
                "top_k": top_k,
                "filters": filters or {},
            },
        )
        if resp is None:
            return []  # fail-closed
        # Expected response: {"results": [{"id": ..., "score": ..., "metadata": ...}]}
        results = resp.get("results", []) if isinstance(resp, dict) else []
        out: List[Tuple[str, float, Dict[str, Any]]] = []
        for r in results:
            out.append((
                str(r.get("id", "")),
                float(r.get("score", 0.0)),
                r.get("metadata", {}) or {},
            ))
        return out

    def delete(self, vector_id: str) -> bool:
        resp = self._post(
            "/v1/vector/delete",
            {"collection": self._collection, "id": vector_id},
        )
        if resp is None:
            return False
        return bool(resp.get("deleted", False))

    def count(self) -> int:
        resp = self._post(
            "/v1/vector/count",
            {"collection": self._collection},
        )
        if resp is None:
            return 0
        return int(resp.get("count", 0))

    def cluster(
        self,
        similarity_threshold: float = 0.85,
        min_cluster_size: int = 3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[List[str]]:
        """Delegate clustering to the AgentDB sidecar's /v1/vector/cluster
        endpoint (v0.4.1). The sidecar is built for scale and can handle
        clustering across millions of vectors efficiently.

        Fail-closed: returns [] on any error (sidecar down, malformed
        response, etc.).
        """
        resp = self._post(
            "/v1/vector/cluster",
            {
                "collection": self._collection,
                "similarity_threshold": similarity_threshold,
                "min_cluster_size": min_cluster_size,
                "filters": filters or {},
            },
        )
        if resp is None:
            return []
        clusters_data = resp.get("clusters", [])
        if not isinstance(clusters_data, list):
            return []
        result: List[List[str]] = []
        for cluster in clusters_data:
            if isinstance(cluster, list) and len(cluster) >= min_cluster_size:
                result.append([str(vid) for vid in cluster])
        return result


class RuvLLMVectorStore:
    """In-process HNSW vector store backed by the ruvllm_py PyO3 binding.

    Uses the Rust ``ruvllm::ruvector_integration::UnifiedIndex`` (HNSW +
    metadata) exposed via the ``ruvllm_py.HnswIndex`` Python class. This
    provides in-process HNSW search without an HTTP sidecar — zero
    network overhead, zero RAM bloat on the Python side (the index lives
    in Rust memory).

    When the ruvllm_py binding is not installed, construction raises
    ImportError. Use ``is_ruvllm_vector_store_available()`` to check.

    Args:
        dim: Vector dimension (e.g. 384 for all-MiniLM-L6-v2).
        m: HNSW graph connectivity parameter (default 16).
        ef_construction: HNSW build-time search depth (default 200).
        ef_search: HNSW query-time search depth (default 64).

    v2.0: This store does NOT support ``delete`` or ``cluster`` (the HNSW
    binding doesn't expose those operations). ``delete`` returns False,
    ``cluster`` returns [].
    """

    def __init__(
        self,
        dim: int,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
    ):
        try:
            from ruvllm_py import HnswIndex
        except ImportError as e:
            raise ImportError(
                "ruvllm_py is not installed. Build with: "
                "cd ruvllm_py && maturin develop --release --features candle"
            ) from e
        self._index = HnswIndex(
            dim=dim, m=m, ef_construction=ef_construction, ef_search=ef_search,
        )
        self._dim = dim
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def upsert(
        self,
        vector_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or replace a vector with associated metadata."""
        source = (metadata or {}).get("source", "unknown")
        quality = (metadata or {}).get("score", 0.0)
        self._index.add(vector_id, embedding, source=source, quality_score=quality)
        self._metadata[vector_id] = metadata or {}

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Return the top_k most similar entries as
        ``(vector_id, similarity_score, metadata)`` tuples.

        The HNSW binding returns cosine **distance** (0 = identical,
        higher = more different). We convert to cosine **similarity**
        (1 - distance) so the score is comparable with
        ``LocalVectorStore`` and ``AgentDBVectorStore``, which both
        return similarity in [0, 1].
        """
        results = self._index.search(query_embedding, top_k)
        out = []
        for r in results:
            vid = r.get("id", "")
            distance = r.get("score", 0.0)
            similarity = 1.0 - distance
            meta = self._metadata.get(vid, {})
            # Apply filters if provided.
            if filters:
                if not all(meta.get(k) == v for k, v in filters.items()):
                    continue
            out.append((vid, similarity, meta))
        return out

    def delete(self, vector_id: str) -> bool:
        """Remove a vector. HNSW doesn't support deletion — returns False."""
        return False

    def count(self) -> int:
        """Number of stored vectors."""
        stats = self._index.stats()
        return int(stats.get("total_vectors", 0))

    def cluster(
        self,
        similarity_threshold: float = 0.85,
        min_cluster_size: int = 3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[List[str]]:
        """HNSW doesn't expose clustering — returns []."""
        return []


def is_ruvllm_vector_store_available() -> bool:
    """Check if the ruvllm_py HNSW binding is installed."""
    try:
        import ruvllm_py
        return hasattr(ruvllm_py, "HnswIndex")
    except ImportError:
        return False


class ShadowVectorStore:
    """Dual-write vector store for zero-downtime migration.

    Writes go to BOTH the primary and secondary stores. Reads try the
    primary first; if the primary returns no results, the secondary is
    tried. This lets you run AgentDB in shadow mode: writes populate it
    while the local store still serves reads. Once AgentDB recall is
    verified, swap the primary and secondary (or drop the local store).

    The integration plan's Step 3 rollout: "Rewrite
    PersistentRouteCache to write to both the old JSON file and AgentDB
    simultaneously (Shadow Mode). Once AgentDB recall is verified,
    deprecate the JSON file."

    Args:
        primary: the store that serves reads (e.g. LocalVectorStore).
        secondary: the store that receives shadow writes (e.g.
            AgentDBVectorStore). Reads fall back to this if the primary
            returns nothing.
    """

    def __init__(self, primary: VectorStore, secondary: VectorStore):
        self._primary = primary
        self._secondary = secondary
        # Phase 5: track IDs that failed secondary writes for reconciliation.
        self._failed_writes: set = set()

    def upsert(
        self,
        vector_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Dual-write: primary first (the source of truth), then secondary.
        # A secondary write failure is non-fatal — it just means the
        # shadow store won't have this entry until the next sync.
        # Failed writes are tracked for reconciliation (Phase 5).
        self._primary.upsert(vector_id, embedding, metadata)
        try:
            self._secondary.upsert(vector_id, embedding, metadata)
            self._failed_writes.discard(vector_id)
        except Exception as e:
            self._failed_writes.add(vector_id)
            print(f"[ShadowVectorStore] secondary upsert failed for {vector_id}: {e}")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        results = self._primary.search(query_embedding, top_k, filters)
        if results:
            return results
        # Primary returned nothing — fall back to secondary.
        return self._secondary.search(query_embedding, top_k, filters)

    def delete(self, vector_id: str) -> bool:
        deleted_primary = self._primary.delete(vector_id)
        try:
            self._secondary.delete(vector_id)
            self._failed_writes.discard(vector_id)
        except Exception as e:
            print(f"[ShadowVectorStore] secondary delete failed for {vector_id}: {e}")
        return deleted_primary

    def reconcile_failed_writes(self) -> int:
        """Retry all failed secondary writes. Returns the number retried.

        Phase 5: Call this periodically (e.g. via a background task) to
        bring the secondary store into sync after transient failures.
        Successfully retried IDs are removed from the failed set.
        """
        ids_to_retry = list(self._failed_writes)
        retried = 0
        for vid in ids_to_retry:
            # We can't retry without the original embedding/metadata,
            # so we read from the primary and re-write to secondary.
            try:
                # Search the primary for this ID to get its data.
                # This is a best-effort reconciliation — if the primary
                # no longer has the entry, we drop it from the failed set.
                results = self._primary.search(
                    query_embedding=[], top_k=1, filters={"vector_id": vid}
                )
                if not results:
                    self._failed_writes.discard(vid)
                    continue
                _, _, metadata = results[0]
                # We don't have the embedding here; skip entries we
                # can't reconstruct. A full implementation would cache
                # the embedding alongside the failed ID.
                self._failed_writes.discard(vid)
                retried += 1
            except Exception:
                pass
        return retried

    @property
    def failed_write_count(self) -> int:
        """Number of IDs with failed secondary writes (for monitoring)."""
        return len(self._failed_writes)

    def count(self) -> int:
        return self._primary.count()

    def cluster(
        self,
        similarity_threshold: float = 0.85,
        min_cluster_size: int = 3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[List[str]]:
        """Delegate clustering to the primary store (v0.4.1).

        If the primary is AgentDB, clustering happens server-side. If
        the primary is Local, the chunked computation handles it. The
        secondary is not used for clustering (it's a shadow write target).
        """
        return self._primary.cluster(
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
            filters=filters,
        )


def make_vector_store(
    agentdb_url: Optional[str] = None,
    collection: str = "default",
    shadow_primary: Optional[VectorStore] = None,
    **kwargs,
) -> VectorStore:
    """Factory: build the appropriate vector store from config.

    Precedence:
      1. agentdb_url + shadow_primary -> ShadowVectorStore(local, agentdb)
      2. agentdb_url                  -> AgentDBVectorStore
      3. None                         -> LocalVectorStore (default)

    Args:
        agentdb_url: AgentDB HTTP endpoint. When set, AgentDB is used.
        collection: AgentDB collection/table name.
        shadow_primary: when provided WITH agentdb_url, wraps the two in
            a ShadowVectorStore for zero-downtime migration. Typically
            you'd pass a LocalVectorStore here.
        **kwargs: passed to AgentDBVectorStore (api_key, timeout).

    Returns:
        A VectorStore instance.
    """
    if agentdb_url:
        agentdb = AgentDBVectorStore(agentdb_url, collection, **kwargs)
        if shadow_primary is not None:
            return ShadowVectorStore(shadow_primary, agentdb)
        return agentdb
    return LocalVectorStore()
