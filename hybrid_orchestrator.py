"""
Hybrid Reasoning Orchestrator for VibeThinker-3B.

Routes queries between:
  - VibeThinker-3B (high-precision reasoning specialist, with optional async CLR)
  - A generalist model (for knowledge, planning, tool use, conversation, etc.)

Includes:
  - Embedding-based semantic router (sentence-transformers) with LRU caching
    of both query embeddings and route decisions.
  - Graceful fallback to keyword routing if embedding deps are missing.
  - Traceable logging hook (JSONL) for a memory vault / bi-temporal KG.

Install (optional, for semantic routing):
  pip install sentence-transformers scikit-learn aiohttp

Bug fixes vs. the original walkthrough version:
  - specialist_plain path: previously called the async _call_model with None
    as the aiohttp session (would crash). Now uses generate_plain() with a
    real session.
  - Generalist stop tokens: removed the bare "]" (corrupted  artifact).
  - Verdict parsing inherited the fix from vibe_clr_async.
  - final_answer "null" normalization inherited from vibe_clr_async.
  - log_to_memory no longer dumps non-serializable raw_traces (which contained
    the full CLRResult dataclass dict) into JSON unconditionally; it stores a
    trimmed, JSON-safe summary.
"""

import asyncio
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from vibe_clr_async import CLRResult, VibeThinkerCLRAsync
from persistent_cache import CLRResultCache, PersistentRouteCache

# Optional dependency — gracefully degrade if not installed
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


# ====================================================================== #
# Result dataclass
# ====================================================================== #
@dataclass
class OrchestratorResult:
    final_answer: str
    route_taken: str
    specialist_used: str
    clr_score: Optional[float] = None
    raw_traces: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    routing_confidence: float = 0.0


# ====================================================================== #
# Embedding router with caching
# ====================================================================== #
class EmbeddingRouter:
    """Semantic router with embedding + decision caching (LRU + disk persistence)."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_size: int = 512,
        persist_path: Optional[str] = "route_cache.json",
        autosave: bool = True,
    ):
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "Install with: pip install sentence-transformers scikit-learn"
            )

        print(f"[EmbeddingRouter] Loading {model_name} with cache_size={cache_size}")
        self.model = SentenceTransformer(model_name)
        self.cache_size = cache_size
        self.model_name = model_name

        # Reference examples (customize for your domain)
        self.specialist_examples: List[str] = [
            "Solve the recurrence a_{n+1} = a_n^2 - a_n + 1",
            "Find the sum of the infinite series 1 + 1/2 + 1/4 + ...",
            "Prove that the sum of angles in a triangle is 180 degrees",
            "What is the time complexity of this DP algorithm?",
            "Calculate the probability of drawing 3 aces without replacement",
            "Find the integral of x^2 sin(x) using integration by parts",
            "LeetCode hard: write an efficient solution",
        ]
        self.generalist_examples: List[str] = [
            "Explain the history of the Riemann Hypothesis",
            "What are the philosophical differences between capitalism and socialism?",
            "Summarize the key ideas in 'The Selfish Gene'",
            "How do transformer models work at a conceptual level?",
            "What caused World War I?",
            "Compare the capabilities of current frontier models",
        ]

        # Pre-compute reference embeddings
        self.specialist_embeddings = self.model.encode(self.specialist_examples)
        self.generalist_embeddings = self.model.encode(self.generalist_examples)

        # Persistent (disk-backed) cache — replaces the in-memory OrderedDicts.
        # Falls back to in-memory only if persist_path is None.
        if persist_path:
            self.persistent = PersistentRouteCache(
                path=persist_path, cache_size=cache_size, autosave=autosave
            )
            self.persistent.model_name = model_name
        else:
            self.persistent = None
            self.embedding_cache: "OrderedDict[str, Any]" = OrderedDict()
            self.route_cache: "OrderedDict[str, Tuple[str, float]]" = OrderedDict()

        print("[EmbeddingRouter] Ready with caching enabled")

    def _normalize(self, text: str) -> str:
        return text.lower().strip()

    def _get_or_compute_embedding(self, query: str):
        key = self._normalize(query)

        # Persistent path
        if self.persistent is not None:
            cached = self.persistent.get_embedding(key)
            if cached is not None:
                return np.array(cached)
            embedding = self.model.encode([query])[0]
            self.persistent.put_embedding(key, embedding)
            return embedding

        # In-memory fallback
        if key in self.embedding_cache:
            self.embedding_cache.move_to_end(key)
            return self.embedding_cache[key]
        embedding = self.model.encode([query])[0]
        if len(self.embedding_cache) >= self.cache_size:
            self.embedding_cache.popitem(last=False)
        self.embedding_cache[key] = embedding
        return embedding

    def classify(
        self, query: str, threshold: float = 0.65
    ) -> Tuple[str, float]:
        key = self._normalize(query)

        # Check route cache first (persistent or in-memory)
        if self.persistent is not None:
            cached_route = self.persistent.get_route(key)
            if cached_route is not None:
                return cached_route
        else:
            if key in self.route_cache:
                self.route_cache.move_to_end(key)
                return self.route_cache[key]

        query_embedding = self._get_or_compute_embedding(query).reshape(1, -1)

        spec_sim = float(cosine_similarity(query_embedding, self.specialist_embeddings).max())
        gen_sim = float(cosine_similarity(query_embedding, self.generalist_embeddings).max())

        confidence = max(spec_sim, gen_sim)

        if spec_sim > gen_sim and spec_sim >= threshold:
            route = "specialist"
        elif gen_sim > spec_sim and gen_sim >= threshold:
            route = "generalist"
        else:
            route = "hybrid"

        result = (route, confidence)

        if self.persistent is not None:
            self.persistent.put_route(key, route, confidence)
        else:
            if len(self.route_cache) >= self.cache_size:
                self.route_cache.popitem(last=False)
            self.route_cache[key] = result

        return result

    def clear_cache(self):
        if self.persistent is not None:
            self.persistent.clear()
        else:
            self.embedding_cache.clear()
            self.route_cache.clear()
        print("[EmbeddingRouter] Caches cleared")

    def save(self):
        """Flush persistent cache to disk (no-op if persistence disabled)."""
        if self.persistent is not None:
            self.persistent.save()


# ====================================================================== #
# Orchestrator
# ====================================================================== #
class HybridReasoningOrchestrator:
    def __init__(
        self,
        vibe_endpoint: str = "http://127.0.0.1:8080",
        generalist_endpoint: str = "http://127.0.0.1:8081",
        use_clr: bool = True,
        clr_k: int = 8,
        max_concurrent_clr: int = 6,
        use_embedding_router: bool = True,
        embedding_model: str = "all-MiniLM-L6-v2",
        router_cache_size: int = 512,
    ):
        self.vibe_endpoint = vibe_endpoint.rstrip("/")
        self.generalist_endpoint = generalist_endpoint.rstrip("/")
        self.use_clr = use_clr

        self.reasoner = VibeThinkerCLRAsync(
            server_url=vibe_endpoint,
            k=clr_k,
            max_concurrent=max_concurrent_clr,
        )

        self.use_embedding_router = use_embedding_router and EMBEDDINGS_AVAILABLE
        if self.use_embedding_router:
            try:
                self.router = EmbeddingRouter(
                    model_name=embedding_model, cache_size=router_cache_size
                )
            except Exception as e:
                print(f"[Warning] Could not load embedding router: {e}")
                self.use_embedding_router = False

        # Keyword fallback
        self.verifiable_keywords = {
            "solve", "calculate", "prove", "find the value", "what is the sum",
            "sequence", "series", "integral", "derivative", "probability",
            "leetcode", "code", "algorithm", "complexity", "math problem",
            "step by step", "rigorous", "formal proof", "recurrence",
        }
        self.generalist_keywords = {
            "explain", "what is", "who is", "history of", "compare",
            "opinion", "tool", "api", "search", "summarize", "describe",
        }

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #
    def _classify_route(self, query: str) -> Tuple[str, float]:
        if self.use_embedding_router:
            return self.router.classify(query)

        # Keyword fallback
        q_lower = query.lower()
        if any(kw in q_lower for kw in self.verifiable_keywords):
            return "specialist", 0.8
        if any(kw in q_lower for kw in self.generalist_keywords):
            return "generalist", 0.75
        return "hybrid", 0.5

    # ------------------------------------------------------------------ #
    # Generalist call
    # ------------------------------------------------------------------ #
    async def _call_generalist(self, query: str) -> str:
        """Call your generalist model endpoint."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "prompt": f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
                "n_predict": 4096,
                "temperature": 0.7,
                "top_p": 0.95,
                "stop": ["<|im_end|>"],
            }
            try:
                async with session.post(
                    f"{self.generalist_endpoint}/completion",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=600),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return data.get("content", "Generalist failed to respond.")
            except Exception as e:
                return f"Generalist call failed: {e}"

    # ------------------------------------------------------------------ #
    # Specialist (plain, no CLR) — fixed
    # ------------------------------------------------------------------ #
    async def _call_specialist_plain(self, query: str, max_tokens: int = 8192) -> str:
        """Plain VibeThinker generation without CLR. Uses a real session."""
        async with aiohttp.ClientSession() as session:
            return await self.reasoner.generate_plain(session, query, max_tokens)

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    async def run(self, query: str, force_route: Optional[str] = None) -> OrchestratorResult:
        if force_route:
            route, confidence = force_route, 1.0
        else:
            route, confidence = self._classify_route(query)

        print(f"\n[Orchestrator] Route: {route.upper()} (conf={confidence:.3f})")

        if route == "specialist":
            if self.use_clr:
                clr_result: CLRResult = await self.reasoner.run(query)
                return OrchestratorResult(
                    final_answer=clr_result.best_answer,
                    route_taken="specialist_clr",
                    specialist_used="VibeThinker-3B + CLR",
                    clr_score=clr_result.best_score,
                    routing_confidence=confidence,
                    raw_traces={"clr_result": self._trim_clr(clr_result)},
                )
            # Plain specialist (fixed: real session via helper)
            answer = await self._call_specialist_plain(query)
            return OrchestratorResult(
                final_answer=answer,
                route_taken="specialist_plain",
                specialist_used="VibeThinker-3B",
                routing_confidence=confidence,
            )

        elif route == "generalist":
            answer = await self._call_generalist(query)
            return OrchestratorResult(
                final_answer=answer,
                route_taken="generalist",
                specialist_used="Generalist Model",
                routing_confidence=confidence,
            )

        else:  # hybrid
            print("[Orchestrator] Hybrid path: Generalist plans -> Specialist solves -> Generalist synthesizes")

            plan = await self._call_generalist(
                f"Break this query into sub-problems and identify which need "
                f"precise reasoning:\n{query}"
            )

            if self.use_clr:
                specialist_result: CLRResult = await self.reasoner.run(query)
                specialist_answer = specialist_result.best_answer
                specialist_score = specialist_result.best_score
                specialist_trace = self._trim_clr(specialist_result)
            else:
                specialist_answer = await self._call_specialist_plain(query)
                specialist_score = None
                specialist_trace = {"raw": specialist_answer}

            final = await self._call_generalist(
                f"Using this plan: {plan}\n"
                f"And this high-quality reasoning result (score "
                f"{specialist_score if specialist_score is not None else 'n/a'}): "
                f"{specialist_answer}\n"
                f"Synthesize the final answer for the original query."
            )

            return OrchestratorResult(
                final_answer=final,
                route_taken="hybrid",
                specialist_used="Generalist + VibeThinker-3B + CLR",
                clr_score=specialist_score,
                routing_confidence=confidence,
                raw_traces={
                    "plan": plan,
                    "specialist_result": specialist_trace,
                },
            )

    # ------------------------------------------------------------------ #
    # Logging helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _trim_clr(clr_result: CLRResult) -> Dict[str, Any]:
        """JSON-safe summary of a CLRResult (drops huge raw traces)."""
        return {
            "best_answer": clr_result.best_answer,
            "best_score": clr_result.best_score,
            "k": clr_result.k,
            "trajectory_count": len(clr_result.all_trajectories),
            "trajectories": [
                {
                    "score": t.get("score"),
                    "answer": t.get("answer"),
                    "claims": t.get("claims"),
                    "verdicts": t.get("verdicts"),
                }
                for t in clr_result.all_trajectories
            ],
        }

    def log_to_memory(self, result: OrchestratorResult, query: str, path: str = "orchestrator_memory.jsonl"):
        """Hook for your memory system / immutable vault."""
        log_entry = {
            "timestamp": result.timestamp,
            "query": query,
            "route": result.route_taken,
            "answer": result.final_answer,
            "clr_score": result.clr_score,
            "routing_confidence": result.routing_confidence,
            "raw_traces": result.raw_traces,
        }
        try:
            with open(path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            print(f"[Memory] Logged to {path}")
        except (TypeError, ValueError) as e:
            # Fall back to a trimmed entry if raw_traces isn't JSON-serializable
            log_entry["raw_traces"] = "<non-serializable>"
            with open(path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            print(f"[Memory] Logged (trimmed) to {path}: {e}")


# ====================== EXAMPLE USAGE ======================

async def main():
    orchestrator = HybridReasoningOrchestrator(
        vibe_endpoint="http://127.0.0.1:8080",
        generalist_endpoint="http://127.0.0.1:8081",  # <- change to your generalist
        use_clr=True,
        clr_k=8,
        use_embedding_router=True,
        router_cache_size=1024,
    )

    queries = [
        "Solve the recurrence: a_1=2, a_{n+1}=a_n^2 - a_n + 1. Find a_5.",
        "Explain the history of the Riemann Hypothesis and its current status.",
        "A complex problem involving both math and conceptual understanding.",
    ]

    for q in queries:
        result = await orchestrator.run(q)
        orchestrator.log_to_memory(result, q)
        print(f"\n>>> Final Answer:\n{result.final_answer}\n")


if __name__ == "__main__":
    asyncio.run(main())
