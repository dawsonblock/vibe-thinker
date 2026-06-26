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
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from vibe_clr_async import CLRResult, VibeThinkerCLRAsync
from persistent_cache import (
    CLRResultCache,
    PersistentRouteCache,
    VerifiedTrajectoryStore,
    should_cache,
)
from verifiers import MathVerifier, CodeVerifier, FactualVerifier
from sandbox import WarmDockerPool
from math_solver import solve as solve_math

# Sentinel for "argument not provided" — distinct from an explicit None
# (which means "disable the code verifier / verified loop").
_UNSET = object()


def select_verifier(task_type: str):
    """Select a deterministic verifier based on the detected task type.

    Returns None if no verifier applies (conversation, summarization, etc.).
    Conversation and summarization tasks do NOT get a verifier — there is
    no deterministic way to verify "explain X" or "summarize Y".
    """
    if task_type == "math":
        return MathVerifier()
    if task_type == "code":
        return CodeVerifier()
    if task_type == "factual":
        return FactualVerifier()
    return None

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
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
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
        code_specialist_endpoint: Optional[str] = None,
        code_candidates: int = 6,
        code_verifier: Optional["CodeVerifier"] = _UNSET,
        use_clr: bool = True,
        clr_k: int = 8,
        max_concurrent_clr: int = 6,
        use_embedding_router: bool = True,
        embedding_model: str = "all-MiniLM-L6-v2",
        router_cache_size: int = 512,
        use_clr_cache: bool = True,
        clr_cache_path: str = "clr_result_cache.json",
        clr_cache_similarity: float = 0.92,
        clr_cache_min_score: float = 0.7,
        use_trajectory_store: bool = True,
        trajectory_store_path: str = "verified_trajectories.json",
        trajectory_retrieval_threshold: float = 0.70,
        trajectory_max_few_shot: int = 3,
        fast_specialist: bool = False,
        local_specialist_model: Optional[str] = None,
        local_specialist_n_ctx: int = 4096,
        local_specialist_n_threads: int = 8,
    ):
        self.vibe_endpoint = vibe_endpoint.rstrip("/")
        self.generalist_endpoint = generalist_endpoint.rstrip("/")
        # Optional dedicated code-generation specialist (e.g. a small, fast
        # coding model like ruvltra-claude-code-0.5b). When set, code tasks are
        # routed here for plain generation instead of the VibeThinker CLR path.
        # Disabled by default; set via CODE_SPECIALIST_URL / --code-specialist.
        self.code_specialist_endpoint = (
            code_specialist_endpoint.rstrip("/") if code_specialist_endpoint else None
        )
        # Multi-candidate code generation: generate N candidates from the code
        # specialist, verify each in the sandbox, return the first that passes.
        # This is the "shotgun + sandbox picks winner" loop. Requires a
        # code_verifier (CodeVerifier with a sandbox executor). If no verifier
        # is configured, falls back to single-candidate plain generation.
        self.code_candidates = max(1, code_candidates)
        # _UNSET -> default CodeVerifier (safe, fail-closed sandbox).
        # None -> explicitly disabled (plain generation, no verification).
        self.code_verifier = CodeVerifier() if code_verifier is _UNSET else code_verifier
        # If the verifier's executor is a WarmDockerPool, start it eagerly so
        # the first code task doesn't pay the cold-start cost.
        if self.code_verifier is not None and hasattr(self.code_verifier, "executor"):
            executor = self.code_verifier.executor
            if isinstance(executor, WarmDockerPool):
                self._warm_pool = executor
            else:
                self._warm_pool = None
        else:
            self._warm_pool = None
        self.use_clr = use_clr

        self.reasoner = VibeThinkerCLRAsync(
            server_url=vibe_endpoint,
            k=clr_k,
            max_concurrent=max_concurrent_clr,
            fast_specialist=fast_specialist,
            local_model=local_specialist_model,
            local_n_ctx=local_specialist_n_ctx,
            local_n_threads=local_specialist_n_threads,
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

        # Semantic CLR result cache (skip re-running high-score CLR for
        # similar/recent problems). Only available with embedding deps.
        self.use_clr_cache = use_clr_cache and EMBEDDINGS_AVAILABLE and use_clr
        if self.use_clr_cache:
            try:
                self.clr_cache = CLRResultCache(
                    path=clr_cache_path,
                    model_name=embedding_model,
                    similarity_threshold=clr_cache_similarity,
                    min_score=clr_cache_min_score,
                )
            except Exception as e:
                print(f"[Warning] Could not load CLR result cache: {e}")
                self.use_clr_cache = False
                self.clr_cache = None
        else:
            self.clr_cache = None

        # Verified trajectory store (self-improving few-shot memory).
        # Stores independently-verified solutions and retrieves them as
        # few-shot context for similar future queries. Only available with
        # embedding deps. Only stores verified=True results — never learns
        # from unverified output.
        self.use_trajectory_store = use_trajectory_store and EMBEDDINGS_AVAILABLE
        if self.use_trajectory_store:
            try:
                self.trajectory_store = VerifiedTrajectoryStore(
                    path=trajectory_store_path,
                    model_name=embedding_model,
                    retrieval_threshold=trajectory_retrieval_threshold,
                    max_few_shot=trajectory_max_few_shot,
                )
            except Exception as e:
                print(f"[Warning] Could not load trajectory store: {e}")
                self.use_trajectory_store = False
                self.trajectory_store = None
        else:
            self.trajectory_store = None

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
    # Non-programming "code" phrases that must NOT route to the code task.
    # These are word-boundary phrase patterns checked before code keywords.
    NON_PROGRAMMING_CODE_PATTERNS = [
        r"\bcode of conduct\b",
        r"\bcode of ethics\b",
        r"\bdress code\b",
        r"\bbuilding code\b",
        r"\blegal code\b",
        r"\bcode of honor\b",
        r"\bcode of practice\b",
        r"\barea code\b",
        r"\bzip code\b",
        r"\bbarcode\b",
        r"\bqr code\b",
        r"\bcode pink\b",
        r"\bcode blue\b",
        r"\bcode red\b",
    ]

    # Strong programming signals that override math intent. If any of these
    # are present, the query is code regardless of math-like words.
    STRONG_CODE_SIGNALS = [
        r"\bleetcode\b",
        r"\bhackerrank\b",
        r"\bcodeforces\b",
        r"\bgithub\b",
        r"\bgitlab\b",
        r"\bnpm\b",
        r"\bpip install\b",
        r"\bstack overflow\b",
        r"\bunit test\b",
        r"\bpytest\b",
        r"\bcompile error\b",
        r"\bsyntax error\b",
        r"\bsegfault\b",
        r"\bdebug\b.*\b(python|javascript|rust|java|c\+\+|go)\b",
    ]

    # Math should require computational intent, not just the word "sum" or
    # "series" appearing in a non-math context (e.g. "sum of human knowledge",
    # "world series", "TV series").
    MATH_INTENT_PATTERNS = [
        r"\bsolve\b.*\b(equation|integral|derivative|sum|series|system|inequality)\b",
        r"\bcompute\b",
        r"\bcalculate\b",
        r"\bprove\b",
        r"\bevaluate\b",
        r"\bwhat is\s+\d+",
        r"\bfind the value\b",
        r"\bfind\s+\w+\s+of\b.*\d",
        r"\b\d+\s*[+\-*/=]\s*\d+",
        r"\\boxed",
        r"\\frac",
        r"\\int",
        r"\\sum",
        r"\bderivative of\b",
        r"\bintegral of\b",
        r"\bsum of\b.*\d",
        r"\bgeometric series\b",
        r"\brecurrence\b",
        r"\bprobability of\b",
        r"\bmatrix\b",
        r"\bvector\b",
        r"\btheorem\b",
        r"\balgebra\b",
        r"\bcalculus\b",
        r"\bcombinatorics\b",
        # Indexed variable notation: a_1, a_{n+1}, a_n, x_0, etc.
        r"\ba_\{?\w+\}?\s*=",
        r"\ba_?\{?\d+\}?\b",
        r"\bx_?\{?\d+\}?\b",
        # Recurrence-like notation: a_{n+1}=a_n^2...
        r"\ba_\{?n\+1\}?\s*=",
        r"\ba_n\s*[+\-*/^]",
        # "find a_5" or "find a_{5}"
        r"\bfind\s+a_?\{?\d+\}?",
        # "solve this step by step" with numbers — canonical math query shape
        r"\bsolve this step by step\b.*\d",
        # Explicit recurrence definition: a_1=... a_{n+1}=...
        r"\ba_1\s*=\s*\d",
    ]

    # Task-type keywords for structured routing. Maps task categories to
    # the tools they typically require for deterministic verification.
    _TASK_TYPE_KEYWORDS = {
        "math": {
            "keywords": {"solve", "calculate", "prove", "find the value", "sum",
                         "sequence", "series", "integral", "derivative", "probability",
                         "recurrence", "equation", "matrix", "vector", "theorem",
                         "geometric", "algebra", "calculus", "combinatorics",
                         "compute", "evaluate"},
            "requires_tools": ["deterministic_check"],
            "requires_model": True,
        },
        "code": {
            "keywords": {"leetcode", "code", "algorithm", "complexity",
                         "implement", "function", "debug", "refactor", "program",
                         "python", "javascript", "rust", "compile"},
            "requires_tools": ["python_exec", "unit_tests"],
            "requires_model": True,
        },
        "planning": {
            "keywords": {"plan", "strategy", "roadmap", "design", "architect",
                         "steps to", "how to", "approach"},
            "requires_tools": [],
            "requires_model": True,
        },
        "retrieval": {
            "keywords": {"search", "find information", "look up", "retrieve",
                         "what does the docs say"},
            "requires_tools": ["search", "retrieval"],
            "requires_model": False,
        },
        "summarization": {
            "keywords": {"summarize", "summarise", "tldr", "brief", "overview",
                         "key points"},
            "requires_tools": [],
            "requires_model": True,
        },
        "conversation": {
            "keywords": {"explain", "what is", "who is", "history of", "compare",
                         "opinion", "describe", "tell me about", "why does"},
            "requires_tools": [],
            "requires_model": True,
        },
    }

    def _detect_task_type(self, query: str) -> Tuple[str, List[str], bool]:
        """Detect task type from query keywords.

        Returns (task_type, requires_tools, requires_model).
        Falls back to ("unknown", [], True) if no match.

        Uses word-boundary matching to avoid false positives (e.g. "summarize"
        containing "sum" should NOT match the math keyword "sum").

        Additional false-positive guards:
          - Non-programming "code" phrases (code of conduct, dress code, etc.)
            are excluded from the code task type.
          - Math requires computational intent (not just the word "sum" or
            "series" in a non-math context like "sum of human knowledge").
        """
        q_lower = query.lower()

        # --- Guard: non-programming "code" phrases ---
        # If the query matches a non-programming code pattern, do NOT classify
        # it as "code" even if the word "code" appears.
        is_non_programming_code = any(
            re.search(pattern, q_lower)
            for pattern in self.NON_PROGRAMMING_CODE_PATTERNS
        )

        # --- Guard: strong programming signals override math intent ---
        # If "leetcode", "debug python", etc. are present, it's code regardless
        # of math-like words like "sum" or "solve".
        has_strong_code_signal = any(
            re.search(pattern, q_lower)
            for pattern in self.STRONG_CODE_SIGNALS
        )

        # --- Guard: math requires computational intent ---
        has_math_intent = any(
            re.search(pattern, q_lower)
            for pattern in self.MATH_INTENT_PATTERNS
        )
        # If there's a strong code signal, suppress math intent
        if has_strong_code_signal:
            has_math_intent = False

        best_type = "unknown"
        best_score = 0
        for task_type, config in self._TASK_TYPE_KEYWORDS.items():
            # Skip code task type if the query is a non-programming "code" phrase
            if task_type == "code" and is_non_programming_code:
                continue
            # Skip math task type if there's no computational intent
            if task_type == "math" and not has_math_intent:
                continue

            score = 0
            for kw in config["keywords"]:
                # Use word boundary matching for single words;
                # substring for multi-word phrases
                if " " in kw:
                    if kw in q_lower:
                        score += 1
                else:
                    if re.search(r"\b" + re.escape(kw) + r"\b", q_lower):
                        score += 1
            if score > best_score:
                best_score = score
                best_type = task_type

        # If math intent was detected but no keywords matched, still classify
        # as math. The intent patterns (indexed variables, recurrence notation,
        # "solve this step by step" with numbers) ARE the math signal.
        if best_type == "unknown" and has_math_intent:
            return "math", ["deterministic_check"], True

        if best_type == "unknown":
            return "unknown", [], True

        config = self._TASK_TYPE_KEYWORDS[best_type]
        return best_type, config["requires_tools"], config["requires_model"]

    def route_structured(self, query: str) -> Dict[str, Any]:
        """Produce a structured routing decision.

        Returns a dict with:
          - route: specialist | generalist | hybrid
          - confidence: float 0-1
          - task_type: math | code | planning | retrieval | summarization | conversation | unknown
          - requires_tools: list of tool names needed for verification
          - requires_model: whether a model call is needed
          - requires_human_review: whether human review is recommended
          - reason: human-readable explanation
        """
        route, confidence = self._classify_route(query)
        task_type, requires_tools, requires_model = self._detect_task_type(query)

        # Human review recommended for low-confidence routing or unknown task types
        requires_human_review = confidence < 0.65 or task_type == "unknown"

        reasons = []
        if self.use_embedding_router:
            reasons.append(f"embedding router (conf={confidence:.2f})")
        else:
            reasons.append("keyword fallback router")
        reasons.append(f"task_type={task_type}")
        if requires_tools:
            reasons.append(f"requires_tools={requires_tools}")

        return {
            "route": route,
            "confidence": round(confidence, 4),
            "task_type": task_type,
            "requires_tools": requires_tools,
            "requires_model": requires_model,
            "requires_human_review": requires_human_review,
            "reason": ", ".join(reasons),
        }

    def _classify_route(self, query: str) -> Tuple[str, float]:
        """Classify a query into a route: specialist, generalist, or hybrid.

        Uses the structured task type to determine the route. This ensures
        that task_type and route agree — "code of conduct" is conversation,
        not code, so it must route to generalist, not specialist.

        Routing rules:
          - math, code -> specialist (these need CLR + deterministic verification)
          - conversation, summarization -> generalist (no verifier needed)
          - planning, retrieval, unknown -> hybrid
        """
        task_type, _, _ = self._detect_task_type(query)

        if task_type in {"math", "code"}:
            # Specialist for verifiable tasks
            if self.use_embedding_router:
                route, conf = self.router.classify(query)
                # Override embedding router if it disagrees with task_type
                if route != "specialist":
                    print(f"[Route] Embedding router said {route} but task_type={task_type} -> specialist")
                return "specialist", max(conf, 0.8)
            return "specialist", 0.8

        if task_type in {"conversation", "summarization"}:
            # Generalist for non-verifiable tasks
            if self.use_embedding_router:
                route, conf = self.router.classify(query)
                if route == "specialist":
                    print(f"[Route] Embedding router said specialist but task_type={task_type} -> generalist")
                return "generalist", max(conf, 0.75)
            return "generalist", 0.75

        # planning, retrieval, unknown -> hybrid
        if self.use_embedding_router:
            return self.router.classify(query)
        return "hybrid", 0.5

    # ------------------------------------------------------------------ #
    # Generalist call
    # ------------------------------------------------------------------ #
    async def _call_generalist(self, query: str, max_tokens: int = 4096) -> str:
        """Call the generalist model via the OpenAI-compatible
        /v1/chat/completions endpoint so llama-server applies the model's
        own baked-in chat template (Llama 3.2 uses <|start_header_id|>, not
        ChatML). Falls back to /completion with ChatML if the chat endpoint
        fails. Raises RuntimeError if both endpoints fail — callers must
        handle the exception, not silently proceed with an error string."""
        async with aiohttp.ClientSession() as session:
            chat_payload = {
                "messages": [{"role": "user", "content": query}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.95,
            }
            try:
                async with session.post(
                    f"{self.generalist_endpoint}/v1/chat/completions",
                    json=chat_payload,
                    timeout=aiohttp.ClientTimeout(total=600),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    if not content:
                        raise RuntimeError("Generalist returned empty response")
                    return content
            except RuntimeError:
                raise
            except Exception as e:
                # Fallback to raw /completion with ChatML
                print(f"[Generalist] chat endpoint failed ({e}), falling back to /completion")
                payload = {
                    "prompt": f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
                    "n_predict": max_tokens,
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
                        content = data.get("content", "")
                        if not content:
                            raise RuntimeError("Generalist returned empty response on fallback")
                        return content
                except RuntimeError:
                    raise
                except Exception as e2:
                    raise RuntimeError(
                        f"Generalist call failed (both endpoints): {e2}"
                    ) from e2

    # ------------------------------------------------------------------ #
    # Specialist (plain, no CLR) — fixed
    # ------------------------------------------------------------------ #
    async def _call_specialist_plain(self, query: str, max_tokens: int = 8192) -> str:
        """Plain VibeThinker generation without CLR. Uses a real session."""
        async with aiohttp.ClientSession() as session:
            return await self.reasoner.generate_plain(session, query, max_tokens)

    # ------------------------------------------------------------------ #
    # Code specialist (dedicated fast code-generation model, e.g. ruvltra)
    # ------------------------------------------------------------------ #
    async def _call_code_specialist(self, query: str, max_tokens: int = 4096) -> str:
        """Call the dedicated code-specialist model via the OpenAI-compatible
        /v1/chat/completions endpoint. Falls back to /completion with ChatML
        if the chat endpoint fails. Raises RuntimeError if both fail.

        Note: this is plain generation — it does NOT run the CLR claim-level
        reliability loop. Code correctness is expected to be checked
        downstream by the CodeVerifier (sandbox), not by self-claims.
        """
        if not self.code_specialist_endpoint:
            raise RuntimeError("No code_specialist_endpoint configured")
        async with aiohttp.ClientSession() as session:
            chat_payload = {
                "messages": [{"role": "user", "content": query}],
                "max_tokens": max_tokens,
                "temperature": 0.2,
                "top_p": 0.95,
            }
            try:
                async with session.post(
                    f"{self.code_specialist_endpoint}/v1/chat/completions",
                    json=chat_payload,
                    timeout=aiohttp.ClientTimeout(total=600),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    if not content:
                        raise RuntimeError("Code specialist returned empty response")
                    return content
            except RuntimeError:
                raise
            except Exception as e:
                print(f"[CodeSpecialist] chat endpoint failed ({e}), falling back to /completion")
                payload = {
                    "prompt": f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n",
                    "n_predict": max_tokens,
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "stop": ["<|im_end|>"],
                }
                try:
                    async with session.post(
                        f"{self.code_specialist_endpoint}/completion",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=600),
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        content = data.get("content", "")
                        if not content:
                            raise RuntimeError("Code specialist returned empty response on fallback")
                        return content
                except RuntimeError:
                    raise
                except Exception as e2:
                    raise RuntimeError(
                        f"Code specialist call failed (both endpoints): {e2}"
                    ) from e2

    # ------------------------------------------------------------------ #
    # Multi-candidate code generation with sandbox verification
    # ------------------------------------------------------------------ #
    _TEST_SPEC_PROMPT = (
        "You are a test engineer. Given the coding request below, write "
        "Python unit tests as bare `assert` statements that verify a correct "
        "solution. Output ONLY a single ```python code block containing the "
        "assert statements — no explanation, no imports unless strictly "
        "needed, no test framework. The asserts must call the functions/"
        "classes the solution is expected to define.\n\n"
        "Coding request:\n{query}\n\n"
        "Output the test asserts now (```python block only):"
    )

    _CODE_GEN_PROMPT = (
        "Write a Python solution for the request below. Output ONLY a "
        "single ```python code block with the complete, runnable code — "
        "no explanation.\n\n"
        "Request:\n{query}\n\n"
        "Your solution must pass these tests:\n{tests}\n\n"
        "Output the ```python code block now:"
    )

    @staticmethod
    def _extract_python_block(text: str) -> str:
        """Extract the first ```python ... ``` block from text, or the whole
        text if no fenced block is found."""
        match = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
        return match[0].strip() if match else text.strip()

    async def _generate_test_spec(self, query: str) -> Optional[str]:
        """Ask the generalist to produce unit-test asserts for the query.

        Returns the extracted Python assert code, or None if the generalist
        failed or produced nothing parseable. This is the "Software Architect"
        step: the generalist defines correctness before the code specialist
        writes any code.
        """
        prompt = self._TEST_SPEC_PROMPT.format(query=query)
        try:
            raw = await self._call_generalist(prompt, max_tokens=1024)
        except Exception as e:
            print(f"[CodeLoop] Test-spec generation failed: {e}")
            return None
        tests = self._extract_python_block(raw)
        if not tests or "assert" not in tests:
            print(f"[CodeLoop] Generalist produced no usable asserts — skipping verification")
            return None
        return tests

    async def _run_code_specialist_verified(
        self, query: str
    ) -> OrchestratorResult:
        """Multi-candidate code generation with sandbox verification.

        Flow (the "shotgun + sandbox picks winner" loop):
          1. Generalist writes unit-test asserts for the query (test spec).
          2. Code specialist (ruvltra) generates N candidate solutions in
             parallel with diverse temperatures.
          3. CodeVerifier runs each candidate against the test spec in the
             sandbox. The first candidate that passes (ALL_TESTS_PASSED)
             wins and is returned with score 1.0.
          4. If no candidate passes, return the first candidate with score
             0.0 and an honest "no candidate passed verification" note.

        If no test spec can be generated, or no verifier/sandbox is
        available, falls back to single-candidate plain generation with
        score 0.0 (unverified — fail-closed, never fake verification).
        """
        tests = await self._generate_test_spec(query)

        # Ensure the warm Docker pool is started before sandbox verification.
        if self._warm_pool is not None and not self._warm_pool._started:
            await self._warm_pool.start()

        # Without tests or a verifier, we cannot verify — fall back to plain.
        if tests is None:
            print("[CodeLoop] No test spec — single-candidate unverified generation")
            answer = await self._call_code_specialist(query)
            return OrchestratorResult(
                final_answer=answer,
                route_taken="code_specialist_unverified",
                specialist_used="Code Specialist (ruvltra-claude-code)",
                clr_score=0.0,
                routing_confidence=0.0,
                raw_traces={"verified": False, "reason": "no test spec generated"},
            )

        # Generate N candidates in parallel with diverse temperatures.
        # Prepend verified trajectories as few-shot context (self-improving).
        temps = [0.2] + [0.7] * (self.code_candidates - 1)
        few_shot = self._get_few_shot_context(query, task_type="code")
        query_with_ctx = f"{few_shot}\n{query}" if few_shot else query
        gen_prompt = self._CODE_GEN_PROMPT.format(query=query_with_ctx, tests=tests)
        if few_shot:
            print(f"[CodeLoop] Retrieved verified examples as few-shot context")
        print(f"[CodeLoop] Generating {self.code_candidates} candidates + test spec verification")
        candidates_raw = await asyncio.gather(
            *[self._call_code_specialist(gen_prompt, max_tokens=2048) for _ in temps],
            return_exceptions=True,
        )
        candidates = [
            (raw, self._extract_python_block(raw))
            for raw in candidates_raw if isinstance(raw, str)
        ]
        if not candidates:
            return OrchestratorResult(
                final_answer="",
                route_taken="code_specialist_failed",
                specialist_used="Code Specialist (ruvltra-claude-code)",
                clr_score=0.0,
                routing_confidence=0.0,
                raw_traces={"verified": False, "reason": "all candidate generations failed"},
            )

        # Verify each candidate against the test spec in the sandbox.
        # First passing candidate wins.
        verification_traces = []
        for i, (raw, code) in enumerate(candidates):
            result = await self.code_verifier.verify(
                query, code, {"unit_tests": tests}
            )
            trace = {
                "candidate_index": i,
                "verified": result.verified,
                "score": result.score,
                "method": result.method,
                "error": result.error,
            }
            verification_traces.append(trace)
            if result.verified:
                print(f"[CodeLoop] Candidate {i} PASSED verification — returning")
                return OrchestratorResult(
                    final_answer=code,
                    route_taken="code_specialist_verified",
                    specialist_used="Code Specialist (ruvltra-claude-code)",
                    clr_score=1.0,
                    routing_confidence=1.0,
                    raw_traces={
                        "verified": True,
                        "verification_method": result.method,
                        "candidates_generated": len(candidates),
                        "candidate_index": i,
                        "unit_tests": tests,
                        "all_verification_traces": verification_traces,
                    },
                )
            print(f"[CodeLoop] Candidate {i} failed: {result.error or 'not verified'}")

        # No candidate passed — return the first with honest score 0.0.
        print(f"[CodeLoop] No candidate passed verification — returning best-effort (unverified)")
        return OrchestratorResult(
            final_answer=candidates[0][1],
            route_taken="code_specialist_unverified",
            specialist_used="Code Specialist (ruvltra-claude-code)",
            clr_score=0.0,
            routing_confidence=0.0,
            raw_traces={
                "verified": False,
                "reason": "no candidate passed sandbox verification",
                "candidates_generated": len(candidates),
                "unit_tests": tests,
                "all_verification_traces": verification_traces,
            },
        )

    # ------------------------------------------------------------------ #
    # CLR with semantic cache
    # ------------------------------------------------------------------ #
    # Answers that must NEVER be cached, regardless of score.
    _UNCACHEABLE_ANSWERS = frozenset({
        "no clear answer found",
        "all trajectories failed",
        "",
        "none",
        "null",
        "n/a",
    })

    def _build_cache_result_dict(self, clr_result: CLRResult) -> dict:
        """Build the result dict that should_cache() evaluates, using
        verification metadata from the CLR result directly."""
        best_traj = max(
            (t for t in clr_result.all_trajectories if isinstance(t, dict)),
            key=lambda x: x.get("score", 0),
            default=None,
        )
        claim_count = (
            len([c for c in (best_traj or {}).get("claims", [])
                 if self.reasoner._is_meaningful_claim(c)])
            if best_traj else 0
        )

        return {
            "answer": clr_result.best_answer,
            "score": clr_result.best_score,
            "answer_present": bool(clr_result.best_answer),
            "claim_count": claim_count,
            "verification_method": clr_result.verification_method,
            "failure": clr_result.failure_reason,
            "transport_failures": clr_result.transport_failures,
            "deterministic_check": clr_result.deterministic_verification,
        }

    def _is_cacheable(self, clr_result: CLRResult, allow_weak_cache: bool = False) -> bool:
        """Return True only if a CLR result is safe to cache.

        Uses the strict :func:`should_cache` policy by default:
          - No answer or sentinel failure strings
          - Score >= 0.75
          - claim_count >= 5
          - No transport failures
          - self_claims_only verification is rejected unless allow_weak_cache

        Args:
            clr_result: the CLR result to evaluate.
            allow_weak_cache: if True, allow self_claims_only verification
                to be cached. Default is False.
        """
        result_dict = self._build_cache_result_dict(clr_result)
        if not should_cache(result_dict, allow_weak_cache=allow_weak_cache):
            return False
        # Also respect the cache's own min_score threshold if set higher
        if self.clr_cache is not None and clr_result.best_score < self.clr_cache.min_score:
            return False
        return True

    def _build_verifier_context(self, query: str, task_type: str) -> Dict[str, Any]:
        """Derive verifier context from the query.

        This is what makes verifiers actually useful instead of ceremonial.
        Without context, MathVerifier has no expected_answer, CodeVerifier
        has no unit_tests, and FactualVerifier has no sources.

        For math: use the deterministic math_solver to derive expected_answer.
        For code: extract code blocks and test assertions from the query.
        For factual: no sources available unless caller provides them.

        Do NOT fake context. If we can't derive it deterministically, leave
        it absent — the verifier will return verified=False, which is honest.
        """
        context: Dict[str, Any] = {}

        if task_type == "math":
            expected = solve_math(query)
            if expected is not None:
                context["expected_answer"] = expected
                print(f"[CLR] Math solver derived expected_answer={expected}")

        elif task_type == "code":
            # Extract code blocks from the query (```python ... ```)
            code_blocks = re.findall(r"```(?:python)?\n(.*?)```", query, re.DOTALL)
            if code_blocks:
                context["expected_output"] = code_blocks[0].strip()

        # Factual: no sources to derive. The verifier will return
        # unsupported_factual, which is the honest result.

        return context

    async def _run_clr_with_cache(self, query: str) -> Tuple[CLRResult, bool]:
        """Run CLR, but return a cached result if a similar high-score
        problem was solved before. Returns (result, cache_hit).

        Selects a deterministic verifier based on the detected task type
        and passes it into the CLR run. This is the ONLY path that allows
        the final score to exceed the self-claims-only cap of 0.65.
        """
        if self.use_clr_cache and self.clr_cache is not None:
            cached = self.clr_cache.lookup(query)
            if cached is not None:
                # Double-check the cached answer isn't a known-bad sentinel.
                # (Defensive: old cache files may contain bad entries.)
                if (cached["best_answer"] or "").strip().lower() in self._UNCACHEABLE_ANSWERS:
                    print(f"[CLRCache] HIT but answer is uncacheable sentinel — ignoring cache")
                else:
                    print(
                        f"[CLRCache] HIT (sim={cached['similarity']:.3f}, "
                        f"score={cached['best_score']:.3f}) — skipping CLR run"
                    )
                    return (
                        CLRResult(
                            best_answer=cached["best_answer"],
                            best_score=cached["best_score"],
                            best_raw_trace="<cached>",
                            all_trajectories=[],
                            k=cached.get("k") or self.reasoner.k,
                            verification_method=cached.get("verification_method", "self_claims_only"),
                            verified=cached.get("verified", False),
                        ),
                        True,
                    )

        # Select a deterministic verifier based on the task type.
        # This is what allows math/code tasks to exceed the 0.65 cap.
        decision = self.route_structured(query)
        task_type = decision["task_type"]
        verifier = select_verifier(task_type)
        verifier_context = self._build_verifier_context(query, task_type)
        if verifier is not None:
            ctx_desc = ", ".join(f"{k}={v!r}" for k, v in verifier_context.items() if v)
            print(f"[CLR] Selected verifier: {verifier.name} (task_type={task_type})"
                  f"{f', context: {ctx_desc}' if ctx_desc else ', no derivable context'}")
        else:
            print(f"[CLR] No verifier for task_type={task_type} — self-claims-only cap applies")

        # Retrieve verified trajectories as few-shot context (self-improving
        # memory). The model sees prior verified solutions to similar problems
        # before attempting this one. This is how the system "learns" — not by
        # updating weights, but by accumulating verified examples.
        few_shot = self._get_few_shot_context(query, task_type=task_type)
        query_with_context = f"{few_shot}\n{query}" if few_shot else query
        if few_shot:
            print(f"[TrajectoryStore] Retrieved {min(self.trajectory_store.max_few_shot, len(self.trajectory_store.retrieve(query, task_type)))} "
                  f"verified examples as few-shot context")

        clr_result = await self.reasoner.run(
            query_with_context, verifier=verifier, task_type=task_type,
            verifier_context=verifier_context,
        )

        # Insert into cache ONLY if the result is cacheable per the strict
        # should_cache policy. Weak self-verification is NOT cached by default.
        if self.use_clr_cache and self.clr_cache is not None and self._is_cacheable(clr_result):
            result_dict = self._build_cache_result_dict(clr_result)
            claim_count = result_dict["claim_count"]

            self.clr_cache.insert(
                problem=query,
                best_answer=clr_result.best_answer,
                best_score=clr_result.best_score,
                k=clr_result.k,
                trajectory_count=len(clr_result.all_trajectories),
                verified=clr_result.verified,
                verification_method=clr_result.verification_method,
                claim_count=claim_count,
                answer_present=True,
                deterministic_check=clr_result.deterministic_verification,
                failure=clr_result.failure_reason,
                transport_failures=clr_result.transport_failures,
                model_failures=clr_result.model_failures,
            )
            print(f"[CLRCache] Stored result (score={clr_result.best_score:.3f}, "
                  f"claims={claim_count}, method={clr_result.verification_method}, "
                  f"verified={clr_result.verified})")
        elif self.use_clr_cache and self.clr_cache is not None:
            print(
                f"[CLRCache] NOT caching (score={clr_result.best_score:.3f}, "
                  f"method={clr_result.verification_method}, "
                  f"verified={clr_result.verified}, "
                  f"answer={clr_result.best_answer[:40]!r}...)"
            )

        return clr_result, False

    # ------------------------------------------------------------------ #
    # Trajectory store helpers (self-improving memory)
    # ------------------------------------------------------------------ #
    def _store_if_verified(
        self, query: str, result: OrchestratorResult,
        clr_result: Optional[CLRResult] = None,
    ) -> None:
        """Store a verified result in the trajectory store.

        Only stores results that were independently verified (verified=True
        with a deterministic verifier, not self_claims_only). This is the
        "learning" step: verified solutions become few-shot context for
        future similar queries. Unverified results are never stored.
        """
        if not self.use_trajectory_store or self.trajectory_store is None:
            return

        # Determine verification status and method
        verified = False
        method = "self_claims_only"
        score = result.clr_score or 0.0
        task_type = "unknown"

        if clr_result is not None:
            verified = clr_result.verified
            method = clr_result.verification_method
        elif result.raw_traces.get("verified"):
            verified = True
            method = result.raw_traces.get("verification_method", "self_claims_only")

        if not verified or method == "self_claims_only":
            return  # Never learn from unverified or self-claimed results

        # Determine task type
        tt, _, _ = self._detect_task_type(query)
        task_type = tt

        self.trajectory_store.store(
            query=query,
            answer=result.final_answer,
            score=score,
            verification_method=method,
            task_type=task_type,
            route_taken=result.route_taken,
        )
        print(f"[TrajectoryStore] Stored verified result (method={method}, "
              f"score={score:.3f}, task_type={task_type}) — "
              f"store now has {len(self.trajectory_store)} entries")

    def _get_few_shot_context(self, query: str, task_type: Optional[str] = None) -> str:
        """Retrieve verified trajectories as few-shot context for the query.

        Returns a string to prepend to the model prompt, or "" if no similar
        verified trajectories exist or the store is disabled.
        """
        if not self.use_trajectory_store or self.trajectory_store is None:
            return ""
        return self.trajectory_store.build_few_shot_prefix(query, task_type=task_type)

    # ------------------------------------------------------------------ #
    # Lifecycle: warm pool start/cleanup
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        """Start background resources (warm Docker pool).

        Call this before submitting code tasks to pre-warm the sandbox
        containers. If no warm pool is configured, this is a no-op.
        """
        if self._warm_pool is not None:
            await self._warm_pool.start()

    async def cleanup(self) -> None:
        """Clean up background resources (warm Docker pool).

        Call this when shutting down to remove warm containers.
        """
        if self._warm_pool is not None:
            await self._warm_pool.cleanup()

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
            # If a dedicated code specialist is configured and this is a code
            # task, route to it. When a code verifier is available, run the
            # multi-candidate sandbox-verified loop; otherwise plain generation.
            # Math/reasoning still uses VibeThinker CLR.
            if self.code_specialist_endpoint:
                task_type, _, _ = self._detect_task_type(query)
                if task_type == "code":
                    if self.code_verifier is not None:
                        print("[Orchestrator] Code task -> code specialist + sandbox verification")
                        result = await self._run_code_specialist_verified(query)
                        self._store_if_verified(query, result)
                        return result
                    print("[Orchestrator] Code task -> code specialist (no verifier)")
                    answer = await self._call_code_specialist(query)
                    return OrchestratorResult(
                        final_answer=answer,
                        route_taken="code_specialist",
                        specialist_used="Code Specialist (ruvltra-claude-code)",
                        routing_confidence=confidence,
                        raw_traces={"task_type": task_type},
                    )
            if self.use_clr:
                clr_result, cache_hit = await self._run_clr_with_cache(query)
                result = OrchestratorResult(
                    final_answer=clr_result.best_answer,
                    route_taken="specialist_clr_cached" if cache_hit else "specialist_clr",
                    specialist_used="VibeThinker-3B + CLR",
                    clr_score=clr_result.best_score,
                    routing_confidence=confidence,
                    raw_traces={"clr_result": self._trim_clr(clr_result), "cache_hit": cache_hit},
                )
                self._store_if_verified(query, result, clr_result=clr_result)
                return result
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
                specialist_result, cache_hit = await self._run_clr_with_cache(query)
                specialist_answer = specialist_result.best_answer
                specialist_score = specialist_result.best_score
                specialist_trace = self._trim_clr(specialist_result)
                specialist_trace["cache_hit"] = cache_hit
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
