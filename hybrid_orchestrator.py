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
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from retrieval import RetrievalBackend

from vibe_clr_async import CLRResult, VibeThinkerCLRAsync
from persistent_cache import (
    CLRResultCache,
    PersistentRouteCache,
    VerifiedTrajectoryStore,
    should_cache,
)
from verifiers import MathVerifier, CodeVerifier, FactualVerifier, SchemaVerifier, LogicVerifier
from sandbox import WarmDockerPool
from math_solver import solve as solve_math

# Sentinel for "argument not provided" — distinct from an explicit None
# (which means "disable the code verifier / verified loop").
_UNSET = object()


def select_verifier(task_type: str, llm_judge=None, prefer_encoder_nli: bool = True):
    """Select a deterministic verifier based on the detected task type.

    Returns None if no verifier applies (conversation, summarization, etc.).
    Conversation and summarization tasks do NOT get a verifier — there is
    no deterministic way to verify "explain X" or "summarize Y".

    Args:
        task_type: the routed task type ("math", "code", "factual",
            "schema", "logic", etc.).
        llm_judge: optional async callable (prompt -> response str) used
            by FactualVerifier as an NLI judge. Typically the
            orchestrator's ``_call_generalist``.
        prefer_encoder_nli: when True (default) and the optional
            ``transformers``+``torch`` deps are installed, prefer an
            :class:`EncoderNLIJudge` (encoder-only NLI model, robust to
            fabrication) over the LLM judge for factual tasks. Phase 3.3:
            this is now the DEFAULT — the encoder NLI judge is preferred
            whenever available, without requiring an opt-in flag. Set to
            False via ``--no-encoder-nli`` / ``VIBE_THINKER_NO_ENCODER_NLI``
            to force the LLM judge. Fail-closed: if the encoder judge
            can't be constructed, falls back to the LLM judge.
    """
    if task_type == "math":
        return MathVerifier()
    if task_type == "code":
        return CodeVerifier()
    if task_type == "factual":
        # Phase 3.3: the encoder-only NLI judge is the DEFAULT when
        # available (no fabrication risk — encoder models can't
        # hallucinate). Fail-closed fallback to the LLM judge when
        # the encoder deps are missing or the judge can't be constructed.
        if prefer_encoder_nli:
            try:
                from verifiers.nli_encoder import EncoderNLIJudge, is_available
                if is_available():
                    return FactualVerifier(
                        llm_judge=EncoderNLIJudge()
                    )
            except Exception as e:
                # Encoder judge unavailable (deps missing, model load
                # failed) — fall back to the LLM judge. Never raise.
                print(f"[select_verifier] encoder NLI unavailable ({e}), "
                      f"falling back to LLM judge")
        return FactualVerifier(llm_judge=llm_judge)
    if task_type == "schema":
        return SchemaVerifier()
    if task_type == "logic":
        return LogicVerifier()
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
# Static analysis fallback (v2.0)
# ====================================================================== #
# Restricted imports that disqualify code from the static analysis
# heuristic score. These modules can access the filesystem, network,
# subprocess, or eval — making them dangerous in unverified code.
_RESTRICTED_IMPORTS = frozenset({
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "http", "urllib", "requests", "ctypes",
    "multiprocessing", "threading", "signal",
    "pickle", "marshal", "shelve",
    "builtins", "importlib",
    "ftplib", "smtplib", "telnetlib", "paramiko",
    "tempfile", "glob",
})


def _static_analysis_fallback(code: str) -> tuple:
    """Static analysis fallback for code verification (v2.0, hardened v3.0).

    When the Generalist fails to generate unit tests, this function
    performs a lightweight static analysis pass on the candidate code:

    1. Parses the code with ``ast.parse`` — if it doesn't parse, score 0.0.
    2. Checks for restricted imports (os, subprocess, socket, etc.) —
       if any are found, score 0.0.
    3. Checks for dynamic import evasion vectors (v3.0):
       - ``__import__('os')`` calls
       - ``importlib.import_module('os')`` calls
       - ``getattr(__builtins__, 'eval')`` style reflection
       - Any reference to ``importlib`` or ``__builtins__`` in the AST
       If any are found, score 0.0.
    4. If the code parses cleanly and has no restricted imports or
       evasion vectors, assign a partial heuristic score of 0.4 (capped
       — this is NOT full verification, just a "syntactically valid and
       not obviously dangerous" signal).

    Returns (score, issues) where issues is a list of strings describing
    any problems found (empty list if score > 0.0).
    """
    import ast as _ast
    issues = []

    # 1. Parse check.
    try:
        tree = _ast.parse(code)
    except SyntaxError as e:
        return (0.0, [f"syntax error: {e}"])

    # 2. Restricted import check.
    restricted_found = []
    evasion_found = []
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _RESTRICTED_IMPORTS:
                    restricted_found.append(alias.name)
        elif isinstance(node, _ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in _RESTRICTED_IMPORTS:
                    restricted_found.append(node.module)
        # 3. Dynamic import evasion checks (v3.0).
        elif isinstance(node, _ast.Call):
            func = node.func
            # __import__('os') — direct builtin call.
            if isinstance(func, _ast.Name) and func.id == "__import__":
                evasion_found.append("__import__() call")
            # importlib.import_module('os') — method call on importlib.
            elif isinstance(func, _ast.Attribute) and func.attr == "import_module":
                if isinstance(func.value, _ast.Name) and func.value.id == "importlib":
                    evasion_found.append("importlib.import_module() call")
            # exec()/eval() — dynamic code execution.
            elif isinstance(func, _ast.Name) and func.id in ("exec", "eval"):
                evasion_found.append(f"{func.id}() call")
            # getattr(__builtins__, 'eval') — reflection-based access.
            elif isinstance(func, _ast.Name) and func.id == "getattr":
                # Check if any argument references __builtins__.
                for arg in node.args:
                    if isinstance(arg, _ast.Name) and arg.id == "__builtins__":
                        evasion_found.append("getattr(__builtins__, ...) call")
                    elif isinstance(arg, _ast.Constant) and isinstance(arg.value, str):
                        if arg.value in ("__import__", "eval", "exec", "compile"):
                            # getattr(x, 'eval') — flag if the target is dangerous.
                            pass  # Caught by the __builtins__ check above.
            # builtins.__import__('os') — attribute access on builtins module.
            elif isinstance(func, _ast.Attribute) and func.attr == "__import__":
                if isinstance(func.value, _ast.Name) and func.value.id == "builtins":
                    evasion_found.append("builtins.__import__() call")
        # Any Name node referencing importlib, __builtins__, or builtins.
        elif isinstance(node, _ast.Name):
            if node.id == "importlib":
                evasion_found.append("importlib reference")
            elif node.id == "__builtins__":
                evasion_found.append("__builtins__ reference")
            elif node.id == "builtins":
                evasion_found.append("builtins reference")

    if restricted_found:
        issues.append(f"restricted imports: {', '.join(restricted_found)}")
        return (0.0, issues)
    if evasion_found:
        issues.append(f"dynamic import evasion: {', '.join(evasion_found)}")
        return (0.0, issues)

    # 4. Clean parse, no restricted imports, no evasion — partial score.
    return (0.4, [])


# ====================================================================== #
# Wasmtime sandbox fallback (v3.1)
# ====================================================================== #
# Score assigned when code runs successfully in a Wasm/Docker sandbox
# without trapping. This is the highest self-claim cap — the code
# actually executed in isolation, proving it doesn't crash or trap.
# It's NOT full verification (no unit tests), but it's a much stronger
# signal than AST static analysis alone.
_WASM_SANDBOX_SCORE = 0.65


async def _wasmtime_sandbox_fallback(
    code: str,
    code_verifier: Optional["CodeVerifier"] = None,
) -> tuple:
    """Sandboxed execution fallback for code verification (v3.1).

    When the Generalist fails to generate unit tests, this function
    provides a real security boundary by executing the candidate code
    in a sandbox where file system and network handles literally do
    not exist. This replaces the v2.0 AST static analysis, which was
    trivially bypassable via obfuscation (e.g.,
    ``__import__(chr(111)+chr(115))``).

    Execution priority:
      1. **Wasmtime** — if the ``wasmtime`` package is installed AND a
         pre-compiled Python-to-Wasm module is available (configured via
         ``VIBE_WASM_PYTHON_MODULE`` env var), run the code in a strict
         Wasm sandbox. File system and network handles do not exist in
         Wasm — the code cannot exfiltrate data even if it tries.
      2. **Docker sandbox** — if a CodeVerifier with a sandbox executor
         is available, execute the code in a Docker container with
         ``--network=none`` and ``--read-only``. This provides the same
         isolation as Wasm (no network, no filesystem writes) using the
         existing sandbox infrastructure.
      3. **None** — if neither wasmtime nor a Docker sandbox is
         available, return ``(None, [])`` so the caller falls back to
         the deprecated AST static analysis with a warning.

    Returns:
        (score, issues) where:
        - score = 0.65 if the code runs without trapping/erroring
        - score = 0.0 if the code traps/errors (with issues describing why)
        - score = None if no sandbox is available (caller falls back to AST)
    """
    # 1. Try wasmtime (if installed + a Wasm Python module is configured).
    wasm_module_path = os.environ.get("VIBE_WASM_PYTHON_MODULE", "")
    if wasm_module_path:
        try:
            import wasmtime  # type: ignore
            from wasmtime import Engine, Module, Store, Instance  # type: ignore

            engine = Engine()
            module = Module.from_file(engine, wasm_module_path)
            store = Store(engine)
            instance = Instance(store, module, [])
            # The Wasm module exposes a `run_python` function that takes
            # the source code as a string and returns 0 on success.
            run = instance.exports(store)["run_python"]
            result = run(store, code)
            if result == 0:
                return (_WASM_SANDBOX_SCORE, [])
            return (0.0, [f"wasm sandbox: code trapped (exit code {result})"])
        except ImportError:
            pass  # wasmtime not installed — fall through to Docker
        except Exception as e:
            # Wasm execution failed — the code is buggy or dangerous.
            return (0.0, [f"wasm sandbox: execution failed: {e}"])

    # 2. Try Docker sandbox execution (if a CodeVerifier is available).
    if code_verifier is not None and code_verifier.executor is not None:
        try:
            # Execute the code with no tests — just check it runs without
            # error. The sandbox provides --network=none and --read-only,
            # so the code cannot exfiltrate data or persist changes.
            result = await code_verifier.executor.execute(
                code, timeout=5.0, network=False, memory_limit="128m",
            )
            if result.exit_code == 0:
                return (_WASM_SANDBOX_SCORE, [])
            # Non-zero exit — the code errored or crashed.
            error_msg = result.stderr.strip()[:200] if result.stderr else ""
            return (0.0, [f"sandbox: exit code {result.exit_code}: {error_msg}"])
        except Exception as e:
            return (0.0, [f"sandbox: execution failed: {e}"])

    # 3. No sandbox available — caller falls back to AST (with warning).
    return (None, [])


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
        max_repair_attempts: int = 2,
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
        local_specialist_pool_size: int = 1,
        agentdb_url: Optional[str] = None,
        retrieval_backend: Optional["RetrievalBackend"] = _UNSET,
        network_allowlist: Optional["NetworkAllowList"] = None,
        dns_resolver: Optional[str] = None,
        sandbox_image: Optional[str] = None,
        proxy_egress: Optional[str] = None,
        use_structured_output: bool = False,
        specialist_transport: str = "completion",
        specialist_api_key: Optional[str] = None,
        specialist_model_name: Optional[str] = None,
        max_parse_repairs: int = 2,
        prefer_encoder_nli: bool = True,
        sona_sync_url: Optional[str] = None,
        sona_sync_interval: int = 3600,
        federation_secret: Optional[str] = None,
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
        # Iterative code repair (v0.4.0): when the best candidate fails with
        # a code bug (ASSERTION_FAILED / IMPORT_ERROR — not a broken test),
        # feed the failing code + error back to the code specialist for a
        # targeted repair. Bounded by max_repair_attempts. Fail-closed: if no
        # repair passes, the best-effort unverified result is returned. Set to
        # 0 to disable. CLI: --max-repair-attempts / MAX_REPAIR_ATTEMPTS env.
        self.max_repair_attempts = max(0, max_repair_attempts)
        # Active retrieval backend (v0.4.0). When configured, factual tasks
        # fetch real source text from a search API (Serper / SearchApi) and
        # feed it to the FactualVerifier's NLI judge. When None (default),
        # no retrieval — the verifier returns unsupported_factual, which is
        # the honest, unchanged fail-closed behavior. _UNSET -> auto-detect
        # from SERPER_API_KEY / SEARCHAPI_API_KEY env vars.
        if retrieval_backend is _UNSET:
            from retrieval import make_retrieval_backend
            self._retrieval_backend = make_retrieval_backend()
        else:
            self._retrieval_backend = retrieval_backend
        # _UNSET -> default CodeVerifier (safe, fail-closed sandbox).
        # None -> explicitly disabled (plain generation, no verification).
        self.code_verifier = CodeVerifier() if code_verifier is _UNSET else code_verifier
        # Apply the network allow-list to the code verifier's executor
        # (v0.4.0). When set, the Docker sandbox uses iptables egress
        # filtering instead of --network=none. Only DockerSandboxExecutor
        # supports allow-lists; WarmDockerPool and others ignore it.
        if network_allowlist is not None and self.code_verifier is not None:
            executor = getattr(self.code_verifier, "executor", None)
            if executor is not None and hasattr(executor, "set_allowlist"):
                executor.set_allowlist(network_allowlist)
                if dns_resolver and hasattr(executor, "set_dns_resolver"):
                    executor.set_dns_resolver(dns_resolver)
                print(f"[Orchestrator] Network allow-list applied to "
                      f"{type(executor).__name__}"
                      + (f" (DNS restricted to {dns_resolver})" if dns_resolver else ""))
            # Override the sandbox image if specified.
            if sandbox_image and executor is not None and hasattr(executor, "image"):
                executor.image = sandbox_image
        # Apply SNI proxy egress mode (v0.4.1). When set, the sandbox
        # routes traffic through the proxy instead of using iptables.
        # This solves CDN IP rotation — the proxy checks the domain (SNI),
        # not the IP. The proxy must be running separately (sandbox.sni_proxy
        # or the --envoy-sidecar flag).
        # v2.0: SNI-proxy is the ONLY egress mode when an allow-list is
        # present. The v0.4.0 iptables path was removed.
        if self.code_verifier is not None:
            executor = getattr(self.code_verifier, "executor", None)
            if executor is not None:
                if proxy_egress and hasattr(executor, "set_proxy_egress"):
                    executor.set_proxy_egress(proxy_egress)
                    print(f"[Orchestrator] SNI proxy egress enabled: {proxy_egress}"
                          + (f" (domain-level filtering)" if network_allowlist else ""))
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
        # v1.1: prefer the encoder-only NLI judge for factual tasks when
        # the optional nli extra is installed. Opt-in (default False) —
        # the encoder model is downloaded from HuggingFace on first use.
        self.prefer_encoder_nli = prefer_encoder_nli

        # Shared HTTP session for all generalist/code-specialist calls.
        # Eagerly created in start() (inside the event loop) to avoid a
        # lazy-init race where multiple concurrent first-requests each
        # create overlapping sessions. A lock guards the fallback lazy
        # path for callers that skip start(). Reused across all calls to
        # avoid TCP connection overhead. Closed in cleanup().
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

        self.reasoner = VibeThinkerCLRAsync(
            server_url=vibe_endpoint,
            k=clr_k,
            max_concurrent=max_concurrent_clr,
            fast_specialist=fast_specialist,
            local_model=local_specialist_model,
            local_n_ctx=local_specialist_n_ctx,
            local_n_threads=local_specialist_n_threads,
            local_pool_size=local_specialist_pool_size,
            use_structured_output=use_structured_output,
            specialist_transport=specialist_transport,
            specialist_api_key=specialist_api_key,
            specialist_model_name=specialist_model_name,
            max_parse_repairs=max_parse_repairs,
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
                    agentdb_url=agentdb_url,
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
                    agentdb_url=agentdb_url,
                )
            except Exception as e:
                print(f"[Warning] Could not load trajectory store: {e}")
                self.use_trajectory_store = False
                self.trajectory_store = None
        else:
            self.trajectory_store = None

        # v2.0: SONA trajectory recorder (optional, in-process PyO3).
        # When the ruvllm_py binding is installed, verified trajectories
        # are also recorded into the SONA learning engine for continuous
        # model improvement (LoRA adaptation, pattern learning).
        self._sona_recorder = None
        try:
            from ruvllm_adapter import is_ruvllm_binding_available
            if is_ruvllm_binding_available():
                from ruvllm_py import SonaRecorder
                self._sona_recorder = SonaRecorder(
                    hidden_dim=256, embedding_dim=384, quality_threshold=0.7,
                )
                print("[Orchestrator] SONA trajectory recorder initialized "
                      "(ruvllm_py binding)")
        except Exception:
            pass  # Fail silently — SONA is optional

        # v3.0: SONA gossip protocol — periodic sync of learned patterns
        # with the federation coordinator. When enabled, the orchestrator
        # exports its SONA patterns and imports global patterns from
        # other nodes every sync_interval_secs seconds.
        self._sona_sync_interval = sona_sync_interval
        self._sona_sync_url = sona_sync_url
        self._sona_sync_task = None
        self._sona_sync_count = 0
        self._node_id = os.uname().nodename

        # v3.0: Fernet instance for decrypting encrypted federation
        # responses (claim responses, SONA sync GET responses).
        self._fernet = None
        if federation_secret:
            try:
                from cryptography.fernet import Fernet
                import base64
                import hashlib
                key = base64.urlsafe_b64encode(
                    hashlib.sha256(federation_secret.encode()).digest()
                )
                self._fernet = Fernet(key)
            except ImportError:
                pass

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
          - compute_limits: suggested sandbox resource limits
            {memory, timeout} for code execution (v0.4.0). The CodeVerifier
            uses these instead of the hardcoded 128m / 5.0s defaults — a
            data-processing script gets more headroom than a one-liner.
          - reason: human-readable explanation
        """
        route, confidence = self._classify_route(query)
        task_type, requires_tools, requires_model = self._detect_task_type(query)

        # Human review recommended for low-confidence routing or unknown task types
        requires_human_review = confidence < 0.65 or task_type == "unknown"

        # Suggest sandbox resource limits based on task complexity (v0.4.0).
        # The CodeVerifier reads these from the verifier context to size the
        # sandbox dynamically instead of using the hardcoded 128m / 5.0s.
        compute_limits = self._suggest_compute_limits(query, task_type)

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
            "compute_limits": compute_limits,
            "reason": ", ".join(reasons),
        }

    # Keywords that signal heavy computation — bump the sandbox limits.
    _HEAVY_COMPUTE_KEYWORDS = frozenset({
        "dataframe", "pandas", "numpy", "tensor", "torch", "tensorflow",
        "ml", "machine learning", "model", "train", "inference",
        "matrix", "large", "dataset", "csv", "json file", "parse",
        "sort", "merge", "graph", "tree", "recursive", "backtracking",
        "dynamic programming", "dp", "simulation", "monte carlo",
        "scrape", "crawl", "download", "process", "transform",
        "encrypt", "decrypt", "hash", "compress", "image", "audio",
    })

    # Default sandbox limits per task type. These are conservative starting
    # points; _suggest_compute_limits bumps them when heavy-compute keywords
    # are detected. The CodeVerifier's own defaults (128m / 5.0s) remain the
    # ultimate fallback when no compute_limits are provided.
    _DEFAULT_COMPUTE_LIMITS = {
        "math": {"memory": "64m", "timeout": 5.0},
        "code": {"memory": "128m", "timeout": 10.0},
        "schema": {"memory": "64m", "timeout": 5.0},
        "logic": {"memory": "128m", "timeout": 10.0},
        "factual": {"memory": "64m", "timeout": 5.0},
        "planning": {"memory": "64m", "timeout": 5.0},
        "retrieval": {"memory": "64m", "timeout": 5.0},
        "summarization": {"memory": "64m", "timeout": 5.0},
        "conversation": {"memory": "64m", "timeout": 5.0},
        "unknown": {"memory": "64m", "timeout": 5.0},
    }

    def _suggest_compute_limits(
        self, query: str, task_type: str
    ) -> Dict[str, Any]:
        """Suggest sandbox resource limits based on task complexity.

        Returns a dict with ``memory`` (Docker memory spec string) and
        ``timeout`` (seconds). The CodeVerifier reads these from the
        verifier context to size the sandbox dynamically.

        Heuristic: start from the task-type default, then bump memory and
        timeout when heavy-computation keywords are detected in the query
        (e.g. "dataframe", "pandas", "matrix", "recursive"). A simple
        math one-liner gets 64m / 5s; a data-processing script gets 512m
        / 30s. The limits are advisory — the CodeVerifier's own defaults
        apply when compute_limits is absent (backward-compatible).
        """
        base = self._DEFAULT_COMPUTE_LIMITS.get(
            task_type, self._DEFAULT_COMPUTE_LIMITS["unknown"]
        )
        memory = base["memory"]
        timeout = base["timeout"]

        query_lower = query.lower()
        heavy_hits = sum(
            1 for kw in self._HEAVY_COMPUTE_KEYWORDS if kw in query_lower
        )
        if heavy_hits > 0:
            # Bump memory: 128m -> 256m -> 512m based on keyword density.
            if heavy_hits >= 3:
                memory = "512m"
            elif heavy_hits >= 2:
                memory = "256m"
            else:
                memory = "256m" if task_type == "code" else "128m"
            # Bump timeout: +10s per heavy keyword, capped at 60s.
            timeout = min(base["timeout"] + heavy_hits * 10.0, 60.0)

        return {
            "memory": memory,
            "timeout": timeout,
            "heavy_compute_keywords": heavy_hits,
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
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get the shared HTTP session, creating it lazily if needed.

        The session is reused across all generalist/code-specialist calls to
        avoid TCP connection overhead. Creation is guarded by a lock to
        prevent concurrent first-requests from creating overlapping sessions.
        Prefer calling ``start()`` before any requests to create the session
        eagerly.
        """
        if self._http_session is None or self._http_session.closed:
            async with self._session_lock:
                # Double-check after acquiring the lock.
                if self._http_session is None or self._http_session.closed:
                    self._http_session = aiohttp.ClientSession()
        return self._http_session

    async def _call_generalist(self, query: str, max_tokens: int = 4096) -> str:
        """Call the generalist model via the OpenAI-compatible
        /v1/chat/completions endpoint so llama-server applies the model's
        own baked-in chat template (Llama 3.2 uses <|start_header_id|>, not
        ChatML). Falls back to /completion with ChatML if the chat endpoint
        fails. Raises RuntimeError if both endpoints fail — callers must
        handle the exception, not silently proceed with an error string."""
        session = await self._get_session()
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
        session = await self._get_session()
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
        session = await self._get_session()
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

    _CODE_REPAIR_PROMPT = (
        "A previous solution to the request below FAILED its unit tests. "
        "Fix the bug. Output ONLY a single ```python code block with the "
        "complete, runnable corrected code — no explanation.\n\n"
        "Request:\n{query}\n\n"
        "Tests it must pass:\n{tests}\n\n"
        "Failing code:\n```python\n{failing_code}\n```\n\n"
        "Error:\n{error}\n\n"
        "Output the corrected ```python code block now:"
    )

    @staticmethod
    def _extract_python_block(text: str) -> str:
        """Extract the best Python code block from text.

        v0.4.0: instead of blindly taking the first fenced block (which
        might be a bash install command or pseudo-code example), parse
        ALL matched blocks with ast.parse() and select the LAST valid
        Python AST. LLMs typically output reasoning/explanation first
        and the final solution last.

        Handles common model output variants:
          - ```python\n...```  (standard)
          - ```py\n...```      (abbreviated language tag)
          - ```\n...```        (no language tag)
          - bare code with no fence (returns the stripped text)

        v0.4.0: line-by-line parsing replaces the regex approach. The
        old regex couldn't distinguish opening and closing fences,
        causing it to match text between a closing ```bash fence and
        an opening ```python fence as if it were a code block.

        Fallback order:
          1. Last fenced block that parses as valid Python AST
          2. First fenced block (if none parse — let the verifier reject it)
          3. The stripped text itself (no fences found)
        """
        import ast as _ast

        # v0.4.0: parse line by line to correctly handle fenced blocks.
        # This avoids the regex ambiguity between opening and closing fences.
        lines = text.split("\n")
        blocks: list = []  # list of (tag, content) tuples
        current_block = None
        current_tag = None
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                if current_block is not None:
                    # Closing fence — save the block
                    blocks.append((current_tag, "\n".join(current_block)))
                    current_block = None
                    current_tag = None
                else:
                    # Opening fence — extract the language tag
                    tag = stripped[3:].strip().lower()
                    current_tag = tag if tag else None
                    current_block = []
            elif current_block is not None:
                current_block.append(line)

        # Filter to only Python blocks (python, py, or no tag).
        # Non-Python blocks (bash, sh, json, etc.) are excluded.
        python_blocks = [
            content for tag, content in blocks
            if tag in ("python", "py", None)
        ]

        if not python_blocks:
            # No fenced Python block found — return the stripped text
            # as-is (the model may have emitted bare code without fences).
            return text.strip()

        # v0.4.0: try all blocks in reverse order (last first). LLMs
        # typically output explanation first, final solution last.
        # Select the last block that parses as valid Python.
        for block in reversed(python_blocks):
            stripped = block.strip()
            try:
                _ast.parse(stripped)
                return stripped
            except SyntaxError:
                continue

        # No block parsed as valid Python. Return the last block anyway —
        # the verifier will reject it, but at least we're evaluating the
        # model's final output, not its first (which might be bash/pseudo).
        return python_blocks[-1].strip()

    @staticmethod
    def _validate_test_spec(tests: str) -> bool:
        """Reject vacuous test specs that would pass any candidate.

        A test spec is vacuous if every ``assert`` statement tests only a
        constant expression (e.g. ``assert True``, ``assert 1 == 1``,
        ``assert "yes"``) without referencing any function call or variable
        from the solution. Such tests would mark garbage code as verified
        (score 1.0), defeating the entire verification loop.

        Returns True if the spec contains at least one non-vacuous assert
        (one whose test expression contains a Call or Name node), False
        otherwise (including unparseable specs).
        """
        import ast as _ast
        try:
            tree = _ast.parse(tests)
        except SyntaxError:
            return False

        def _contains_ref(node):
            """True if the expression contains a Name or Call node."""
            for child in _ast.walk(node):
                if isinstance(child, (_ast.Name, _ast.Call)):
                    return True
            return False

        has_nonvacuous = False
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Assert):
                if _contains_ref(node.test):
                    has_nonvacuous = True
                    break
        return has_nonvacuous

    async def _generate_test_spec(self, query: str) -> Optional[str]:
        """Ask the generalist to produce unit-test asserts for the query.

        Returns the extracted Python assert code, or None if the generalist
        failed or produced nothing parseable. This is the "Software Architect"
        step: the generalist defines correctness before the code specialist
        writes any code.

        Validates the spec with :meth:`_validate_test_spec` to reject
        vacuous tests (e.g. ``assert True``) that would pass any candidate.
        If the first attempt fails validation, retries once with feedback
        telling the generalist what was wrong (up to 2 attempts total).
        """
        feedback: Optional[str] = None
        for attempt in range(2):
            if feedback is not None:
                prompt = (
                    self._TEST_SPEC_PROMPT.format(query=query)
                    + f"\n\n--- FEEDBACK ---\n"
                    f"The previous test spec was rejected because: {feedback}\n"
                    f"Write tests that actually call the functions/classes "
                    f"the solution should define. Do NOT use vacuous asserts "
                    f"like `assert True`."
                )
            else:
                prompt = self._TEST_SPEC_PROMPT.format(query=query)
            try:
                raw = await self._call_generalist(prompt, max_tokens=1024)
            except Exception as e:
                print(f"[CodeLoop] Test-spec generation failed (attempt "
                      f"{attempt + 1}): {e}")
                return None
            tests = self._extract_python_block(raw)
            if not tests or "assert" not in tests:
                feedback = "no assert statements found in the output"
                print(f"[CodeLoop] Generalist produced no usable asserts "
                      f"(attempt {attempt + 1}) — {'retrying' if attempt == 0 else 'giving up'}")
                if attempt == 0:
                    continue
                return None
            if not self._validate_test_spec(tests):
                feedback = ("all asserts are vacuous (e.g. `assert True`) — "
                            "they must reference functions or variables from "
                            "the solution")
                print(f"[CodeLoop] Generalist produced vacuous test spec "
                      f"(attempt {attempt + 1}) — {'retrying' if attempt == 0 else 'giving up'}")
                if attempt == 0:
                    continue
                return None
            return tests
        return None

    async def _mutation_check(
        self, code: str, tests: str
    ) -> tuple[bool, Optional[dict]]:
        """Mutation testing for vacuous-test detection (Phase 3.1).

        Injects a known bug into ``code`` and re-runs ``tests`` against
        the mutated code. If the mutated (broken) code still passes the
        tests, the tests are vacuous — they cannot distinguish correct
        from incorrect code.

        Returns:
            (tests_are_vacuous, mutation_details)
            - tests_are_vacuous: True if the mutated code passed (tests
              are vacuous), False if the mutated code correctly failed
              (tests are meaningful) or no mutation could be applied.
            - mutation_details: a dict with the mutation operator and
              description for audit, or None if no mutation was applied.
        """
        from verifiers.mutation import mutate_code
        mutation = mutate_code(code)
        if not mutation.applied:
            # Code too simple to mutate — not a failure, skip the check.
            return False, None
        try:
            mutated_result = await self.code_verifier.verify(
                "", mutation.mutated_code, {"unit_tests": tests}
            )
        except Exception as e:
            # If the mutation check itself errors, fail-safe: treat the
            # tests as NOT vacuous (don't reject a passing candidate on
            # an infrastructure error). Log the issue.
            print(f"[CodeLoop] Mutation check errored (treating as non-vacuous): {e}")
            return False, {"operator": mutation.operator, "error": str(e)}
        details = {
            "operator": mutation.operator,
            "description": mutation.description,
            "mutated_passed": mutated_result.verified,
        }
        if mutated_result.verified:
            # The mutated (broken) code passed the tests -> VACUOUS.
            print(f"[CodeLoop] VACUOUS TESTS detected: mutated code "
                  f"({mutation.operator}: {mutation.description}) still "
                  f"passed — rejecting candidate, triggering test retry")
            return True, details
        print(f"[CodeLoop] Mutation check passed: mutated code "
              f"({mutation.operator}) correctly failed — tests are meaningful")
        return False, details

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

        Test-feedback loop (v0.3.6): if ALL candidates fail with TEST_ERROR
        (the test harness itself crashed — not an assertion failure), the
        generalist gets ONE retry to rewrite the tests with the error fed
        back as context. This distinguishes "bad tests" from "bad code":
          - ASSERTION_FAILED / IMPORT_ERROR = candidate is wrong → no retry
          - TEST_ERROR = the test spec is broken → retry once with feedback

        Iterative code repair (v0.4.0): if the best candidate fails with a
        code bug (ASSERTION_FAILED / IMPORT_ERROR — a real defect in the
        candidate, NOT a broken test), the failing code + error are fed
        back to the code specialist for a targeted repair. Up to
        max_repair_attempts rounds. This is distinct from the test-feedback
        loop above (which only fires when ALL candidates fail with
        TEST_ERROR). Fail-closed: if no repair passes, the best-effort
        unverified result is returned with score 0.0.

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
            # v3.1: Sandbox fallback. If the Generalist fails to generate
            # unit tests, execute the candidate code in a sandbox (Wasm or
            # Docker) where file system and network handles do not exist.
            # This replaces the v2.0 AST static analysis, which was trivially
            # bypassable via obfuscation (e.g., __import__(chr(111)+chr(115))).
            # If the code runs without trapping, assign a partial score of
            # 0.65 (the highest self-claim cap — NOT full verification, but
            # a real security boundary).
            sandbox_score, sandbox_issues = await _wasmtime_sandbox_fallback(
                answer, code_verifier=self.code_verifier,
            )
            if sandbox_score is not None and sandbox_score > 0.0:
                print(f"[CodeLoop] Sandbox fallback: score={sandbox_score} "
                      f"(code ran in sandbox without trapping)")
                return OrchestratorResult(
                    final_answer=answer,
                    route_taken="code_specialist_sandbox",
                    specialist_used="Code Specialist (ruvltra-claude-code)",
                    clr_score=sandbox_score,
                    routing_confidence=0.3,
                    raw_traces={
                        "verified": False,
                        "reason": "no test spec generated — sandbox fallback",
                        "sandbox": True,
                        "sandbox_score": sandbox_score,
                        "sandbox_issues": sandbox_issues,
                    },
                )
            if sandbox_score is not None and sandbox_score == 0.0:
                # The code trapped/errored in the sandbox — it's buggy or
                # dangerous. Return with score 0.0 and the error.
                print(f"[CodeLoop] Sandbox fallback: code trapped — "
                      f"{sandbox_issues}")
                return OrchestratorResult(
                    final_answer=answer,
                    route_taken="code_specialist_sandbox_failed",
                    specialist_used="Code Specialist (ruvltra-claude-code)",
                    clr_score=0.0,
                    routing_confidence=0.0,
                    raw_traces={
                        "verified": False,
                        "reason": "code trapped in sandbox",
                        "sandbox": True,
                        "sandbox_issues": sandbox_issues,
                    },
                )
            # No sandbox available — fall back to deprecated AST static
            # analysis with a warning. AST is NOT a security boundary.
            print("[CodeLoop] WARNING: no sandbox available — falling back "
                  "to AST static analysis (NOT a security boundary)")
            static_score, static_issues = _static_analysis_fallback(answer)
            if static_score > 0.0:
                print(f"[CodeLoop] AST fallback: score={static_score} "
                      f"(parses OK, no restricted imports)")
                return OrchestratorResult(
                    final_answer=answer,
                    route_taken="code_specialist_static_analysis",
                    specialist_used="Code Specialist (ruvltra-claude-code)",
                    clr_score=static_score,
                    routing_confidence=0.3,
                    raw_traces={
                        "verified": False,
                        "reason": "no test spec generated — AST static analysis fallback (no sandbox available)",
                        "static_analysis": True,
                        "static_score": static_score,
                        "static_issues": static_issues,
                    },
                )
            return OrchestratorResult(
                final_answer=answer,
                route_taken="code_specialist_unverified",
                specialist_used="Code Specialist (ruvltra-claude-code)",
                clr_score=0.0,
                routing_confidence=0.0,
                raw_traces={"verified": False, "reason": "no test spec generated"},
            )

        # --- Test-feedback loop: up to 2 attempts (initial + 1 retry) ---
        # On the first attempt, use the original tests. If ALL candidates fail
        # with TEST_ERROR (the test harness crashed, not the candidate), feed
        # the error back to the generalist and retry test generation once.
        test_error_feedback: Optional[str] = None
        for attempt in range(2):
            if test_error_feedback is not None:
                print(f"[CodeLoop] Tests crashed on attempt {attempt} — asking "
                      f"generalist to rewrite tests with error feedback")
                tests = await self._generate_test_spec(
                    f"{query}\n\n--- TEST FEEDBACK ---\n"
                    f"The previous test spec crashed with this error:\n"
                    f"{test_error_feedback}\n\n"
                    f"Rewrite the unit tests to be syntactically correct and "
                    f"robust. Ensure all referenced functions/classes exist "
                    f"in the solution before asserting on them."
                )
                if tests is None:
                    print("[CodeLoop] Retry test generation failed — "
                          "single-candidate unverified generation")
                    answer = await self._call_code_specialist(query)
                    return OrchestratorResult(
                        final_answer=answer,
                        route_taken="code_specialist_unverified",
                        specialist_used="Code Specialist (ruvltra-claude-code)",
                        clr_score=0.0,
                        routing_confidence=0.0,
                        raw_traces={
                            "verified": False,
                            "reason": "test spec retry failed after TEST_ERROR",
                        },
                    )

            # Generate N candidates in parallel with diverse temperatures.
            # Prepend verified trajectories as few-shot context (self-improving).
            temps = [0.2] + [0.7] * (self.code_candidates - 1)
            few_shot = self._get_few_shot_context(query, task_type="code")
            query_with_ctx = f"{few_shot}\n{query}" if few_shot else query
            gen_prompt = self._CODE_GEN_PROMPT.format(query=query_with_ctx, tests=tests)
            if few_shot:
                print(f"[CodeLoop] Retrieved verified examples as few-shot context")
            print(f"[CodeLoop] Generating {self.code_candidates} candidates + "
                  f"test spec verification (attempt {attempt + 1}/2)")
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
            # Parallel verification: all candidates verified concurrently via
            # asyncio.gather. The warm pool's multiple containers handle
            # parallelism naturally (round-robin dispatch). This replaces the
            # old sequential loop — with 6 candidates and 3 warm containers,
            # verification time drops from ~1.2s to ~0.4s.
            async def _verify_one(idx: int, code: str):
                result = await self.code_verifier.verify(
                    query, code, {"unit_tests": tests}
                )
                return idx, result

            verify_tasks = [
                _verify_one(i, code) for i, (raw, code) in enumerate(candidates)
            ]
            verify_results = await asyncio.gather(*verify_tasks)

            # Process results in candidate order (gather preserves order).
            verification_traces = []
            test_error_count = 0
            first_test_error: Optional[str] = None
            for i, result in verify_results:
                trace = {
                    "candidate_index": i,
                    "verified": result.verified,
                    "score": result.score,
                    "method": result.method,
                    "error": result.error,
                }
                verification_traces.append(trace)
                if result.verified:
                    # Phase 3.1: Mutation testing — before accepting a
                    # 1.0 verified score, inject a known bug into the
                    # winning code and re-run the tests. If the mutated
                    # (broken) code still passes, the tests are vacuous.
                    vacuous, mutation_details = await self._mutation_check(
                        candidates[i][1], tests
                    )
                    if vacuous:
                        # Tests are vacuous — reject, score 0.0, trigger
                        # the test-feedback loop (same as TEST_ERROR).
                        test_error_count = len(candidates)
                        first_test_error = (
                            f"VACUOUS_TESTS: mutation ({mutation_details['operator']}: "
                            f"{mutation_details['description']}) still passed — "
                            f"tests cannot distinguish correct from incorrect code"
                        )
                        verification_traces[-1]["mutation_check"] = mutation_details
                        verification_traces[-1]["vacuous_tests"] = True
                        if attempt == 0:
                            test_error_feedback = first_test_error
                            break  # break out of the candidate loop, retry tests
                        # Already retried — fall through to best-effort.
                        break
                    print(f"[CodeLoop] Candidate {i} PASSED verification — returning")
                    return OrchestratorResult(
                        final_answer=candidates[i][1],
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
                            "test_spec_retry": attempt,
                            "mutation_check": mutation_details,
                        },
                    )
                print(f"[CodeLoop] Candidate {i} failed: {result.error or 'not verified'}")
                # Detect test-harness crashes (TEST_ERROR = the test itself is
                # broken, not the candidate). IMPORT_ERROR and ASSERTION_FAILED
                # are candidate problems — they don't trigger a test retry.
                if result.error and "TEST_ERROR" in result.error:
                    test_error_count += 1
                    if first_test_error is None:
                        first_test_error = result.error

            # If ALL candidates failed with TEST_ERROR, the test spec is likely
            # broken (not the code). Retry test generation once with feedback.
            if (
                test_error_count == len(candidates)
                and first_test_error is not None
                and attempt == 0
            ):
                test_error_feedback = first_test_error
                continue  # retry with rewritten tests

            # Not all TEST_ERROR, or we already retried — return best-effort.
            break

        # --- Iterative code repair (v0.4.0) ---
        # If the best candidate failed with a code bug (ASSERTION_FAILED or
        # IMPORT_ERROR — a real defect in the candidate, NOT a broken test),
        # feed the failing code + error back to the code specialist for a
        # targeted repair. This is distinct from the test-feedback loop above
        # (which fires only when ALL candidates fail with TEST_ERROR = the
        # tests themselves are broken). Bounded by max_repair_attempts;
        # fail-closed: if no repair passes, the best-effort unverified result
        # below is returned with score 0.0.
        repair_attempts_made = 0
        for repair_attempt in range(self.max_repair_attempts):
            # Pick the candidate to repair: first one whose error indicates a
            # code bug. If none are code bugs (e.g. all timeouts or all
            # TEST_ERROR — the latter already handled above), stop: repair
            # cannot fix a timeout or a broken test spec.
            repair_idx: Optional[int] = None
            repair_error: Optional[str] = None
            for i, result in verify_results:
                if result.error and any(
                    m in result.error for m in ("ASSERTION_FAILED", "IMPORT_ERROR")
                ):
                    repair_idx = i
                    repair_error = result.error
                    break
            if repair_idx is None:
                break  # nothing repairable

            failing_code = candidates[repair_idx][1]
            print(f"[CodeLoop] Repair attempt {repair_attempt + 1}/"
                  f"{self.max_repair_attempts}: feeding error back to code "
                  f"specialist (candidate {repair_idx} failed: {repair_error})")
            repair_prompt = self._CODE_REPAIR_PROMPT.format(
                query=query_with_ctx, tests=tests,
                failing_code=failing_code, error=repair_error,
            )
            repair_raw = await asyncio.gather(
                *[self._call_code_specialist(repair_prompt, max_tokens=2048)
                  for _ in temps],
                return_exceptions=True,
            )
            repair_candidates = [
                (raw, self._extract_python_block(raw))
                for raw in repair_raw if isinstance(raw, str)
            ]
            if not repair_candidates:
                continue

            async def _repair_verify(idx: int, code: str):
                r = await self.code_verifier.verify(
                    query, code, {"unit_tests": tests}
                )
                return idx, r

            repair_results = await asyncio.gather(
                *[_repair_verify(i, code)
                  for i, (raw, code) in enumerate(repair_candidates)]
            )
            for i, result in repair_results:
                verification_traces.append({
                    "candidate_index": i,
                    "verified": result.verified,
                    "score": result.score,
                    "method": result.method,
                    "error": result.error,
                    "repair_attempt": repair_attempt + 1,
                })
                if result.verified:
                    # Phase 3.1: Mutation testing on repaired candidates too.
                    vacuous, mutation_details = await self._mutation_check(
                        repair_candidates[i][1], tests
                    )
                    if vacuous:
                        print(f"[CodeLoop] Repaired candidate {i} passed but "
                              f"tests are VACUOUS — rejecting")
                        verification_traces[-1]["mutation_check"] = mutation_details
                        verification_traces[-1]["vacuous_tests"] = True
                        continue  # try next repair candidate
                    print(f"[CodeLoop] Repaired candidate {i} PASSED — returning")
                    return OrchestratorResult(
                        final_answer=repair_candidates[i][1],
                        route_taken="code_specialist_verified",
                        specialist_used="Code Specialist (ruvltra-claude-code)",
                        clr_score=1.0,
                        routing_confidence=1.0,
                        raw_traces={
                            "verified": True,
                            "verification_method": result.method,
                            "candidates_generated": len(candidates)
                            + len(repair_candidates),
                            "candidate_index": i,
                            "unit_tests": tests,
                            "all_verification_traces": verification_traces,
                            "test_spec_retry": attempt,
                            "repair_attempts": repair_attempt + 1,
                            "mutation_check": mutation_details,
                        },
                    )
                print(f"[CodeLoop] Repaired candidate {i} failed: "
                      f"{result.error or 'not verified'}")
            # Use the repaired candidates as the basis for the next repair
            # iteration (feed the new failing code + error back).
            verify_results = repair_results
            candidates = repair_candidates
            repair_attempts_made = repair_attempt + 1

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
                "test_spec_retries": attempt,
                "repair_attempts": repair_attempts_made,
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

    async def _build_verifier_context(
        self, query: str, task_type: str,
        compute_limits: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Derive verifier context from the query.

        This is what makes verifiers actually useful instead of ceremonial.
        Without context, MathVerifier has no expected_answer, CodeVerifier
        has no unit_tests, and FactualVerifier has no sources.

        For math: use the deterministic math_solver to derive expected_answer.
        For code: extract code blocks and test assertions from the query.
        For factual: if a retrieval backend is configured, fetch real source
          text from a search API (Serper / SearchApi) and feed it to the
          FactualVerifier's NLI judge. If no backend is configured, or the
          backend returns no results, no sources are set — the verifier
          returns unsupported_factual, which is the honest result.

        Do NOT fake context. If we can't derive it deterministically (or
        retrieve it from a real source), leave it absent — the verifier
        will return verified=False, which is honest.
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
            # Pass dynamic sandbox resource limits (v0.4.0) to the CodeVerifier.
            # When present, the verifier sizes the sandbox accordingly instead
            # of using its hardcoded 128m / 5.0s defaults. Backward-compatible:
            # absent compute_limits -> the verifier uses its own defaults.
            if compute_limits is not None:
                context["compute_limits"] = compute_limits

        elif task_type == "factual":
            # Active retrieval (v0.4.0): fetch real source text from a search
            # API so the FactualVerifier's NLI judge can classify entailment
            # / contradiction. Fail-closed: if no backend is configured, or
            # the backend returns no results, sources stays absent and the
            # verifier returns unsupported_factual (unchanged honest behavior).
            if self._retrieval_backend is not None:
                try:
                    sources = await self._retrieval_backend.search(query)
                except Exception as e:
                    print(f"[CLR] Retrieval failed: {e} — no sources (fail-closed)")
                    sources = []
                if sources:
                    context["sources"] = sources
                    print(f"[CLR] Retrieved {len(sources)} source(s) via "
                          f"{self._retrieval_backend.name} for factual verification")
                else:
                    print(f"[CLR] Retrieval returned no sources — "
                          f"verifier will return unsupported_factual")

        elif task_type == "logic":
            # Z3 constraint translation (v0.4.1): the LogicVerifier needs
            # constraints, variables, and values in Z3-compatible format.
            # We prompt the generalist to translate the natural-language
            # query into a JSON block with these fields. Fail-closed: if
            # the generalist is unreachable or returns malformed JSON, no
            # constraints are set — the verifier returns verified=False
            # (honest — we can't verify what we can't parse).
            #
            # Phase 3.2: translation retry loop. If the generalist's
            # constraints fail to parse as valid Z3, the parse error is
            # fed back to the generalist for a corrected attempt (up to
            # max_logic_translation_retries). This separates "bad
            # translation" from "bad answer" — a parse error means the
            # constraint set is malformed, not that the answer is wrong.
            constraints = await self._translate_logic_constraints_with_retry(
                query
            )
            if constraints is not None:
                context["constraints"] = constraints.get("constraints", [])
                context["variables"] = constraints.get("variables", {})
                context["values"] = constraints.get("values", {})
                print(f"[CLR] Logic constraints translated: "
                      f"{len(context['constraints'])} constraint(s), "
                      f"{len(context['variables'])} variable(s)")
            else:
                print(f"[CLR] Logic constraint translation failed — "
                      f"verifier will return verified=False (no constraints)")

        return context

    # Prompt for translating natural-language logic problems into Z3
    # constraints. The generalist must output a JSON block with
    # "constraints" (list of Z3 assertion strings), "variables" (name ->
    # sort), and "values" (name -> numeric, the answer's values).
    _LOGIC_CONSTRAINT_PROMPT = (
        "Translate the following logic problem into Z3 SMT constraints.\n"
        "Output ONLY a JSON object with these keys (no explanation, no markdown):\n"
        '  "constraints": list of Z3 assertion strings (e.g. "x > 0", "x + y == 10")\n'
        '  "variables": object mapping variable names to their Z3 sort ("Int", "Real", or "Bool")\n'
        '  "values": object mapping variable names to the answer\'s numeric values\n'
        "\n"
        "Examples:\n"
        "\n"
        'Problem: "I have 3 apples and give 1 away. How many do I have?"\n'
        "Output:\n"
        '{{"constraints": ["apples == 3", "given == 1", "remaining == apples - given"], '
        '"variables": {{"apples": "Int", "given": "Int", "remaining": "Int"}}, '
        '"values": {{"apples": 3, "given": 1, "remaining": 2}}}}\n'
        "\n"
        'Problem: "If it is raining then the ground is wet. The ground is not wet. Is it raining?"\n'
        "Output:\n"
        '{{"constraints": ["Implies(raining, wet)", "Not(wet)"], '
        '"variables": {{"raining": "Bool", "wet": "Bool"}}, '
        '"values": {{"raining": 0}}}}\n'
        "\n"
        'Problem: "If x is a positive integer and x + y = 10 and y < x, what are x and y?"\n'
        "Output:\n"
        '{{"constraints": ["x > 0", "x + y == 10", "y < x"], '
        '"variables": {{"x": "Int", "y": "Int"}}, "values": {{"x": 7, "y": 3}}}}\n'
        "\n"
        "Problem: {query}\n"
        "Output:"
    )

    async def _translate_logic_constraints(
        self, query: str
    ) -> Optional[Dict[str, Any]]:
        """Translate a natural-language logic problem into Z3 constraints.

        Prompts the generalist model to output a JSON block with
        constraints, variables, and values. Fail-closed: returns None
        on any error (network, parse, validation) — the verifier will
        return verified=False (honest — we can't verify what we can't
        parse).
        """
        try:
            prompt = self._LOGIC_CONSTRAINT_PROMPT.format(query=query)
            response = await self._call_generalist(prompt, max_tokens=512)
            if not response:
                return None
            # Extract JSON from the response (it may be wrapped in
            # markdown code blocks or have trailing text).
            text = response.strip()
            # Strip markdown code fences if present.
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            # Find the first { and last } to extract the JSON object.
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1:
                return None
            result = json.loads(text[start:end + 1])
            # Validate the structure.
            if not isinstance(result, dict):
                return None
            constraints = result.get("constraints")
            if not isinstance(constraints, list) or not constraints:
                return None
            variables = result.get("variables", {})
            if not isinstance(variables, dict):
                variables = {}
            values = result.get("values", {})
            if not isinstance(values, dict):
                values = {}
            return {
                "constraints": constraints,
                "variables": variables,
                "values": values,
            }
        except Exception as e:
            print(f"[CLR] Logic constraint translation error: {e}")
            return None

    async def _translate_logic_constraints_with_retry(
        self, query: str, max_retries: int = 2
    ) -> Optional[Dict[str, Any]]:
        """Translate logic constraints with a Z3 parse-validation retry loop.

        Phase 3.2: wraps _translate_logic_constraints with a validation
        step. After the generalist produces a constraint set, we validate
        that the constraints actually parse as Z3 expressions (using
        LogicVerifier.validate_constraints). If parsing fails, the error
        is fed back to the generalist for a corrected attempt. Bounded by
        max_retries. Fail-closed: returns None if all retries fail.

        This separates "bad translation" (the generalist wrote invalid Z3
        syntax — retryable) from "bad answer" (the answer's values violate
        valid constraints — not retryable, that's a verification failure).
        """
        from verifiers.logic_verifier import LogicVerifier, _Z3_AVAILABLE

        # If Z3 is not installed, no point retrying — the verifier will
        # return smt_unavailable anyway.
        if not _Z3_AVAILABLE:
            return await self._translate_logic_constraints(query)

        feedback: Optional[str] = None
        for attempt in range(max_retries + 1):
            # On retry, append the parse error feedback to the query.
            if feedback is not None:
                print(f"[CLR] Logic translation retry {attempt}/"
                      f"{max_retries}: feeding parse error back to generalist")
                result = await self._translate_logic_constraints(
                    f"{query}\n\n--- Z3 PARSE ERROR ---\n"
                    f"The previous constraint translation failed to parse "
                    f"as valid Z3:\n{feedback}\n\n"
                    f"Rewrite the constraints using only valid Z3 Python "
                    f"syntax. Ensure all variables are declared in the "
                    f"'variables' object and referenced correctly in the "
                    f"constraints."
                )
            else:
                result = await self._translate_logic_constraints(query)

            if result is None:
                # Translation itself failed (network, JSON parse, etc.).
                # No point validating — retry if we have attempts left.
                if attempt < max_retries:
                    feedback = "Previous response was not valid JSON or "
                    "did not contain a constraints array."
                    continue
                return None

            # Validate that the constraints parse as Z3.
            error = LogicVerifier.validate_constraints(
                result.get("constraints", []),
                result.get("variables", {}),
            )
            if error is None:
                # All constraints parse — return the validated set.
                if attempt > 0:
                    print(f"[CLR] Logic translation succeeded on retry "
                          f"{attempt}")
                return result

            # Parse error — feed it back and retry.
            print(f"[CLR] Logic constraint parse error (attempt "
                  f"{attempt + 1}): {error}")
            feedback = error

        # All retries exhausted — return the last result anyway. The
        # LogicVerifier will catch the parse error and return
        # verified=False (fail-closed). This is better than returning
        # None (which would skip verification entirely) because the
        # error trace will be visible in the verification result.
        print(f"[CLR] Logic translation retries exhausted — returning "
              f"best-effort (will fail at verification)")
        return result
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
        verifier = select_verifier(
            task_type, llm_judge=self._call_generalist,
            prefer_encoder_nli=self.prefer_encoder_nli,
        )
        verifier_context = await self._build_verifier_context(
            query, task_type,
            compute_limits=decision.get("compute_limits"),
        )
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
        verifier_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a verified result in the trajectory store.

        Only stores results that were independently verified (verified=True
        with a deterministic verifier, not self_claims_only). This is the
        "learning" step: verified solutions become few-shot context for
        future similar queries. Unverified results are never stored.

        Args:
            verifier_context: optional dict of verification context
                (e.g. ``{"expected_answer": "42"}`` for math,
                ``{"unit_tests": "..."}`` for code). Stored with the
                entry so synthesized masters can be re-verified against
                the children's ground truths (v3.1). If None, the
                context is extracted from result.raw_traces (for code
                tasks, the unit_tests are stored in raw_traces).
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

        # Extract verification context for re-verification of synthesized
        # masters (v3.1). Priority: explicit verifier_context > raw_traces.
        vctx = verifier_context
        if vctx is None:
            # For code tasks, the unit_tests are in raw_traces.
            tests = result.raw_traces.get("unit_tests")
            if tests:
                vctx = {"unit_tests": tests}

        self.trajectory_store.store(
            query=query,
            answer=result.final_answer,
            score=score,
            verification_method=method,
            task_type=task_type,
            route_taken=result.route_taken,
            verification_context=vctx,
        )
        print(f"[TrajectoryStore] Stored verified result (method={method}, "
              f"score={score:.3f}, task_type={task_type}) — "
              f"store now has {len(self.trajectory_store)} entries")

        # v2.0/v3.0: Also record into the SONA learning engine if available.
        # The SONA recorder uses the query + response embeddings for
        # continuous model improvement (LoRA adaptation, pattern learning).
        #
        # Embedding source priority (v3.0):
        #   1. RuvLLMBinding.get_embeddings() — Rust-native, no Python deps.
        #   2. trajectory_store.model.encode() — sentence-transformers.
        #   3. Skip — no embedding source available.
        if self._sona_recorder is not None:
            try:
                import uuid as _uuid
                req_id = str(_uuid.uuid4())

                q_emb = None
                a_emb = None

                # 1. Try Rust-native embeddings from the RuvLLMBinding.
                local_llm = getattr(self.reasoner, "_local_llm", None)
                if local_llm is not None and hasattr(local_llm, "get_embeddings"):
                    q_emb = local_llm.get_embeddings(query)
                    a_emb = local_llm.get_embeddings(result.final_answer)

                # 2. Fall back to sentence-transformers for any missing embeddings.
                model = getattr(self.trajectory_store, "model", None)
                if q_emb is None and model is not None:
                    q_emb = model.encode([query])[0].tolist()
                if a_emb is None and model is not None:
                    a_emb = model.encode([result.final_answer])[0].tolist()

                # 3. Record into SONA if we have embeddings.
                if q_emb and a_emb:
                    self._sona_recorder.record(
                        request_id=req_id,
                        session_id="default",
                        query_embedding=q_emb,
                        response_embedding=a_emb,
                        quality_score=score,
                    )
            except Exception as e:
                print(f"[SONA] Warning: failed to record trajectory: {e}")

    def _decrypt_federation_response(self, data: dict) -> dict:
        """Decrypt a federation response payload if encrypted (v3.0)."""
        if not isinstance(data, dict) or "__encrypted__" not in data:
            return data
        if self._fernet is None:
            print("[SONA] Warning: received encrypted response but no "
                  "federation_secret configured")
            return data
        try:
            import json as _json
            ciphertext = data["__encrypted__"].encode("ascii")
            plaintext = self._fernet.decrypt(ciphertext).decode("utf-8")
            result = _json.loads(plaintext)
            if not isinstance(result, dict):
                print("[SONA] Warning: decrypted response is not a dict")
                return data
            return result
        except Exception as e:
            print(f"[SONA] Warning: failed to decrypt response: {e}")
            return data

    def _sona_export_patterns(self) -> list:
        """Export learned SONA patterns for gossip protocol sync (v3.0).

        Returns a list of pattern dicts (id, centroid, cluster_size,
        avg_quality) that can be POSTed to the federation coordinator's
        /api/sona/sync endpoint.

        Uses search_patterns with a high limit to retrieve all learned
        patterns. The zero-vector query only affects sort order (nearest
        to origin first), not which patterns are returned — HNSW returns
        the top-k patterns regardless of query, so a high limit returns
        all patterns.
        """
        if self._sona_recorder is None:
            return []
        try:
            # search_patterns returns the top-k nearest patterns to the
            # query. With a very high limit, this returns ALL patterns
            # (sorted by distance to the zero vector, which is arbitrary
            # but doesn't filter any out). The zero vector is used because
            # we don't have a meaningful query — we want all patterns.
            dim = 384  # default embedding dim
            zero_query = [0.0] * dim
            _EXPORT_LIMIT = 100000
            patterns = self._sona_recorder.search_patterns(zero_query, _EXPORT_LIMIT)
            if len(patterns) >= _EXPORT_LIMIT:
                print(f"[SONA] Warning: export hit limit ({_EXPORT_LIMIT}), "
                      f"some patterns may be truncated")
            return patterns
            return patterns
        except Exception as e:
            print(f"[SONA] Warning: failed to export patterns: {e}")
            return []

    def _sona_import_patterns(self, patterns: list) -> int:
        """Import global SONA patterns from the federation coordinator (v3.0).

        Records each global pattern as a trajectory in the local SONA
        engine, allowing the node to learn from patterns discovered by
        other nodes in the swarm.

        Returns the number of patterns successfully imported.
        """
        if self._sona_recorder is None or not patterns:
            return 0
        imported = 0
        for p in patterns:
            try:
                import uuid as _uuid
                centroid = p.get("centroid", [])
                quality = p.get("avg_quality", 0.5)
                if isinstance(centroid, list) and centroid and isinstance(quality, (int, float)) and quality > 0:
                    self._sona_recorder.record(
                        request_id=str(_uuid.uuid4()),
                        session_id="global_sync",
                        query_embedding=centroid,
                        response_embedding=centroid,
                        quality_score=quality,
                    )
                    imported += 1
            except Exception:
                pass
        return imported

    async def sona_sync_once(self) -> dict:
        """Perform one SONA gossip sync cycle (v3.0).

        Exports local patterns to the coordinator and imports the
        global pattern set. Returns a dict with sync statistics.
        """
        if not self._sona_sync_url or self._sona_recorder is None:
            return {"status": "disabled"}
        try:
            import aiohttp
            import json as _json

            # Export local patterns.
            local_patterns = self._sona_export_patterns()
            local_stats = self._sona_recorder.stats()
            if not isinstance(local_stats, dict):
                local_stats = {}

            export_payload = {
                "worker_id": self._node_id,
                "patterns": local_patterns,
                "stats": local_stats,
            }

            timeout = aiohttp.ClientTimeout(total=10.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # POST local patterns.
                async with session.post(
                    f"{self._sona_sync_url}/api/sona/sync",
                    json=export_payload,
                ) as resp:
                    post_result = await resp.json()

                # GET global patterns (may be encrypted).
                async with session.get(
                    f"{self._sona_sync_url}/api/sona/sync",
                ) as resp:
                    global_data = await resp.json()
                    global_data = self._decrypt_federation_response(global_data)

            # Import global patterns.
            global_patterns = global_data.get("patterns", [])
            imported = self._sona_import_patterns(global_patterns)

            self._sona_sync_count += 1
            result = {
                "status": "ok",
                "exported": len(local_patterns),
                "imported": imported,
                "global_patterns": len(global_patterns),
                "sync_count": self._sona_sync_count,
            }
            print(f"[SONA] Sync #{self._sona_sync_count}: exported "
                  f"{len(local_patterns)}, imported {imported}, "
                  f"global pool: {len(global_patterns)}")
            return result
        except Exception as e:
            print(f"[SONA] Sync failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _sona_sync_loop(self):
        """Background loop for periodic SONA gossip sync (v3.0)."""
        import asyncio as _asyncio
        while True:
            await _asyncio.sleep(self._sona_sync_interval)
            await self.sona_sync_once()

    def _get_few_shot_context(self, query: str, task_type: Optional[str] = None) -> str:
        """Retrieve verified trajectories as few-shot context for the query.

        Returns a string to prepend to the model prompt, or "" if no similar
        verified trajectories exist or the store is disabled.
        """
        if not self.use_trajectory_store or self.trajectory_store is None:
            return ""
        return self.trajectory_store.build_few_shot_prefix(query, task_type=task_type)

    # ------------------------------------------------------------------ #
    # Trajectory synthesis (v0.4.0) — memory pruning
    # ------------------------------------------------------------------ #
    async def _reverify_synthesized_master(
        self, master: str, children: List[Dict[str, Any]],
    ) -> bool:
        """Re-verify a synthesized master against the children's ground truths (v3.1).

        For each child trajectory, the master is run against the child's
        verification context (unit_tests for code, expected_answer for math).
        If the master passes ALL children's criteria, it is considered
        "synthesized_and_proven" and can be stored as verified=True.

        Fail-closed: if any child lacks a verification_context, or if any
        child's verification fails, the master is NOT re-verified (returns
        False). This preserves the v0.4.0 behavior (verified=False,
        provenance only) as the fallback.

        Args:
            master: the synthesized master trajectory text.
            children: the child trajectory entries (each a dict with
                ``query``, ``answer``, ``verification_context``, etc.).

        Returns:
            True if the master passes ALL children's verification criteria.
            False if any child lacks context, any verification fails, or
            no verifier is available.
        """
        if not children:
            return False

        # Collect the verification contexts from the children.
        contexts = []
        for child in children:
            vctx = child.get("verification_context")
            if not vctx:
                # No context for this child — can't re-verify. Fail-closed.
                return False
            contexts.append((child, vctx))

        if not contexts:
            return False

        # Determine the task type from the first child.
        task_type = children[0].get("task_type", "unknown")

        # Select the appropriate verifier.
        verifier = select_verifier(
            task_type, llm_judge=self._call_generalist,
            prefer_encoder_nli=self.prefer_encoder_nli,
        )
        if verifier is None:
            # No verifier for this task type — can't re-verify.
            return False

        # Run the master against each child's verification context.
        for child, vctx in contexts:
            try:
                result = await verifier.verify(
                    child["query"], master, vctx,
                )
                if not result.verified:
                    print(f"[Synthesis] Master failed re-verification against "
                          f"child {child['query'][:50]!r}: {result.error or 'not verified'}")
                    return False
            except Exception as e:
                print(f"[Synthesis] Re-verification error for child "
                      f"{child['query'][:50]!r}: {e}")
                return False

        # All children passed — the master is proven.
        return True

    _SYNTHESIS_PROMPT = (
        "You are a knowledge distillation engine. Below are {n} similar "
        "problems that were each independently verified correct, along with "
        "their verified answers. Synthesize them into a single general "
        "rule or master solution that captures the common pattern.\n\n"
        "The master solution should:\n"
        "  - Be more general than any individual solution\n"
        "  - Capture the reusable pattern across all examples\n"
        "  - Be concise (shorter than the {n} solutions combined)\n"
        "  - Not reference specific problem instances — generalize\n\n"
        "Verified examples:\n{examples}\n\n"
        "Output the synthesized master solution now (no preamble):"
    )

    async def synthesize_trajectories(
        self,
        similarity_threshold: float = 0.85,
        min_cluster_size: int = 3,
        task_type: Optional[str] = None,
        max_clusters: int = 5,
    ) -> Dict[str, Any]:
        """Synthesize clusters of similar verified trajectories into master
        trajectories to prune memory.

        For each cluster of highly-similar verified trajectories, the
        generalist model distills them into a single "master trajectory"
        (a general rule capturing the common pattern). The master is stored
        with ``verification_method="synthesized"`` (NOT independently
        verified — it is a compression of verified data, not a new proof).
        The raw entries are then removed to save memory.

        Trust model (fail-closed, no epistemic contamination):
          - Synthesized masters are stored with ``verified=False`` and
            ``verification_method="synthesized"``. The existing trust check
            in ``retrieve()`` (``is_cache_entry_trustworthy``) filters them
            out — they are NEVER served as few-shot "verified examples."
            They exist for provenance/audit and potential future re-
            verification, not as substitute evidence.
          - Only independently-verified entries (``verified=True`` with a
            deterministic method) are eligible for clustering. Synthesized
            entries themselves are never re-synthesized.
          - If the generalist fails to produce a synthesis for a cluster,
            the raw entries are kept (no data loss).

        Args:
            similarity_threshold: cosine similarity to consider two
                trajectories "similar" (default 0.85 — high, only near-
                duplicates are merged).
            min_cluster_size: minimum cluster size to synthesize (default 3
                — merging 2 into 1 saves little).
            task_type: restrict synthesis to one task type (default None =
                all task types).
            max_clusters: cap on clusters processed per call (default 5 —
                bounded to avoid unbounded generalist calls).

        Returns:
            A summary dict: {clusters_found, clusters_synthesized,
            entries_removed, masters_stored, errors}.
        """
        if not self.use_trajectory_store or self.trajectory_store is None:
            return {"error": "trajectory store not enabled"}
        if self.trajectory_store is None:
            return {"error": "trajectory store not enabled"}

        clusters = self.trajectory_store.find_clusters(
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
            task_type=task_type,
        )
        if not clusters:
            print("[Synthesis] No clusters of similar trajectories found — "
                  "nothing to synthesize")
            return {
                "clusters_found": 0, "clusters_synthesized": 0,
                "entries_removed": 0, "masters_stored": 0,
                "masters_reverified": 0, "errors": [],
            }

        clusters = clusters[:max_clusters]
        print(f"[Synthesis] Found {len(clusters)} cluster(s) to synthesize "
              f"(similarity >= {similarity_threshold}, min_size {min_cluster_size})")

        masters_stored = 0
        masters_reverified = 0
        entries_removed = 0
        errors: List[str] = []
        all_removed_indices: List[int] = []

        for cluster_idx, cluster in enumerate(clusters):
            # Gather the verified entries in this cluster.
            entries = [
                self.trajectory_store.entries[i]
                for i in cluster
                if i < len(self.trajectory_store.entries)
            ]
            # Skip any already-synthesized entries (don't re-synthesize).
            entries = [e for e in entries if not e.get("synthesized")]
            if len(entries) < min_cluster_size:
                continue

            # Build the examples string for the generalist.
            examples = "\n\n".join(
                f"Problem {i+1}: {e['query']}\n"
                f"Verified answer: {e['answer']}\n"
                f"(verified via {e.get('verification_method', '?')})"
                for i, e in enumerate(entries)
            )
            prompt = self._SYNTHESIS_PROMPT.format(
                n=len(entries), examples=examples,
            )

            try:
                master = await self._call_generalist(prompt, max_tokens=1024)
            except Exception as e:
                msg = f"cluster {cluster_idx}: generalist failed: {e}"
                print(f"[Synthesis] {msg} — keeping raw entries")
                errors.append(msg)
                continue

            if not master or not master.strip():
                msg = f"cluster {cluster_idx}: empty synthesis"
                errors.append(msg)
                continue

            # Use the highest-score entry's query as the representative query.
            representative = max(entries, key=lambda e: e.get("score", 0.0))
            source_queries = [e["query"] for e in entries]
            master_text = master.strip()

            # v3.1: Re-verify the synthesized master against the children's
            # ground truths. If the master passes ALL children's verification
            # criteria, store it as verified=True with
            # verification_method="synthesized_and_proven" — it becomes
            # retrievable as few-shot context (the memory flywheel works).
            # If re-verification fails or no verification context is
            # available, fall back to the v0.4.0 behavior (verified=False,
            # provenance only).
            reverified = await self._reverify_synthesized_master(
                master_text, entries,
            )
            if reverified:
                self.trajectory_store.store_synthesized_verified(
                    query=representative["query"],
                    answer=master_text,
                    task_type=representative.get("task_type", "unknown"),
                    source_count=len(entries),
                    source_queries=source_queries,
                )
                masters_reverified += 1
                print(f"[Synthesis] Cluster {cluster_idx}: synthesized "
                      f"{len(entries)} trajectories into 1 RE-VERIFIED master "
                      f"(synthesized_and_proven)")
            else:
                # Store the synthesized master (NOT verified — provenance only).
                self.trajectory_store.store_synthesized(
                    query=representative["query"],
                    answer=master_text,
                    task_type=representative.get("task_type", "unknown"),
                    source_count=len(entries),
                    source_queries=source_queries,
                )
                print(f"[Synthesis] Cluster {cluster_idx}: synthesized "
                      f"{len(entries)} trajectories into 1 master (unverified)")
            masters_stored += 1

            # Mark the raw entries for removal. Adjust indices for already-
            # removed entries (removal shifts indices, so we collect all and
            # remove in one batch at the end).
            all_removed_indices.extend(cluster)
            entries_removed += len(entries)

        # Remove all synthesized-from entries in one batch.
        if all_removed_indices:
            # Deduplicate (a cluster index could appear once; but be safe).
            unique_indices = sorted(set(all_removed_indices), reverse=True)
            self.trajectory_store.remove_entries(unique_indices)

        return {
            "clusters_found": len(clusters),
            "clusters_synthesized": masters_stored,
            "entries_removed": entries_removed,
            "masters_stored": masters_stored,
            "masters_reverified": masters_reverified,
            "errors": errors,
        }

    # ------------------------------------------------------------------ #
    # Lifecycle: warm pool start/cleanup
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        """Start background resources (warm Docker pool, shared HTTP session).

        Call this before submitting tasks to pre-warm the sandbox containers
        and eagerly create the shared aiohttp session inside the event loop.
        Eager session creation avoids a lazy-init race where multiple
        concurrent first-requests each create overlapping sessions.
        """
        # Eagerly create the shared HTTP session (must be inside the event loop).
        async with self._session_lock:
            if self._http_session is None or self._http_session.closed:
                self._http_session = aiohttp.ClientSession()
        if self._warm_pool is not None:
            await self._warm_pool.start()

    async def cleanup(self) -> None:
        """Clean up background resources (warm Docker pool, shared HTTP session).

        Call this when shutting down to remove warm containers and close the
        shared aiohttp session.
        """
        if self._warm_pool is not None:
            await self._warm_pool.cleanup()
        if self._http_session is not None and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None

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
