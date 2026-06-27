"""Active retrieval abstraction for factual verification.

The FactualVerifier's NLI judge needs real source text to classify
ENTAILMENT / CONTRADICTION / NEUTRAL. Without sources, it fail-closes to
``unsupported_factual`` — the honest result when no evidence exists. This
module provides the pluggable backends that fetch that evidence from real
search APIs, so factual claims can actually be verified instead of
ceremonially failing.

Trust model (fail-closed, no epistemic contamination):
  - Every backend returns ``[]`` on any failure: missing API key, network
    error, timeout, non-2xx response, malformed JSON, empty results. The
    caller (the orchestrator's ``_build_verifier_context``) treats ``[]`` as
    "no sources" — the FactualVerifier then returns ``unsupported_factual``,
    which is the honest, unchanged behavior. No backend ever fabricates
    sources or returns hardcoded text.
  - The sources returned are real text snippets from search-engine results
    (titles + snippets from organic results). These are genuine web text,
    not model-generated — the NLI judge classifies the model's answer
    against them, same as a human checking a citation.
  - API keys are read from constructor args or environment variables. They
    are NEVER hardcoded, logged, or committed to the repo.

Backends:
  - :class:`SerperBackend`   — google.serper.dev (POST, X-API-KEY header).
  - :class:`SearchApiBackend` — www.searchapi.io (GET, api_key query param).
  - :class:`DuckDuckGoBackend` — free DuckDuckGo HTML search via the
    ``duckduckgo_search`` package (no API key required). Fail-closed to
    ``[]`` when the package is not installed or the search fails.
  - :class:`WikipediaBackend` — free Wikipedia article summaries via the
    ``wikipedia`` package (no API key required). Fail-closed to ``[]``
    when the package is not installed or the lookup fails.
  - ``None`` / no key configured and free backends disabled — no
    retrieval (unchanged fail-closed).

Factory: :func:`make_retrieval_backend` — precedence: explicit backend >
  serper_key > searchapi_key > env SERPER_API_KEY > env SEARCHAPI_API_KEY >
  DuckDuckGo (free, unless ``allow_free=False``) > Wikipedia (free, unless
  ``allow_free=False``) > None.

Requires aiohttp (already a core dep of vibe-thinker). The free backends
additionally require the optional ``duckduckgo_search`` and ``wikipedia``
packages respectively — when absent, they fail-closed to ``[]`` (no
sources), preserving the unchanged fail-closed behavior.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class RetrievalBackend(Protocol):
    """Minimal async retrieval protocol.

    Implementations must be safe for concurrent calls and must fail-closed
    (return ``[]``) on any error — never raise, never fabricate.
    """

    name: str

    async def search(
        self, query: str, max_results: int = 5
    ) -> List[str]:
        """Search for ``query`` and return up to ``max_results`` source texts.

        Each returned string is a real text snippet suitable for feeding to
        the FactualVerifier's NLI judge. Returns ``[]`` on any failure
        (missing key, network error, timeout, empty results) — the caller
        treats this as "no sources available" and the verifier fail-closes.
        """
        ...


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _format_result(title: str, snippet: str, link: str) -> str:
    """Format a single search result as a source string for the NLI judge.

    Returns an empty string if no snippet is present — a title alone is too
    thin for the NLI judge to classify entailment against. The snippet is
    the substantive source text; without it the result is not useful as
    evidence and is skipped.
    """
    if not snippet:
        return ""
    parts = []
    if title:
        parts.append(f"Title: {title}")
    parts.append(f"Snippet: {snippet}")
    if link:
        parts.append(f"Source: {link}")
    return "\n".join(parts)


def _extract_serper_sources(data: Dict[str, Any], max_results: int) -> List[str]:
    """Extract source texts from a Serper.dev JSON response.

    Serper returns ``organic`` (list of {title, link, snippet, ...}) and
    optionally ``knowledgeGraph`` ({title, description, ...}). The knowledge
    graph description is high-quality source text when present.
    """
    sources: List[str] = []

    # Knowledge graph — often the most authoritative single source.
    kg = data.get("knowledgeGraph")
    if isinstance(kg, dict):
        kg_title = kg.get("title", "")
        kg_desc = kg.get("description") or kg.get("text") or ""
        if kg_desc:
            sources.append(_format_result(kg_title, kg_desc, ""))

    organic = data.get("organic")
    if isinstance(organic, list):
        for item in organic:
            if not isinstance(item, dict):
                continue
            src = _format_result(
                item.get("title", ""),
                item.get("snippet", ""),
                item.get("link", ""),
            )
            if src:
                sources.append(src)
            if len(sources) >= max_results:
                break

    return sources[:max_results]


def _extract_searchapi_sources(data: Dict[str, Any], max_results: int) -> List[str]:
    """Extract source texts from a SearchApi.io JSON response.

    SearchApi returns ``organic_results`` (list of {title, link, snippet,
    ...}) and optionally ``knowledge_graph`` ({title, description, ...}).
    """
    sources: List[str] = []

    kg = data.get("knowledge_graph")
    if isinstance(kg, dict):
        kg_title = kg.get("title", "")
        kg_desc = kg.get("description") or kg.get("text") or ""
        if kg_desc:
            sources.append(_format_result(kg_title, kg_desc, ""))

    organic = data.get("organic_results")
    if isinstance(organic, list):
        for item in organic:
            if not isinstance(item, dict):
                continue
            src = _format_result(
                item.get("title", ""),
                item.get("snippet", ""),
                item.get("link", ""),
            )
            if src:
                sources.append(src)
            if len(sources) >= max_results:
                break

    return sources[:max_results]


# ---------------------------------------------------------------------- #
# Serper.dev backend
# ---------------------------------------------------------------------- #
class SerperBackend:
    """Retrieval backend using the Serper.dev Google Search API.

    POSTs to ``https://google.serper.dev/search`` with the ``X-API-KEY``
    header. Returns formatted source strings from organic results + the
    knowledge graph.

    Fail-closed: returns ``[]`` on missing key, network error, timeout,
    non-2xx status, or malformed response. Never raises.

    Args:
        api_key: Serper API key. If None, reads ``SERPER_API_KEY`` env.
            If still None, all searches return ``[]``.
        timeout: HTTP timeout in seconds (default 10.0 — search APIs can
            be slower than a local sidecar).
        base_url: override the Serper endpoint (for testing).
    """

    name = "serper"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        base_url: str = "https://google.serper.dev",
    ):
        self._api_key = api_key or os.environ.get("SERPER_API_KEY")
        self._timeout = timeout
        self._base_url = base_url.rstrip("/")

    async def search(
        self, query: str, max_results: int = 5
    ) -> List[str]:
        if not self._api_key:
            return []
        if not query or not query.strip():
            return []
        try:
            import aiohttp
        except ImportError:
            return []

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout)
            ) as session:
                async with session.post(
                    f"{self._base_url}/search",
                    json={"q": query, "num": max_results},
                    headers={
                        "X-API-KEY": self._api_key,
                        "Content-Type": "application/json",
                    },
                ) as resp:
                    if resp.status >= 400:
                        print(f"[Retrieval/Serper] HTTP {resp.status} — "
                              f"fail-closed (no sources)")
                        return []
                    data = await resp.json()
                    if not isinstance(data, dict):
                        return []
                    return _extract_serper_sources(data, max_results)
        except Exception as e:
            print(f"[Retrieval/Serper] search failed: {e} — fail-closed")
            return []


# ---------------------------------------------------------------------- #
# SearchApi.io backend
# ---------------------------------------------------------------------- #
class SearchApiBackend:
    """Retrieval backend using the SearchApi.io Google Search API.

    GETs ``https://www.searchapi.io/api/v1/search`` with the ``api_key``
    query parameter. Returns formatted source strings from organic results
    + the knowledge graph.

    Fail-closed: returns ``[]`` on missing key, network error, timeout,
    non-2xx status, or malformed response. Never raises.

    Args:
        api_key: SearchApi API key. If None, reads ``SEARCHAPI_API_KEY``
            env. If still None, all searches return ``[]``.
        timeout: HTTP timeout in seconds (default 10.0).
        base_url: override the SearchApi endpoint (for testing).
    """

    name = "searchapi"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        base_url: str = "https://www.searchapi.io",
    ):
        self._api_key = api_key or os.environ.get("SEARCHAPI_API_KEY")
        self._timeout = timeout
        self._base_url = base_url.rstrip("/")

    async def search(
        self, query: str, max_results: int = 5
    ) -> List[str]:
        if not self._api_key:
            return []
        if not query or not query.strip():
            return []
        try:
            import aiohttp
        except ImportError:
            return []

        try:
            params = {
                "engine": "google",
                "q": query,
                "num": max_results,
                "api_key": self._api_key,
            }
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout)
            ) as session:
                async with session.get(
                    f"{self._base_url}/api/v1/search",
                    params=params,
                ) as resp:
                    if resp.status >= 400:
                        print(f"[Retrieval/SearchApi] HTTP {resp.status} — "
                              f"fail-closed (no sources)")
                        return []
                    data = await resp.json()
                    if not isinstance(data, dict):
                        return []
                    return _extract_searchapi_sources(data, max_results)
        except Exception as e:
            print(f"[Retrieval/SearchApi] search failed: {e} — fail-closed")
            return []


# ---------------------------------------------------------------------- #
# DuckDuckGo backend (free, no API key)
# ---------------------------------------------------------------------- #
class DuckDuckGoBackend:
    """Retrieval backend using the free DuckDuckGo search (no API key).

    Uses the ``duckduckgo_search`` Python package (HTML scraping under the
    hood). No paid API key is required — this is the open-source fallback
    when Serper/SearchApi keys are not configured, so the FactualVerifier
    can still fetch real source text without a paid account.

    Fail-closed: returns ``[]`` on missing package, network error, timeout,
    rate-limit, or empty results. Never raises, never fabricates.

    Args:
        timeout: search timeout in seconds (default 10.0).
        max_results: default cap on results per query (default 5).
    """

    name = "duckduckgo"

    def __init__(self, timeout: float = 10.0, max_results: int = 5):
        self._timeout = timeout
        self._max_results = max_results

    async def search(
        self, query: str, max_results: int = 5
    ) -> List[str]:
        if not query or not query.strip():
            return []
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except ImportError:
            return []

        try:
            import asyncio
            cap = max_results or self._max_results

            def _sync_search() -> List[str]:
                sources: List[str] = []
                with DDGS() as ddgs:
                    results = ddgs.text(query, max_results=cap)
                    for item in results:
                        if not isinstance(item, dict):
                            continue
                        src = _format_result(
                            item.get("title", ""),
                            item.get("body", "") or item.get("snippet", ""),
                            item.get("href", "") or item.get("link", ""),
                        )
                        if src:
                            sources.append(src)
                        if len(sources) >= cap:
                            break
                return sources[:cap]

            return await asyncio.wait_for(
                asyncio.to_thread(_sync_search),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            print(f"[Retrieval/DuckDuckGo] timeout after {self._timeout}s — "
                  f"fail-closed (no sources)")
            return []
        except Exception as e:
            print(f"[Retrieval/DuckDuckGo] search failed: {e} — fail-closed")
            return []


# ---------------------------------------------------------------------- #
# Wikipedia backend (free, no API key)
# ---------------------------------------------------------------------- #
class WikipediaBackend:
    """Retrieval backend using the free Wikipedia API (no API key).

    Uses the ``wikipedia`` Python package to query article summaries
    directly. No paid API key is required. Best for factual/entity
    queries (people, places, concepts) — returns the article summary as
    a single high-quality source.

    Fail-closed: returns ``[]`` on missing package, network error,
    disambiguation, page-not-found, or empty results. Never raises,
    never fabricates.

    Args:
        timeout: lookup timeout in seconds (default 10.0).
        max_results: cap on results per query (default 5). Wikipedia
            typically returns 1-2 summaries; extra slots are filled by
            related page summaries when available.
    """

    name = "wikipedia"

    def __init__(self, timeout: float = 10.0, max_results: int = 5):
        self._timeout = timeout
        self._max_results = max_results

    async def search(
        self, query: str, max_results: int = 5
    ) -> List[str]:
        if not query or not query.strip():
            return []
        try:
            import wikipedia  # type: ignore
        except ImportError:
            return []

        try:
            import asyncio
            cap = max_results or self._max_results

            def _sync_lookup() -> List[str]:
                sources: List[str] = []
                # First try a direct summary (most authoritative).
                try:
                    summary = wikipedia.summary(query, auto_suggest=False)
                    if summary:
                        sources.append(_format_result(query, summary, ""))
                except wikipedia.exceptions.DisambiguationError as d:
                    # Pick the first disambiguated option and summarize it.
                    if d.options:
                        try:
                            summary = wikipedia.summary(
                                d.options[0], auto_suggest=False,
                            )
                            if summary:
                                sources.append(_format_result(
                                    d.options[0], summary, "",
                                ))
                        except Exception:
                            pass
                except wikipedia.exceptions.PageError:
                    pass
                except Exception:
                    pass

                # Fill remaining slots with search results' summaries.
                if len(sources) < cap:
                    try:
                        titles = wikipedia.search(query, results=cap)
                    except Exception:
                        titles = []
                    for title in titles:
                        if len(sources) >= cap:
                            break
                        try:
                            summary = wikipedia.summary(title, auto_suggest=False)
                            if summary:
                                sources.append(_format_result(title, summary, ""))
                        except Exception:
                            continue
                return sources[:cap]

            return await asyncio.wait_for(
                asyncio.to_thread(_sync_lookup),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            print(f"[Retrieval/Wikipedia] timeout after {self._timeout}s — "
                  f"fail-closed (no sources)")
            return []
        except Exception as e:
            print(f"[Retrieval/Wikipedia] search failed: {e} — fail-closed")
            return []


# ---------------------------------------------------------------------- #
# Factory
# ---------------------------------------------------------------------- #
def make_retrieval_backend(
    serper_key: Optional[str] = None,
    searchapi_key: Optional[str] = None,
    timeout: float = 10.0,
    allow_free: bool = True,
) -> Optional[RetrievalBackend]:
    """Build a retrieval backend from explicit keys or environment variables.

    Precedence: explicit serper_key > explicit searchapi_key >
    ``SERPER_API_KEY`` env > ``SEARCHAPI_API_KEY`` env > DuckDuckGo
    (free, unless ``allow_free=False``) > Wikipedia (free, unless
    ``allow_free=False``) > None (no retrieval — unchanged fail-closed
    behavior).

    Returns None when no key is configured and free backends are
    disabled, which means the orchestrator skips retrieval entirely and
    the FactualVerifier returns ``unsupported_factual`` as before.

    The free backends (DuckDuckGo, Wikipedia) require their respective
    optional packages (``duckduckgo_search``, ``wikipedia``). When a
    package is not installed, the backend still constructs but every
    search fail-closes to ``[]`` — preserving the unchanged behavior.

    Args:
        serper_key: explicit Serper.dev API key (overrides env).
        searchapi_key: explicit SearchApi.io API key (overrides env).
        timeout: HTTP timeout for the chosen backend.
        allow_free: when True (default), fall back to the free
            DuckDuckGo/Wikipedia backends when no paid key is set. Set
            to False to preserve the pre-v3.1 behavior (None when no
            paid key is configured).
    """
    serper_key = serper_key or os.environ.get("SERPER_API_KEY")
    searchapi_key = searchapi_key or os.environ.get("SEARCHAPI_API_KEY")
    if serper_key:
        return SerperBackend(api_key=serper_key, timeout=timeout)
    if searchapi_key:
        return SearchApiBackend(api_key=searchapi_key, timeout=timeout)
    if allow_free:
        # Prefer DuckDuckGo (broader coverage) then Wikipedia (high-
        # quality entity summaries). Both fail-closed to [] when their
        # optional packages are absent, so constructing them is safe.
        return DuckDuckGoBackend(timeout=timeout)
    return None
