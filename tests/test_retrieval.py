"""Tests for the active retrieval abstraction (retrieval.py) and its
integration into the orchestrator's factual verification path.

All tests use mocked HTTP — no real API calls are made. The fail-closed
contract is the core invariant: every failure mode returns ``[]``, never
raises, never fabricates sources.
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from retrieval import (
    RetrievalBackend,
    SerperBackend,
    SearchApiBackend,
    make_retrieval_backend,
    _extract_serper_sources,
    _extract_searchapi_sources,
    _format_result,
)


# ---------------------------------------------------------------------- #
# Source extraction helpers
# ---------------------------------------------------------------------- #
class TestExtractSerperSources:
    def test_organic_results_extracted(self):
        data = {
            "organic": [
                {"title": "Apple Inc.", "link": "https://apple.com",
                 "snippet": "Apple Inc. is a technology company."},
                {"title": "Apple - Wikipedia", "link": "https://wikipedia.org/apple",
                 "snippet": "Apple was founded in 1976."},
            ]
        }
        sources = _extract_serper_sources(data, max_results=5)
        assert len(sources) == 2
        assert "Title: Apple Inc." in sources[0]
        assert "Snippet: Apple Inc. is a technology company." in sources[0]
        assert "Source: https://apple.com" in sources[0]

    def test_knowledge_graph_included_first(self):
        data = {
            "knowledgeGraph": {
                "title": "Apple Inc.",
                "description": "American technology company.",
            },
            "organic": [
                {"title": "Other", "snippet": "Other snippet", "link": "url"},
            ],
        }
        sources = _extract_serper_sources(data, max_results=5)
        assert len(sources) == 2
        # Knowledge graph is first (highest quality source).
        assert "American technology company." in sources[0]

    def test_max_results_cap(self):
        data = {
            "organic": [
                {"title": f"Result {i}", "snippet": f"Snippet {i}", "link": f"url{i}"}
                for i in range(10)
            ]
        }
        sources = _extract_serper_sources(data, max_results=3)
        assert len(sources) == 3

    def test_empty_response(self):
        assert _extract_serper_sources({}, 5) == []

    def test_missing_snippet_skipped(self):
        data = {"organic": [{"title": "No snippet", "link": "url"}]}
        assert _extract_serper_sources(data, 5) == []

    def test_non_dict_items_skipped(self):
        data = {"organic": ["not a dict", 42, None]}
        assert _extract_serper_sources(data, 5) == []


class TestExtractSearchApiSources:
    def test_organic_results_extracted(self):
        data = {
            "organic_results": [
                {"title": "Apple Inc.", "link": "https://apple.com",
                 "snippet": "Technology company."},
            ]
        }
        sources = _extract_searchapi_sources(data, max_results=5)
        assert len(sources) == 1
        assert "Title: Apple Inc." in sources[0]

    def test_knowledge_graph(self):
        data = {
            "knowledge_graph": {"title": "Apple", "description": "Tech company."},
            "organic_results": [],
        }
        sources = _extract_searchapi_sources(data, 5)
        assert len(sources) == 1
        assert "Tech company." in sources[0]

    def test_empty(self):
        assert _extract_searchapi_sources({}, 5) == []


class TestFormatResult:
    def test_all_fields(self):
        result = _format_result("Title", "Snippet", "https://link")
        assert "Title: Title" in result
        assert "Snippet: Snippet" in result
        assert "Source: https://link" in result

    def test_no_snippet_returns_empty(self):
        """A title without a snippet is too thin for NLI — skip it."""
        assert _format_result("Title", "", "https://link") == ""

    def test_empty_returns_empty_string(self):
        assert _format_result("", "", "") == ""


# ---------------------------------------------------------------------- #
# SerperBackend
# ---------------------------------------------------------------------- #
class TestSerperBackend:
    @pytest.mark.asyncio
    async def test_missing_key_returns_empty(self):
        backend = SerperBackend(api_key=None)
        # Ensure env doesn't leak in.
        with patch.dict(os.environ, {}, clear=True):
            result = await backend.search("test query")
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self):
        backend = SerperBackend(api_key="fake-key")
        assert await backend.search("") == []
        assert await backend.search("   ") == []

    @pytest.mark.asyncio
    async def test_successful_search(self):
        backend = SerperBackend(api_key="fake-key", base_url="http://localhost:9999")
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "organic": [
                {"title": "Paris", "link": "https://paris.fr",
                 "snippet": "Paris is the capital of France."},
            ]
        })
        mock_post = MagicMock()
        mock_post.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.__aexit__ = AsyncMock(return_value=None)
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            sources = await backend.search("capital of France")
        assert len(sources) == 1
        assert "Paris is the capital of France." in sources[0]

    @pytest.mark.asyncio
    async def test_http_error_fail_closed(self):
        backend = SerperBackend(api_key="fake-key", base_url="http://localhost:9999")
        mock_response = MagicMock()
        mock_response.status = 403
        mock_post = MagicMock()
        mock_post.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.__aexit__ = AsyncMock(return_value=None)
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            assert await backend.search("test") == []

    @pytest.mark.asyncio
    async def test_network_error_fail_closed(self):
        backend = SerperBackend(api_key="fake-key", base_url="http://localhost:9999")
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(side_effect=OSError("connection refused"))
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            assert await backend.search("test") == []

    @pytest.mark.asyncio
    async def test_malformed_json_fail_closed(self):
        backend = SerperBackend(api_key="fake-key", base_url="http://localhost:9999")
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value="not a dict")
        mock_post = MagicMock()
        mock_post.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.__aexit__ = AsyncMock(return_value=None)
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            assert await backend.search("test") == []


# ---------------------------------------------------------------------- #
# SearchApiBackend
# ---------------------------------------------------------------------- #
class TestSearchApiBackend:
    @pytest.mark.asyncio
    async def test_missing_key_returns_empty(self):
        with patch.dict(os.environ, {}, clear=True):
            backend = SearchApiBackend(api_key=None)
            assert await backend.search("test") == []

    @pytest.mark.asyncio
    async def test_successful_search(self):
        backend = SearchApiBackend(api_key="fake-key", base_url="http://localhost:9999")
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "organic_results": [
                {"title": "France", "link": "https://france.fr",
                 "snippet": "France's capital is Paris."},
            ]
        })
        mock_get = MagicMock()
        mock_get.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get.__aexit__ = AsyncMock(return_value=None)
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_get)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            sources = await backend.search("capital of France")
        assert len(sources) == 1
        assert "France's capital is Paris." in sources[0]

    @pytest.mark.asyncio
    async def test_http_error_fail_closed(self):
        backend = SearchApiBackend(api_key="fake-key", base_url="http://localhost:9999")
        mock_response = MagicMock()
        mock_response.status = 429
        mock_get = MagicMock()
        mock_get.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get.__aexit__ = AsyncMock(return_value=None)
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_get)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            assert await backend.search("test") == []


# ---------------------------------------------------------------------- #
# Factory
# ---------------------------------------------------------------------- #
class TestMakeRetrievalBackend:
    def test_no_keys_returns_none(self):
        with patch.dict(os.environ, {}, clear=True):
            assert make_retrieval_backend() is None

    def test_serper_key_takes_precedence(self):
        with patch.dict(os.environ, {}, clear=True):
            backend = make_retrieval_backend(
                serper_key="serper", searchapi_key="searchapi"
            )
        assert isinstance(backend, SerperBackend)
        assert backend.name == "serper"

    def test_searchapi_key_when_no_serper(self):
        with patch.dict(os.environ, {}, clear=True):
            backend = make_retrieval_backend(searchapi_key="searchapi")
        assert isinstance(backend, SearchApiBackend)
        assert backend.name == "searchapi"

    def test_env_serper_key(self):
        with patch.dict(os.environ, {"SERPER_API_KEY": "env-serper"}, clear=True):
            backend = make_retrieval_backend()
        assert isinstance(backend, SerperBackend)

    def test_env_searchapi_key(self):
        env = {"SEARCHAPI_API_KEY": "env-searchapi"}
        with patch.dict(os.environ, env, clear=True):
            backend = make_retrieval_backend()
        assert isinstance(backend, SearchApiBackend)

    def test_serper_env_overrides_searchapi_env(self):
        env = {"SERPER_API_KEY": "s", "SEARCHAPI_API_KEY": "sa"}
        with patch.dict(os.environ, env, clear=True):
            backend = make_retrieval_backend()
        assert isinstance(backend, SerperBackend)


# ---------------------------------------------------------------------- #
# Protocol compliance
# ---------------------------------------------------------------------- #
class TestProtocolCompliance:
    def test_serper_satisfies_protocol(self):
        assert isinstance(SerperBackend(api_key="k"), RetrievalBackend)

    def test_searchapi_satisfies_protocol(self):
        assert isinstance(SearchApiBackend(api_key="k"), RetrievalBackend)


# ---------------------------------------------------------------------- #
# Orchestrator integration
# ---------------------------------------------------------------------- #
class TestOrchestratorRetrievalIntegration:
    @pytest.mark.asyncio
    async def test_no_backend_no_retrieval(self):
        """Without a retrieval backend, factual tasks get no sources —
        unchanged fail-closed behavior (unsupported_factual)."""
        from hybrid_orchestrator import HybridReasoningOrchestrator
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            use_clr=False,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
            retrieval_backend=None,
        )
        ctx = await o._build_verifier_context("What is the capital of France?", "factual")
        assert "sources" not in ctx

    @pytest.mark.asyncio
    async def test_backend_populates_sources(self):
        """With a retrieval backend, factual tasks get real sources fed
        into the verifier context."""
        from hybrid_orchestrator import HybridReasoningOrchestrator
        mock_backend = MagicMock()
        mock_backend.search = AsyncMock(return_value=[
            "Title: France\nSnippet: Paris is the capital of France.",
        ])
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            use_clr=False,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
            retrieval_backend=mock_backend,
        )
        ctx = await o._build_verifier_context("What is the capital of France?", "factual")
        assert "sources" in ctx
        assert len(ctx["sources"]) == 1
        assert "Paris is the capital" in ctx["sources"][0]

    @pytest.mark.asyncio
    async def test_backend_returns_empty_no_sources(self):
        """If the backend returns [] (fail-closed), no sources are set —
        the verifier will return unsupported_factual."""
        from hybrid_orchestrator import HybridReasoningOrchestrator
        mock_backend = MagicMock()
        mock_backend.search = AsyncMock(return_value=[])
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            use_clr=False,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
            retrieval_backend=mock_backend,
        )
        ctx = await o._build_verifier_context("obscure query", "factual")
        assert "sources" not in ctx

    @pytest.mark.asyncio
    async def test_backend_exception_fail_closed(self):
        """If the backend raises, the orchestrator catches it and
        fail-closes — no sources, no crash."""
        from hybrid_orchestrator import HybridReasoningOrchestrator
        mock_backend = MagicMock()
        mock_backend.search = AsyncMock(side_effect=RuntimeError("network down"))
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            use_clr=False,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
            retrieval_backend=mock_backend,
        )
        ctx = await o._build_verifier_context("test", "factual")
        assert "sources" not in ctx

    @pytest.mark.asyncio
    async def test_math_task_does_not_trigger_retrieval(self):
        """Retrieval is factual-only — math tasks must not call the backend."""
        from hybrid_orchestrator import HybridReasoningOrchestrator
        mock_backend = MagicMock()
        mock_backend.search = AsyncMock(return_value=["should not be called"])
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            use_clr=False,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
            retrieval_backend=mock_backend,
        )
        ctx = await o._build_verifier_context("2 + 2", "math")
        assert "sources" not in ctx
        mock_backend.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_code_task_does_not_trigger_retrieval(self):
        """Retrieval is factual-only — code tasks must not call the backend."""
        from hybrid_orchestrator import HybridReasoningOrchestrator
        mock_backend = MagicMock()
        mock_backend.search = AsyncMock(return_value=["should not be called"])
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            use_clr=False,
            use_embedding_router=False,
            use_clr_cache=False,
            use_trajectory_store=False,
            retrieval_backend=mock_backend,
        )
        ctx = await o._build_verifier_context("Write a function", "code")
        assert "sources" not in ctx
        mock_backend.search.assert_not_called()
