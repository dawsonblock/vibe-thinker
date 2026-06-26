"""Pytest tests for the REPL (no model servers needed)."""

import asyncio
import os
import pytest

from rfsn_cli import _env_bool, _split_flags, build_argparser, JobQueueREPL


class TestFlagParsing:
    @pytest.mark.parametrize("tokens,expected_flags,expected_rest", [
        (["-p", "5", "hello"], {"priority": 5}, ["hello"]),
        (["hello", "-p", "5"], {"priority": 5}, ["hello"]),
        (["a", "-r", "specialist", "-p", "3"], {"force_route": "specialist", "priority": 3}, ["a"]),
        (["--axis", "transaction", "job1"], {"axis": "transaction"}, ["job1"]),
    ])
    def test_flag_extraction(self, tokens, expected_flags, expected_rest):
        flags, rest = _split_flags(tokens)
        assert flags == expected_flags
        assert rest == expected_rest

    def test_dangling_flag_no_value(self):
        flags, _ = _split_flags(["-p"])
        assert flags == {}


class TestCLIFlags:
    """Tests for the CLI boolean flag parsing (--clr/--no-clr, etc.).

    The old ``--clr/--no-clr`` argparse syntax was broken: ``--no-clr`` was
    not recognized. These tests verify the split flag approach works.
    """

    def test_cli_parses_no_clr(self):
        args = build_argparser().parse_args(["--no-clr"])
        assert args.use_clr is False

    def test_cli_parses_clr(self):
        args = build_argparser().parse_args(["--clr"])
        assert args.use_clr is True

    def test_cli_default_clr_is_true(self):
        args = build_argparser().parse_args([])
        assert args.use_clr is True

    def test_cli_parses_no_embedding_router(self):
        args = build_argparser().parse_args(["--no-embedding-router"])
        assert args.use_embedding_router is False

    def test_cli_parses_embedding_router(self):
        args = build_argparser().parse_args(["--embedding-router"])
        assert args.use_embedding_router is True

    def test_cli_default_embedding_router_is_true(self):
        args = build_argparser().parse_args([])
        assert args.use_embedding_router is True

    def test_cli_parses_both_flags(self):
        args = build_argparser().parse_args(["--no-clr", "--no-embedding-router"])
        assert args.use_clr is False
        assert args.use_embedding_router is False


class TestEnvVarSupport:
    """Tests for environment variable loading and CLI override precedence."""

    def test_env_bool_parsing(self):
        assert _env_bool("NONEXISTENT_VAR", True) is True
        assert _env_bool("NONEXISTENT_VAR", False) is False
        os.environ["TEST_BOOL_TRUE"] = "true"
        assert _env_bool("TEST_BOOL_TRUE", False) is True
        os.environ["TEST_BOOL_FALSE"] = "false"
        assert _env_bool("TEST_BOOL_FALSE", True) is False
        os.environ["TEST_BOOL_1"] = "1"
        assert _env_bool("TEST_BOOL_1", False) is True
        os.environ["TEST_BOOL_YES"] = "yes"
        assert _env_bool("TEST_BOOL_YES", False) is True
        for k in ("TEST_BOOL_TRUE", "TEST_BOOL_FALSE", "TEST_BOOL_1", "TEST_BOOL_YES"):
            del os.environ[k]

    def test_env_vibe_url_loaded(self, monkeypatch):
        monkeypatch.setenv("VIBE_THINKER_URL", "http://env-specialist:9999")
        args = build_argparser().parse_args([])
        assert args.vibe == "http://env-specialist:9999"

    def test_env_generalist_url_loaded(self, monkeypatch):
        monkeypatch.setenv("GENERALIST_URL", "http://env-generalist:8888")
        args = build_argparser().parse_args([])
        assert args.generalist == "http://env-generalist:8888"

    def test_env_code_specialist_url_loaded(self, monkeypatch):
        monkeypatch.setenv("CODE_SPECIALIST_URL", "http://env-code-specialist:8082")
        args = build_argparser().parse_args([])
        assert args.code_specialist == "http://env-code-specialist:8082"

    def test_code_specialist_defaults_empty(self):
        args = build_argparser().parse_args([])
        assert args.code_specialist == ""

    def test_cli_overrides_env(self, monkeypatch):
        """CLI flags take precedence over environment variables."""
        monkeypatch.setenv("VIBE_THINKER_URL", "http://env-specialist:9999")
        args = build_argparser().parse_args(["--vibe", "http://cli-specialist:7777"])
        assert args.vibe == "http://cli-specialist:7777"

    def test_env_max_concurrent_loaded(self, monkeypatch):
        monkeypatch.setenv("RFSN_MAX_CONCURRENT", "5")
        args = build_argparser().parse_args([])
        assert args.max_concurrent == 5

    def test_env_use_clr_false(self, monkeypatch):
        monkeypatch.setenv("RFSN_USE_CLR", "false")
        args = build_argparser().parse_args([])
        assert args.use_clr is False

    def test_env_use_clr_true(self, monkeypatch):
        monkeypatch.setenv("RFSN_USE_CLR", "true")
        args = build_argparser().parse_args([])
        assert args.use_clr is True

    def test_env_clr_k_loaded(self, monkeypatch):
        monkeypatch.setenv("RFSN_CLR_K", "16")
        args = build_argparser().parse_args([])
        assert args.clr_k == 16

    def test_cli_overrides_env_clr(self, monkeypatch):
        """CLI --no-clr overrides env RFSN_USE_CLR=true."""
        monkeypatch.setenv("RFSN_USE_CLR", "true")
        args = build_argparser().parse_args(["--no-clr"])
        assert args.use_clr is False

    def test_boolean_env_parsing_on(self, monkeypatch):
        monkeypatch.setenv("RFSN_USE_EMBEDDING_ROUTER", "on")
        args = build_argparser().parse_args([])
        assert args.use_embedding_router is True

    def test_fast_specialist_defaults_off(self):
        args = build_argparser().parse_args([])
        assert args.fast_specialist is False

    def test_fast_specialist_cli_flag(self):
        args = build_argparser().parse_args(["--fast-specialist"])
        assert args.fast_specialist is True

    def test_fast_specialist_env_true(self, monkeypatch):
        monkeypatch.setenv("RFSN_FAST_SPECIALIST", "true")
        args = build_argparser().parse_args([])
        assert args.fast_specialist is True

    def test_local_specialist_model_defaults_empty(self):
        args = build_argparser().parse_args([])
        assert args.local_specialist_model == ""

    def test_local_specialist_model_cli(self):
        args = build_argparser().parse_args(
            ["--local-specialist-model", "/tmp/tiny.gguf"]
        )
        assert args.local_specialist_model == "/tmp/tiny.gguf"

    def test_local_specialist_model_env(self, monkeypatch):
        monkeypatch.setenv("VIBE_THINKER_LOCAL_MODEL", "ruv/ruvltra-claude-code-0.5b-q4_k_m.gguf")
        args = build_argparser().parse_args([])
        assert args.local_specialist_model == "ruv/ruvltra-claude-code-0.5b-q4_k_m.gguf"

    def test_local_specialist_n_ctx_env(self, monkeypatch):
        monkeypatch.setenv("VIBE_THINKER_LOCAL_N_CTX", "8192")
        args = build_argparser().parse_args([])
        assert args.local_specialist_n_ctx == 8192

    def test_local_specialist_pool_size_defaults_to_1(self):
        args = build_argparser().parse_args([])
        assert args.local_specialist_pool_size == 1

    def test_local_specialist_pool_size_cli(self):
        args = build_argparser().parse_args(["--local-specialist-pool-size", "4"])
        assert args.local_specialist_pool_size == 4

    def test_local_specialist_pool_size_env(self, monkeypatch):
        monkeypatch.setenv("VIBE_THINKER_LOCAL_POOL_SIZE", "8")
        args = build_argparser().parse_args([])
        assert args.local_specialist_pool_size == 8




class FakeResult:
    final_answer = "42 is the answer"


class FakeJob:
    job_id = "abc123"
    priority = 5
    force_route = None
    status = type("S", (), {"value": "completed"})()
    query = "test query here"
    result = FakeResult()
    error = None

    def to_dict(self):
        return {"job_id": self.job_id, "status": "completed", "priority": 5,
                "force_route": None, "query": "test query here",
                "created_at": "t1", "started_at": "t2", "finished_at": "t3",
                "error": None, "has_result": True}


class FakeQueue:
    bitemporal = None

    def submit(self, q, priority=0, force_route=None):
        return FakeJob()

    def list_jobs(self):
        return [FakeJob().to_dict()]

    def get(self, jid):
        return FakeJob() if jid == "abc123" else None

    def cancel(self, jid):
        return jid == "abc123"

    def job_history(self, jid, axis="valid"):
        if jid == "abc123":
            return [{"valid_time": "2026-01-01T00:00:00+00:00",
                     "transaction_time": "2026-01-01T00:00:01+00:00",
                     "event": "submitted", "status": "pending",
                     "extra": {"route": "specialist"}}]
        return []

    def state_as_of(self, jid, t, axis="valid"):
        if jid == "abc123":
            return {"valid_time": "2026-01-01T00:00:00+00:00",
                    "event": "submitted", "status": "pending"}
        return None


@pytest.fixture
def repl():
    return JobQueueREPL(FakeQueue())


class TestDispatchEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_string_no_crash(self, repl):
        await repl._dispatch("")

    @pytest.mark.asyncio
    async def test_whitespace_only_no_crash(self, repl):
        await repl._dispatch("   ")

    @pytest.mark.asyncio
    async def test_unknown_command(self, repl, capsys):
        await repl._dispatch("bogus")
        out = capsys.readouterr().out
        assert "unknown command" in out

    @pytest.mark.asyncio
    async def test_submit_no_query(self, repl, capsys):
        await repl._dispatch("submit")
        out = capsys.readouterr().out
        assert "usage" in out

    @pytest.mark.asyncio
    async def test_status_unknown_job(self, repl, capsys):
        await repl._dispatch("status nope")
        out = capsys.readouterr().out
        assert "no such job" in out

    @pytest.mark.asyncio
    async def test_result_unknown_job(self, repl, capsys):
        await repl._dispatch("result nope")
        out = capsys.readouterr().out
        assert "no such job" in out


class TestDispatchCommands:
    @pytest.mark.asyncio
    async def test_submit(self, repl, capsys):
        await repl._dispatch("submit hello -p 5 -r specialist")
        out = capsys.readouterr().out
        assert "submitted job" in out

    @pytest.mark.asyncio
    async def test_list(self, repl, capsys):
        await repl._dispatch("list")
        out = capsys.readouterr().out
        assert "abc123" in out

    @pytest.mark.asyncio
    async def test_status(self, repl, capsys):
        await repl._dispatch("status abc123")
        out = capsys.readouterr().out
        assert "abc123" in out

    @pytest.mark.asyncio
    async def test_result(self, repl, capsys):
        await repl._dispatch("result abc123")
        out = capsys.readouterr().out
        assert "42 is the answer" in out

    @pytest.mark.asyncio
    async def test_history(self, repl, capsys):
        await repl._dispatch("history abc123")
        out = capsys.readouterr().out
        assert "submitted" in out

    @pytest.mark.asyncio
    async def test_help(self, repl, capsys):
        await repl._dispatch("help")
        out = capsys.readouterr().out
        assert "submit" in out
        assert "quit" in out
