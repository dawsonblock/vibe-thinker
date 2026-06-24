"""Pytest tests for the REPL (no model servers needed)."""

import asyncio
import pytest

from rfsn_cli import _split_flags, JobQueueREPL


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
