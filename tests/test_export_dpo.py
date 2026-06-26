"""Tests for the DPO/SFT exporter (scripts/export_dpo.py)."""

import importlib.util
import json
import os
import sys

import pytest

# scripts/ has no __init__.py — load export_dpo.py as a module by path.
_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "export_dpo.py",
)
_spec = importlib.util.spec_from_file_location("export_dpo", _SCRIPT_PATH)
export_dpo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_dpo)


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def _append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


# ---------------------------------------------------------------------- #
# load_chosen
# ---------------------------------------------------------------------- #
class TestLoadChosen:
    def test_filters_unverified_and_self_claims(self, tmp_path):
        tpath = tmp_path / "verified_trajectories.json"
        _write_json(tpath, {
            "entries": [
                {  # trustworthy chosen
                    "query": "What is 2+2?",
                    "answer": "4",
                    "score": 0.95,
                    "verification_method": "math_verifier",
                    "task_type": "math",
                    "verified": True,
                },
                {  # unverified — must be dropped
                    "query": "What is 3+3?",
                    "answer": "6",
                    "score": 0.9,
                    "verification_method": "math_verifier",
                    "verified": False,
                },
                {  # self_claims_only — must be dropped
                    "query": "What is 4+4?",
                    "answer": "8",
                    "score": 0.9,
                    "verification_method": "self_claims_only",
                    "verified": True,
                },
                {  # empty answer — must be dropped
                    "query": "What is 5+5?",
                    "answer": "",
                    "score": 0.9,
                    "verification_method": "math_verifier",
                    "verified": True,
                },
            ]
        })
        chosen = export_dpo.load_chosen(str(tpath))
        assert len(chosen) == 1
        assert chosen[0]["query"] == "What is 2+2?"
        assert chosen[0]["answer"] == "4"
        assert chosen[0]["normalized"] == "what is 2+2?"

    def test_missing_file_returns_empty(self, tmp_path):
        assert export_dpo.load_chosen(str(tmp_path / "nope.json")) == []

    def test_corrupt_file_returns_empty(self, tmp_path, capsys):
        tpath = tmp_path / "bad.json"
        tpath.write_text("{not valid json")
        assert export_dpo.load_chosen(str(tpath)) == []

    def test_best_answer_alias_accepted(self, tmp_path):
        """Older entries may store the answer under 'best_answer'."""
        tpath = tmp_path / "verified_trajectories.json"
        _write_json(tpath, {
            "entries": [{
                "query": "Q", "best_answer": "A", "best_score": 0.9,
                "verification_method": "unit_tests", "verified": True,
            }]
        })
        chosen = export_dpo.load_chosen(str(tpath))
        assert len(chosen) == 1
        assert chosen[0]["answer"] == "A"


# ---------------------------------------------------------------------- #
# load_rejected
# ---------------------------------------------------------------------- #
class TestLoadRejected:
    def test_clr_low_score_trajectories_are_rejected(self, tmp_path):
        mpath = tmp_path / "orchestrator_memory.jsonl"
        _append_jsonl(mpath, {
            "query": "Explain gravity",
            "answer": "good answer",
            "raw_traces": {
                "clr_result": {
                    "best_answer": "good answer",
                    "best_score": 0.9,
                    "trajectories": [
                        {"answer": "good answer", "score": 0.9},
                        {"answer": "wrong answer", "score": 0.2},
                        {"answer": "another wrong", "score": 0.1},
                    ],
                },
            },
        })
        rej = export_dpo.load_rejected(str(mpath), reject_threshold=0.5,
                                       min_score=0.75)
        norm = export_dpo._normalize_query("Explain gravity")
        assert norm in rej
        assert "wrong answer" in rej[norm]
        assert "another wrong" in rej[norm]
        # The high-scoring near-tie is NOT rejected.
        assert "good answer" not in rej[norm]

    def test_clr_near_tie_not_rejected(self, tmp_path):
        """A trajectory scoring 0.72 vs best 0.75 is a near-tie, not a
        preference signal — it must not be labeled rejected."""
        mpath = tmp_path / "orchestrator_memory.jsonl"
        _append_jsonl(mpath, {
            "query": "Q",
            "answer": "best",
            "raw_traces": {
                "clr_result": {
                    "best_answer": "best", "best_score": 0.75,
                    "trajectories": [
                        {"answer": "best", "score": 0.75},
                        {"answer": "almost as good", "score": 0.72},
                    ],
                },
            },
        })
        rej = export_dpo.load_rejected(str(mpath), reject_threshold=0.5,
                                       min_score=0.75)
        norm = export_dpo._normalize_query("Q")
        assert rej.get(norm, []) == []

    def test_unverified_clr_best_answer_below_min_score(self, tmp_path):
        mpath = tmp_path / "orchestrator_memory.jsonl"
        _append_jsonl(mpath, {
            "query": "Q",
            "answer": "low confidence guess",
            "raw_traces": {
                "clr_result": {
                    "best_answer": "low confidence guess",
                    "best_score": 0.4,
                    "trajectories": [],
                },
            },
        })
        rej = export_dpo.load_rejected(str(mpath), reject_threshold=0.5,
                                       min_score=0.75)
        norm = export_dpo._normalize_query("Q")
        assert rej[norm] == ["low confidence guess"]

    def test_unverified_code_answer_is_rejected(self, tmp_path):
        mpath = tmp_path / "orchestrator_memory.jsonl"
        _append_jsonl(mpath, {
            "query": "Write a sort function",
            "answer": "def sort(x): return x  # buggy",
            "raw_traces": {
                "verified": False,
                "all_verification_traces": [
                    {"candidate_index": 0, "verified": False,
                     "error": "ASSERTION_FAILED: ..."},
                ],
            },
        })
        rej = export_dpo.load_rejected(str(mpath), reject_threshold=0.5,
                                       min_score=0.75)
        norm = export_dpo._normalize_query("Write a sort function")
        assert rej[norm] == ["def sort(x): return x  # buggy"]

    def test_verified_code_not_rejected(self, tmp_path):
        """A verified code run must not contribute a rejected completion."""
        mpath = tmp_path / "orchestrator_memory.jsonl"
        _append_jsonl(mpath, {
            "query": "Write a sort function",
            "answer": "def sort(x): return sorted(x)",
            "raw_traces": {
                "verified": True,
                "all_verification_traces": [
                    {"candidate_index": 0, "verified": True},
                ],
            },
        })
        rej = export_dpo.load_rejected(str(mpath), reject_threshold=0.5,
                                       min_score=0.75)
        assert rej == {}

    def test_non_serializable_raw_traces_skipped(self, tmp_path):
        mpath = tmp_path / "orchestrator_memory.jsonl"
        _append_jsonl(mpath, {
            "query": "Q", "answer": "x",
            "raw_traces": "<non-serializable>",
        })
        rej = export_dpo.load_rejected(str(mpath), reject_threshold=0.5,
                                       min_score=0.75)
        assert rej == {}

    def test_missing_file_returns_empty(self, tmp_path):
        assert export_dpo.load_rejected(
            str(tmp_path / "nope.jsonl"), 0.5, 0.75) == {}

    def test_dedup_rejected_answers(self, tmp_path):
        mpath = tmp_path / "orchestrator_memory.jsonl"
        for _ in range(3):
            _append_jsonl(mpath, {
                "query": "Q", "answer": "wrong",
                "raw_traces": {"verified": False,
                               "all_verification_traces": [{"verified": False}]},
            })
        rej = export_dpo.load_rejected(str(mpath), reject_threshold=0.5,
                                       min_score=0.75)
        norm = export_dpo._normalize_query("Q")
        assert rej[norm] == ["wrong"]  # deduped, not 3 copies


# ---------------------------------------------------------------------- #
# build_dpo_pairs / build_sft_examples
# ---------------------------------------------------------------------- #
class TestBuildPairs:
    def _chosen(self, query="Q1", answer="correct"):
        return [{
            "query": query, "normalized": export_dpo._normalize_query(query),
            "answer": answer, "score": 0.95,
            "verification_method": "math_verifier", "task_type": "math",
        }]

    def test_pairs_match_by_normalized_query(self):
        chosen = self._chosen("  What is 2+2?  ")
        rej_map = {export_dpo._normalize_query("what is 2+2?"): ["wrong"]}
        pairs = export_dpo.build_dpo_pairs(chosen, rej_map, max_pairs_per_query=3)
        assert len(pairs) == 1
        assert pairs[0]["prompt"] == "  What is 2+2?  "
        assert pairs[0]["chosen"] == "correct"
        assert pairs[0]["rejected"] == "wrong"

    def test_never_rejects_the_chosen_answer(self):
        chosen = self._chosen("Q", answer="correct")
        rej_map = {export_dpo._normalize_query("Q"): ["correct", "wrong"]}
        pairs = export_dpo.build_dpo_pairs(chosen, rej_map, max_pairs_per_query=3)
        assert len(pairs) == 1
        assert pairs[0]["rejected"] == "wrong"

    def test_max_pairs_per_query_cap(self):
        chosen = self._chosen("Q")
        rej_map = {export_dpo._normalize_query("Q"):
                   [f"wrong{i}" for i in range(10)]}
        pairs = export_dpo.build_dpo_pairs(chosen, rej_map, max_pairs_per_query=3)
        assert len(pairs) == 3

    def test_no_rejected_yields_no_pairs(self):
        chosen = self._chosen("Q")
        pairs = export_dpo.build_dpo_pairs(chosen, {}, max_pairs_per_query=3)
        assert pairs == []

    def test_sft_examples_format(self):
        chosen = self._chosen("What is 2+2?", answer="4")
        examples = export_dpo.build_sft_examples(chosen)
        assert len(examples) == 1
        msgs = examples[0]["messages"]
        assert msgs[0] == {"role": "user", "content": "What is 2+2?"}
        assert msgs[1] == {"role": "assistant", "content": "4"}


# ---------------------------------------------------------------------- #
# main() end-to-end
# ---------------------------------------------------------------------- #
class TestMain:
    def test_both_format_writes_two_files(self, tmp_path):
        tpath = tmp_path / "verified_trajectories.json"
        _write_json(tpath, {
            "entries": [{
                "query": "What is 2+2?", "answer": "4", "score": 0.95,
                "verification_method": "math_verifier", "verified": True,
            }],
        })
        mpath = tmp_path / "orchestrator_memory.jsonl"
        _append_jsonl(mpath, {
            "query": "what is 2+2?", "answer": "5",
            "raw_traces": {"verified": False,
                           "all_verification_traces": [{"verified": False}]},
        })
        out = tmp_path / "dataset"
        rc = export_dpo.main([
            "--trajectories", str(tpath), "--memory", str(mpath),
            "--out", str(out), "--format", "both",
        ])
        assert rc == 0
        dpo_path = str(out) + ".dpo.jsonl"
        sft_path = str(out) + ".sft.jsonl"
        assert os.path.exists(dpo_path)
        assert os.path.exists(sft_path)
        with open(dpo_path) as f:
            dpo = [json.loads(l) for l in f]
        with open(sft_path) as f:
            sft = [json.loads(l) for l in f]
        assert len(dpo) == 1
        assert dpo[0]["chosen"] == "4"
        assert dpo[0]["rejected"] == "5"
        assert len(sft) == 1
        assert sft[0]["messages"][1]["content"] == "4"

    def test_no_chosen_returns_nonzero(self, tmp_path, capsys):
        tpath = tmp_path / "empty.json"
        _write_json(tpath, {"entries": []})
        mpath = tmp_path / "mem.jsonl"
        mpath.write_text("")
        rc = export_dpo.main([
            "--trajectories", str(tpath), "--memory", str(mpath),
            "--out", str(tmp_path / "out"), "--format", "dpo",
        ])
        assert rc == 1

    def test_sft_only_no_memory_file(self, tmp_path):
        """SFT export works with no memory log at all (chosen-only)."""
        tpath = tmp_path / "verified_trajectories.json"
        _write_json(tpath, {
            "entries": [{
                "query": "Q", "answer": "A", "score": 0.9,
                "verification_method": "math_verifier", "verified": True,
            }],
        })
        out = tmp_path / "out"
        rc = export_dpo.main([
            "--trajectories", str(tpath),
            "--memory", str(tmp_path / "nonexistent.jsonl"),
            "--out", str(out), "--format", "sft",
        ])
        assert rc == 0
        with open(str(out) + ".jsonl") as f:
            lines = [json.loads(l) for l in f]
        assert len(lines) == 1
        assert lines[0]["messages"][1]["content"] == "A"
