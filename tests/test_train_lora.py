"""Tests for the manual train_lora subcommand (v3.2 Step 4.1).

Tests the diversity-stats computation and the human-in-the-loop gate.
Does NOT test the actual trainer invocation (mlx_lm.lora / unsloth) —
those require installed ML backends and are integration-level.
"""

import json
import os
import sys
import pytest

# Make scripts/ importable.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))


@pytest.fixture
def sft_dataset(tmp_path):
    """A small SFT dataset with mixed task types."""
    path = tmp_path / "ds.sft.jsonl"
    examples = [
        {"messages": [{"role": "user", "content": "def fib(n): return the nth fibonacci"},
                      {"role": "assistant", "content": "def fib(n): ..."}]},
        {"messages": [{"role": "user", "content": "Solve the equation 2x + 3 = 7"},
                      {"role": "assistant", "content": "x = 2"}]},
        {"messages": [{"role": "user", "content": "Prove that if P implies Q and not Q, then not P"},
                      {"role": "assistant", "content": "Modus tollens..."}]},
        {"messages": [{"role": "user", "content": "def fib(n): return the nth fibonacci"},
                      {"role": "assistant", "content": "def fib(n): ..."}]},  # dup
        {"messages": [{"role": "user", "content": "What is the capital of France?"},
                      {"role": "assistant", "content": "Paris"}]},
    ]
    path.write_text("\n".join(json.dumps(e) for e in examples))
    return str(path)


@pytest.fixture
def dpo_dataset(tmp_path):
    """A small DPO dataset with chosen/rejected pairs."""
    path = tmp_path / "ds.dpo.jsonl"
    examples = [
        {"prompt": "def add(a, b):", "chosen": "def add(a,b): return a+b",
         "rejected": "def add(a,b): return a-b"},
        {"prompt": "Solve 2x=8", "chosen": "x=4", "rejected": "x=2"},
    ]
    path.write_text("\n".join(json.dumps(e) for e in examples))
    return str(path)


class TestDiversityStats:
    def test_sft_dataset_stats(self, sft_dataset):
        import train_lora
        examples = train_lora._load_dataset(sft_dataset)
        stats = train_lora.compute_diversity_stats(examples)
        assert stats["total_examples"] == 5
        assert stats["format"] == "sft"
        assert stats["unique_queries"] < 5  # there's a dup
        assert stats["query_diversity_ratio"] < 1.0
        # Task types detected.
        ttypes = stats["task_type_distribution"]
        assert ttypes.get("code", 0) >= 1
        assert ttypes.get("math", 0) >= 1
        assert ttypes.get("logic", 0) >= 1
        assert ttypes.get("factual", 0) >= 1

    def test_dpo_dataset_stats(self, dpo_dataset):
        import train_lora
        examples = train_lora._load_dataset(dpo_dataset)
        stats = train_lora.compute_diversity_stats(examples)
        assert stats["total_examples"] == 2
        assert stats["format"] == "dpo"
        assert stats["dpo_pairs_complete"] == 2
        assert stats["dpo_pairs_incomplete"] == 0

    def test_empty_dataset_stats(self):
        import train_lora
        stats = train_lora.compute_diversity_stats([])
        assert stats["total_examples"] == 0
        assert stats["query_diversity_ratio"] == 0.0


class TestHumanInTheLoopGate:
    def test_no_yes_flag_exits_after_stats(self, sft_dataset, capsys):
        """Without --yes, the script prints stats and exits 0 WITHOUT
        training. This is the human-review boundary."""
        import train_lora
        rc = train_lora.main([
            "--dataset", sft_dataset,
            "--model", "fake.gguf",
            "--output-dir", "/tmp/lora-out-test",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "DIVERSITY REPORT" in out
        assert "--yes" in out  # tells the user how to proceed
        # Must NOT have invoked a trainer.
        assert "invoking:" not in out

    def test_yes_flag_proceeds_to_trainer_detection(self, sft_dataset, capsys, monkeypatch):
        """With --yes, the script proceeds past the gate. Since no
        trainer is installed in the test env, it exits with code 3
        (no trainer found) — but only AFTER printing the stats."""
        import train_lora
        # Force the auto-detection to find no trainer.
        monkeypatch.setattr(train_lora, "_detect_trainer", lambda: "none")
        rc = train_lora.main([
            "--dataset", sft_dataset,
            "--model", "fake.gguf",
            "--output-dir", "/tmp/lora-out-test",
            "--yes",
        ])
        assert rc == 3
        out = capsys.readouterr().out
        assert "DIVERSITY REPORT" in out  # stats still printed first

    def test_missing_dataset_returns_2(self, capsys):
        import train_lora
        rc = train_lora.main([
            "--dataset", "/nonexistent/path.jsonl",
            "--model", "fake.gguf",
            "--output-dir", "/tmp/x",
        ])
        assert rc == 2

    def test_empty_dataset_returns_2(self, tmp_path, capsys):
        import train_lora
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        rc = train_lora.main([
            "--dataset", str(p),
            "--model", "fake.gguf",
            "--output-dir", "/tmp/x",
        ])
        assert rc == 2


class TestTaskTypeDetection:
    def test_code_detection(self):
        import train_lora
        assert train_lora._detect_task_type("def foo(): pass") == "code"
        assert train_lora._detect_task_type("write a python function") == "code"

    def test_math_detection(self):
        import train_lora
        assert train_lora._detect_task_type("solve 2x + 3 = 7") == "math"
        assert train_lora._detect_task_type("calculate the integral") == "math"

    def test_logic_detection(self):
        import train_lora
        assert train_lora._detect_task_type("prove that P implies Q") == "logic"
        assert train_lora._detect_task_type("translate to z3 constraints") == "logic"

    def test_factual_detection(self):
        import train_lora
        assert train_lora._detect_task_type("what is the capital of France") == "factual"


class TestValidateDatasetFormat:
    """v3.2: dataset-format validation against trainer expectations."""

    def test_sft_format_valid_for_mlx(self, sft_dataset):
        import train_lora
        examples = train_lora._load_dataset(sft_dataset)
        errors = train_lora.validate_dataset_format(examples, "mlx_lm")
        assert errors == []

    def test_dpo_format_invalid_for_mlx(self, dpo_dataset):
        """mlx_lm.lora doesn't accept DPO (prompt/chosen/rejected) — it
        wants SFT (messages) or prompt/completion. This must fail fast."""
        import train_lora
        examples = train_lora._load_dataset(dpo_dataset)
        errors = train_lora.validate_dataset_format(examples, "mlx_lm")
        assert len(errors) > 0
        assert "mlx_lm" in errors[0] or "SFT" in errors[0]

    def test_sft_format_valid_for_unsloth(self, sft_dataset):
        import train_lora
        examples = train_lora._load_dataset(sft_dataset)
        errors = train_lora.validate_dataset_format(examples, "unsloth")
        assert errors == []

    def test_dpo_format_valid_for_unsloth(self, dpo_dataset):
        """unsloth accepts both SFT and DPO."""
        import train_lora
        examples = train_lora._load_dataset(dpo_dataset)
        errors = train_lora.validate_dataset_format(examples, "unsloth")
        assert errors == []

    def test_empty_dataset_invalid(self):
        import train_lora
        errors = train_lora.validate_dataset_format([], "mlx_lm")
        assert len(errors) == 1
        assert "empty" in errors[0]

    def test_malformed_messages_caught(self, tmp_path):
        """An SFT example with a malformed messages entry is caught."""
        import train_lora
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps({"messages": [{"role": "user"}]}))  # no content
        examples = train_lora._load_dataset(str(path))
        errors = train_lora.validate_dataset_format(examples, "mlx_lm")
        assert any("content" in e for e in errors)

    def test_neither_format_caught(self, tmp_path):
        """A dataset that's neither SFT nor DPO is caught for both trainers."""
        import train_lora
        path = tmp_path / "neither.jsonl"
        path.write_text(json.dumps({"foo": "bar"}))
        examples = train_lora._load_dataset(str(path))
        errs_mlx = train_lora.validate_dataset_format(examples, "mlx_lm")
        errs_unsloth = train_lora.validate_dataset_format(examples, "unsloth")
        assert len(errs_mlx) > 0
        assert len(errs_unsloth) > 0


class TestDryRun:
    """v3.2: --dry-run validates format + prints command, no execution."""

    def test_dry_run_sft_mlx_passes(self, sft_dataset, capsys, monkeypatch):
        import train_lora
        monkeypatch.setattr(train_lora, "_detect_trainer", lambda: "mlx_lm")
        rc = train_lora.main([
            "--dataset", sft_dataset,
            "--model", "fake.gguf",
            "--output-dir", "/tmp/lora-dryrun",
            "--dry-run",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "DRY RUN" in out
        assert "FORMAT VALIDATION: PASSED" in out
        assert "mlx_lm.lora" in out
        assert "no training executed" in out.lower()

    def test_dry_run_dpo_mlx_fails_format(self, dpo_dataset, capsys, monkeypatch):
        """DPO format + mlx_lm -> format validation fails, exit 1."""
        import train_lora
        monkeypatch.setattr(train_lora, "_detect_trainer", lambda: "mlx_lm")
        rc = train_lora.main([
            "--dataset", dpo_dataset,
            "--model", "fake.gguf",
            "--output-dir", "/tmp/lora-dryrun",
            "--dry-run",
        ])
        assert rc == 1
        out = capsys.readouterr().out
        assert "FORMAT VALIDATION FAILED" in out

    def test_dry_run_dpo_unsloth_passes(self, dpo_dataset, capsys, monkeypatch):
        import train_lora
        monkeypatch.setattr(train_lora, "_detect_trainer", lambda: "unsloth")
        rc = train_lora.main([
            "--dataset", dpo_dataset,
            "--model", "fake-model",
            "--output-dir", "/tmp/lora-dryrun",
            "--dry-run",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "FORMAT VALIDATION: PASSED" in out
        assert "unsloth" in out

    def test_dry_run_no_trainer_returns_3(self, sft_dataset, capsys, monkeypatch):
        import train_lora
        monkeypatch.setattr(train_lora, "_detect_trainer", lambda: "none")
        rc = train_lora.main([
            "--dataset", sft_dataset,
            "--model", "fake.gguf",
            "--output-dir", "/tmp/lora-dryrun",
            "--dry-run",
        ])
        assert rc == 3

    def test_dry_run_does_not_execute_trainer(self, sft_dataset, capsys, monkeypatch):
        """Dry-run must NEVER shell out — even with --yes also set."""
        import train_lora
        monkeypatch.setattr(train_lora, "_detect_trainer", lambda: "mlx_lm")
        called = []
        monkeypatch.setattr(train_lora, "train_with_mlx",
                            lambda *a, **k: called.append(1) or 0)
        rc = train_lora.main([
            "--dataset", sft_dataset,
            "--model", "fake.gguf",
            "--output-dir", "/tmp/lora-dryrun",
            "--dry-run", "--yes",
        ])
        assert rc == 0
        assert called == []  # train_with_mlx was NOT called
        assert train_lora._detect_task_type("explain why the sky is blue") == "factual"
