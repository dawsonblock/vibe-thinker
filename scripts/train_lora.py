#!/usr/bin/env python3
"""Manual LoRA training subcommand for the vibe-thinker data flywheel.

This is the HUMAN-IN-THE-LOOP boundary of the flywheel. It does NOT:
  - auto-train when "X new patterns accumulate" (the original plan's idea,
    which bakes in selection bias toward verifiable task types)
  - hot-swap adapters into a running llama-cpp-python / ruvllm_py instance
    (that's a model reload, not a primitive — and silently swapping a
    learned adapter into production is an unreviewed deployment)

It DOES:
  - read an exported DPO/SFT dataset (from scripts/export_dpo.py)
  - compute and print DIVERSITY STATS before training (task-type
    distribution, unique-query count, verification-method breakdown) so
    the operator can decide whether the dataset is worth training on
  - require an explicit --yes flag to proceed past the stats review
  - shell out to the chosen trainer (mlx_lm.lora on Apple Silicon,
    unsloth on CUDA) with the dataset + a config
  - write the LoRA adapter to --output-dir and STOP

The operator then reviews the adapter and loads it manually (a model
reload). This keeps the export as the reviewable boundary and makes
every trained adapter an explicit, reviewed deployment.

Usage:
    # 1. Export a dataset (review the trajectories first):
    python3 scripts/export_dpo.py --format both --out dataset

    # 2. Review diversity stats BEFORE training:
    python3 scripts/train_lora.py --dataset dataset.sft.jsonl \\
        --model ./my-model.gguf --output-dir ./lora-adapter

    # 3. After reviewing the stats, explicitly proceed:
    python3 scripts/train_lora.py --dataset dataset.sft.jsonl \\
        --model ./my-model.gguf --output-dir ./lora-adapter --yes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------- #
# Diversity stats — computed BEFORE training, printed for human review.
# ---------------------------------------------------------------------- #

def _load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL dataset (SFT or DPO format)."""
    examples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def _extract_query(ex: Dict[str, Any]) -> str:
    """Extract the user query from an SFT or DPO example."""
    if "prompt" in ex:
        return str(ex["prompt"])[:200]
    if "messages" in ex:
        for m in ex["messages"]:
            if m.get("role") == "user":
                return str(m.get("content", ""))[:200]
    return ""


def _detect_task_type(query: str) -> str:
    """Coarse task-type classification for diversity stats."""
    q = query.lower()
    if any(k in q for k in ("def ", "function", "class ", "import ", "code",
                            "python", "algorithm", "leetcode")):
        return "code"
    if any(k in q for k in ("prove", "theorem", "implies", "constraint",
                            "logic", "z3", "smt")):
        return "logic"
    if any(k in q for k in ("calculate", "sum", "product", "equation",
                            "solve", "integral", "derivative", "matrix")):
        return "math"
    if any(k in q for k in ("explain", "what is", "why", "describe",
                            "compare", "summarize")):
        return "factual"
    return "other"


def compute_diversity_stats(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute diversity stats for a dataset.

    Reports:
      - total examples
      - unique queries (dedup by normalized first-200-chars) — a dataset
        of 100 identical queries teaches nothing new
      - task-type distribution (code/math/logic/factual/other)
      - format (sft messages vs dpo prompt/chosen/rejected)
      - DPO pair completeness (how many have both chosen + rejected)
    """
    queries = [_extract_query(ex) for ex in examples]
    unique_queries = len(set(q.strip().lower() for q in queries if q))
    task_types = Counter(_detect_task_type(q) for q in queries)

    is_dpo = any("chosen" in ex and "rejected" in ex for ex in examples)
    is_sft = any("messages" in ex for ex in examples)
    dpo_pairs = sum(1 for ex in examples
                    if ex.get("chosen") and ex.get("rejected"))

    return {
        "total_examples": len(examples),
        "unique_queries": unique_queries,
        "query_diversity_ratio": (
            unique_queries / len(examples) if examples else 0.0
        ),
        "task_type_distribution": dict(task_types),
        "format": "dpo" if is_dpo else ("sft" if is_sft else "unknown"),
        "dpo_pairs_complete": dpo_pairs,
        "dpo_pairs_incomplete": len(examples) - dpo_pairs if is_dpo else 0,
    }


def validate_dataset_format(examples: List[Dict[str, Any]],
                            trainer: str) -> List[str]:
    """Validate that the dataset matches what the chosen trainer expects.

    Returns a list of error strings (empty list = valid). This runs BEFORE
    shelling out to the trainer so a malformed dataset fails fast with a
    clear message instead of a cryptic traceback inside mlx_lm/unsloth.

    mlx_lm.lora: expects an SFT JSONL where each line has a "messages" key
      (list of {role, content}) OR a "prompt"/"completion" pair. DPO format
      (prompt/chosen/rejected) is NOT accepted by mlx_lm.lora directly.
    unsloth: accepts either SFT (messages) or DPO (prompt/chosen/rejected)
      via the UnslothTrainer; the driver script in train_with_unsloth
      loads the JSONL as a list and passes it to the trainer, which handles
      both shapes.
    """
    errors: List[str] = []
    if not examples:
        errors.append("dataset is empty")
        return errors

    is_sft = any("messages" in ex for ex in examples)
    is_dpo = any("chosen" in ex and "rejected" in ex for ex in examples)

    if trainer == "mlx_lm":
        # mlx_lm.lora wants SFT (messages) or prompt/completion.
        if not is_sft and not any("prompt" in ex and "completion" in ex
                                  for ex in examples):
            errors.append(
                "mlx_lm.lora expects SFT format: each line needs a "
                "'messages' key (list of {role, content}) OR a "
                "'prompt'+'completion' pair. DPO format "
                "(prompt/chosen/rejected) is not accepted by mlx_lm.lora "
                "directly — re-export with --format sft, or use unsloth."
            )
        # Check messages structure if SFT.
        for i, ex in enumerate(examples):
            msgs = ex.get("messages")
            if msgs is not None:
                if not isinstance(msgs, list) or not msgs:
                    errors.append(f"example {i}: 'messages' must be a "
                                  f"non-empty list")
                    continue
                for j, m in enumerate(msgs):
                    if not isinstance(m, dict) or "role" not in m or "content" not in m:
                        errors.append(f"example {i} message {j}: each "
                                      f"message must have 'role' and "
                                      f"'content'")
                        break
    elif trainer == "unsloth":
        # unsloth accepts either; just validate the shape is one of them.
        if not is_sft and not is_dpo:
            errors.append(
                "unsloth expects SFT ('messages') or DPO "
                "('prompt'+'chosen'+'rejected') format. Found neither."
            )
    else:
        errors.append(f"unknown trainer: {trainer}")
    return errors


def print_diversity_report(stats: Dict[str, Any]) -> None:
    """Print a human-readable diversity report."""
    print("=" * 60)
    print("DATASET DIVERSITY REPORT — review BEFORE training")
    print("=" * 60)
    print(f"  Total examples:      {stats['total_examples']}")
    print(f"  Unique queries:      {stats['unique_queries']}")
    ratio = stats["query_diversity_ratio"]
    print(f"  Query diversity:     {ratio:.1%} "
          f"({'GOOD' if ratio > 0.8 else 'LOW — may overfit' if ratio > 0.5 else 'POOR — do not train'})")
    print(f"  Format:              {stats['format']}")
    if stats["format"] == "dpo":
        print(f"  DPO pairs complete:  {stats['dpo_pairs_complete']}")
        print(f"  DPO pairs incomplete:{stats['dpo_pairs_incomplete']}")
    print()
    print("  Task-type distribution:")
    for ttype, count in sorted(stats["task_type_distribution"].items(),
                               key=lambda x: -x[1]):
        pct = count / stats["total_examples"] * 100 if stats["total_examples"] else 0
        bar = "#" * int(pct / 2)
        print(f"    {ttype:10s} {count:4d} ({pct:5.1f}%) {bar}")
    # Selection-bias warning: if one task type is >70%, warn.
    dist = stats["task_type_distribution"]
    if dist:
        top_type, top_count = max(dist.items(), key=lambda x: x[1])
        if stats["total_examples"] and top_count / stats["total_examples"] > 0.7:
            print()
            print(f"  WARNING: '{top_type}' is {top_count}/{stats['total_examples']} "
                  f"({top_count/stats['total_examples']:.0%}) of the dataset.")
            print(f"  Training on this will sharpen the model on {top_type} "
                  f"and leave other task types under-served. This is the "
                  f"known selection-bias failure mode of verification-driven "
                  f"training. Consider collecting more diverse data before "
                  f"proceeding.")
    print("=" * 60)


# ---------------------------------------------------------------------- #
# Trainer invocation — mlx_lm.lora (Apple Silicon) or unsloth (CUDA).
# ---------------------------------------------------------------------- #

def _detect_trainer() -> str:
    """Detect which LoRA trainer is available."""
    # Prefer mlx_lm on Apple Silicon (no CUDA), unsloth on CUDA.
    if sys.platform == "darwin":
        try:
            import mlx_lm  # noqa: F401
            return "mlx_lm"
        except ImportError:
            pass
    try:
        import unsloth  # noqa: F401
        return "unsloth"
    except ImportError:
        pass
    return "none"


def train_with_mlx(dataset_path: str, model_path: str, output_dir: str,
                   epochs: int, batch_size: int, lr: float) -> int:
    """Train a LoRA adapter with mlx_lm.lora. Returns exit code."""
    # mlx_lm.lora expects a config file. Write one.
    config = {
        "model": model_path,
        "train": True,
        "data": dataset_path,
        "iters": epochs * 10,  # coarse mapping
        "batch_size": batch_size,
        "lora_parameters": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.0,
            "scale": 10.0,
        },
        "lr": lr,
        "steps_per_report": 10,
        "steps_per_eval": 50,
        "adapter_path": output_dir,
    }
    config_path = os.path.join(output_dir, "lora_config.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[train_lora] mlx_lm config written to {config_path}")
    print(f"[train_lora] invoking: python3 -m mlx_lm.lora --config {config_path}")
    import subprocess
    return subprocess.call([sys.executable, "-m", "mlx_lm.lora",
                            "--config", config_path])


def train_with_unsloth(dataset_path: str, model_path: str, output_dir: str,
                       epochs: int, batch_size: int, lr: float) -> int:
    """Train a LoRA adapter with unsloth. Returns exit code.

    unsloth doesn't have a CLI lora subcommand; we write a small driver
    script that imports unsloth and runs the training loop, then shell
    out to it. This keeps the training logic in a reviewable file
    rather than inline string-eval.
    """
    driver = os.path.join(output_dir, "_unsloth_driver.py")
    os.makedirs(output_dir, exist_ok=True)
    driver_src = f'''
import json
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("{model_path}")
model = FastLanguageModel.get_peft_model(model, r=8, target_modules=[
    "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
# Load the SFT dataset (unsloth expects the HF datasets format or a list).
data = []
with open("{dataset_path}") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))
from unsloth import UnslothTrainer, UnslothTrainingArguments
trainer = UnslothTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=data,
    args=UnslothTrainingArguments(
        per_device_train_batch_size={batch_size},
        num_train_epochs={epochs},
        learning_rate={lr},
        output_dir="{output_dir}",
    ),
)
trainer_stats = trainer.train()
model.save_pretrained("{output_dir}")
print(f"[train_lora] unsloth adapter saved to {output_dir}")
'''
    with open(driver, "w") as f:
        f.write(driver_src)
    print(f"[train_lora] unsloth driver written to {driver}")
    print(f"[train_lora] invoking: python3 {driver}")
    import subprocess
    return subprocess.call([sys.executable, driver])


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   help="Path to the exported JSONL dataset (from export_dpo.py).")
    p.add_argument("--model", required=True,
                   help="Path/ID of the base model to train on (.gguf for mlx, "
                        "HF repo id for unsloth).")
    p.add_argument("--output-dir", required=True,
                   help="Where to write the LoRA adapter.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--yes", action="store_true",
                   help="Proceed with training after reviewing the diversity "
                        "stats. WITHOUT this flag, only the stats are printed "
                        "and the script exits (human-in-the-loop gate).")
    p.add_argument("--trainer", choices=["auto", "mlx_lm", "unsloth"],
                   default="auto")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate the dataset format for the chosen trainer "
                        "and print the exact training command WITHOUT "
                        "executing it. Useful for CI / pre-flight checks "
                        "where no ML backend is available. Exits 0 if the "
                        "format is valid, 1 if not.")
    args = p.parse_args(argv)

    # 1. Load + report diversity stats (always — even with --yes).
    if not os.path.exists(args.dataset):
        print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
        return 2
    examples = _load_dataset(args.dataset)
    if not examples:
        print(f"ERROR: dataset is empty: {args.dataset}", file=sys.stderr)
        return 2
    stats = compute_diversity_stats(examples)
    print_diversity_report(stats)

    # 2. Human-in-the-loop gate: without --yes, stop after stats.
    if not args.yes and not args.dry_run:
        print()
        print("Diversity stats printed. Review them above.")
        print("If the dataset is worth training on, re-run with --yes.")
        print("This is the human-review boundary — no auto-training, no "
              "hot-swap. The adapter must be loaded manually after review.")
        return 0

    # 2b. Dry-run: validate format + print the exact command, no execution.
    # Used for CI / pre-flight checks where no ML backend is available.
    if args.dry_run:
        trainer = args.trainer
        if trainer == "auto":
            trainer = _detect_trainer()
        print()
        print("=" * 60)
        print(f"DRY RUN — trainer: {trainer}")
        print("=" * 60)
        # Validate dataset format against the trainer's expectations.
        errors = validate_dataset_format(examples, trainer)
        if errors:
            print("FORMAT VALIDATION FAILED:")
            for e in errors:
                print(f"  - {e}")
            return 1
        print("FORMAT VALIDATION: PASSED")
        # Print the exact command that would run.
        if trainer == "mlx_lm":
            config_path = os.path.join(args.output_dir, "lora_config.json")
            print(f"Would write config to: {config_path}")
            print(f"Would invoke: python3 -m mlx_lm.lora --config {config_path}")
        elif trainer == "unsloth":
            driver = os.path.join(args.output_dir, "_unsloth_driver.py")
            print(f"Would write driver to: {driver}")
            print(f"Would invoke: python3 {driver}")
        else:
            print(f"Would fail: no trainer found (install mlx-lm or unsloth)")
            return 3
        print("DRY RUN complete — no training executed.")
        return 0

    # 3. Detect + invoke the trainer.
    trainer = args.trainer
    if trainer == "auto":
        trainer = _detect_trainer()
    if trainer == "none":
        print("ERROR: no LoRA trainer found. Install one of:", file=sys.stderr)
        print("  Apple Silicon: pip install mlx-lm", file=sys.stderr)
        print("  CUDA:          pip install unsloth", file=sys.stderr)
        return 3
    print(f"[train_lora] using trainer: {trainer}")
    if trainer == "mlx_lm":
        return train_with_mlx(args.dataset, args.model, args.output_dir,
                              args.epochs, args.batch_size, args.lr)
    if trainer == "unsloth":
        return train_with_unsloth(args.dataset, args.model, args.output_dir,
                                  args.epochs, args.batch_size, args.lr)
    print(f"ERROR: unknown trainer: {trainer}", file=sys.stderr)
    return 4


if __name__ == "__main__":
    sys.exit(main())
