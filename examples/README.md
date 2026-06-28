# vibe-thinker examples

Three self-contained demos that showcase the system's core pipelines.
Each runs **without a live model server** (using mock/synthetic data) and
includes a `--live` mode for testing against a real llama-server or
ruvllm_py instance.

## Quick start

```bash
# 1. Math reasoning — routing + \boxed{} extraction + verification
python examples/demo_math_reasoning.py

# 2. Code verification — static analysis + sandbox + golden-set regression
python examples/demo_code_verification.py

# 3. TurboQuant PPL — perplexity math + comparison + logprob extraction
python examples/demo_turboquant_ppl.py
```

## Demo 1: Math Reasoning (`demo_math_reasoning.py`)

Shows the specialist routing pipeline:
- The router classifies math queries → specialist route (conf=0.800)
- The model produces step-by-step reasoning with `\boxed{}` answers
- The orchestrator extracts and verifies the final answer

Three problems: divisor sum (number theory), derivative (calculus),
combinatorics (arrangements with constraints).

**Live mode:**
```bash
llama-server -m model.gguf --port 8080
python examples/demo_math_reasoning.py --live --vibe http://127.0.0.1:8080
```

## Demo 2: Code Verification (`demo_code_verification.py`)

Shows the defense-in-depth code verification pipeline:
- **Static analysis**: AST parse + restricted-import check + evasion detection
- **Wasmtime sandbox**: fuel-limited execution (if configured)
- **Verifier golden-set**: 9 curated verified/hallucinated pairs that guard
  against verifier regressions

Catches: `os`/`subprocess`/`socket` imports, `__import__` evasion,
`importlib` evasion, `__builtins__` reflection, syntax errors.

No live mode needed — all checks are local.

## Demo 3: TurboQuant PPL (`demo_turboquant_ppl.py`)

Shows the KV-cache compression perplexity validation:
- **PPL math**: `compute_ppl()` from per-token log-probabilities
- **Comparison**: `compare_ppl()` with tolerance-based pass/fail
- **Logprob extraction**: parsing llama-server `/completion` responses
- **End-to-end simulation**: baseline (f16) vs candidate (q8_0/turbo3)

**Live mode (HTTP):**
```bash
llama-server -m model.gguf --port 8081 --logprobs
python examples/demo_turboquant_ppl.py --live --base-url http://127.0.0.1:8081
```

**Live mode (in-process, Apple Silicon):**
```bash
cd ruvllm_py && maturin develop --release --features inference-metal
python examples/demo_turboquant_ppl.py --inprocess --model model.gguf --metal
```
