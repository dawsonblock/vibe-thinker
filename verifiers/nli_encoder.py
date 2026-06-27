"""Optional encoder-only NLI judge for the FactualVerifier (v1.1).

An alternative to the LLM-judge NLI path that uses a dedicated encoder-only
model fine-tuned for Natural Language Inference (e.g. DeBERTa-v3-base-mnli).
Encoder-only models output fixed probabilities for
[entailment, neutral, contradiction] — they are not generative, so they
cannot hallucinate a verdict. This is a robustness win, not a determinism
win (a generative judge at temperature=0 is also deterministic; the
problem the citation-backed path solves is fabrication, not sampling noise).

Trust model (fail-closed, no epistemic contamination):
  - When ``transformers`` / ``torch`` is not installed, ``is_available()``
    returns False and constructing :class:`EncoderNLIJudge` raises
    ``ImportError`` with an install hint. The orchestrator falls back to
    the existing LLM judge (or ``nli_unavailable``). This mirrors the
    project's pattern for ``cryptography``, ``z3-solver``, and
    ``llama-cpp-python`` — heavy deps are optional extras, never core.
  - The judge returns the same JSON shape the FactualVerifier expects
    (``{"verdict": "...", "supporting_quote": ""}``), but with an empty
    ``supporting_quote`` — encoder models classify the (source, claim)
    pair but do not extract supporting spans. The FactualVerifier's
    citation check therefore falls back to the un-cited path (score 0.7,
    below the 0.75 cache threshold). This is honest: an encoder NLI
    verdict is stronger than self-claims but does not carry a verifiable
    citation, so it does not get the 0.8 citation-backed score.
  - On ANY failure (model load, inference, unexpected output), the judge
    raises — the FactualVerifier catches it and returns ``nli_judge_error``
    (fail-closed, score 0.0). No fabrication.

Determinism note: on CPU the encoder is deterministic. On GPU/MPS, set
``torch.manual_seed`` and use deterministic algorithms for true
reproducibility — the constructor does this when ``deterministic=True``.

Install (optional extra):
  pip install "vibe-thinker[nli]"   # transformers + torch
  # or explicitly:
  pip install transformers torch
"""

from __future__ import annotations

import json
import os
from typing import Optional


# Default model: a small, fast NLI cross-encoder. Smaller than
# deberta-v3-large-mnli (~440MB) but strong on the MNLI benchmark. Override
# via the ``VIBE_THINKER_NLI_MODEL`` env var or the constructor arg.
_DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


def is_available() -> bool:
    """Return True if the optional transformers + torch deps are installed.

    Used by the orchestrator to decide whether to prefer the encoder NLI
    judge over the LLM judge. Never raises.
    """
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class EncoderNLIJudge:
    """Async NLI judge backed by an encoder-only transformers model.

    Drop-in replacement for the FactualVerifier's ``llm_judge`` callable:
    takes a prompt string (the same ``_NLI_JUDGE_PROMPT`` the LLM judge
    receives), extracts the SOURCE and CLAIM, classifies the pair with the
    encoder model, and returns a JSON string
    ``{"verdict": "...", "supporting_quote": ""}``.

    The model is loaded lazily on first call (so importing this module is
    cheap even when transformers is installed but the judge isn't used).

    Args:
        model_name: HuggingFace model id for an NLI cross-encoder. Defaults
            to ``cross-encoder/nli-deberta-v3-base`` (override via
            ``VIBE_THINKER_NLI_MODEL`` env var).
        device: "cpu", "cuda", "mps", or None (auto-detect). Default "cpu"
            for determinism and to avoid GPU OOM on small machines.
        deterministic: if True, set torch seeds and enable deterministic
            algorithms (CPU only; may raise on GPU if unsupported).
        threshold: confidence threshold below which the verdict is
            downgraded to NEUTRAL (the encoder is uncertain). Default 0.6.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = "cpu",
        deterministic: bool = True,
        threshold: float = 0.6,
    ):
        if not is_available():
            raise ImportError(
                "EncoderNLIJudge requires the optional 'nli' extra. "
                "Install with: pip install \"vibe-thinker[nli]\" "
                "(or: pip install transformers torch). "
                "Without it, the FactualVerifier falls back to the LLM "
                "judge or fail-closed nli_unavailable."
            )
        # Store config; load the model lazily on first __call__ so
        # constructing this object is cheap (no network, no RAM) even when
        # transformers is installed. The orchestrator can construct it
        # speculatively without paying the model-load cost unless it's
        # actually used for a factual verification.
        self._threshold = threshold
        self._device = device or "cpu"
        self._deterministic = deterministic
        self._model_name = model_name or os.environ.get(
            "VIBE_THINKER_NLI_MODEL", _DEFAULT_NLI_MODEL
        )
        self._tokenizer = None
        self._model = None
        self._id2label = None

    def _ensure_loaded(self) -> None:
        """Lazily load the model on first use (network + RAM cost)."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self._deterministic:
            torch.manual_seed(0)
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name
        )
        self._model.eval()
        if self._device != "cpu":
            try:
                self._model.to(self._device)
            except Exception:
                self._device = "cpu"
        self._id2label = self._model.config.id2label

    async def __call__(self, prompt: str) -> str:
        """Classify the (source, claim) pair in ``prompt`` and return JSON.

        Returns a JSON string matching the FactualVerifier's expected judge
        output: ``{"verdict": "ENTAILMENT"|"CONTRADICTION"|"NEUTRAL",
        "supporting_quote": ""}``. The quote is always empty (encoder
        models don't extract spans) — the FactualVerifier treats this as
        an un-cited verdict (score 0.7, below the cache threshold).

        Raises on any inference failure — the FactualVerifier catches and
        returns ``nli_judge_error`` (fail-closed).
        """
        self._ensure_loaded()
        source, claim = self._extract_source_claim(prompt)
        if not source or not claim:
            # Can't classify without both — return NEUTRAL (fail-closed
            # in the verifier: all-NEUTRAL → nli_neutral, score 0.0).
            return json.dumps({"verdict": "NEUTRAL", "supporting_quote": ""})

        import torch

        # Run inference (synchronous — the FactualVerifier calls this via
        # `await self._llm_judge(prompt)`, but the encoder call itself is
        # CPU-bound and fast on a base model; the GIL is held briefly).
        with torch.no_grad():
            inputs = self._tokenizer(
                source, claim, return_tensors="pt", truncation=True,
                max_length=512, padding=True,
            )
            if self._device != "cpu":
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        # Map the model's label order to canonical verdicts.
        verdict, confidence = self._pick_verdict(probs)
        # Downgrade low-confidence verdicts to NEUTRAL (honest uncertainty).
        if confidence < self._threshold:
            verdict = "NEUTRAL"
        return json.dumps({"verdict": verdict, "supporting_quote": ""})

    def _pick_verdict(self, probs) -> tuple:
        """Map model probabilities to (verdict, confidence).

        Cross-encoder NLI models use label names like 'ENTAILMENT',
        'NEUTRAL', 'CONTRADICTION'. We match case-insensitively and fall
        back to the argmax if the labels don't match the expected names.
        """
        idx = int(probs.argmax().item())
        conf = float(probs[idx].item())
        label = str(self._id2label.get(idx, "")).lower()

        if "entail" in label:
            return "ENTAILMENT", conf
        if "contrad" in label or "contradict" in label:
            return "CONTRADICTION", conf
        if "neutral" in label:
            return "NEUTRAL", conf
        # Unknown label scheme — fall back to argmax position heuristics
        # (common order: 0=entailment, 1=neutral, 2=contradiction).
        if idx == 0:
            return "ENTAILMENT", conf
        if idx == 2:
            return "CONTRADICTION", conf
        return "NEUTRAL", conf

    @staticmethod
    def _extract_source_claim(prompt: str) -> tuple:
        """Extract the SOURCE and CLAIM texts from the NLI judge prompt.

        The prompt format is defined by ``_NLI_JUDGE_PROMPT`` in
        factual_verifier.py. We parse it defensively — if the format
        changes, returning empty strings causes a NEUTRAL verdict
        (fail-closed), never a false positive.
        """
        source = ""
        claim = ""
        # The prompt has "SOURCE: {source}\nCLAIM: {claim}\n\nRespond..."
        # Extract each field up to the next label or end.
        s_idx = prompt.find("SOURCE:")
        c_idx = prompt.find("CLAIM:")
        if s_idx != -1:
            end = c_idx if c_idx != -1 else len(prompt)
            source = prompt[s_idx + len("SOURCE:"):end].strip()
        if c_idx != -1:
            # Claim ends at the next blank line or "Respond with".
            rest = prompt[c_idx + len("CLAIM:"):]
            for terminator in ("\n\nRespond", "\n\nOutput", "\n\n", "\n"):
                t_idx = rest.find(terminator)
                if t_idx != -1:
                    claim = rest[:t_idx].strip()
                    break
            else:
                claim = rest.strip()
        return source, claim
