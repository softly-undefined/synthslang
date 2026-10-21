# Stage 3 metrics: GEMBA-DA, a binary understanding judge, CometKiwi QE,
# ROUGE-L, and optional BERTScore/embedding similarity. Heavy dependencies load
# lazily; unavailable metrics report a skip reason instead of aborting the run.

from __future__ import annotations

import sys
from pathlib import Path

# inference.py lives two levels up; prompts.py is in this directory.
SYNTHSLANG_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SYNTHSLANG_DIR))

import inference  # noqa: E402
import prompts  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# GEMBA-DA  (reference-free direct assessment, 0–100)
# ─────────────────────────────────────────────────────────────────────────────

def gemba_da(client, model, source, mt_translation,
             src_lang="zh", tgt_lang="en",
             term: str | None = None,
             definition: str | None = None,
             temperature: float | None = 0.0) -> dict:
    """Run GEMBA-DA, optionally conditioning the judge on the slang definition."""
    if term is not None and definition is not None:
        prompt = prompts.build_gemba_da_def_prompt(
            source, mt_translation, term, definition, src_lang, tgt_lang)
    else:
        prompt = prompts.build_gemba_da_prompt(source, mt_translation, src_lang, tgt_lang)
    raw = inference.chat(client, model, prompt, temperature=temperature)

    score: float | None = None
    try:
        data = inference.extract_json(raw)
        if isinstance(data, dict):
            v = data.get("score")
            if isinstance(v, (int, float)):
                score = float(v)
    except ValueError:
        # model returned a bare number — use the validation helper
        parsed = prompts.validate_number(raw.strip())
        if parsed is not None:
            score = float(parsed)

    return {"gemba_da_score": score, "raw": raw}


# ─────────────────────────────────────────────────────────────────────────────
# Binary understanding judge for the two identify paths.
# ─────────────────────────────────────────────────────────────────────────────

def understanding_judge(client, model, term, reference_definition, mt_understanding) -> dict:
    prompt = prompts.build_understanding_judge_prompt(
        term, reference_definition, mt_understanding)
    raw = inference.chat(client, model, prompt)
    data = inference.extract_json(raw)
    score = data.get("score") if isinstance(data, dict) else None
    reason = data.get("reason", "") if isinstance(data, dict) else ""
    return {"understanding_score": score, "reason": reason, "raw": raw}


# ─────────────────────────────────────────────────────────────────────────────
# CometKiwi  (reference-free QE: source + MT, no gold translation)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_COMETKIWI = "Unbabel/wmt22-cometkiwi-da"


class CometKiwiQE:
    """Lazy CometKiwi wrapper that records a skip reason when unavailable."""

    def __init__(self, model_name: str = DEFAULT_COMETKIWI):
        self.model_name = model_name
        self.model = None
        self.gpus = 0
        self.skip_reason: str | None = None
        try:
            from comet import download_model, load_from_checkpoint
            try:
                import torch
                self.gpus = 1 if torch.cuda.is_available() else 0
            except ImportError:
                self.gpus = 0
            self.model = load_from_checkpoint(download_model(model_name))
        except Exception as err:  # missing package, no HF auth, no checkpoint
            self.skip_reason = f"{type(err).__name__}: {err}"

    @property
    def available(self) -> bool:
        return self.model is not None

    def score(self, pairs: list[tuple[str, str]], batch_size: int = 8) -> list[float]:
        """Score (source, translation) pairs; callers must check `available` first."""
        data = [{"src": s, "mt": m} for s, m in pairs]
        out = self.model.predict(data, batch_size=batch_size, gpus=self.gpus)
        if isinstance(out, dict):
            return list(out["scores"])
        if hasattr(out, "scores"):
            return list(out.scores)
        return list(out)


# ─────────────────────────────────────────────────────────────────────────────
# ROUGE-L  (pure-python; lower-bound surface baseline for the understanding judge:
# the model's slang definition vs the gold English definition. Both sides are
# English, so whitespace tokens are fine.)
# ─────────────────────────────────────────────────────────────────────────────

def _lcs_length(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for token in a:
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            cur[j] = prev[j - 1] + 1 if token == b[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[len(b)]


def rouge_l_f(hypothesis: str, reference: str) -> float:
    """Token-level ROUGE-L F-measure."""
    hyp, ref = hypothesis.split(), reference.split()
    if not hyp or not ref:
        return 0.0
    lcs = _lcs_length(hyp, ref)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(hyp), lcs / len(ref)
    return 2 * prec * rec / (prec + rec)


# ─────────────────────────────────────────────────────────────────────────────
# Optional extras (off by default): BERTScore and embedding similarity.
# ─────────────────────────────────────────────────────────────────────────────

def bertscore_f(hypotheses: list[str], references: list[str], lang: str = "en"):
    """BERTScore F1 per pair — model definition vs gold definition, same
    pairing as rouge_l (needs bert_score + torch). Returns a list, or None with no
    crash if the package is absent."""
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        return None
    _, _, f1 = bert_score_fn(hypotheses, references, lang=lang, verbose=False)
    return [float(x) for x in f1]


class EmbeddingSim:
    """Cosine similarity between two texts (needs sentence-transformers). A cheap
    correlation check against the understanding judge, not a primary metric.
    Disabled with a skip_reason when the package is unavailable."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = None
        self.skip_reason: str | None = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except Exception as err:
            self.skip_reason = f"{type(err).__name__}: {err}"

    @property
    def available(self) -> bool:
        return self.model is not None

    def similarity(self, text_a: str, text_b: str) -> float:
        # normalize_embeddings=True → dot product is cosine similarity
        emb = self.model.encode([text_a, text_b], normalize_embeddings=True)
        return float((emb[0] * emb[1]).sum())
