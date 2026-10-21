# synthslang

Created by [Eric Bennett](https://softly-undefined.github.io), Summer 2026

The data-generation and evaluation **pipeline** behind the synthslang
slang-translation benchmark. It generates realistic-but-fake slang in Chinese,
Russian, and Farsi, proves each term unseen with a contamination gate, runs
translation inference across a panel of models, and scores the results with
reference-free LLM-judge metrics.

The only released data are the final, human-accepted benchmark items in
[`benchmark-data/`](benchmark-data/); its [description](benchmark-data/DESCRIPTION.md)
defines the slang categories and English gloss fields. Intermediate generations,
model outputs, review workbooks, caches, and analysis artifacts are intentionally
excluded. Human-annotation records and the private metric-validation workflow
reported in the paper are also outside the scope of this release.

## What it measures

How well LLMs translate and understand **synthetic slang** — terms that are
realistic but freshly coined, so no model can have memorised them. That isolates
*in-context* slang comprehension from training-data recall.

## Released pipeline (three stages)

Each stage is a self-contained directory with its own `DESCRIPTION.md` — what it
does, its inputs/outputs, and a **Running** section with the commands.

| Stage | Directory | What it does |
|---|---|---|
| 1 | [`1_slang_generation/`](1_slang_generation/DESCRIPTION.md) | Generate fake slang + definitions; contamination-gate them unseen; add in-context usage; human-validate. |
| 2 | [`2_slang_translation/`](2_slang_translation/DESCRIPTION.md) | Run translation inference over the items across models × five inference conditions. |
| 3 | [`3_slang_translation_evaluation/`](3_slang_translation_evaluation/DESCRIPTION.md) | Score the translations (GEMBA-DA, understanding judge, ROUGE-L, CometKiwi) and aggregate into leaderboards. |

Stages run in order — Stage N consumes Stage N−1's output.

## Inference setup

Every stage that calls a model goes through one place: `inference.py`, a thin
wrapper over the OpenAI SDK. Point it at any OpenAI-compatible engine with two
standard environment variables:

```bash
export OPENAI_API_KEY=...       # your key
export OPENAI_BASE_URL=...       # optional — omit for the official OpenAI API; set it
                                 # to a local vLLM / TGI / LiteLLM / other
                                 # OpenAI-compatible server
```

Model ids are passed through verbatim (`--model` / `--models` / `--judge-model`),
so use whatever id your endpoint serves. See each stage's **Running** section for
the exact commands.

## Requirements

Python 3.10+. The core dependency is the `openai` SDK. Individual stages add
`sentence-transformers` / `torch` (local embeddings for the contamination gate and
surface metrics), a web-search client (existence checks), and CometKiwi (optional,
GPU) for the QE metric. Install what a stage needs as you run it.

## Layout

```
inference.py                                  # shared OpenAI-compatible inference seam
benchmark-data/                               # final accepted CSVs (zh, ru, fa)
1_slang_generation/                           # Stage 1 — generation + contamination gate
2_slang_translation/                          # Stage 2 — translation inference
3_slang_translation_evaluation/               # Stage 3 — scoring + leaderboards
```

Generated inputs and outputs are written below each stage when its scripts run;
they are not part of the public repository.
