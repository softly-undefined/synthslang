# Stage 3 — Translation Evaluation

Scores the Stage-2 translations and aggregates them into the benchmark's
results. The full data-flow diagram of `slang_translation_evaluation.py` is
in `FLOW.md`.

## What it does

For every aligned Stage-2 record, scores the applicable metrics —
`gemba_da` / `gemba_da_def` (LLM-judge translation quality, reference-free),
`understanding` + `rouge_l` (the identify paths' definitions vs gold),
`cometkiwi` (QE), plus the reported `gemba_da_def_big` / `_avg3` judge variants,
then aggregates per run into `output-data/<run-name>/`:
`per_item.json`, `leaderboard.json`, `run_variability.json`, `summary.json`.

`metrics.py` and `prompts.py` contain the metric implementations and judge
prompts. Human-annotation data and validation tooling are not included in this
release.

## Input (`input-data/`)

The `-aligned.json` files written by `../2_slang_translation/align_outputs.py`
(one per model × track, intersected to the shared error-free segment set), plus
`--items` pointing at the matching CSV in `../benchmark-data/`.

## Output (`output-data/`)

One directory per run, named `<model>_fake[_<lang>]` (e.g. `gemma3_4b_fake_ru`);
`metric_cache.json` inside each is the resume cache. Generated evaluation output
and paper-figure analysis artifacts are not included in the public repository.

## Running

```bash
export OPENAI_API_KEY=...            # + OPENAI_BASE_URL to target a non-OpenAI engine

python3 slang_translation_evaluation.py \
    --results input-data/<results-aligned.json> \
    --items   ../benchmark-data/<lang>.csv \
    --metrics gemba_da,gemba_da_def,understanding,rouge_l,cometkiwi \
    --judge-model <id> \
    --run-name <run-name> \
    --outdir  output-data/<run-name>
```

`--run-name` names the output directory (convention: `<model>_fake[_<lang>]`);
a completed run resumes from its `metric_cache.json`. `FLOW.md` diagrams the
full data flow.
