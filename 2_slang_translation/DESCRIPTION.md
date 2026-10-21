# Stage 2 — Slang Translation Inference

Runs translation inference over the benchmark items, across a panel of models
and the five inference conditions. Output feeds Stage 3 directly.

## What it does

For each item × model, `slang_translation.py` runs **five paths**:

| Path | What the model gets | Output field |
|---|---|---|
| `Direct` | sentence only, translate cold | `mt_translation` |
| `DirectWithHint` | + gold term & definition (upper bound) | `mt_translation` |
| `SlangDITIdentify` | term provided; define it, no translation | `mt_slang`, `mt_definition` |
| `DirectContext` | Direct + the item's paragraph as reference | `mt_translation` |
| `SlangDITIdentifyContext` | Identify + that paragraph | `mt_slang`, `mt_definition` |

The 3 regular paths run per `--modalities` (sentence, paragraph); the two
`*Context` paths are sentence-anchored (once per item/model) and skip items
with no `paragraph_in_context`. Per model:
`N × (3 × 2 + 2) = N × 8` calls.

## Input (`../benchmark-data/`)

One CSV per language (`zh.csv`, `ru.csv`, `fa.csv`). Key fields: `item_id`, `term`, `lang`,
`definition_tar_lang` (the gold hint), `sentence_in_context`,
`paragraph_in_context`.

## Output (`output-data/`)

One JSON file **per model** (`<output-stem>-<model-slug>.json`), one record per
(item × model × path × modality × run): the input keys plus `model`, `path`,
`modality`, `mt_slang`, `mt_definition`, `mt_translation`, `raw_response`,
`error`. Resume is keyed on that tuple; saving is atomic. `align_outputs.py`
then intersects all per-model files to the shared segment
set and writes the `-aligned.json` inputs for Stage 3.

## Running

```bash
export OPENAI_API_KEY=...            # + OPENAI_BASE_URL to target a non-OpenAI engine

# Translate. Each model writes its own per-model file into the --output base, so
# several models can run concurrently without colliding; add --run 2/3 for
# repeated runs of the same model.
python3 slang_translation.py \
    --input ../benchmark-data/zh.csv \
    --models <id> --src-lang zh --modalities sentence,paragraph \
    --output output-data/zh_en_translations.json

# Align all per-model files into Stage 3's input set (writes -aligned.json).
python3 align_outputs.py output-data/zh_en_translations-*.json
```
