# Stage 1 — Synthetic Slang Generation

Generates realistic-but-fake slang, guarantees it's unseen via a three-part
contamination gate, and produces the in-context usage that makes each item a
benchmark record.

## What it does

Three scripts, run in order per language × slang type:

1. **`slang-generation.py`** — LLM emits `(term, definition_tar_lang)`
   candidates (definitions English-only; no sentences yet). Prompts per
   language/type live in `prompts.py` (zh: 4 types; ru: 4 types; fa: 3 types;
   Arabic prompts exist in the code but that track is not part of this release).
2. **`contamination-gate.py`** — runs on term + definition only, three
   sub-gates cheapest-first:
   - *near-dup* — embedding near-duplicate removal within the batch (local, free)
   - *guessability* (Gate 2) — term shown with zero context; FAIL if the
     meaning is decodable from form alone (guesses embedded vs definition,
     cosine ≥ 0.55)
   - *existence* (Gate 1) — web search (`existence_search.py`, DDG) + a
     consensus sense judge (local embedding + gpt-5.4); FAIL at ≥3 independent
     in-sense sources; judge disagreements escalate to human review.
   Support tooling: `review_gate.py` re-tunes cutoffs from saved evidence with
   no API calls.
3. **`in-context-generation.py`** — for gate survivors only: a
   sentence-in-context and paragraph-in-context (each with an English gloss),
   without paraphrasing the definition.

## Input

None — generation is prompt-driven. Everything configurable is a flag
(`--lang`, `--type`, `--model`, `--n`).

## Output (`output-data/<lang>_400/`)

Generated per `<lang>_<type>` stem (these artifacts are not committed): raw candidates (`.json` + `.runlog.jsonl`), gate
outputs (`_pass` / `_rejected` / `_escalate` / `_gate_cache`), final benchmark
items (`_items.json`: term · definition · sentence · paragraph), and the
human-review CSV (`_items_review.csv`, via
`human_validation/export_review_csv.py`). The final human-accepted release is
stored by language in `../benchmark-data/`.

## Running

Point at any OpenAI-compatible engine (see `../inference.py`):

```bash
export OPENAI_API_KEY=...            # + OPENAI_BASE_URL to target a non-OpenAI engine
```

Per language × slang type, run the three scripts in order (all resumable). Add
`--no-temperature` if `--model` is a reasoning model that rejects a set temperature:

```bash
LANG=fa; T=light_verb; OUT=output-data/${LANG}_400
# 1. generate raw (term, definition) candidates
python slang-generation.py --lang $LANG --type $T --model <id> --n 400 \
    --output $OUT/${LANG}_${T}.json \
# 2. contamination gate: near-dup -> guessability -> existence (consensus judge)
&& python contamination-gate.py --input $OUT/${LANG}_${T}.json \
    --gates neardup,guess,exist --sense-judge consensus --judge-model <id> \
    --guess-sim-threshold 0.55 --snippet-sim-threshold 0.5 --in-sense-threshold 3 \
# 3. in-context sentence + paragraph for the gate survivors
&& python in-context-generation.py --input $OUT/${LANG}_${T}_pass.json \
    --model <id> --output $OUT/${LANG}_${T}_items.json
```

Slang types per language — zh: `semantic_shift` `morph_blend` `affixation`
`sentence_compression`; ru: `semantic_shift` `morph_blend` `suffixation`
`foreign_verb`; fa: `light_verb` `affixation` `semantic_shift`.

Then the human-validation gate (fully local, no key) turns `*_items.json` into
the accepted items Stage 2 consumes:

```bash
cd human_validation
python3 export_review_csv.py --items ../output-data/<lang>_400/<stem>_items.json
# reviewer fills the CSV (accept/reject per item), then:
python3 ingest_review_csv.py --items ../output-data/<lang>_400/<stem>_items.json \
    --reviewed <filled-review.csv>          # -> <stem>_items_accepted.json
```

Gate re-tuning from saved evidence (no API calls): `review_gate.py`.
