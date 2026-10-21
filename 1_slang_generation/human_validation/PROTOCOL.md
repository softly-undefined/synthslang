Created: 2026-06-10

# Human validation protocol

> **Scope:** this folder contains only Stage 1's analyst gate for deciding which
> generated items enter the benchmark.

The **generation gate** (Mode A) is a CSV the reviewer fills in Google Sheets
(`export_review_csv.py` → `ingest_review_csv.py`; run commands in the Stage-1
`DESCRIPTION.md`).

## Mode A — generation validation (the analyst gate) — CSV flow

**What it decides:** whether a synthetic item (term · definition · sentence + gloss ·
paragraph + gloss) ships as a benchmark item. This is the final gate of Stage 1.

`export_review_csv.py` dumps the items to a review CSV; the reviewer fills the
`REVIEW_*` columns in Google Sheets; `ingest_review_csv.py` folds the verdicts back
into `benchmark_items_accepted.json`. **Per item:**

1. `REVIEW_faithful_use` — *Do the two in-context examples faithfully use the term
   in its defined sense?* — yes / no
2. `REVIEW_exists_check` — *Does this slang genuinely not already exist?* — yes /
   unsure / it exists
3. `REVIEW_verdict` — *Approve as a benchmark item?* — accept / reject
4. `REVIEW_note` — one line, **required on reject**.

(The old Streamlit Mode A also had an optional §6 four-dimension 1–5 rubric toggle —
plausibility, translation opacity, contextual recoverability, novelty. If you want
those rubric stats for the paper, add the four columns to the sheet; that path
lived in the retired Streamlit app.)

**Reviewers:** ≥2 per item where possible — each downloads their own filled CSV and
you pass all of them to `ingest_review_csv.py`, which accepts iff **all** reviewers
accept (`--rule majority` to relax) and reports **Cohen's κ** per reviewer pair on
the accept/reject verdict.

**Output chain:** review CSV → `ingest_review_csv.py` → `<items>_accepted.json` →
Stage 2 `--input`.
