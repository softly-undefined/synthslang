# Stage 3 evaluation — what `slang_translation_evaluation.py` does

```text
  --results  (Stage 2 translations)          --items  (benchmark items w/ refs)
        │                                            │
        ▼                                            ▼
  dedupe_results()                            index_items()
  one record per (item,model,path,modality)   item_id → reference fields
        │                                            │
        └─────────────────────┬──────────────────────┘
                              ▼
                   load_cache()  ── resume: skip (result_key, metric) pairs
                              │   already scored in output-data/<run>/metric_cache.json
                              ▼
        ┌─────────────── per-record scoring ───────────────┐
        │  for each result × each selected metric:          │
        │                                                   │
        │   gemba_da        source + MT          → 0–100    │  (needs mt_translation)
        │   gemba_da_def    + term + definition  → 0–100    │  (+ reference definition)
        │   understanding   gold def vs model's  → 0/1      │  UNDERSTANDING PATHS ONLY (needs mt_definition)
        │                   definition                      │
        │   rouge_l         model definition     → F        │  UNDERSTANDING PATHS ONLY (needs gold definition)
        │                   vs gold definition                │
        │   embedding*      def vs definition    → cosine   │  UNDERSTANDING PATHS ONLY  (* off by default)
        │                                                   │
        │   missing field → skip + count why (not scored)   │
        │   one bad call  → store None+error, run continues  │
        │   save cache after every record (crash-safe)      │
        └───────────────────────┬───────────────────────────┘
                                ▼
        cometkiwi*  + bertscore*   batch QE/surface metrics
        (lazy heavy deps; skip with a reason if unavailable)   (* opt-in)
                                │
                                ▼
                       attach_metrics()  → rows = results + their scores
                                │
        ┌───────────────────────┴─────────────────────┐
        ▼                                             ▼
  leaderboard()                                 too_easy_flags()
  mean per slice:                               gemba_da_def==100
  model×path×modality                           on ALL translating
  ×lang×type                                    paths (≥2) →
                                                contamination suspect
        │                                             │
        └───────────────────────┬─────────────────────┘
                                ▼
   write output-data/<run>/ :  per_item.json · leaderboard.json
                       summary.json
```

**Note:** the understanding-only paths (`SlangDITIdentify`,
`SlangDITIdentifyContext`) carry no translation by design, so they only feed
`understanding`; their rows appear under `metrics_skipped_missing_refs` for the
translation metrics.
