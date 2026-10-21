# Stage 3 scoring and aggregation. Understanding paths use definition-matching
# metrics; translation paths use reference-free GEMBA-DA and optional CometKiwi.
# Outputs include per-item scores, summaries, variability, and sliced leaderboards.

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

SYNTHSLANG_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SYNTHSLANG_DIR))

import inference  # noqa: E402
import metrics  # noqa: E402  (same directory)

try:
    from twoqdm import tqdm
except ImportError:
    def tqdm(iterable, **_):
        return iterable

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTDIR = BASE_DIR / "output-data"

ALL_METRICS = ("gemba_da", "gemba_da_def", "gemba_da_def_big", "gemba_da_def_avg3",
               "understanding", "cometkiwi", "rouge_l", "bertscore", "embedding")
DEFAULT_METRICS = "gemba_da,gemba_da_def,understanding,cometkiwi,rouge_l"

# What each metric requires; the scorer skips (and counts why) when a field is absent.
# The understanding family (judge + the embedding/rouge_l/bertscore surface baselines)
# all score the model's slang definition against the gold DEFINITION, which every
# item has — so no metric depends on a gold reference translation.
DEFINITION_METRICS = {"understanding", "embedding", "rouge_l", "bertscore"}    # need mt_definition (SlangDITIdentify / SlangDITIdentifyContext)
NEEDS_REFERENCE_DEFINITION = {"gemba_da_def", "gemba_da_def_big", "gemba_da_def_avg3",
                              "understanding", "embedding",
                              "rouge_l", "bertscore"}                          # gold meaning anchor

# gemba_da_def_avg3: same judge as gemba_da_def, sampled this many times at the
# backend's default temperature and averaged (greedy repeats would be identical).
AVG_SAMPLES = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Stage 2 translations (GEMBA-DA, CometKiwi QE, "
        "understanding judge, ROUGE-L) + real-vs-fake control and leaderboard.")
    p.add_argument("--results", type=Path, required=True, help="Stage 2 results JSON")
    p.add_argument("--items", type=Path, default=None,
                   help="benchmark-items CSV or JSON — supplies reference definition/gloss "
                        "(without it, only reference-free metrics like CometKiwi run)")
    p.add_argument("--metrics", default=DEFAULT_METRICS,
                   help=f"comma list from {','.join(ALL_METRICS)}")
    p.add_argument("--judge-model", default="gpt-5.4",
                   help="model id for the LLM-judge metrics (gemba, understanding); also the "
                        "judge sampled by gemba_da_def_avg3. The paper used claude-sonnet-4-6.")
    p.add_argument("--big-judge-model", default="gpt-5.4",
                   help="stronger judge for the gemba_da_def_big metric")
    p.add_argument("--paths", default=None,
                   help="comma list — score only these inference paths (e.g. Direct,DirectWithHint)")
    p.add_argument("--modalities", default=None,
                   help="comma list — score only these modalities (sentence,paragraph)")
    p.add_argument("--runs", default=None,
                   help="comma list of run numbers to keep (e.g. 1). NOTE: filters shrink the "
                        "whole run (per_item.json etc.), so point a filtered run at its OWN "
                        "--outdir rather than an existing full run's")
    p.add_argument("--run-name", default=None)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return p.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else [data]


def load_items(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    return load_json(path)


def result_key(r: dict[str, Any]) -> str:
    # `run` is part of the key so repeated Stage-2 runs of the same model are scored
    # and cached separately (they pool into the same leaderboard cells, raising n);
    # dedupe still collapses retries WITHIN a run. Legacy records with no
    # `run` field default to 1.
    return f"{r['item_id']}|{r['model']}|{r['path']}|{r['modality']}|{r.get('run', 1)}"


def mt_definition_of(r: dict[str, Any]) -> str | None:
    # The model's own definition of the slang (understanding-only paths). Canonical
    # field is mt_definition; older Stage-2 outputs spelled it mt_explanation.
    return r.get("mt_definition") or r.get("mt_explanation")


def dedupe_results(results: list[dict]) -> list[dict]:
    """One record per (item,model,path,modality); a success replaces an errored dup."""
    best: dict[str, dict] = {}
    for r in results:
        k = result_key(r)
        if r.get("mt_translation") is not None and not r.get("error"):
            best[k] = r
        else:
            best.setdefault(k, r)
    return list(best.values())


def index_items(items: list[dict]) -> dict[str, dict]:
    idx = {}
    for it in items:
        iid = str(it.get("id") or it.get("item_id") or "")
        if iid:
            idx[iid] = it
    return idx


def reference_definition_for(result: dict, items: dict[str, dict]) -> str | None:
    """Return the gold slang definition for a result, if available."""
    it = items.get(result["item_id"], {})
    # Canonical field is definition_tar_lang; accept the legacy spelling too.
    return it.get("definition_tar_lang") or it.get("explanation_tar_lang")


# ── resume cache ──────────────────────────────────────────────────────────────

def load_cache(path: Path) -> dict[tuple[str, str], dict]:
    """Load cached metric scores from a JSON array."""
    cache: dict[tuple[str, str], dict] = {}
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            for rec in json.loads(text):
                cache[(rec["result_key"], rec["metric"])] = rec
    return cache


def save_cache(cache: dict[tuple[str, str], dict], path: Path) -> None:
    path.write_text(
        json.dumps(list(cache.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def store(cache: dict, rk: str, metric: str, value, extra=None) -> None:
    rec = {"result_key": rk, "metric": metric, "value": value}
    if extra:
        rec.update(extra)
    cache[(rk, metric)] = rec


# ── scoring ──────────────────────────────────────────────────────────────────

def score_per_result(results, items, selected, cache, cache_path, judge_model,
                     big_judge_model=None):
    """Score records with lazy clients and save the cache after each result."""
    per_result = [m for m in selected if m not in ("cometkiwi", "bertscore")]
    skips: dict[str, int] = defaultdict(int)
    if not per_result:
        return skips

    client = None
    embedder = None

    def ensure_client():
        nonlocal client
        if client is None:
            client = inference.make_client()
        return client

    def ensure_embedder():
        nonlocal embedder
        if embedder is None:
            embedder = metrics.EmbeddingSim()
            if not embedder.available:
                print(f"[embedding] skipped — {embedder.skip_reason}")
        return embedder

    for r in tqdm(results, desc="scoring"):
        rk = result_key(r)
        ref_def = reference_definition_for(r, items)
        mt = r.get("mt_translation")
        dirty = False
        for m in per_result:
            if (rk, m) in cache:
                continue
            if m in DEFINITION_METRICS and not mt_definition_of(r):
                continue
            if m in NEEDS_REFERENCE_DEFINITION and not ref_def:
                skips[m] += 1; continue
            try:
                if m == "gemba_da":
                    if not mt: skips[m] += 1; continue
                    out = metrics.gemba_da(ensure_client(), judge_model, r["source_text"], mt,
                                           src_lang=r["lang"], tgt_lang=r.get("tgt_lang", "en"))
                    store(cache, rk, m, out["gemba_da_score"])
                elif m == "gemba_da_def":
                    if not mt: skips[m] += 1; continue
                    out = metrics.gemba_da(ensure_client(), judge_model, r["source_text"], mt,
                                           src_lang=r["lang"], tgt_lang=r.get("tgt_lang", "en"),
                                           term=r["term"], definition=ref_def)
                    store(cache, rk, m, out["gemba_da_score"])
                elif m == "gemba_da_def_big":
                    if not mt: skips[m] += 1; continue
                    out = metrics.gemba_da(ensure_client(), big_judge_model, r["source_text"], mt,
                                           src_lang=r["lang"], tgt_lang=r.get("tgt_lang", "en"),
                                           term=r["term"], definition=ref_def)
                    store(cache, rk, m, out["gemba_da_score"],
                          {"judge_model": big_judge_model})
                elif m == "gemba_da_def_avg3":
                    if not mt: skips[m] += 1; continue
                    samples = []
                    for _ in range(AVG_SAMPLES):
                        out = metrics.gemba_da(ensure_client(), judge_model, r["source_text"], mt,
                                               src_lang=r["lang"], tgt_lang=r.get("tgt_lang", "en"),
                                               term=r["term"], definition=ref_def,
                                               temperature=None)
                        samples.append(out["gemba_da_score"])
                    valid = [s for s in samples if s is not None]
                    value = round(mean(valid), 4) if valid else None
                    store(cache, rk, m, value, {"samples": samples})
                elif m == "understanding":
                    und = mt_definition_of(r)
                    if not und: skips[m] += 1; continue
                    out = metrics.understanding_judge(ensure_client(), judge_model,
                                                      r["term"], ref_def, und)
                    store(cache, rk, m, out["understanding_score"], {"reason": out["reason"]})
                elif m == "rouge_l":
                    # surface lower-bound for the understanding judge: lexical overlap
                    # of the model's slang definition against the gold definition
                    # (mt_definition guaranteed by the DEFINITION_METRICS guard above)
                    store(cache, rk, m, metrics.rouge_l_f(mt_definition_of(r), ref_def))
                elif m == "embedding":
                    und = mt_definition_of(r)
                    emb = ensure_embedder()
                    if not und or not emb.available:
                        skips[m] += 1; continue
                    store(cache, rk, m, emb.similarity(und, ref_def))
                dirty = True
            except Exception as err:  # one bad call shouldn't sink the run
                store(cache, rk, m, None, {"error": str(err)})
                dirty = True
        if dirty:
            save_cache(cache, cache_path)
    return skips


def score_cometkiwi(results, cache, cache_path) -> str:
    """Batch reference-free QE. Loads comet only if there's uncached work."""
    jobs = [r for r in results if (result_key(r), "cometkiwi") not in cache
            and r.get("mt_translation") and r.get("source_text")]
    if not jobs:
        return "nothing to score"
    qe = metrics.CometKiwiQE()
    if not qe.available:
        print(f"[cometkiwi] skipped {len(jobs)} segments — {qe.skip_reason}")
        return f"skipped — {qe.skip_reason}"
    print(f"[cometkiwi] scoring {len(jobs)} segments on {'GPU' if qe.gpus else 'CPU'} ...", flush=True)
    scores = qe.score([(r["source_text"], r["mt_translation"]) for r in jobs])
    for r, s in zip(jobs, scores):
        store(cache, result_key(r), "cometkiwi", float(s))
    save_cache(cache, cache_path)
    return f"scored {len(jobs)}"


def score_bertscore(results, items, cache, cache_path) -> str:
    """Surface baseline for the understanding judge (SlangDITIdentify/SlangDITIdentifyContext): BERTScore
    between the model's definition and the gold definition — same pairing as rouge_l."""
    jobs = []
    for r in results:
        if (result_key(r), "bertscore") in cache:
            continue
        ref_def = reference_definition_for(r, items)
        und = mt_definition_of(r)
        if und and ref_def:
            jobs.append((r, und, ref_def))
    if not jobs:
        return "nothing to score"
    f1 = metrics.bertscore_f([h for _, h, _ in jobs], [ref for _, _, ref in jobs])
    if f1 is None:
        print("[bertscore] skipped — bert_score not installed")
        return "skipped — bert_score not installed"
    for (r, _, _), s in zip(jobs, f1):
        store(cache, result_key(r), "bertscore", float(s))
    save_cache(cache, cache_path)
    return f"scored {len(jobs)}"


# ── aggregation ───────────────────────────────────────────────────────────────

def attach_metrics(results, cache) -> list[dict]:
    rows = []
    for r in results:
        rk = result_key(r)
        vals = {m: cache[(rk, m)]["value"] for m in ALL_METRICS if (rk, m) in cache}
        rows.append({**r, "metrics": vals})
    return rows


def leaderboard(rows, selected) -> list[dict]:
    """Mean of each metric, sliced by model × path × modality × lang × type."""
    groups: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    counts: dict[tuple, int] = defaultdict(int)
    for row in rows:
        key = (row["model"], row["path"], row["modality"], row["lang"], row["type"])
        counts[key] += 1
        for m, v in row["metrics"].items():
            if v is not None:
                groups[key][m].append(v)
    out = []
    for key in sorted(groups, key=str):
        model, path, mod, lang, typ = key
        entry = {"model": model, "path": path, "modality": mod, "lang": lang,
                 "type": typ, "n": counts[key]}
        for m in selected:
            if groups[key].get(m):
                entry[f"{m}_mean"] = round(mean(groups[key][m]), 4)
        out.append(entry)
    return out


def run_variability(rows, selected) -> list[dict]:
    """Summarize aggregate and within-item variation across repeated model runs."""
    # (cell) -> metric -> run -> item_id -> value
    cells: dict[tuple, dict[str, dict[int, dict[str, float]]]] = \
        defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for row in rows:
        cell = (row["model"], row["path"], row["modality"], row["lang"],
                row["type"])
        run = row.get("run", 1)
        for m, v in row["metrics"].items():
            if v is not None:
                cells[cell][m][run][row["item_id"]] = v

    out = []
    for cell in sorted(cells, key=str):
        model, path, mod, lang, typ = cell
        for m in selected:
            by_run = cells[cell].get(m)
            if not by_run or len(by_run) < 2:  # need >=2 runs for any variability
                continue
            runs = sorted(by_run)
            per_run_mean = {str(r): round(mean(by_run[r].values()), 4) for r in runs}
            run_means = [mean(by_run[r].values()) for r in runs]

            item_ids = set().union(*(set(by_run[r]) for r in runs))
            item_stds, item_ranges = [], []
            for iid in item_ids:
                vals = [by_run[r][iid] for r in runs if iid in by_run[r]]
                if len(vals) >= 2:
                    item_stds.append(stdev(vals))
                    item_ranges.append(max(vals) - min(vals))

            entry = {"model": model, "path": path, "modality": mod, "lang": lang,
                     "type": typ, "metric": m, "n_runs": len(runs),
                     "per_run_mean": per_run_mean,
                     "across_run_mean_std": round(stdev(run_means), 4),
                     "n_items_multi_run": len(item_stds)}
            if item_stds:
                entry["within_item_std_mean"] = round(mean(item_stds), 4)
                entry["within_item_std_max"] = round(max(item_stds), 4)
                entry["within_item_range_mean"] = round(mean(item_ranges), 4)
            out.append(entry)
    return out


# The understanding-only paths (SlangDITIdentify, SlangDITIdentifyContext) emit no
# translation, so they can't hit the GEMBA ceiling — the "too easy" check only
# considers the paths that actually translate.
TRANSLATING_PATHS = {"Direct", "DirectWithHint", "DirectContext"}


def too_easy_flags(rows) -> list[dict]:
    """GEMBA-DA-def at ceiling (100) on EVERY translating path scored for the same
    item/model/modality (≥2 paths required) — too easy / possibly contaminated.
    If even direct translation nails it, context inference earned nothing."""
    by: dict[tuple, dict[str, float]] = defaultdict(dict)
    for row in rows:
        g = row["metrics"].get("gemba_da_def")
        if g is not None and row["path"] in TRANSLATING_PATHS:
            by[(row["item_id"], row["model"], row["modality"])][row["path"]] = g
    return [{"item_id": iid, "model": model, "modality": mod,
             "paths_at_ceiling": sorted(p),
             "note": "GEMBA-DA-def ceiling on all translating paths — too easy / possibly contaminated"}
            for (iid, model, mod), p in by.items()
            if len(p) >= 2 and all(v == 100 for v in p.values())]


def main() -> None:
    args = parse_args()
    selected = [m.strip() for m in args.metrics.split(",") if m.strip()]
    bad = [m for m in selected if m not in ALL_METRICS]
    if bad:
        raise SystemExit(f"--metrics: unknown {bad}; allowed: {', '.join(ALL_METRICS)}")

    results = dedupe_results(load_json(args.results))
    if args.paths:
        keep = {p.strip() for p in args.paths.split(",")}
        results = [r for r in results if r["path"] in keep]
    if args.modalities:
        keep = {m.strip() for m in args.modalities.split(",")}
        results = [r for r in results if r["modality"] in keep]
    if args.runs:
        keep = {int(x) for x in args.runs.split(",")}
        results = [r for r in results if r.get("run", 1) in keep]
    if args.paths or args.modalities or args.runs:
        print(f"filters (paths={args.paths} modalities={args.modalities} runs={args.runs}) "
              f"-> {len(results)} results")
    items = index_items(load_items(args.items)) if args.items else {}
    if not items and NEEDS_REFERENCE_DEFINITION & set(selected):
        print("note: no --items given; reference-based metrics will be skipped for all rows.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    cache_path = args.outdir / "metric_cache.json"
    cache = load_cache(cache_path)
    print(f"{len(results)} results, {len(items)} items, metrics={selected}, "
          f"{len(cache)} cached scores", flush=True)

    skips = score_per_result(results, items, selected, cache, cache_path,
                             args.judge_model, args.big_judge_model)
    cometkiwi_status = score_cometkiwi(results, cache, cache_path) if "cometkiwi" in selected else "not selected"
    bertscore_status = score_bertscore(results, items, cache, cache_path) if "bertscore" in selected else "not selected"

    rows = attach_metrics(results, cache)
    lb = leaderboard(rows, selected)
    runvar = run_variability(rows, selected)
    flags = too_easy_flags(rows)

    value_counts = {m: sum(1 for (_, mm), rec in cache.items()
                           if mm == m and rec.get("value") is not None) for m in selected}
    summary = {
        "run_name": args.run_name or args.outdir.name,
        "results_deduped": len(results),
        "items_loaded": len(items),
        "metrics_selected": selected,
        "metric_value_counts": value_counts,
        "metrics_skipped_missing_refs": dict(skips),
        "cometkiwi_status": cometkiwi_status,
        "bertscore_status": bertscore_status,
        "too_easy_flags": flags,
        "leaderboard_rows": len(lb),
        "run_variability_rows": len(runvar),
    }

    def dump(name, obj):
        path = args.outdir / name
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    dump("per_item.json", rows)
    dump("leaderboard.json", lb)
    dump("run_variability.json", runvar)
    dump("summary.json", summary)
    print(f"\nscored {len(rows)} results | {len(flags)} too-easy flags | "
          f"{len(runvar)} run-variability rows | cometkiwi: {cometkiwi_status}")
    print(f"outputs in {args.outdir}/ (per_item.json, leaderboard.json, "
          f"run_variability.json, summary.json)")


if __name__ == "__main__":
    main()
