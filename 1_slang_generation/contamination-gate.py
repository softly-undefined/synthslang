# Stage 1 contamination gate: near-duplicate definitions, bare-term guessability,
# then web existence. Evidence is cached for resume and threshold tuning. Survivors,
# failures, and human-review escalations are written separately.

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SYNTHSLANG_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SYNTHSLANG_DIR))

import inference  # noqa: E402  (after sys.path insert)
import prompts  # noqa: E402
import existence_search  # noqa: E402  (same directory)

try:
    from twoqdm import tqdm
except ImportError:
    def tqdm(iterable, **_):
        return iterable

BASE_DIR = Path(__file__).resolve().parent

GATES = ("neardup", "guess", "exist")

# A compact multilingual sentence model — handles Chinese + English, mean-pooled.
# Run locally via transformers; no remote model or OPENAI_API_KEY needed.
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Near-duplicates must exceed both mean + K*std and an absolute floor.
DEFAULT_K = 3.0        # std multiplier for the relative (outlier) component
DEFAULT_FLOOR = 0.5    # absolute floor on the centered-cosine scale

# Gate 2 compares repeated bare-term guesses with the intended definition.
DEFAULT_GUESS_RUNS = 3
# Temperature is omitted by default for reasoning-model compatibility.
DEFAULT_GUESS_TEMPERATURE = None
DEFAULT_GUESS_SIM = 0.6           # FAIL above this avg cosine — UNCALIBRATED placeholder;
                                  # tune empirically against a labeled sample

# Gate 1's default cross-lingual embedding threshold requires empirical tuning.
DEFAULT_SNIPPET_SIM = 0.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Contamination gate: near-dup (local embeddings) → Gate 2 "
        "LLM guessability → Gate 1 existence search, cheapest first. PASS = passes "
        "all selected gates (--gates). `--gates neardup` alone runs the standalone "
        "near-dup mode with its original _dedup.json outputs and --annotate-only."
    )
    p.add_argument("--input", type=Path, required=True,
                   help="raw candidates JSON from slang-generation.py")
    p.add_argument("--gates", default="neardup,guess,exist",
                   help="comma subset of neardup,guess,exist (default all three; "
                        "run order is always neardup → guess → exist)")
    p.add_argument("--output", type=Path, default=None,
                   help="PASS JSON, feeds in-context-generation.py (default: "
                        "<input-stem>_pass.json; in neardup-only mode the legacy "
                        "<input-stem>_dedup.json)")
    p.add_argument("--rejected", type=Path, default=None,
                   help="FAIL JSON with gate_failed + gate_reason per record "
                        "(default: <input-stem>_rejected.json)")
    p.add_argument("--cache", type=Path, default=None,
                   help="API-call cache for resume, per (term, sub-gate, model) "
                        "(default: <input-stem>_gate_cache.json)")
    # gate 2 — guessability by embedding similarity
    p.add_argument("--guess-model", default=None,
                   help="model id for the zero-context guess (default: each record's "
                        "own generation model, record['model']). The guess is embedded "
                        "and compared to the real definition — no judge model")
    p.add_argument("--guess-runs", type=int, default=DEFAULT_GUESS_RUNS,
                   help=f"how many sampled guesses per term (default {DEFAULT_GUESS_RUNS})")
    p.add_argument("--guess-temperature", type=float, default=DEFAULT_GUESS_TEMPERATURE,
                   help="sampling temperature for the guesses. Omitted by default — the "
                        "gpt-5.x guesser rejects an explicit temperature and varies on its "
                        "own; set a value only for a non-reasoning guess model")
    p.add_argument("--guess-sim-threshold", default=str(DEFAULT_GUESS_SIM),
                   help=f"FAIL gate 2 when the avg cosine(def, guess) ≥ this. A bare "
                        f"number applies to every language; 'zh=0.55,ru=0.65' sets "
                        f"per-language cutoffs (zh/ru leak differently). Default "
                        f"{DEFAULT_GUESS_SIM} — UNCALIBRATED; tune with review_gate.py")
    # gate 1 — existence search
    p.add_argument("--sense-judge", choices=("embedding", "llm", "consensus"),
                   default="embedding",
                   help="how gate 1 decides if a snippet uses the assigned sense: "
                        "'embedding' (default) = local cosine(def, snippet), free/keyless, "
                        "no collision detection; 'llm' = gpt-5.x judge + form-collision flags; "
                        "'consensus' = run BOTH off one retrieval — drop only when both flag "
                        "existence, escalate embedding-keeps/llm-flags to human review")
    p.add_argument("--snippet-sim-threshold", type=float, default=DEFAULT_SNIPPET_SIM,
                   help=f"(embedding sense judge) a snippet is in-sense when cosine(def, "
                        f"snippet) ≥ this (default {DEFAULT_SNIPPET_SIM} — UNCALIBRATED, "
                        f"cross-lingual; tune empirically)")
    p.add_argument("--judge-model", default="gpt-5.4-mini",
                   help="(--sense-judge llm only) model id that adjudicates gate-1 snippets")
    p.add_argument("--retriever", choices=tuple(existence_search.RETRIEVERS),
                   default="ddg",
                   help="gate-1 retriever (pluggable; 'ddg' is free/keyless, "
                        "'brave' needs BRAVE_API_KEY)")
    p.add_argument("--search-count", type=int, default=10,
                   help="results per gate-1 query (default 10)")
    p.add_argument("--in-sense-threshold", type=int, default=3,
                   help="FAIL gate 1 at this many independent in-sense sources "
                        "(default 3 — the conservative end of the 3–5 range)")
    # near-dup sub-gate
    p.add_argument("--annotate-only", action="store_true",
                   help="(neardup-only mode) do not remove; keep every record and tag "
                        "near-dups with near_dup_flag + matched term + similarity")
    p.add_argument("--field", default="definition_tar_lang",
                   help="record field to embed (default: definition_tar_lang — the "
                        "meaning; that's what 'duplicate concept' lives in)")
    p.add_argument("--model", default=DEFAULT_EMBED_MODEL,
                   help=f"local HF embedding model (default: {DEFAULT_EMBED_MODEL})")
    p.add_argument("--center", action=argparse.BooleanOptionalAction, default=True,
                   help="mean-center embeddings before comparing (--no-center to disable). "
                        "Subtracts the batch centroid so the shared theme is removed and "
                        "genuine concept-twins separate from merely-related items.")
    p.add_argument("--k", type=float, default=DEFAULT_K,
                   help=f"std multiplier for the relative cutoff mean+K·std (default {DEFAULT_K})")
    p.add_argument("--floor", type=float, default=DEFAULT_FLOOR,
                   help=f"absolute similarity floor; a diverse batch never clears it, so "
                        f"nothing is removed (default {DEFAULT_FLOOR}, centered-cosine scale)")
    p.add_argument("--top-pairs", type=int, default=15,
                   help="how many most-similar pairs to print for calibration (default 15)")
    return p.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    data = json.loads(text) if text else []
    if not isinstance(data, list):
        raise SystemExit(f"{path} is not a JSON array of records")
    return data


# ── near-dup sub-gate (local embeddings) ─────────────────────────────────────

# The local embedder (load_embedder / embed_texts) lives in existence_search.py — shared
# by near-dup, Gate 2 guessability, and the Gate 1 embedding sense judge.


def find_near_duplicates(
    sim: np.ndarray, cutoff: float
) -> tuple[list[int], list[tuple[int, int, float]]]:
    """Greedily retain the first representative of each similarity cluster."""
    kept: list[int] = []
    rejected: list[tuple[int, int, float]] = []
    for i in range(sim.shape[0]):
        if kept:
            sims_to_kept = sim[i, kept]
            j = int(np.argmax(sims_to_kept))
            best, matched = float(sims_to_kept[j]), kept[j]
        else:
            best, matched = -1.0, -1
        if matched != -1 and best >= cutoff:
            rejected.append((i, matched, best))
        else:
            kept.append(i)
    return kept, rejected


def top_similar_pairs(sim: np.ndarray, k: int) -> list[tuple[int, int, float]]:
    """The k highest-similarity (i, j) pairs, i < j — for threshold calibration."""
    n = sim.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    order = np.argsort(sim[iu, ju])[::-1][:k]
    return [(int(iu[o]), int(ju[o]), float(sim[iu[o], ju[o]])) for o in order]


def neardup_scan(
    records: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[list[tuple[int, int, float]], float, str]:
    """Embed → pairwise similarity → data-relative cutoff → greedy clustering.
    Prints the calibration stats and top pairs. Returns (flagged, cutoff, scale)
    where flagged is [(dup_index, matched_kept_index, similarity), ...]."""
    texts = [str(r.get(args.field, "")) for r in records]
    missing = sum(1 for t in texts if not t.strip())
    if missing:
        print(f"warning: {missing}/{len(texts)} records have an empty '{args.field}'")

    print(f"{len(records)} records | embedding field '{args.field}' with {args.model}")
    emb = existence_search.embed_texts(texts, args.model)
    if args.center:
        emb = emb - emb.mean(axis=0, keepdims=True)
        emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
    sim = emb @ emb.T                       # unit vectors → cosine
    np.fill_diagonal(sim, 0.0)

    # Require both the distribution-relative cutoff and the absolute floor.
    iu, ju = np.triu_indices(len(records), k=1)
    pair_sims = sim[iu, ju]
    mean, std, mx = float(pair_sims.mean()), float(pair_sims.std()), float(pair_sims.max())
    relative = mean + args.k * std
    cutoff = max(args.floor, relative)
    scale = "centered cosine" if args.center else "cosine"

    print(f"\npairwise {scale}: mean {mean:.3f} | std {std:.3f} | max {mx:.3f}")
    print(f"cutoff = max(floor {args.floor}, mean+{args.k}·std {relative:.3f}) = {cutoff:.3f}")

    print(f"\nTop {args.top_pairs} most-similar pairs:")
    for i, j, s in top_similar_pairs(sim, args.top_pairs):
        flag = "  <-- flagged" if s >= cutoff else ""
        print(f"  {s:.3f}  {records[i]['term']}  ~  {records[j]['term']}{flag}")
        print(f"          {texts[i][:70]}")
        print(f"          {texts[j][:70]}")

    _, flagged = find_near_duplicates(sim, cutoff)
    return flagged, cutoff, scale


def run_neardup_legacy(args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    """The original standalone near-dup mode (`--gates neardup`): writes
    _dedup.json / _dedup_removed.json (or _dedup_annotated.json with --annotate-only),
    which the smoke analysis scripts read."""
    stem = args.input.stem
    if args.annotate_only:
        output = args.output or args.input.with_name(f"{stem}_dedup_annotated.json")
        removed_path = None
    else:
        output = args.output or args.input.with_name(f"{stem}_dedup.json")
        removed_path = args.input.with_name(f"{stem}_dedup_removed.json")

    flagged, cutoff, scale = neardup_scan(records, args)
    flagged_map = {dup_i: (kept_i, s) for dup_i, kept_i, s in flagged}
    texts = [str(r.get(args.field, "")) for r in records]

    def with_near_dup_fields(idx: int) -> dict[str, Any]:
        """A near-dup record with the matched term + similarity attached."""
        kept_i, s = flagged_map[idx]
        out = dict(records[idx])
        out["near_dup_of"] = records[kept_i].get("term")
        out["near_dup_of_index"] = kept_i
        out["near_dup_similarity"] = round(s, 4)
        out["near_dup_note"] = (f"{scale} {s:.3f} ≥ cutoff {cutoff:.3f} vs "
                                f"'{records[kept_i].get('term')}' (embedded field: {args.field})")
        return out

    if args.annotate_only:
        # Keep everything; tag near-dups so a human (or the analyst gate) decides.
        annotated = []
        for idx, rec in enumerate(records):
            out = with_near_dup_fields(idx) if idx in flagged_map else dict(rec)
            out["near_dup_flag"] = idx in flagged_map
            annotated.append(out)
        output.write_text(json.dumps(annotated, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\ncutoff {cutoff:.3f}: {len(annotated)} records kept (none removed), "
              f"{len(flagged)} flagged for human review")
        mark = "⚑"
    else:
        # Remove near-dups: keep the earlier representatives, drop the later near-dups.
        kept = [rec for idx, rec in enumerate(records) if idx not in flagged_map]
        removed = [with_near_dup_fields(idx) for idx in sorted(flagged_map)]
        output.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
        removed_path.write_text(json.dumps(removed, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\ncutoff {cutoff:.3f}: kept {len(kept)}, removed {len(removed)} near-duplicate(s)")
        mark = "✕"

    for dup_i, kept_i, s in flagged:
        print(f"  {mark} {records[dup_i].get('term')}  ~  {records[kept_i].get('term')}  ({round(s, 4)})")
    print(f"\n{'annotated' if args.annotate_only else 'deduped'} → {output}")
    if not args.annotate_only:
        print(f"removed → {removed_path}")
    print("note: this was the near-dup sub-gate only — the full gate is "
          "`--gates neardup,guess,exist` (the default).")


# ── gate cache (resume for the API-calling sub-gates) ────────────────────

def load_gate_cache(path: Path) -> dict[str, dict[str, Any]]:
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return {rec["key"]: rec for rec in json.loads(text)}
    return {}


def save_gate_cache(cache: dict[str, dict[str, Any]], path: Path) -> None:
    # Atomic replacement preserves the prior cache if a write is interrupted.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(list(cache.values()), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(path)


# ── GATE 2 — LLM guessability ────────────────────────────────────────────────

def gate2_guessability(client, record: dict[str, Any], embed, *, guess_model: str | None,
                       n_runs: int, temperature: float, sim_threshold: float,
                       cache: dict, cache_path: Path) -> dict[str, Any]:
    """Reject terms whose context-free meaning guesses resemble the definition."""
    term = record["term"]
    lang = record.get("lang", "zh")
    definition = record.get("definition_tar_lang") or ""
    pos = record.get("pos") if lang == "ru" else None
    model = guess_model or record.get("model")
    if not model:
        raise ValueError("no guess model: record has no 'model' field and "
                         "--guess-model was not set")

    guesses: list[str] = []
    for i in range(n_runs):
        gkey = f"{term}|gate2|guess{i}|{model}"
        if gkey not in cache:
            raw = inference.chat(client, model,
                               prompts.build_gate_guess_prompt(term, lang, pos),
                               temperature=temperature)
            data = inference.extract_json(raw)
            # Raise on an empty guess so it is retried instead of cached as a false pass.
            if not isinstance(data, dict) or not data.get("guess"):
                raise ValueError(f"guesser {model} run {i} returned no guess: {raw[:200]!r}")
            cache[gkey] = {"key": gkey, "guess": str(data["guess"])}
            save_gate_cache(cache, cache_path)
        guesses.append(cache[gkey]["guess"])

    # Unit-normalized vectors make each definition/guess dot product a cosine score.
    vecs = embed([definition, *guesses])
    sims = [float(vecs[0] @ vecs[i + 1]) for i in range(n_runs)]
    avg = sum(sims) / len(sims)
    failed = avg >= sim_threshold
    reason = (f"{'guessable' if failed else 'opaque'} from form alone "
              f"(avg cosine {avg:.3f} {'≥' if failed else '<'} {sim_threshold}; "
              f"sims {', '.join(f'{s:.3f}' for s in sims)})")

    return {
        "gate2_guess_model": model,
        "gate2_guesses": guesses,
        "gate2_guess_sims": sims,
        "gate2_guess_sim_avg": avg,
        "gate2_sim_threshold": sim_threshold,
        "gate2_rule_used": f"FAIL if avg cosine(def, guess) over {n_runs} runs ≥ {sim_threshold}",
        "gate2_pass": not failed,
        "gate2_reason": reason,
    }


# ── threshold-tuning summaries ───────────────────────────────────────────────

def parse_lang_thresholds(spec: str, fallback: float) -> dict[str, float]:
    """A bare number ('0.6') applies to every language; a key=value list
    ('zh=0.55,ru=0.65') sets per-language cutoffs. Returns a dict carrying a 'default'
    key, so callers resolve with thr.get(lang, thr['default'])."""
    spec = str(spec).strip()
    if "=" not in spec:
        return {"default": float(spec)}
    out: dict[str, float] = {}
    for part in spec.split(","):
        key, _, val = part.partition("=")
        out[key.strip()] = float(val)
    out.setdefault("default", fallback)
    return out


def _sweep(label: str, values: list[float], current: float,
           grid: list[float], fmt) -> None:
    """Print score quantiles and removal counts across candidate thresholds."""
    if not values:
        return
    arr = sorted(values)
    n = len(arr)
    mean = sum(arr) / n
    print(f"\n{label}: n={n}  min {fmt(arr[0])}  median {fmt(arr[n // 2])}  "
          f"mean {fmt(mean)}  max {fmt(arr[-1])}")
    cells = [f"{fmt(t)}{'*' if abs(t - current) < 1e-9 else ''}→{sum(1 for v in arr if v >= t)}"
             for t in grid]
    print("  removed at threshold:  " + "   ".join(cells))


def sweep_float(label: str, values: list[float], current: float) -> None:
    grid = sorted({round(0.30 + 0.05 * i, 2) for i in range(11)} | {round(current, 2)})
    _sweep(label, values, round(current, 2), grid, lambda v: f"{v:.2f}")


def sweep_int(label: str, values: list[int], current: int) -> None:
    grid = list(range(1, max([*values, current, 1]) + 1))
    _sweep(label, values, current, grid, lambda v: f"{int(v)}")


# ── integrated PASS/FAIL pipeline ────────────────────────────────────────────

def run_gate_pipeline(args: argparse.Namespace, gates: list[str],
                      records: list[dict[str, Any]]) -> None:
    stem = args.input.stem
    output = args.output or args.input.with_name(f"{stem}_pass.json")
    rejected_path = args.rejected or args.input.with_name(f"{stem}_rejected.json")
    cache_path = args.cache or args.input.with_name(f"{stem}_gate_cache.json")
    cache = load_gate_cache(cache_path)
    print(f"{len(records)} records | gates: {' → '.join(gates)} | "
          f"{len(cache)} cached API calls ({cache_path.name})")

    funnel: list[tuple[str, int]] = [("in", len(records))]
    rejected: list[dict[str, Any]] = []
    survivors = list(records)

    if "neardup" in gates:
        flagged, cutoff, scale = neardup_scan(survivors, args)
        flagged_map = {dup_i: (kept_i, s) for dup_i, kept_i, s in flagged}
        for dup_i in sorted(flagged_map):
            kept_i, s = flagged_map[dup_i]
            rejected.append({
                **survivors[dup_i],
                "near_dup_of": survivors[kept_i].get("term"),
                "near_dup_similarity": round(s, 4),
                "gate_failed": "near_dup",
                "gate_reason": (f"{scale} {s:.3f} ≥ cutoff {cutoff:.3f} vs "
                                f"'{survivors[kept_i].get('term')}'"),
            })
        survivors = [r for i, r in enumerate(survivors) if i not in flagged_map]
        print(f"near-dup: kept {len(survivors)}, removed {len(flagged_map)}\n")
        funnel.append(("near-dup", len(survivors)))

    # Build the client and embedder only if the selected gates need them.
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
            embedder = existence_search.load_embedder(args.model)
        return embedder

    if "guess" in gates:
        embed = ensure_embedder()   # local model, loaded once for all records
        thr_by_lang = parse_lang_thresholds(args.guess_sim_threshold, DEFAULT_GUESS_SIM)
        keep = []
        scored: list[tuple[str, float]] = []   # (lang, avg) for the per-language sweep
        for rec in tqdm(survivors, desc="gate 2 (guessability)"):
            lang = rec.get("lang", "zh")
            sim_threshold = thr_by_lang.get(lang, thr_by_lang["default"])
            try:
                fields = gate2_guessability(
                    ensure_client(), rec, embed, guess_model=args.guess_model,
                    n_runs=args.guess_runs, temperature=args.guess_temperature,
                    sim_threshold=sim_threshold,
                    cache=cache, cache_path=cache_path)
            except Exception as err:  # failed item is retried on the next run
                rejected.append({**rec, "gate_failed": "error",
                                 "gate_reason": f"gate2: {err}"})
                print(f"  ! {rec.get('term')}: gate2 error: {err}", flush=True)
                continue
            scored.append((lang, fields["gate2_guess_sim_avg"]))
            rec = {**rec, **fields}
            if fields["gate2_pass"]:
                keep.append(rec)
            else:
                rejected.append({**rec, "gate_failed": "guessability",
                                 "gate_reason": fields["gate2_reason"]})
                print(f"  ✕ {rec.get('term')} — {fields['gate2_reason']}", flush=True)
        survivors = keep
        funnel.append(("guessability", len(survivors)))
        for lang in sorted({l for l, _ in scored}):  # one sweep per language, its own cutoff
            sweep_float(f"gate 2 [{lang}] — avg cosine(def, guess)",
                        [a for l, a in scored if l == lang],
                        thr_by_lang.get(lang, thr_by_lang["default"]))

    if "exist" in gates:
        consensus_mode = args.sense_judge == "consensus"
        # The snapshot date stamps the unseen claim: "not in public text as of D."
        snapshot_date = datetime.now(timezone.utc).date().isoformat()
        keep = []
        escalated: list[dict[str, Any]] = []   # consensus: embedding keeps, llm flags → human
        source_counts: list[int] = []          # for the tuning sweep — every item scored here
        # Resolve local and remote judges lazily.
        embed = ensure_embedder() if args.sense_judge in ("embedding", "consensus") else None
        for rec in tqdm(survivors, desc="gate 1 (existence)"):
            key = f"{rec['term']}|gate1"
            try:
                if key not in cache:
                    # Cache shared retrieval and both judge signals; apply cutoffs below.
                    if consensus_mode:
                        evidence = existence_search.gather_gate1_both(
                            rec, retriever=args.retriever, count=args.search_count,
                            snapshot_date=snapshot_date, embed=embed,
                            client=ensure_client(), judge_model=args.judge_model)
                    else:
                        evidence = existence_search.gather_gate1(
                            rec, retriever=args.retriever, count=args.search_count,
                            snapshot_date=snapshot_date, sense_judge=args.sense_judge,
                            embed=embed,
                            client=ensure_client() if args.sense_judge == "llm" else None,
                            judge_model=args.judge_model)
                    cache[key] = {"key": key, "fields": evidence}
                    save_gate_cache(cache, cache_path)
                evidence = cache[key]["fields"]
                if consensus_mode:
                    fields = {**evidence, **existence_search.decide_consensus(
                        evidence, args.in_sense_threshold,
                        snippet_sim_threshold=args.snippet_sim_threshold)}
                else:
                    fields = {**evidence, **existence_search.decide_gate1(
                        evidence, args.in_sense_threshold,
                        snippet_sim_threshold=args.snippet_sim_threshold)}
            except Exception as err:  # failed item is retried on the next run
                rejected.append({**rec, "gate_failed": "error",
                                 "gate_reason": f"gate1: {err}"})
                print(f"  ! {rec.get('term')}: gate1 error: {err}", flush=True)
                continue
            # consensus carries per-judge source lists; single-judge carries one.
            source_counts.append(len(fields.get("gate1_sources")
                                     or fields.get("gate1_llm_sources") or []))
            rec = {**rec, **fields}
            if fields["gate1_pass"]:
                keep.append(rec)
                if fields.get("gate1_consensus") == "escalate":
                    escalated.append(rec)
                    print(f"  ⚑ {rec.get('term')} → human review — {fields['gate1_llm_evidence']}",
                          flush=True)
            else:
                rejected.append({**rec,
                                 "gate_failed": "existence_consensus" if consensus_mode
                                                else "existence",
                                 "gate_reason": fields["gate1_reason"]})
                print(f"  ✕ {rec.get('term')} — {fields['gate1_reason']}", flush=True)
        survivors = keep
        funnel.append(("existence", len(survivors)))
        if consensus_mode:
            escalate_path = args.input.with_name(f"{stem}_escalate.json")
            escalate_path.write_text(json.dumps(escalated, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
            print(f"\nconsensus: {len(survivors)} kept "
                  f"({len(escalated)} flagged for human review → {escalate_path.name}), "
                  f"{sum(1 for r in rejected if r.get('gate_failed') == 'existence_consensus')} "
                  f"removed (both judges agreed)")
        sweep_int("gate 1 — independent in-sense source count "
                  "(threshold rule only; adjacent-collision fails counted separately)",
                  source_counts, args.in_sense_threshold)

    output.write_text(json.dumps(survivors, ensure_ascii=False, indent=2), encoding="utf-8")
    rejected_path.write_text(json.dumps(rejected, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nfunnel: " + " → ".join(f"{n} after {name}" if name != "in" else f"{n} in"
                                    for name, n in funnel))
    print(f"PASS {len(survivors)} → {output}")
    print(f"FAIL {len(rejected)} (gate_failed + gate_reason kept as diagnostics) → {rejected_path}")
    print("Next: in-context-generation.py on the PASS file, then the validator UI "
          "(human_validation/).")


def main() -> None:
    args = parse_args()
    gates = [g.strip() for g in args.gates.split(",") if g.strip()]
    bad = [g for g in gates if g not in GATES]
    if bad:
        raise SystemExit(f"--gates: unknown {bad}; allowed: {', '.join(GATES)}")
    gates = [g for g in GATES if g in gates]   # enforce cheapest-first order

    records = load_records(args.input)
    if not records:
        raise SystemExit(f"no records in {args.input}")

    if gates == ["neardup"]:
        run_neardup_legacy(args, records)
        return
    if args.annotate_only:
        raise SystemExit("--annotate-only applies only to `--gates neardup` "
                         "(the integrated pipeline removes failures)")
    run_gate_pipeline(args, gates, records)


if __name__ == "__main__":
    main()
