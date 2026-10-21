# Stage 1 candidate generation. Prompts are selected by language and slang type.
#
# Each model call returns a small batch of terms and English definitions. Existing
# output is loaded for resume and exact-term deduplication, and prior terms are sent
# back as an avoid-list. The file is saved after every batch. Quality, semantic
# duplication, and contamination are handled by contamination-gate.py.

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from twoqdm import tqdm
except ImportError:
    tqdm = None


def log(msg: str) -> None:
    """Print without clobbering the tqdm bar (falls back to plain print)."""
    if tqdm is not None:
        tqdm.write(msg)
    else:
        print(msg, flush=True)

# inference.py lives one level up; prompts.py is in this directory.
SYNTHSLANG_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SYNTHSLANG_DIR))

import inference  # noqa: E402  (after sys.path insert)
import prompts  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent

# Just for --type's allowed values. prompts.py is the real source of truth for
# which (lang, type) pairs actually have a registered prompt.
ZH_TYPES = ("semantic_shift", "morph_blend", "affixation", "sentence_compression")
RU_TYPES = ("semantic_shift", "morph_blend", "suffixation", "foreign_verb")
AR_TYPES = ("semantic_shift", "morph_blend")
FA_TYPES = ("light_verb", "affixation", "semantic_shift")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1, step 1: generate raw (term + definition) slang candidates "
        "for one (lang, type). No usage sentences yet — those come later, after the "
        "contamination gate. Output is a resumable JSON array."
    )
    p.add_argument("--lang", choices=("zh", "ru", "ar", "fa"), required=True)
    # zh / ru / ar / fa share some type names, so dedupe for the choices list.
    all_types = tuple(dict.fromkeys(ZH_TYPES + RU_TYPES + AR_TYPES + FA_TYPES))
    p.add_argument("--type", choices=all_types,
                   help="phenomenon type. Required. zh: " + " / ".join(ZH_TYPES)
                        + "; ru: " + " / ".join(RU_TYPES)
                        + "; ar: " + " / ".join(AR_TYPES)
                        + "; fa: " + " / ".join(FA_TYPES) + ".")
    p.add_argument("--n", type=int, default=150,
                   help="how many NEW unique terms to add this run (default 150 — the "
                        "starting target per category; note the contamination gate trims "
                        "this, ~60%% for zh / ~10%% for ru, so the surviving count is lower)")
    p.add_argument("--model", required=True,
                   help="model id served by your endpoint (e.g. gpt-4o-mini)")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="sampling temperature. KEEP THIS ABOVE 0: every call returns a "
                        "fresh batch of terms and we dedupe by term, so a temperature "
                        "near 0 makes each batch identical and the run stalls on duplicates.")
    p.add_argument("--no-temperature", action="store_true",
                   help="do NOT send a temperature at all. Required for reasoning models "
                        "(gpt-5.x, o-series, gpt-oss) — they reject any non-default "
                        "temperature and the call errors. They sample at their default "
                        "(diverse) temperature, so the dedup loop still gets fresh batches.")
    p.add_argument("--max-rounds", type=int, default=50,
                   help="safety cap on model calls, in case it keeps repeating terms (default 50)")
    p.add_argument("--output", type=Path, default=None,
                   help="output JSON (default: output-data/<lang>_<type>.json)")
    p.add_argument("--run-name", default=None,
                   help="run label stored on each record (default: output stem)")
    return p.parse_args()


def load_output(path: Path) -> list[dict[str, Any]]:
    """Existing records, so a rerun appends instead of overwriting (resume)."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    return json.loads(text) if text else []


def call_model(client, model: str, prompt: str, temperature: float | None) -> str:
    """Return one raw generation response for parsing and logging."""
    return inference.chat(client, model, prompt, temperature=temperature)


def parse_terms(raw: str) -> list[dict[str, Any]]:
    """Extract and validate the generated JSON array of term records."""
    data = inference.extract_json(raw)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):     # a model returned a single object instead of an array
        return [data]
    return []


def build_record(item: dict[str, Any], *, lang: str, type_: str, model: str,
                 run_name: str, source_file: str, generated_at: str,
                 round_: int) -> dict[str, Any]:
    """Add language, type, and run metadata to a generated term record."""
    record: dict[str, Any] = {"term": item.get("term"), "lang": lang, "type": type_}
    for key, value in item.items():
        if key != "term":          # term is already placed first
            record[key] = value
    record.update({
        "round": round_,
        "model": model,
        "run_name": run_name,
        "source_file": source_file,
        "generated_at": generated_at,
    })
    return record


def main() -> None:
    args = parse_args()

    # --type is required for all languages.
    type_ = args.type
    if type_ is None:
        if args.lang == "zh":
            valid = ZH_TYPES
        elif args.lang == "ru":
            valid = RU_TYPES
        elif args.lang == "fa":
            valid = FA_TYPES
        else:
            valid = AR_TYPES
        raise SystemExit(f"--type is required for {args.lang} ({' / '.join(valid)})")

    # Pull the matching prompt. prompts.py raises a clear ValueError listing what's
    # available if this (lang, type) pair has no registered prompt (e.g. zh + suffixation).
    try:
        prompt = prompts.get_generation_prompt(args.lang, type_)
    except ValueError as e:
        raise SystemExit(str(e))

    output = args.output or (BASE_DIR / "output-data" / f"{args.lang}_{type_}.json")
    run_name = args.run_name or output.stem
    # Store one telemetry record per model call beside the term output.
    runlog = output.with_name(output.stem + ".runlog.jsonl")

    # Reasoning models (gpt-5.x, o-series, gpt-oss) reject a non-default temperature,
    # so --no-temperature sends none at all (None) and lets the model use its default.
    temperature = None if args.no_temperature else args.temperature

    # Resume: load whatever is already there and remember every term we've seen, so
    # we never write the same term twice (within this run or across reruns).
    existing = load_output(output)
    seen = {r.get("term") for r in existing}
    all_records = list(existing)
    print(f"{len(existing)} existing records in {output} | generating {args.n} new {args.lang}/{type_} terms")
    print(f"model={args.model} temperature={temperature} batch={prompts.TERMS_PER_PROMPT} | runlog → {runlog}")

    output.parent.mkdir(parents=True, exist_ok=True)
    client = inference.make_client()
    generated_at = datetime.now(timezone.utc).isoformat()

    # Keep calling the model until we've added --n new unique terms, or we hit the
    # round cap (guards against a model that just keeps repeating itself).
    added = 0
    rounds = 0
    bar = tqdm(total=args.n, desc=f"{args.lang}/{type_}", unit="term") if tqdm else None
    while added < args.n and rounds < args.max_rounds:
        rounds += 1
        # Grow the avoid-list across rounds, including terms loaded on resume.
        avoid_terms = sorted(t for t in seen if t)
        round_prompt = prompts.with_avoid_list(prompt, avoid_terms)

        # Call and parse separately so the raw response is captured for the run log
        # even when the API call errors or the JSON fails to parse.
        raw, error = "", None
        batch: list[dict[str, Any]] = []
        try:
            raw = call_model(client, args.model, round_prompt, temperature)
            batch = parse_terms(raw)
        except Exception as e:
            error = str(e)
            log(f"  ! round {rounds}: {e}")

        # Walk the batch, splitting it into new / duplicate / blank so the run log
        # records exactly what the dedup step rejected this round.
        new_terms: list[str] = []
        dup_terms: list[str] = []
        blanks = 0
        for item in batch:
            term = item.get("term")
            if not term:
                blanks += 1
                continue
            if term in seen:
                dup_terms.append(term)
                continue
            seen.add(term)
            all_records.append(build_record(
                item, lang=args.lang, type_=type_, model=args.model,
                run_name=run_name, source_file=str(output), generated_at=generated_at,
                round_=rounds,
            ))
            new_terms.append(term)
            added += 1
            if added >= args.n:
                break

        # Append this round's telemetry. prompt_chars/response_chars are a cheap proxy
        # for the token-cost question (the avoid-list inflates prompt_chars as N grows).
        log_entry = {
            "round": rounds,
            "run_name": run_name,
            "model": args.model,
            "temperature": temperature,
            "avoid_list_size": len(avoid_terms),
            "prompt_chars": len(round_prompt),
            "response_chars": len(raw),
            "n_returned": len(batch),
            "n_new": len(new_terms),
            "n_duplicate": len(dup_terms),
            "n_blank": blanks,
            "new_terms": new_terms,
            "duplicate_terms": dup_terms,
            "error": error,
            "raw_response": raw,
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }
        with runlog.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        # Write the term file after every round so an interrupted run keeps what it made.
        output.write_text(json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8")
        if bar is not None:
            bar.update(len(new_terms))
            bar.set_postfix(rnd=rounds, dup=len(dup_terms), avoid=len(avoid_terms))
        log(f"  round {rounds}: returned {len(batch)}, +{len(new_terms)} new, "
            f"{len(dup_terms)} dup ({added}/{args.n}) | avoid-list {len(avoid_terms)}")

    if bar is not None:
        bar.close()
    if added < args.n:
        print(f"  stopped at {added}/{args.n} after {rounds} rounds — model kept repeating "
              f"terms. Raise --max-rounds (or --temperature, unless --no-temperature).")
    print(f"\nwrote {added} new terms ({len(all_records)} total) → {output}")
    print(f"per-round telemetry ({rounds} rounds) → {runlog}")
    print("Next: run contamination-gate.py on this file before generating sentences.")


if __name__ == "__main__":
    main()
