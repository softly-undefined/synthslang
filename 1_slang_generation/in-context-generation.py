# Stage 1 context generation for terms that pass the contamination gate.
#
# Generates sentence and paragraph examples plus English glosses, preserving all
# input metadata. Context must make the intended slang sense recoverable without
# directly defining it. Blank analyst_* fields are filled by the CSV-based human
# review before accepted items move to Stage 2. Runs resume by item ID.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SYNTHSLANG_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SYNTHSLANG_DIR))

import inference  # noqa: E402
import prompts  # noqa: E402

try:
    from twoqdm import tqdm
except ImportError:
    def tqdm(iterable, **_):
        return iterable

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE_DIR / "output-data" / "benchmark_items.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate in-context sentences and paragraphs for contamination-gate "
        "PASS items; resumable JSON output."
    )
    p.add_argument("--input", type=Path, required=True,
                   help="PASS JSON from contamination-gate.py")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--model", required=True, help="model id")
    p.add_argument("--limit", type=int, default=None,
                   help="only process the first N input records")
    p.add_argument("--run-name", default=None,
                   help="run label stored on each record (default: output stem)")
    p.add_argument("--no-temperature", action="store_true",
                   help="do NOT send a temperature at all. Required for reasoning models "
                        "(gpt-5.x, o-series, gpt-oss) — they reject a non-default temperature "
                        "and every call errors otherwise. (inference.chat defaults to 0.0.)")
    return p.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_output(output: Path) -> list[dict[str, Any]]:
    """Load existing output records (empty list if file absent or empty)."""
    if not output.exists():
        return []
    text = output.read_text(encoding="utf-8").strip()
    return json.loads(text) if text else []


def done_ids(records: list[dict[str, Any]]) -> set[str]:
    """IDs of successfully generated records (skipped on rerun; errors are retried)."""
    return {r["item_id"] for r in records if r.get("sentence_in_context") is not None}


def make_item_id(rec: dict[str, Any], index: int, stem: str) -> str:
    return str(rec.get("id") or rec.get("item_id") or f"{stem}:{index}")


def generate_usage(client, model: str, rec: dict[str, Any],
                   temperature: float | None) -> dict[str, Any]:
    """One model call: sentence + paragraph in context. temperature=None omits it
    (required for reasoning models like gpt-5.x — see --no-temperature)."""
    is_compression = rec.get("type") == "sentence_compression"
    prompt = prompts.build_sentence_paragraph_prompt(
        term=rec["term"],
        definition=rec["definition_tar_lang"],
        lang=rec.get("lang", "zh"),
        source=rec.get("source") if is_compression else None,
        source_gloss=rec.get("source_gloss") if is_compression else None,
    )
    raw = inference.chat(client, model, prompt, temperature=temperature)
    data = inference.extract_json(raw)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object, got {type(data).__name__}: {raw[:200]!r}")
    return {
        "sentence_in_context":        data.get("sentence_in_context"),
        "sentence_in_context_gloss":  data.get("sentence_in_context_gloss"),
        "paragraph_in_context":       data.get("paragraph_in_context"),
        "paragraph_in_context_gloss": data.get("paragraph_in_context_gloss"),
        "generation_raw":             raw,
    }


def build_record(
    rec: dict[str, Any],
    item_id: str,
    usage: dict[str, Any] | None,
    *,
    model: str,
    run_name: str,
    source_file: str,
    error: str | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"item_id": item_id}
    for k, v in rec.items():
        if k not in ("id", "item_id"):
            out[k] = v

    out.update({
        "sentence_in_context":        usage["sentence_in_context"]        if usage else None,
        "sentence_in_context_gloss":  usage["sentence_in_context_gloss"]  if usage else None,
        "paragraph_in_context":       usage["paragraph_in_context"]       if usage else None,
        "paragraph_in_context_gloss": usage["paragraph_in_context_gloss"] if usage else None,
        # Blank analyst fields — filled by human_validation/ (validator app
        # + adjudication) before passing to Stage 2.
        "analyst_plausibility":              None,
        "analyst_translation_opacity":       None,
        "analyst_contextual_recoverability": None,
        "analyst_novelty":                   None,
        "analyst_status":                    None,
        "model":       model,
        "run_name":    run_name,
        "source_file": source_file,
        "error":       error,
    })
    return out


def main() -> None:
    args = parse_args()
    run_name = args.run_name or args.output.stem
    # Reasoning models (gpt-5.x) reject a non-default temperature; --no-temperature sends
    # none (None) and lets the model use its default. Otherwise inference.chat's 0.0.
    temperature = None if args.no_temperature else 0.0

    records = load_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]

    stem = args.input.stem
    labeled = [(make_item_id(r, i, stem), r) for i, r in enumerate(records)]
    existing = load_output(args.output)
    already = done_ids(existing)
    todo = [(iid, r) for iid, r in labeled if iid not in already]

    print(f"{len(records)} input records | already done: {len(already)} | to run: {len(todo)}")
    if not todo:
        print("nothing to do.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = inference.make_client()
    failures = 0
    all_records = list(existing)

    for iid, rec in tqdm(todo, desc="generating"):
        usage: dict[str, Any] | None = None
        err: str | None = None
        try:
            usage = generate_usage(client, args.model, rec, temperature)
        except Exception as e:
            err = str(e)
            failures += 1
            print(f"  ! [{iid}] {rec.get('term')}: {err}", flush=True)

        record = build_record(
            rec, iid, usage,
            model=args.model,
            run_name=run_name,
            source_file=str(args.input),
            error=err,
        )
        all_records.append(record)
        args.output.write_text(
            json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    written = len(todo)
    print(f"\nwrote {written} records ({failures} failed) → {args.output}")
    print(f"Next: human-validate in human_validation/ (export_review_csv.py → "
          f"Google Sheets → ingest_review_csv.py) — the accepted file is Stage 2's input.")


if __name__ == "__main__":
    main()
