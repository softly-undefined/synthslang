# Stage 2 translation inference across five SlangDIT-derived prompt paths.
# Path, modality, and run remain explicit for Stage 3 analysis. Each model has an
# atomically replaced, resumable output file; failed records are retried.

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# inference.py lives one level up; prompts.py is in this directory.
SYNTHSLANG_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SYNTHSLANG_DIR))

import inference  # noqa: E402  (after sys.path insert)
import prompts  # noqa: E402

try:
    from twoqdm import tqdm
except ImportError:  # tqdm is optional; degrade to a no-op iterator
    def tqdm(iterable, **_):
        return iterable

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE_DIR / "output-data" / "translations.json"

PATHS = ("Direct", "DirectWithHint", "SlangDITIdentify", "DirectContext", "SlangDITIdentifyContext")
MODALITIES = ("sentence", "paragraph")

# Context paths are sentence-anchored and require both context fields.
CONTEXT_PATHS = ("DirectContext", "SlangDITIdentifyContext")
# Paths that emit a definition but no translation (understanding-only).
NO_TRANSLATION_PATHS = ("SlangDITIdentify", "SlangDITIdentifyContext")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run slang translation inference (Direct / DirectWithHint / SlangDITIdentify / "
        "DirectContext / SlangDITIdentifyContext) across models and modalities; resumable JSON output."
    )
    p.add_argument("--input", type=Path, required=True,
                   help="benchmark-items CSV or JSON")
    p.add_argument("--models", required=True,
                   help="comma-separated model ids, passed through verbatim to the endpoint set by "
                        "OPENAI_BASE_URL (e.g. gpt-4o-mini, or a served name on a local vLLM server)")
    p.add_argument("--paths", default="Direct,DirectWithHint,SlangDITIdentify,DirectContext,SlangDITIdentifyContext",
                   help="comma-separated subset of Direct,DirectWithHint,SlangDITIdentify,DirectContext,"
                        "SlangDITIdentifyContext (default all). DirectContext/SlangDITIdentifyContext run once per "
                        "item/model (sentence target + paragraph context) regardless of --modalities")
    p.add_argument("--modalities", default="sentence,paragraph",
                   help="comma-separated subset of sentence,paragraph")
    p.add_argument("--src-lang", default="zh",
                   help="source language when an item record omits `lang` (zh/ru)")
    p.add_argument("--tgt-lang", default="en", help="target language")
    p.add_argument("--limit", type=int, default=None,
                   help="only run the first N input records (handy for a small run)")
    p.add_argument("--run", type=int, default=1,
                   help="run number for repeated runs of the SAME model on the SAME data "
                        "(default 1). It is part of the resume key, so --run 2 produces its own "
                        "records instead of being skipped as already-done. Use it for the extra "
                        "gpt-4o-mini / gemma-3-4b runs.")
    p.add_argument("--temperature", default="0.0",
                   help="sampling temperature sent to the model (default 0.0). Pass 'none' "
                        "to omit it entirely for models that "
                        "reject an explicit temperature (some frontier/reasoning models). Raise it "
                        "above 0 to sample diversity across repeated --run values.")
    p.add_argument("--max-tokens", default="4096",
                   help="completion-token cap sent with every call (default 4096 — clears every "
                        "legitimate response observed across the done models, including the "
                        "verbose SlangDITIdentify outliers). Bounds runaway generation loops (a "
                        "confused model ruminating for hours on one item, e.g. Qwen3.5-27B on "
                        "zh_morph_blend_pass:26); a capped runaway errors fast and is recorded "
                        "instead of hanging the run. Pass 'none' to omit the field.")
    p.add_argument("--reasoning-effort", nargs="?", const="medium", default=None,
                   choices=["low", "medium", "high"],
                   help="enable extended thinking for reasoning-capable models. Bare flag "
                        "(--reasoning-effort) defaults to 'medium'; pass low/medium/high to set "
                        "the level; omit entirely for no thinking (default). Thinking requires "
                        "temperature=1 — a --temperature of 0.0 is auto-bumped to 1.0 (the backend "
                        "rejects temp 0 with thinking on). The reasoning trace, when the model "
                        "returns one, is saved to each record's reasoning_trace field. Give a "
                        "thinking run a distinct --output base (its own per-model file) so it is "
                        "evaluated separately — Stage 3 POOLS --run values into one cell, so a "
                        "thinking run sharing a file with a plain run would be averaged together.")
    p.add_argument("--run-name", default=None,
                   help="free-text run label stored on each record (default: output stem)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="results-JSON BASE path. The file actually written is "
                        "<stem>-<model-slug><suffix>, ONE PER MODEL, so different models can "
                        "run concurrently without overwriting each other. Realign afterward "
                        "with align_outputs.py.")
    return p.parse_args()


def parse_temperature(value: str) -> float | None:
    # "none" or "default" omits temperature for endpoint compatibility.
    if value.strip().lower() in {"none", "default"}:
        return None
    try:
        return float(value)
    except ValueError:
        raise SystemExit(f"--temperature: expected a number or 'none', got {value!r}")


def parse_max_tokens(value: str) -> int | None:
    # "none" or "default" omits max_tokens.
    if value.strip().lower() in {"none", "default"}:
        return None
    try:
        return int(value)
    except ValueError:
        raise SystemExit(f"--max-tokens: expected an integer or 'none', got {value!r}")


def split_csv(value: str, allowed: tuple[str, ...], name: str) -> list[str]:
    # validates that every comma-separated value is in the allowed set
    items = [v.strip() for v in value.split(",") if v.strip()]
    bad = [v for v in items if v not in allowed]
    if bad:
        raise SystemExit(f"--{name}: unknown {bad}; allowed: {', '.join(allowed)}")
    return items


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    # JSON accepts both an array and a single object for backward compatibility.
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else [data]


def source_for_modality(rec: dict[str, Any], modality: str) -> str | None:
    # sentence: the in-context sentence. paragraph: the in-context paragraph.
    # returns None so the caller can report the skip rather than silently substituting
    if modality == "sentence":
        return rec.get("sentence_in_context")
    if modality == "paragraph":
        return rec.get("paragraph_in_context") or None
    return None


def cache_key(item_id: str, model: str, path: str, modality: str, run: int) -> str:
    # Resume key for a single experimental cell.
    return f"{item_id}|{model}|{path}|{modality}|run{run}"


def model_slug(model: str) -> str:
    # Filesystem-safe model ID.
    return re.sub(r"[^0-9A-Za-z]+", "-", model).strip("-")


def output_for_model(base: Path, model: str) -> Path:
    # One result file per model prevents concurrent writers from colliding.
    return base.with_name(f"{base.stem}-{model_slug(model)}{base.suffix}")


def load_output(output: Path) -> list[dict[str, Any]]:
    # returns whatever was already written so we can append to it on resume
    if not output.exists():
        return []
    text = output.read_text(encoding="utf-8").strip()
    return json.loads(text) if text else []


def done_keys(records: list[dict[str, Any]]) -> set[str]:
    # Retry errored records; definition-only paths succeed without a translation.
    return {
        cache_key(r["item_id"], r["model"], r["path"], r["modality"], r.get("run", 1))
        for r in records
        if not r.get("error") and (
            r.get("mt_translation") is not None
            or r.get("path") in NO_TRANSLATION_PATHS
        )
    }


def build_tasks(records, models, paths, modalities, *, source_file, args):
    # Regular paths cross items with modalities; context paths run once per item
    # and require both sentence and paragraph fields.
    regular_paths = [p for p in paths if p not in CONTEXT_PATHS]
    context_paths = [p for p in paths if p in CONTEXT_PATHS]
    tasks: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []
    for index, rec in enumerate(records):
        item_id = str(rec.get("id") or rec.get("item_id") or f"{source_file.stem}:{index}")
        term = rec.get("term") or rec.get("slang") or ""
        # Definitions are English-only (target language). Prefer the canonical
        # definition_tar_lang; fall back to the older explanation_tar_lang spelling.
        definition_tar_lang = rec.get("definition_tar_lang") or rec.get("explanation_tar_lang") or ""
        lang = rec.get("lang") or args.src_lang
        type_ = rec.get("type") or "unknown"
        base = {
            "item_id": item_id, "term": term, "lang": lang, "type": type_,
            "tgt_lang": args.tgt_lang, "run": args.run,
            "definition_tar_lang": definition_tar_lang,
        }

        for modality in modalities:
            source = source_for_modality(rec, modality)
            if not source:
                skipped.append((item_id, modality))
                continue
            for model in models:
                for path in regular_paths:
                    tasks.append({**base, "model": model, "path": path,
                                  "modality": modality, "source_text": source,
                                  "context_text": None})

        if context_paths:
            sentence = source_for_modality(rec, "sentence")
            paragraph = rec.get("paragraph_in_context")
            if sentence and paragraph:
                for model in models:
                    for path in context_paths:
                        tasks.append({**base, "model": model, "path": path,
                                      "modality": "sentence", "source_text": sentence,
                                      "context_text": paragraph})
            else:
                missing = "sentence" if not sentence else "paragraph_in_context"
                for path in context_paths:
                    skipped.append((item_id, f"{path} (no {missing})"))
    return tasks, skipped


def run_task(client, task: dict[str, Any], *, run_name: str, source_file: Path,
             temperature: float | None, max_tokens: int | None,
             reasoning_effort: str | None) -> dict[str, Any]:
    """Execute one translation call and return the output record (error captured, never raises)."""
    path, source = task["path"], task["source_text"]
    # Keep a stable output schema even on errors.
    record = {
        "item_id": task["item_id"], "term": task["term"], "lang": task["lang"],
        "type": task["type"], "model": task["model"],
        "path": path, "modality": task["modality"], "run": task["run"],
        "temperature": temperature, "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "tgt_lang": task["tgt_lang"],
        "source_text": source, "mt_slang": None, "mt_definition": None,
        "mt_translation": None, "raw_response": None, "reasoning_trace": None,
        "run_name": run_name, "source_file": str(source_file), "error": None,
    }
    last_err: Exception | None = None
    for attempt in range(5):
        if attempt:
            print(f"  retry {attempt}/4 {task['item_id']} {path}", flush=True)
        try:
            last_err = None
            # Retry parse failures; preserve successfully parsed null fields.
            if path == "SlangDITIdentify":
                # term is provided; explain it (English) — no translation
                prompt = prompts.build_identify_prompt(
                    source, task["term"], src_lang=task["lang"])
                raw, record["reasoning_trace"] = inference.chat_full(
                    client, task["model"], prompt, temperature=temperature,
                    max_tokens=max_tokens, reasoning_effort=reasoning_effort)
                record["raw_response"] = raw
                data = prompts.parse_identify_response(raw)
                record["mt_slang"] = data.get("slang")
                record["mt_definition"] = data.get("definition")
                if data.get("salvaged"):
                    print(f"  ~ salvaged malformed identify JSON {task['item_id']} "
                          f"{path}/{task['modality']} (raw kept in raw_response)", flush=True)
            elif path == "SlangDITIdentifyContext":
                # like SlangDITIdentify, plus the paragraph as reference context — no translation
                prompt = prompts.build_identify_context_prompt(
                    source, task["term"], task["context_text"], src_lang=task["lang"])
                raw, record["reasoning_trace"] = inference.chat_full(
                    client, task["model"], prompt, temperature=temperature,
                    max_tokens=max_tokens, reasoning_effort=reasoning_effort)
                record["raw_response"] = raw
                data = prompts.parse_identify_response(raw)
                record["mt_slang"] = data.get("slang")
                record["mt_definition"] = data.get("definition")
                if data.get("salvaged"):
                    print(f"  ~ salvaged malformed identify JSON {task['item_id']} "
                          f"{path}/{task['modality']} (raw kept in raw_response)", flush=True)
            elif path == "DirectWithHint":
                # Andrew Ng prompt + gold slang term and definition appended — upper-bound comparison
                system, prompt = prompts.build_translate_direct_with_hint_prompt(
                    source, task["term"], task["definition_tar_lang"],
                    src_lang=task["lang"], tgt_lang=task["tgt_lang"])
                raw, record["reasoning_trace"] = inference.chat_full(
                    client, task["model"], prompt, system=system, temperature=temperature,
                    max_tokens=max_tokens, reasoning_effort=reasoning_effort)
                record["raw_response"] = raw
                record["mt_translation"] = raw.strip()
            elif path == "DirectContext":
                # Andrew Ng prompt + a same-source paragraph as reference context; translate the sentence only
                system, prompt = prompts.build_translate_direct_context_prompt(
                    source, task["context_text"],
                    src_lang=task["lang"], tgt_lang=task["tgt_lang"])
                raw, record["reasoning_trace"] = inference.chat_full(
                    client, task["model"], prompt, system=system, temperature=temperature,
                    max_tokens=max_tokens, reasoning_effort=reasoning_effort)
                record["raw_response"] = raw
                record["mt_translation"] = raw.strip()
            else:
                # Direct: no slang hint — model translates cold (Andrew Ng prompt), plain text response
                system, prompt = prompts.build_translate_direct_prompt(
                    source, src_lang=task["lang"], tgt_lang=task["tgt_lang"])
                raw, record["reasoning_trace"] = inference.chat_full(
                    client, task["model"], prompt, system=system, temperature=temperature,
                    max_tokens=max_tokens, reasoning_effort=reasoning_effort)
                record["raw_response"] = raw
                record["mt_translation"] = raw.strip()
        except Exception as err:  # catches API errors and bad responses — retried, then logged
            last_err = err
            continue
        break  # success — exit retry loop
    if last_err is not None:
        record["error"] = str(last_err)
    return record


def save_records(records: list[dict[str, Any]], output: Path) -> None:
    # Replace atomically so interruption cannot truncate the existing results file.
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, output)


def main() -> None:
    args = parse_args()
    if args.run < 1:
        raise SystemExit(f"--run must be >= 1, got {args.run}")
    temperature = parse_temperature(args.temperature)
    max_tokens = parse_max_tokens(args.max_tokens)
    reasoning_effort = args.reasoning_effort
    # This backend requires temperature 1 when reasoning is enabled.
    if reasoning_effort is not None and temperature == 0.0:
        print(f"note: --reasoning-effort {reasoning_effort} requires temperature=1 "
              f"(the backend rejects temp 0 with thinking on); bumping temperature 0.0 -> 1.0.",
              flush=True)
        temperature = 1.0
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    paths = split_csv(args.paths, PATHS, "paths")
    modalities = split_csv(args.modalities, MODALITIES, "modalities")
    run_name = args.run_name or args.output.stem

    records = load_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Separate output files let model runs execute concurrently without collisions.
    for model in models:
        # Every model in a single invocation is served by the one endpoint the
        # client points at (OPENAI_BASE_URL); to run models on a different engine,
        # re-run with that engine's OPENAI_BASE_URL.
        client = inference.make_client()
        out_path = output_for_model(args.output, model)
        tasks, skipped = build_tasks(records, [model], paths, modalities,
                                     source_file=args.input, args=args)
        existing = load_output(out_path)
        already = done_keys(existing)
        todo = [t for t in tasks
                if cache_key(t["item_id"], t["model"], t["path"], t["modality"], t["run"]) not in already]

        print(f"\n[{model}] -> {out_path}")
        print(f"  {len(tasks)} calls: {len(records)} records × {len(paths)} paths "
              f"(modalities {','.join(modalities)} for regular paths; context paths run once) "
              f"| run {args.run}, temperature {temperature}, max_tokens {max_tokens or 'off'}, "
              f"reasoning_effort {reasoning_effort or 'off'}")
        print(f"  already done: {len(already)} | to run: {len(todo)}", flush=True)
        if skipped:
            print(f"  skipped {len(skipped)} (item, reason) — no usable source text "
                  f"(e.g. an item with no paragraph): {skipped[:5]}{' ...' if len(skipped) > 5 else ''}")
        if not todo:
            print("  nothing to do.")
            continue

        failures = 0
        all_records = list(existing)
        for task in tqdm(todo, desc=model):
            record = run_task(client, task, run_name=run_name, source_file=args.input,
                              temperature=temperature, max_tokens=max_tokens,
                              reasoning_effort=reasoning_effort)
            if record["error"]:
                failures += 1
                print(f"  ! {task['model']} {task['path']}/{task['modality']} "
                      f"{task['item_id']}: {record['error']}", flush=True)
            all_records.append(record)
            # save after every record so a crash mid-run loses at most one result (atomic write)
            save_records(all_records, out_path)

        print(f"  wrote {len(todo)} records ({failures} failed) -> {out_path}")


if __name__ == "__main__":
    main()
