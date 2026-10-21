# Align Stage 2 model outputs to item IDs present and error-free in every input.
# Null output fields are allowed. Aligned copies are written to the Stage 3 input
# directory unless --outdir is set; originals remain unchanged.
#
#   python3 align_outputs.py zh_en_translations-*.json
#   python3 align_outputs.py ru_en_translations-*.json

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# .../2_slang_translation/  ->  .../3_slang_translation_evaluation/input-data/
DEFAULT_OUTDIR = (Path(__file__).resolve().parent.parent
                  / "3_slang_translation_evaluation" / "input-data")


def load(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def item_ids(records: list[dict]) -> set[str]:
    return {r["item_id"] for r in records}


def failed_ids(records: list[dict]) -> set[str]:
    return {r["item_id"] for r in records if r.get("error")}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Align per-model Stage 2 outputs to a common, error-free segment set.")
    p.add_argument("files", nargs="+", type=Path,
                   help="Stage 2 output JSONs (e.g. the gpt and llama files)")
    p.add_argument("--suffix", default="-aligned",
                   help="appended to each input stem for its output file (default: -aligned)")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                   help="where to write the aligned files (default: Stage 3 input-data/)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [(f, load(f)) for f in args.files]

    common = set.intersection(*(item_ids(recs) for _, recs in datasets))
    failed = set().union(*(failed_ids(recs) for _, recs in datasets))
    keep = common - failed

    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"{len(common)} segments common to all {len(datasets)} files; "
          f"{len(failed & common)} dropped for a failure -> {len(keep)} kept")
    print(f"writing to {args.outdir}")
    for f, recs in datasets:
        out = args.outdir / f"{f.stem}{args.suffix}{f.suffix}"
        kept = [r for r in recs if r["item_id"] in keep]
        out.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {out.name}: {len(recs)} -> {len(kept)} records")


if __name__ == "__main__":
    main()
