# Export benchmark items to a UTF-8 review CSV for Google Sheets or Excel.
# Reviewers fill the REVIEW_* columns documented in PROTOCOL.md; rejected items
# require a note. Records without generated sentence context are skipped.

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
STAGE1_DIR = BASE_DIR.parent          # this folder lives inside 1_slang_generation/
DEFAULT_ITEMS = STAGE1_DIR / "output-data" / "benchmark_items.json"

# Consensus-mode escalations include the LLM's existence evidence for review.
# These fields remain blank when both judges agree or consensus mode was not used.
CONTEXT_FIELDS = [
    "item_id", "term", "lang", "type", "definition_tar_lang",
    "sentence_in_context", "sentence_in_context_gloss",
    "paragraph_in_context", "paragraph_in_context_gloss",
    "gate1_consensus", "gate1_llm_evidence",
]
# Columns the reviewer fills in (exported blank).
REVIEW_FIELDS = [
    "REVIEW_faithful_use",   # yes | no
    "REVIEW_exists_check",   # yes | unsure | it exists
    "REVIEW_verdict",        # accept | reject
    "REVIEW_note",           # required on reject
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export benchmark items to a Google-Sheets-friendly review CSV.")
    p.add_argument("--items", type=Path, default=DEFAULT_ITEMS,
                   help="benchmark-items JSON (in-context-generation.py output)")
    p.add_argument("--output", type=Path, default=None,
                   help="review CSV (default: <items-stem>_review.csv next to --items)")
    return p.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else [data]


def main() -> None:
    args = parse_args()
    if not args.items.exists():
        raise SystemExit(f"items file not found: {args.items}\n"
                         "Run in-context-generation.py first (Stage 1, step 2).")
    output = args.output or args.items.with_name(args.items.stem + "_review.csv")

    items = [it for it in load_json(args.items) if it.get("sentence_in_context")]
    skipped = sum(1 for it in load_json(args.items) if not it.get("sentence_in_context"))
    # Escalated items (embedding kept / LLM flagged) go to the BOTTOM so the reviewer
    # works the clean both-judges-passed items first. Stable sort preserves order within
    # each group; only gate1_consensus=="escalate" is pushed down.
    items.sort(key=lambda it: it.get("gate1_consensus") == "escalate")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CONTEXT_FIELDS + REVIEW_FIELDS)
        writer.writeheader()
        for it in items:
            row = {k: it.get(k, "") for k in CONTEXT_FIELDS}
            row.update({k: "" for k in REVIEW_FIELDS})
            writer.writerow(row)

    print(f"wrote {len(items)} items → {output}"
          + (f"  ({skipped} generation-error records skipped)" if skipped else ""))
    print("Next: upload to Google Sheets (File → Import), have the reviewer fill the "
          "REVIEW_* columns, download as CSV, then run ingest_review_csv.py.")


if __name__ == "__main__":
    main()
