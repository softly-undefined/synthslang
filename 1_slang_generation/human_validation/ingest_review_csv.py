# Merge one or more reviewed CSVs into validated and accepted benchmark JSON.
# The default acceptance rule is unanimity; --rule majority relaxes it. Annotator
# IDs come from the sheet or, if absent, its filename. Pairwise Cohen's kappa is
# reported when multiple reviewers are present.

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
STAGE1_DIR = BASE_DIR.parent          # this folder lives inside 1_slang_generation/
DEFAULT_ITEMS = STAGE1_DIR / "output-data" / "benchmark_items.json"

ACCEPT_WORDS = {"accept", "accepted", "keep", "yes", "y", "✓", "approve", "approved"}
REJECT_WORDS = {"reject", "rejected", "drop", "no", "n", "✗", "x", "deny", "denied"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply reviewed CSV verdicts to the benchmark items; write the "
        "validated + accepted JSON and report inter-annotator agreement.")
    p.add_argument("--items", type=Path, default=DEFAULT_ITEMS,
                   help="benchmark-items JSON (in-context-generation.py output)")
    p.add_argument("--reviewed", type=Path, nargs="+", required=True,
                   help="one or more filled review CSVs (download from Google Sheets)")
    p.add_argument("--rule", choices=("all", "majority"), default="all",
                   help="accept when ALL reviewers accept (default) or a strict majority")
    p.add_argument("--output", type=Path, default=None,
                   help="validated items JSON (default: <items-stem>_validated.json)")
    p.add_argument("--accepted", type=Path, default=None,
                   help="accepted-only JSON, feeds Stage 2 (default: <items-stem>_accepted.json)")
    return p.parse_args()


def norm_verdict(raw: str) -> str | None:
    """Map a free-text verdict cell to 'accept' / 'reject' / None (blank)."""
    v = (raw or "").strip().lower()
    if not v:
        return None
    if v in ACCEPT_WORDS:
        return "accept"
    if v in REJECT_WORDS:
        return "reject"
    raise SystemExit(f"unrecognized REVIEW_verdict value {raw!r} — "
                     "use accept / reject (or leave blank for not-yet-reviewed)")


def blank_to_none(raw: str) -> str | None:
    v = (raw or "").strip()
    return v or None


def read_reviews(path: Path) -> list[dict[str, Any]]:
    """One annotation dict per reviewed (non-blank-verdict) row."""
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if rows and "item_id" not in rows[0]:
        raise SystemExit(f"{path} has no item_id column — is this the review CSV?")
    out = []
    for r in rows:
        verdict = norm_verdict(r.get("REVIEW_verdict", ""))
        if verdict is None:
            continue  # not yet reviewed
        annotator = (r.get("annotator") or r.get("REVIEW_annotator")
                     or path.stem).strip()
        out.append({
            "item_id": (r.get("item_id") or "").strip(),
            "annotator": annotator,
            "faithful_use": blank_to_none(r.get("REVIEW_faithful_use", "")),
            "exists_check": blank_to_none(r.get("REVIEW_exists_check", "")),
            "verdict": verdict,
            "note": (r.get("REVIEW_note") or "").strip(),
        })
    return out


def consensus(answers: list[str | None]) -> str | None:
    """Unanimous answer, or the distinct answers joined — disagreement stays visible."""
    distinct = sorted({a for a in answers if a})
    if not distinct:
        return None
    return distinct[0] if len(distinct) == 1 else " / ".join(distinct)


def cohens_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """κ for two raters over paired labels; None on no overlap or degenerate pe=1."""
    n = len(pairs)
    if n == 0:
        return None
    labels = sorted({lab for pair in pairs for lab in pair})
    po = sum(1 for a, b in pairs if a == b) / n
    pe = sum((sum(1 for a, _ in pairs if a == lab) / n)
             * (sum(1 for _, b in pairs if b == lab) / n) for lab in labels)
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def main() -> None:
    args = parse_args()
    output = args.output or args.items.with_name(args.items.stem + "_validated.json")
    accepted_path = args.accepted or args.items.with_name(args.items.stem + "_accepted.json")

    items = json.loads(args.items.read_text(encoding="utf-8"))
    annotations: list[dict[str, Any]] = []
    for csv_path in args.reviewed:
        if not csv_path.exists():
            raise SystemExit(f"reviewed CSV not found: {csv_path}")
        annotations += read_reviews(csv_path)
    if not annotations:
        raise SystemExit("no reviewed rows (every REVIEW_verdict is blank) — "
                         "fill the sheet before ingesting.")

    # Last annotation wins per (item, annotator) — a reviewer can redo a row.
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for a in annotations:
        latest[(a["item_id"], a["annotator"])] = a
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a in latest.values():
        by_item[a["item_id"]].append(a)

    n_accepted = n_rejected = n_unannotated = 0
    no_note_rejects: list[str] = []
    validated: list[dict[str, Any]] = []
    for item in items:
        anns = by_item.get(item.get("item_id"), [])
        out = dict(item)
        if not anns:
            n_unannotated += 1
            validated.append(out)
            continue
        verdicts = [a["verdict"] for a in anns]
        accepts = verdicts.count("accept")
        if args.rule == "all":
            status = "accepted" if accepts == len(verdicts) else "rejected"
        else:
            status = "accepted" if accepts * 2 > len(verdicts) else "rejected"
        out["analyst_contextual_recoverability"] = consensus([a["faithful_use"] for a in anns])
        out["analyst_novelty"] = consensus([a["exists_check"] for a in anns])
        out["analyst_status"] = status
        out["analyst_note"] = "; ".join(f"{a['annotator']}: {a['note']}"
                                        for a in anns if a["note"]) or None
        out["analyst_annotators"] = sorted(a["annotator"] for a in anns)
        validated.append(out)
        n_accepted += status == "accepted"
        n_rejected += status == "rejected"
        if status == "rejected" and not out["analyst_note"]:
            no_note_rejects.append(item.get("item_id"))

    by_annotator: dict[str, dict[str, str]] = defaultdict(dict)
    for (iid, who), a in latest.items():
        by_annotator[who][iid] = a["verdict"]
    print(f"{len(items)} items | {len(items) - n_unannotated} reviewed by "
          f"{len(by_annotator)} reviewer(s) | rule: {args.rule}")
    print(f"accepted {n_accepted} | rejected {n_rejected} | unreviewed {n_unannotated}")
    for a1, a2 in combinations(sorted(by_annotator), 2):
        overlap = sorted(set(by_annotator[a1]) & set(by_annotator[a2]))
        kappa = cohens_kappa([(by_annotator[a1][i], by_annotator[a2][i]) for i in overlap])
        print(f"  κ({a1}, {a2}) = {kappa}  (overlap n={len(overlap)})")
    if no_note_rejects:
        print(f"note: {len(no_note_rejects)} rejected item(s) have no REVIEW_note "
              f"(e.g. {no_note_rejects[0]})")

    output.write_text(json.dumps(validated, ensure_ascii=False, indent=2), encoding="utf-8")
    accepted = [it for it in validated if it.get("analyst_status") == "accepted"]
    accepted_path.write_text(json.dumps(accepted, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nvalidated → {output}")
    print(f"accepted ({len(accepted)}) → {accepted_path}   ← Stage 2 --input")


if __name__ == "__main__":
    main()
