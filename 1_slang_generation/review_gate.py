# Inspect saved contamination-gate scores without making API or search calls.
# Prints threshold sweeps and the items nearest a chosen cutoff. Apply a new
# threshold by rerunning contamination-gate.py; cached evidence is reused.
#
#   python review_gate.py --input output-data/run/zh_affixation.json --gate guess
#   python review_gate.py --input ...                --gate guess --threshold 0.55
#   python review_gate.py --input ...                --gate exist --show 20

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# gate -> (record field holding the score, recorded-threshold field, human label)
GATES = {
    "guess": ("gate2_guess_sim_avg", "gate2_sim_threshold", "avg cosine(def, guess)"),
    "exist": ("gate1_sources", "gate1_threshold", "independent in-sense sources"),
}

# record fields you can break the sweep out by — language track and slang type both
# have different leakage profiles, so a single pooled cutoff blends populations.
GROUP_FIELDS = ("lang", "type")


def load(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    return json.loads(text) if text else []


def score_of(rec: dict[str, Any], gate: str) -> float | None:
    """The gate's tunable value for one record (source COUNT for exist; raw avg for
    guess). None if the record never reached this gate."""
    raw = rec.get(GATES[gate][0])
    if raw is None:
        return None
    return float(len(raw)) if gate == "exist" else float(raw)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Review/tune a contamination-gate run from its saved output — no API."
    )
    p.add_argument("--input", type=Path,
                   help="raw candidates path; _pass.json & _rejected.json are derived from it")
    p.add_argument("--pass-file", type=Path, default=None, help="override the _pass.json path")
    p.add_argument("--rejected", type=Path, default=None, help="override the _rejected.json path")
    p.add_argument("--gate", choices=tuple(GATES), default="guess")
    p.add_argument("--threshold", default=None,
                   help="preview this cutoff (default: the one recorded in the run). For "
                        "the guess gate a per-language spec 'zh=0.55,ru=0.65' is allowed, "
                        "resolved per group when grouping by lang")
    p.add_argument("--show", type=int, default=12,
                   help="how many items nearest the cutoff to list, per group (default 12)")
    p.add_argument("--by", default="lang,type",
                   help="break the sweep out by these record fields (comma list of "
                        f"{', '.join(GROUP_FIELDS)}; 'none' to pool). Default: lang,type")
    return p.parse_args()


def parse_lang_thresholds(spec: str) -> dict[str, float]:
    """'0.6' -> {'default':0.6}; 'zh=0.55,ru=0.65' -> per-language. Mirrors the same
    parser in contamination-gate.py so preview matches enforcement."""
    spec = str(spec).strip()
    if "=" not in spec:
        return {"default": float(spec)}
    out: dict[str, float] = {}
    for part in spec.split(","):
        key, _, val = part.partition("=")
        out[key.strip()] = float(val)
    return out


def parse_by(spec: str) -> list[str]:
    if not spec or spec.strip().lower() == "none":
        return []
    keys = [k.strip() for k in spec.split(",") if k.strip()]
    bad = [k for k in keys if k not in GROUP_FIELDS]
    if bad:
        raise SystemExit(f"--by: unknown {bad}; allowed: {', '.join(GROUP_FIELDS)} or 'none'")
    return keys


def resolve_paths(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    pass_path, rejected_path = args.pass_file, args.rejected
    if args.input:
        stem = args.input.stem
        pass_path = pass_path or args.input.with_name(f"{stem}_pass.json")
        rejected_path = rejected_path or args.input.with_name(f"{stem}_rejected.json")
    return pass_path, rejected_path


def print_sweep(gate: str, scored: list[dict[str, Any]], threshold: float) -> None:
    vals = sorted(score_of(r, gate) for r in scored)
    n = len(vals)
    label = GATES[gate][2]
    print(f"\n{label}: n={n}  min {vals[0]:.3f}  median {vals[n // 2]:.3f}  "
          f"mean {sum(vals) / n:.3f}  max {vals[-1]:.3f}")
    if gate == "exist":
        grid: list[float] = [float(t) for t in range(1, int(vals[-1]) + 1)]
        fmt = lambda v: f"{int(v)}"
    else:
        grid = sorted({round(0.30 + 0.05 * i, 2) for i in range(11)} | {round(threshold, 2)})
        fmt = lambda v: f"{v:.2f}"
    cells = [f"{fmt(t)}{'*' if abs(t - threshold) < 1e-9 else ''}→{sum(1 for v in vals if v >= t)}"
             for t in grid]
    print("  removed at threshold:  " + "   ".join(cells))
    print(f"  (* = previewed threshold {fmt(threshold)}; raise to remove fewer, lower to remove more)")


def print_borderline(gate: str, scored: list[dict[str, Any]], threshold: float, show: int) -> None:
    """The items closest to the cutoff — where a threshold tweak flips PASS/FAIL, so this
    is what to eyeball. Sorted by |score - threshold|."""
    ranked = sorted(scored, key=lambda r: abs(score_of(r, gate) - threshold))[:show]
    print(f"\n{show} items nearest the cutoff (✕ = would be removed at {threshold:g}):")
    for r in ranked:
        v = score_of(r, gate)
        mark = "✕" if v >= threshold else "·"
        was = r.get("gate_failed", "kept")   # actual disposition from the saved run
        print(f"  {mark} {v:.3f}  {r.get('term')}  [{was}]  {r.get('definition_tar_lang', '')[:60]}")
        if gate == "guess":
            for g in r.get("gate2_guesses", []):
                print(f"        guess: {g[:80]}")
        else:
            srcs = r.get("gate1_sources", [])
            print(f"        sources({len(srcs)}): {', '.join(srcs) or '—'}")


def main() -> None:
    args = parse_args()
    pass_path, rejected_path = resolve_paths(args)
    records = load(pass_path) + load(rejected_path)
    scored = [r for r in records if score_of(r, args.gate) is not None]
    if not scored:
        raise SystemExit(
            f"no items carry the {args.gate} score in {pass_path} / {rejected_path}. "
            "Run contamination-gate.py with that gate first."
        )

    spec = parse_lang_thresholds(args.threshold) if args.threshold else None
    rec_field = GATES[args.gate][1]
    hard_default = 3.0 if args.gate == "exist" else 0.6

    def threshold_for(group: list[dict[str, Any]]) -> float:
        """The cutoff to preview for this group. With no --threshold, use what the run
        recorded (now per-language). With a per-language --threshold, resolve to the
        group's language when the group is a single language; else fall to 'default'."""
        recorded = next((r.get(rec_field) for r in group if r.get(rec_field) is not None),
                        hard_default)
        if spec is None:
            t = recorded
        else:
            langs = {r.get("lang") for r in group}
            lang = next(iter(langs)) if len(langs) == 1 else None
            t = spec.get(lang, spec.get("default", recorded)) if lang else spec.get("default", recorded)
        return int(round(t)) if args.gate == "exist" else float(t)

    print(f"{len(scored)} items scored at gate '{args.gate}'  "
          f"(from {pass_path} + {rejected_path})")

    by = parse_by(args.by)
    if not by:
        t = threshold_for(scored)
        print_sweep(args.gate, scored, t)
        print_borderline(args.gate, scored, t, args.show)
    else:
        groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for r in scored:
            groups.setdefault(tuple(str(r.get(k, "?")) for k in by), []).append(r)
        for key in sorted(groups):
            label = "  ".join(f"{k}={v}" for k, v in zip(by, key))
            t = threshold_for(groups[key])
            print(f"\n{'═' * 4} {label}  (n={len(groups[key])}, cutoff {t:g}) {'═' * 4}")
            print_sweep(args.gate, groups[key], t)
            print_borderline(args.gate, groups[key], t, args.show)

    flag = "guess-sim-threshold" if args.gate == "guess" else "in-sense-threshold"
    print(f"\nTo apply: re-run contamination-gate.py with --{flag} "
          f"{args.threshold or '<value, e.g. zh=0.55,ru=0.65>'}  "
          "(cached evidence → no API calls repeat).")


if __name__ == "__main__":
    main()
