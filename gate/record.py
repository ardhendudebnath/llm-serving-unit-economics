"""Record a baseline, and measure the noise floor that makes it usable.

    python -m gate.record --runs results/eval/*.json --precision fp16

Takes several runs of the **same configuration** and writes `gate/baseline.json`
from them. The scores are the mean across runs; the tolerance is the observed
spread. Both matter, but the spread is the one that makes the gate honest --
see the module docstring in `gate/compare.py` for why a chosen threshold does
not work here.

Repeats must differ only in random seed. Feeding this runs from two different
precisions would record the difference between them as noise, and the gate
would then wave through exactly the regression it exists to catch -- so the
provenance fields are checked and a mismatch is an error.
"""

from __future__ import annotations

import argparse
import glob
import json
from datetime import UTC, datetime
from pathlib import Path

from gate.compare import DEFAULT_BASELINE, METRICS, MIN_REPEATS, GateError


def _load_runs(patterns: list[str]) -> list[dict]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(p) for p in sorted(glob.glob(pattern)))
    if not paths:
        raise GateError(f"no run files matched {patterns}")

    runs = []
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not raw.get("summary"):
            raise GateError(f"{path} has no summary block")
        raw["_path"] = str(path)
        runs.append(raw)
    return runs


def _check_same_configuration(runs: list[dict]) -> None:
    """Every run must be the same thing measured repeatedly."""
    for field_name in ("dataset_sha", "prompt_version", "model_key"):
        values = {r.get(field_name, "") for r in runs}
        if len(values) > 1:
            raise GateError(
                f"runs disagree on {field_name}: {sorted(values)}. A noise "
                "floor measured across different configurations would record "
                "the difference between them as noise, and the gate would "
                "then pass a real regression of that size."
            )


def build(runs: list[dict], precision: str, note: str = "") -> dict:
    if len(runs) < MIN_REPEATS:
        raise GateError(
            f"{len(runs)} run(s) given; at least {MIN_REPEATS} are needed. "
            "Two runs give one gap, which is not a spread."
        )
    _check_same_configuration(runs)

    scores: dict[str, float] = {}
    noise: dict[str, float] = {}
    observed: dict[str, list[float]] = {}

    for metric in METRICS:
        values = [
            float(r["summary"][metric.key])
            for r in runs
            if metric.key in r["summary"]
        ]
        if len(values) != len(runs):
            continue
        scale = 100.0 if metric.key.endswith(("_acc", "_rate", "_f1")) else 1.0
        scores[metric.key] = sum(values) / len(values)
        # Spread, not standard deviation. With three to five runs a standard
        # deviation is barely estimable, and the range is what a reviewer can
        # check against the per-run numbers printed below it.
        noise[metric.key] = (max(values) - min(values)) * scale
        observed[metric.key] = [round(v * scale, 2) for v in values]

    first = runs[0]
    return {
        "model": first.get("served_model_id") or first.get("model_id", ""),
        "precision": precision,
        "dataset_sha": first.get("dataset_sha", ""),
        "prompt_version": first.get("prompt_version", ""),
        "recorded_at": datetime.now(UTC).date().isoformat(),
        "n_repeats": len(runs),
        "scores": {k: round(v, 6) for k, v in scores.items()},
        "noise": {k: round(v, 3) for k, v in noise.items()},
        # Kept so a reviewer can see the runs the tolerance came from rather
        # than taking the spread on trust.
        "observed_per_run": observed,
        # `_path` is stamped on by _load_runs. build() is callable with plain
        # run dicts too, so its absence is normal rather than an error.
        "source_runs": [Path(p).name for r in runs if (p := r.get("_path"))],
        "note": note,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="glob(s) matching repeat runs of one configuration")
    ap.add_argument("--precision", required=True, choices=["fp16", "int8", "int4"])
    ap.add_argument("--out", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    try:
        runs = _load_runs(args.runs)
        baseline = build(runs, args.precision, args.note)
    except GateError as exc:
        print(f"\n  cannot record a baseline: {exc}\n")
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")

    print(f"\n  baseline from {baseline['n_repeats']} runs -> {args.out}\n")
    for metric in METRICS:
        if metric.key not in baseline["scores"]:
            continue
        scale = 100.0 if metric.key.endswith(("_acc", "_rate", "_f1")) else 1.0
        runs_str = ", ".join(f"{v:g}" for v in baseline["observed_per_run"][metric.key])
        print(f"    {metric.label:<24} {baseline['scores'][metric.key] * scale:>6.1f} "
              f"± {baseline['noise'][metric.key]:<5.1f}  [{runs_str}]")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
