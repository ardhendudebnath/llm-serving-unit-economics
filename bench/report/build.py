"""Render the published charts from whatever has actually been measured.

    python -m bench.report.build --gpu rtx5070ti-laptop --api haiku-4-5

Reads `results/sweeps/*.json` and writes to `docs/charts/`. Deliberately
partial: it renders what it can and says plainly what it could not, rather than
failing entirely or -- far worse -- filling a gap with a default.

The crossover chart in particular is refused rather than approximated when the
GPU has no rate read from a provider, when no load point met the SLO, or when
the API model carries no dated price in Project 01's registry. Each of those is
a missing measurement, and a chart drawn over one is a chart that invents the
thing it exists to show.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from bench.config import GPUS, LAPTOP, get_gpu, priced
from bench.cost import ApiPricing, Capacity, UnpricedError, api_pricing_from_harness
from bench.report.charts import crossover_chart, latency_vs_load_chart

SWEEPS = Path("results/sweeps")
OUT = Path("docs/charts")

#: Profile order, so the latency panels always read short -> long_in ->
#: long_out rather than in whatever order the filesystem returns.
PROFILE_ORDER = ("short", "long_in", "long_out")


def load_sweeps(directory: Path, precision: str | None = None) -> list[dict]:
    if not directory.exists():
        return []
    sweeps = []
    for path in sorted(directory.glob("sweep_*.json")):
        sweep = json.loads(path.read_text(encoding="utf-8"))
        if precision and sweep.get("precision") != precision:
            continue
        if sweep.get("points"):
            sweeps.append(sweep)
    return sorted(
        sweeps,
        key=lambda s: (
            PROFILE_ORDER.index(s["profile"])
            if s["profile"] in PROFILE_ORDER
            else len(PROFILE_ORDER)
        ),
    )


def capacity_from(sweep: dict, gpu_key: str) -> Capacity | None:
    """Build a Capacity from a sweep's measured knee.

    None when no arrival rate met the SLO. That is a real outcome -- the
    deployment cannot serve this profile at this target at any rate -- and it
    must not be turned into a capacity of zero, which would divide into an
    infinite cost.
    """
    knee_rps = sweep.get("knee_rps")
    if not knee_rps:
        return None

    knee = next(
        (p for p in sweep["points"] if p["target_rate_rps"] == knee_rps), None
    )
    if knee is None:
        return None

    # Token counts come from the server's own usage accounting at the knee, so
    # both sides of the comparison are charged for the same work.
    tokens_in = knee.get("mean_tokens_in")
    tokens_out = knee.get("mean_tokens_out")
    if not tokens_in or not tokens_out:
        # A sweep recorded before these were captured. Refused rather than
        # defaulted: guessing the token counts would silently move the API
        # side of the crossover, which is the whole comparison.
        return None

    return Capacity(
        gpu=get_gpu(gpu_key),
        profile=sweep["profile"],
        precision=sweep["precision"],
        knee_rps=float(knee_rps),
        slo_p95_s=float(sweep["slo_p95_s"]),
        mean_tokens_in=float(tokens_in),
        mean_tokens_out=float(tokens_out),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweeps", type=Path, default=SWEEPS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--gpu", default="rtx5070ti-laptop", choices=sorted(GPUS))
    ap.add_argument("--api", default="haiku-4-5",
                    help="model key in Project 01's registry to compare against")
    ap.add_argument("--precision", default=None,
                    help="restrict to one rung of the ladder")
    ap.add_argument("--profile", default="long_in",
                    help="which profile the crossover chart is drawn for")
    ap.add_argument("--peak-to-mean", type=float, default=1.0)
    ap.add_argument("--currency", default="INR", choices=["INR", "USD"])
    args = ap.parse_args()

    sweeps = load_sweeps(args.sweeps, args.precision)
    if not sweeps:
        print(
            f"\n  no sweeps in {args.sweeps}.\n"
            "  Nothing has been measured yet, so there is nothing to chart.\n"
            "      python -m bench.sweep --all-profiles\n"
        )
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    skipped: list[str] = []

    # --- latency vs load ---------------------------------------------------
    written.append(latency_vs_load_chart(sweeps, args.out / "latency-vs-load.png"))

    # --- the crossover chart -----------------------------------------------
    target = next((s for s in sweeps if s["profile"] == args.profile), None)
    if target is None:
        skipped.append(f"crossover: no sweep for profile {args.profile!r}")
    else:
        capacity = capacity_from(target, args.gpu)
        if capacity is None:
            skipped.append(
                f"crossover: {args.profile} has no priceable knee -- either no "
                f"arrival rate met the {target['slo_p95_s']:g}s SLO, or the "
                "sweep predates per-request token accounting and would have to "
                "be re-run"
            )
        else:
            if args.gpu == LAPTOP.gpu_key:
                # One chart, priced around the clock like a rented card. Idle
                # time is already on the chart as low utilisation; pricing the
                # laptop at a partial duty cycle too would count it twice. See
                # OwnedHardware.priced_gpu().
                capacity = replace(capacity, gpu=LAPTOP.priced_gpu())

            if not priced(capacity.gpu):
                skipped.append(
                    f"crossover: {args.gpu} has no hourly rate read from a "
                    "provider. Set market_usd_per_hour and rate_read_on in "
                    "bench/config.py, or describe it as OwnedHardware if it "
                    "cannot be rented"
                )
            else:
                try:
                    api: ApiPricing = api_pricing_from_harness(args.api)
                except UnpricedError as exc:
                    skipped.append(f"crossover: {exc}")
                else:
                    written.append(crossover_chart(
                        capacity, api, args.out / "crossover.png",
                        peak_to_mean=args.peak_to_mean, currency=args.currency,
                    ))

    print(f"\n  charts -> {args.out}\n")
    for path in written:
        print(f"    written  {path.name}")
    for reason in skipped:
        print(f"    SKIPPED  {reason}")
    print()

    # Exit 0: a partial report is the correct output when only part of the
    # measurement exists. The skips are printed, not swallowed.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
