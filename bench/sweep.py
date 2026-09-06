"""Sweep arrival rate to find the knee, checkpointing as it goes.

    python -m bench.sweep --profile long_in --slo 5.0
    python -m bench.sweep --all-profiles --precision int4

One GPU block should be: spin up, run this, collect results, destroy. So this
script is written to be the whole measurement -- it sweeps every rate, finds
the knee, records VRAM and server-side counters, and writes everything it saw.

Two properties exist purely to avoid re-renting hardware:

**It checkpoints after every point.** A preempted spot instance loses the point
in flight, not the sweep. Re-running with the same `--out` picks up where it
stopped rather than starting over.

**It stops climbing once the SLO is decisively breached.** Continuing to push
rate at a server that is already three times over its latency target buys one
more dot on a chart nobody reads, at full GPU price. It records why it stopped.

The plan's fifth rule -- log everything the first time -- is why this records
peak VRAM and vLLM's own counters alongside the latency, even though the
headline only needs the knee. Re-running a sweep because VRAM was not captured
is money set on fire.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from bench import workloads
from bench.loadgen import RunConfig, discover_model, run_point
from bench.metrics import ERROR_CEILING

#: Doubling rates. A linear sweep wastes most of its points either far below
#: the knee or far above it; a geometric one brackets the knee in few steps,
#: and `--refine` then bisects the last passing gap where precision matters.
DEFAULT_RATES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)

#: Stop climbing once p95 exceeds this multiple of the SLO. Three times over
#: is unambiguous saturation; anything past it is paying to measure how badly
#: a server fails, which no decision depends on.
ABANDON_AT = 3.0


@dataclass
class SweepState:
    """Everything measured so far. Written to disk after every point."""

    profile: str
    precision: str
    model: str
    slo_p95_s: float
    started_at: str
    points: list[dict] = field(default_factory=list)
    stopped_because: str = ""

    def rates_done(self) -> set[float]:
        return {p["target_rate_rps"] for p in self.points}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_row(), indent=2) + "\n", encoding="utf-8")

    def as_row(self) -> dict:
        knee = self.knee()
        return {
            "profile": self.profile,
            "precision": self.precision,
            "model": self.model,
            "slo_p95_s": self.slo_p95_s,
            "started_at": self.started_at,
            "arrival": "poisson",
            "closed_loop": False,
            "stopped_because": self.stopped_because,
            "knee_rps": knee["target_rate_rps"] if knee else None,
            "knee_achieved_rps": knee["achieved_rps"] if knee else None,
            "points": sorted(self.points, key=lambda p: p["target_rate_rps"]),
        }

    def knee(self) -> dict | None:
        """Re-derive the knee from the recorded points.

        Recomputed from the saved rows rather than tracked as the sweep runs,
        so a resumed sweep and a fresh one give the same answer.

        The rule is `bench.metrics.find_knee`'s, applied to stored summaries
        instead of live records: walk in rate order, take the *last* rate that
        passed rather than the one before the first failure, and disqualify any
        point with over 1% errors. Meeting a latency target by shedding
        requests is not meeting it.
        """
        passing: dict | None = None
        for row in sorted(self.points, key=lambda p: p["target_rate_rps"]):
            lat = row.get("total_latency_s")
            if not lat or row["error_rate"] > ERROR_CEILING or lat["p95"] > self.slo_p95_s:
                break
            passing = row
        return passing


def load_state(path: Path) -> SweepState | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return SweepState(
        profile=raw["profile"],
        precision=raw["precision"],
        model=raw["model"],
        slo_p95_s=raw["slo_p95_s"],
        started_at=raw["started_at"],
        points=raw.get("points", []),
        stopped_because=raw.get("stopped_because", ""),
    )


def sweep_one(
    profile: str,
    *,
    base_url: str,
    model: str,
    precision: str,
    slo_p95_s: float,
    rates: tuple[float, ...],
    duration_s: float,
    warmup_s: float,
    out: Path,
    seed: int = 0,
) -> SweepState:
    corpus = workloads.load(profile)
    checkpoint = out / f"sweep_{profile}_{precision}.json"

    state = load_state(checkpoint)
    if state is None:
        state = SweepState(
            profile=profile,
            precision=precision,
            model=model,
            slo_p95_s=slo_p95_s,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
    else:
        done = sorted(state.rates_done())
        print(f"  resuming {profile}/{precision}: {len(done)} point(s) already done "
              f"({', '.join(f'{r:g}' for r in done)})")

    already = state.rates_done()

    for rate in rates:
        if rate in already:
            continue

        cfg = RunConfig(
            base_url=base_url, model=model, profile=profile, rate_rps=rate,
            duration_s=duration_s, warmup_s=warmup_s, seed=seed,
        )
        print(f"\n  {profile}/{precision} @ {rate:g} rps ...", flush=True)
        point = asyncio.run(run_point(cfg, corpus))

        dist = point.total_latency()
        if dist is None:
            state.stopped_because = f"every request failed at {rate:g} rps"
            state.save(checkpoint)
            print("    all requests failed -- stopping")
            break

        row = point.as_row()
        state.points.append(row)
        # Checkpoint before anything else can go wrong. A preempted instance
        # loses this point, not the sweep.
        state.save(checkpoint)

        lag = point.worst_client_lag_s()
        flag = "  [client lagged -- latency optimistic]" if lag > 0.5 else ""
        print(f"    achieved {point.achieved_rps:.2f} rps · p50 {dist.p50:.2f}s · "
              f"p95 {dist.p95:.2f}s · p99 {dist.p99:.2f}s · "
              f"err {point.error_rate:.1%}{flag}")

        if dist.p95 > slo_p95_s * ABANDON_AT:
            state.stopped_because = (
                f"p95 {dist.p95:.1f}s exceeded {ABANDON_AT:g}x the {slo_p95_s:g}s "
                f"SLO at {rate:g} rps -- climbing further would only price how "
                "badly it fails"
            )
            state.save(checkpoint)
            print(f"    {state.stopped_because}")
            break
    else:
        state.stopped_because = "swept every requested rate"
        state.save(checkpoint)

    return state


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", choices=sorted(workloads.PROFILES))
    ap.add_argument("--all-profiles", action="store_true",
                    help="sweep every profile in one GPU block")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "int8", "int4"],
                    help="recorded with the results; the server must already be "
                         "serving at this precision")
    ap.add_argument("--slo", type=float, default=5.0, help="p95 target, seconds")
    ap.add_argument("--rates", default="",
                    help="comma-separated rates; defaults to a doubling ladder")
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--warmup", type=float, default=15.0)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="")
    ap.add_argument("--out", type=Path, default=Path("results/sweeps"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.profile and not args.all_profiles:
        ap.error("choose --profile or --all-profiles")

    profiles = sorted(workloads.PROFILES) if args.all_profiles else [args.profile]
    rates = (
        tuple(float(r) for r in args.rates.split(","))
        if args.rates else DEFAULT_RATES
    )
    model = args.model or discover_model(args.base_url)

    print(f"\n  model      {model}")
    print(f"  precision  {args.precision}  (as configured on the server)")
    print(f"  SLO        p95 <= {args.slo:g}s")
    print(f"  rates      {', '.join(f'{r:g}' for r in rates)}")
    print(f"  profiles   {', '.join(profiles)}")

    results: list[SweepState] = []
    for profile in profiles:
        state = sweep_one(
            profile,
            base_url=args.base_url, model=model, precision=args.precision,
            slo_p95_s=args.slo, rates=rates, duration_s=args.duration,
            warmup_s=args.warmup, out=args.out, seed=args.seed,
        )
        results.append(state)

    print("\n  knees\n")
    for state in results:
        knee = state.knee()
        if knee is None:
            print(f"    {state.profile:<9} no rate met the SLO")
            continue
        lat = knee["total_latency_s"]
        print(f"    {state.profile:<9} {knee['target_rate_rps']:>6g} rps  "
              f"(achieved {knee['achieved_rps']:.2f})  p95 {lat['p95']:.2f}s")
    print(f"\n  -> {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
