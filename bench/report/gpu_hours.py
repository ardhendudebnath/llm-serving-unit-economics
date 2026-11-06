"""GPU-hours spent measuring, recomputed from the committed results.

    python -m bench.report.gpu_hours

`docs/cost-log.md` publishes how many GPU-hours the measurements took. Until
this module existed that figure was arithmetic done by hand and typed into
prose, and two of its inputs -- the warm-up and cooldown each load point cost --
were recorded nowhere. This recomputes it, and says out loud which parts are
recorded and which are assumed.

Three components, each from a committed file:

  sweeps         each published load point: its measured duration, plus the
                 warm-up and cooldown it cost
  contaminated   points re-measured because the card was not in the state the
                 measurement claimed, kept under results/sweeps/contaminated/
                 as evidence
  harness        each eval run under results/eval/, from its own recorded
                 start and finish

**Warm-up and cooldown are recorded per point from now on** (see
`bench/sweep.py`). Every committed sweep predates that, so for those points
the values are assumptions, applied by the rules below, and the report counts
how many points rest on them. A total that quietly mixed recorded and assumed
seconds would be exactly the kind of number this repo refuses to publish.

Stdlib only, so it runs in the dependency-free CI job.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

SWEEPS = Path("results/sweeps")
CONTAMINATED = SWEEPS / "contaminated"
EVAL = Path("results/eval")

#: The sweep's `--warmup` default, and the warm-up every committed sweep was
#: run with. Assumed only for points that do not record their own.
LEGACY_WARMUP_S = 15.0

#: Cooldown for published points that predate recording it. 60 s is the method
#: the README states for every published point ("150 s of Poisson arrivals
#: after 60 s of idle") and the `--cooldown 60` in its reproduce command. That
#: is a statement in prose, not a recorded value -- which is the gap that
#: recording `cooldown_s` per point now closes.
LEGACY_COOLDOWN_PUBLISHED_S = 60.0

#: Cooldown for contaminated points that predate recording it. Zero: these are
#: the points measured back to back, which is why they were contaminated.
#:
#: **One file is an exception by name and is flagged rather than guessed:**
#: `sweep_short_fp16_cooled_lowrates.json`. Its name says it was cooled, the
#: published total treats it as not, and nothing recorded settles which. The
#: report prints what the total would be either way.
LEGACY_COOLDOWN_CONTAMINATED_S = 0.0
AMBIGUOUS = frozenset({"sweep_short_fp16_cooled_lowrates.json"})


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(point: dict) -> tuple:
    """What makes two recorded points the same measurement.

    Used to tell which points of an evidence copy were the ones replaced: the
    copy is the published sweep from before one point was re-measured, so every
    other point in it is byte-for-byte the published one.
    """
    lat = point.get("total_latency_s") or {}
    return (point["target_rate_rps"], point["duration_s"], lat.get("p95"),
            point.get("requests"))


@dataclass
class Component:
    name: str
    items: int = 0
    seconds: float = 0.0
    #: Points whose warm-up or cooldown had to be assumed.
    assumed: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def hours(self) -> float:
        return self.seconds / 3600.0


def _point_seconds(point: dict, legacy_cooldown_s: float) -> tuple[float, bool]:
    """A point's GPU time, and whether any of it was assumed."""
    warm = point.get("warmup_s")
    cool = point.get("cooldown_s")
    assumed = warm is None or cool is None
    warm = LEGACY_WARMUP_S if warm is None else warm
    cool = legacy_cooldown_s if cool is None else cool
    return point["duration_s"] + warm + cool, assumed


def published(root: Path = Path(".")) -> Component:
    comp = Component("published sweeps")
    for path in sorted((root / SWEEPS).glob("sweep_*.json")):
        for point in _load(path)["points"]:
            secs, assumed = _point_seconds(point, LEGACY_COOLDOWN_PUBLISHED_S)
            comp.items += 1
            comp.seconds += secs
            comp.assumed += assumed
    return comp


def contaminated(root: Path = Path(".")) -> Component:
    """Contaminated points, counting each re-measurement once.

    A file with a published twin of the same name is an evidence copy: only its
    points that differ from the twin were measured again, so only they count.
    A file with no twin is a run of its own and counts whole.
    """
    comp = Component("contaminated points")
    for path in sorted((root / CONTAMINATED).glob("*.json")):
        points = _load(path)["points"]
        twin = root / SWEEPS / path.name
        if twin.exists():
            seen = {_identity(p) for p in _load(twin)["points"]}
            points = [p for p in points if _identity(p) not in seen]
        for point in points:
            secs, assumed = _point_seconds(point, LEGACY_COOLDOWN_CONTAMINATED_S)
            comp.items += 1
            comp.seconds += secs
            comp.assumed += assumed
        if path.name in AMBIGUOUS and points:
            extra = sum(LEGACY_COOLDOWN_PUBLISHED_S for p in points if p.get("cooldown_s") is None)
            if extra:
                comp.notes.append(
                    f"{path.name}: counted uncooled; if it was cooled as its name "
                    f"says, add {extra / 3600:.2f} h"
                )
    return comp


def harness(root: Path = Path(".")) -> Component:
    """Each harness run once, by run_id.

    A pull request that rolls out a rung attaches a *copy* of a run already
    committed under that rung's directory -- the gate's convention is to judge
    the newest file in results/eval/. The copy is the same run, not more GPU
    time, so it is counted once.
    """
    comp = Component("harness runs")
    seen: set[str] = set()
    for path in sorted((root / EVAL).rglob("*_shared.json")):
        run = _load(path)
        run_id = run.get("run_id", path.stem)
        if run_id in seen:
            continue
        seen.add(run_id)
        start = datetime.fromisoformat(run["started_at"])
        end = datetime.fromisoformat(run["finished_at"])
        comp.items += 1
        comp.seconds += (end - start).total_seconds()
    return comp


def breakdown(root: Path = Path(".")) -> list[Component]:
    return [published(root), contaminated(root), harness(root)]


def hours_2dp(seconds: float) -> Decimal:
    """Hours to two places, half-up -- the way the cost log rounds.

    Python's round() is half-to-even on a binary float, and the harness total
    (1,746 s) sits exactly on 0.485 h, where the two conventions disagree.
    """
    return (Decimal(str(seconds)) / Decimal(3600)).quantize(Decimal("0.01"), ROUND_HALF_UP)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    parts = breakdown(args.root)
    total = sum(p.seconds for p in parts)

    if args.json:
        print(json.dumps({
            "components": [
                {"name": p.name, "items": p.items, "hours": round(p.hours, 4),
                 "assumed_points": p.assumed, "notes": p.notes}
                for p in parts
            ],
            "total_hours": round(total / 3600, 4),
        }, indent=2))
        return 0

    print()
    for p in parts:
        flag = f"   ({p.assumed} with assumed warm-up/cooldown)" if p.assumed else ""
        print(f"  {p.name:<20} {p.items:>3}  {hours_2dp(p.seconds):>5} h{flag}")
    print(f"  {'total':<20} {'':>3}  {hours_2dp(total):>5} h")
    for note in (n for p in parts for n in p.notes):
        print(f"\n  note: {note}")
    assumed = sum(p.assumed for p in parts)
    if assumed:
        print(f"\n  {assumed} point(s) predate per-point warm-up/cooldown recording;")
        print(f"  for those, warm-up {LEGACY_WARMUP_S:g} s and cooldown "
              f"{LEGACY_COOLDOWN_PUBLISHED_S:g} s (published) / "
              f"{LEGACY_COOLDOWN_CONTAMINATED_S:g} s (contaminated) are assumed.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
