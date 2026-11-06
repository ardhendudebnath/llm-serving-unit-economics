"""The cost log's GPU-hour figures are the ones the committed files produce.

Until bench/report/gpu_hours.py existed, docs/cost-log.md's 4.72 GPU-hours was
hand arithmetic in prose -- the one published figure in the repo that no test
recomputed. These recompute it from results/, and check that recorded
warm-up and cooldown take precedence over the legacy assumptions.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path

import pytest

from bench.config import LAPTOP, USD_TO_INR
from bench.report import gpu_hours as gh

ROOT = Path(__file__).resolve().parent.parent
COST_LOG = re.sub(r"\s+", " ", (ROOT / "docs" / "cost-log.md").read_text(encoding="utf-8"))


def _row(label: str) -> tuple[int, Decimal]:
    m = re.search(rf"\| {label}[^|]*\| (\d+) \w+ \| ([\d.]+) \|", COST_LOG)
    assert m, f"cost-log.md no longer has a {label!r} row"
    return int(m.group(1)), Decimal(m.group(2))


# ------------------------------------------------- the published figures --

@pytest.mark.parametrize(("label", "component"), [
    ("Published sweeps", gh.published),
    ("Contaminated points", gh.contaminated),
    ("Harness runs", gh.harness),
])
def test_each_cost_log_row_is_what_the_files_give(label, component):
    items, hours = _row(label)
    comp = component(ROOT)
    assert (comp.items, gh.hours_2dp(comp.seconds)) == (items, hours)


def test_the_published_total_is_what_the_files_give():
    m = re.search(r"\*\*([\d.]+) GPU-hours\*\*", COST_LOG)
    assert m, "cost-log.md no longer states a GPU-hour total"
    total = sum(c.seconds for c in gh.breakdown(ROOT))
    assert gh.hours_2dp(total) == Decimal(m.group(1))


def test_the_rupee_figures_follow_from_the_hours():
    total_h = sum(c.seconds for c in gh.breakdown(ROOT)) / 3600
    contaminated_h = gh.contaminated(ROOT).hours
    rate = LAPTOP.priced_gpu().market_usd_per_hour * USD_TO_INR

    worth = re.search(r"\*\*about ₹(\d+)\*\*", COST_LOG)
    share = re.search(r"contaminated points are (\d+) % of it", COST_LOG)
    wasted = re.search(r"about ₹(\d+) of GPU time spent measuring", COST_LOG)
    assert worth and share and wasted, "the cost log's rupee figures were reworded"

    assert int(worth.group(1)) == round(total_h * rate)
    assert int(share.group(1)) == round(contaminated_h / total_h * 100)
    assert int(wasted.group(1)) == round(contaminated_h * rate)


def test_the_legacy_points_are_counted_as_assumed_not_passed_off_as_recorded():
    # Every committed sweep predates per-point recording. That has to show in
    # the output, or the total reads as fully measured when 64 points of it are
    # not.
    parts = gh.breakdown(ROOT)
    assert sum(p.assumed for p in parts[:2]) == sum(p.items for p in parts[:2])


def test_the_ambiguous_file_is_flagged_rather_than_guessed():
    notes = gh.contaminated(ROOT).notes
    assert any("cooled_lowrates" in n for n in notes)


# ------------------------------------------------------------- the rules --

def _sweep(points: list[dict]) -> dict:
    return {"profile": "short", "precision": "fp16", "points": points}


def _pt(rate: float, duration: float, p95: float, **extra) -> dict:
    return {"target_rate_rps": rate, "duration_s": duration, "requests": 10,
            "total_latency_s": {"p95": p95}, **extra}


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_recorded_warmup_and_cooldown_beat_the_assumptions(tmp_path):
    _write(tmp_path / gh.SWEEPS / "sweep_short_fp16.json", _sweep([
        _pt(1.0, 100.0, 1.0, warmup_s=5.0, cooldown_s=0.0),  # recorded
        _pt(2.0, 100.0, 1.0),                                # legacy
    ]))
    comp = gh.published(tmp_path)
    assert comp.seconds == 100 + 5 + 0 + 100 + gh.LEGACY_WARMUP_S + gh.LEGACY_COOLDOWN_PUBLISHED_S
    assert comp.assumed == 1


def test_an_evidence_copy_counts_only_the_points_it_replaced(tmp_path):
    shared = _pt(1.0, 100.0, 1.0)
    _write(tmp_path / gh.SWEEPS / "sweep_short_fp16.json",
           _sweep([shared, _pt(2.0, 100.0, 1.1)]))
    _write(tmp_path / gh.CONTAMINATED / "sweep_short_fp16.json",
           _sweep([shared, _pt(2.0, 100.0, 37.5)]))   # the hot point
    _write(tmp_path / gh.CONTAMINATED / "sweep_short_fp16_extra.json",
           _sweep([_pt(0.5, 50.0, 2.0)]))             # no twin: counts whole
    comp = gh.contaminated(tmp_path)
    assert comp.items == 2
    assert comp.seconds == (100 + gh.LEGACY_WARMUP_S) + (50 + gh.LEGACY_WARMUP_S)


def test_hours_round_half_up_like_the_cost_log():
    # 1,746 s is exactly 0.485 h; Python's round() would print 0.48.
    assert gh.hours_2dp(1746) == Decimal("0.49")
