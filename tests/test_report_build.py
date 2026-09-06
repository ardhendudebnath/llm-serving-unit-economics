from __future__ import annotations

import json

from bench.metrics import LoadPoint, RequestRecord
from bench.report.build import capacity_from, load_sweeps


def _point(rate: float, p95: float, *, tokens_in=1800.0, tokens_out=40.0) -> dict:
    return {
        "profile": "long_in",
        "target_rate_rps": rate,
        "achieved_rps": rate * 0.96,
        "duration_s": 120.0,
        "completed": 100,
        "error_rate": 0.0,
        "mean_tokens_in": tokens_in,
        "mean_tokens_out": tokens_out,
        "total_latency_s": {"p50": p95 * 0.5, "p95": p95, "p99": p95 * 1.2},
    }


def _sweep(profile: str = "long_in", *, knee: float | None = 2.0, **over) -> dict:
    sweep = {
        "profile": profile,
        "precision": "fp16",
        "model": "test",
        "slo_p95_s": 5.0,
        "knee_rps": knee,
        "points": [_point(1.0, 1.4), _point(2.0, 3.1), _point(4.0, 12.0)],
    }
    sweep.update(over)
    return sweep


# ------------------------------------------------------------- token log ----

def test_mean_token_counts_are_recorded_in_every_load_point():
    """The gap that running the pipeline exposed.

    The cost model prices the API side per token, so both sides have to be
    charged for the same work. These are not derivable from anything else in
    the summary, so a sweep that omitted them would have to be re-run --
    exactly the waste the plan's "log everything the first time" rule exists
    to prevent.
    """
    records = [
        RequestRecord(profile="long_in", queued_at=0.0, sent_at=0.0, done_at=1.0,
                      tokens_in=1800, tokens_out=40, status=200),
        RequestRecord(profile="long_in", queued_at=0.0, sent_at=0.0, done_at=1.0,
                      tokens_in=2200, tokens_out=60, status=200),
        # A failure must not drag the mean down -- it consumed no tokens.
        RequestRecord(profile="long_in", queued_at=0.0, sent_at=0.0, done_at=1.0,
                      tokens_in=0, tokens_out=0, status=500, error="boom"),
    ]
    point = LoadPoint(profile="long_in", target_rate_rps=2.0, duration_s=60.0,
                      records=records)

    assert point.mean_tokens_in == 2000.0
    assert point.mean_tokens_out == 50.0
    row = point.as_row()
    assert row["mean_tokens_in"] == 2000.0
    assert row["mean_tokens_out"] == 50.0


def test_mean_tokens_are_zero_rather_than_dividing_by_zero():
    point = LoadPoint(profile="short", target_rate_rps=1.0, duration_s=10.0,
                      records=[])
    assert point.mean_tokens_in == 0.0
    assert point.mean_tokens_out == 0.0


# --------------------------------------------------------------- capacity --

def test_capacity_is_built_from_the_measured_knee():
    cap = capacity_from(_sweep(), "rtx5070ti-laptop")
    assert cap is not None
    assert cap.knee_rps == 2.0
    assert cap.mean_tokens_in == 1800.0
    assert cap.mean_tokens_out == 40.0
    assert cap.slo_p95_s == 5.0


def test_no_knee_yields_no_capacity_rather_than_a_capacity_of_zero():
    # "No rate met the SLO" is a real outcome. A capacity of zero would divide
    # into an infinite cost and render as a chart.
    assert capacity_from(_sweep(knee=None), "rtx5070ti-laptop") is None


def test_a_sweep_without_token_accounting_is_refused_not_defaulted():
    # Guessing token counts would silently move the API side of the crossover,
    # which is the entire comparison.
    sweep = _sweep()
    for p in sweep["points"]:
        p.pop("mean_tokens_in")
        p.pop("mean_tokens_out")
    assert capacity_from(sweep, "rtx5070ti-laptop") is None


def test_a_knee_naming_a_rate_that_was_never_swept_yields_nothing():
    assert capacity_from(_sweep(knee=99.0), "rtx5070ti-laptop") is None


# ---------------------------------------------------------------- loading --

def test_sweeps_load_in_profile_order_not_filesystem_order(tmp_path):
    # short -> long_in -> long_out, so the latency panels always read in
    # increasing-work order regardless of how the files sort.
    for profile in ("long_out", "short", "long_in"):
        (tmp_path / f"sweep_{profile}_fp16.json").write_text(
            json.dumps(_sweep(profile)), encoding="utf-8"
        )
    assert [s["profile"] for s in load_sweeps(tmp_path)] == [
        "short", "long_in", "long_out"
    ]


def test_empty_sweeps_are_skipped(tmp_path):
    # A checkpoint written before the first point completed carries no data.
    (tmp_path / "sweep_short_fp16.json").write_text(
        json.dumps(_sweep("short") | {"points": []}), encoding="utf-8"
    )
    assert load_sweeps(tmp_path) == []


def test_precision_filter_selects_one_rung(tmp_path):
    for precision in ("fp16", "int4"):
        (tmp_path / f"sweep_long_in_{precision}.json").write_text(
            json.dumps(_sweep() | {"precision": precision}), encoding="utf-8"
        )
    assert len(load_sweeps(tmp_path)) == 2
    assert len(load_sweeps(tmp_path, precision="int4")) == 1


def test_a_missing_directory_is_not_an_error(tmp_path):
    assert load_sweeps(tmp_path / "nope") == []
