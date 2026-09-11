"""The report across the quantisation ladder: per-rung charts, in ladder order."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from bench.cost import ApiPricing
from bench.report import build

API = ApiPricing("test-api", usd_in_per_m=1.0, usd_out_per_m=5.0, read_on="2026-06-24")
CHARTS = ("latency_vs_load_chart", "throughput_vs_precision_chart",
          "quality_vs_precision_chart", "crossover_chart")


def _sweep(profile: str, precision: str, knee: float | None) -> dict:
    return {
        "profile": profile, "precision": precision, "model": f"gst-4b-{precision}",
        "slo_p95_s": 10.0, "knee_rps": knee,
        "points": [{
            "profile": profile, "target_rate_rps": knee or 1.0, "achieved_rps": 0.95,
            "duration_s": 150.0, "completed": 140, "error_rate": 0.0,
            "mean_tokens_in": 1620.0, "mean_tokens_out": 48.0,
            "output_tokens_per_s": 45.0,
            "total_latency_s": {"p50": 3.0, "p95": 4.8 if knee else 22.0,
                                "p99": 5.2, "n": 140},
        }],
    }


def _record(precision: str, slab: float) -> dict:
    return {
        "precision": precision, "n_repeats": 5,
        "scores": {"slab_acc": slab}, "noise": {"slab_acc": 3.57},
        "observed_per_run": {"slab_acc": [slab * 100] * 5},
    }


@pytest.fixture
def repo(tmp_path):
    sweeps = tmp_path / "sweeps"
    sweeps.mkdir()
    # int8 on long_in met the SLO at no rate: a real outcome, not a gap.
    for profile, precision, knee in (
        ("long_in", "int4", 4.0), ("short", "fp16", 8.0),
        ("long_in", "fp16", 2.0), ("long_in", "int8", None),
    ):
        (sweeps / f"sweep_{profile}_{precision}.json").write_text(
            json.dumps(_sweep(profile, precision, knee)), encoding="utf-8"
        )
    return tmp_path


@pytest.fixture
def run(repo, monkeypatch):
    """Run build.main() over `repo`, recording (chart, file name, data) per call."""
    calls: list[tuple[str, str, object]] = []

    def capture(kind: str):
        def fake(*args, **_kw):
            out = next(a for a in args if isinstance(a, Path))
            calls.append((kind, out.name, args[0]))
            return out
        return fake

    for name in CHARTS:
        monkeypatch.setattr(build, name, capture(name))
    monkeypatch.setattr(build, "api_pricing_from_harness", lambda key: API)

    def go() -> list[tuple[str, str, object]]:
        monkeypatch.setattr(sys, "argv", [
            "build", "--sweeps", str(repo / "sweeps"), "--out", str(repo / "charts"),
            "--baseline", str(repo / "baseline.json"), "--eval", str(repo / "eval"),
        ])
        assert build.main() == 0
        return calls

    return go


def test_sweeps_sort_by_profile_then_ladder_order(repo):
    order = [(s["profile"], s["precision"]) for s in build.load_sweeps(repo / "sweeps")]
    # By filename int4 sorts before int8. Every chart reads in ladder order.
    assert order == [("short", "fp16"), ("long_in", "fp16"),
                     ("long_in", "int8"), ("long_in", "int4")]


def test_load_quality_reads_the_baseline_and_each_rung_that_has_run(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_record("fp16", 0.414)), encoding="utf-8")
    (tmp_path / "eval" / "int4").mkdir(parents=True)
    (tmp_path / "eval" / "int4" / "spread.json").write_text(
        json.dumps(_record("int4", 0.35)), encoding="utf-8"
    )
    quality = build.load_quality(baseline, tmp_path / "eval")
    # int8 has not run, so it is absent -- not present at zero.
    assert sorted(quality) == ["fp16", "int4"]
    assert quality["int4"]["scores"]["slab_acc"] == 0.35


def test_load_quality_is_empty_when_nothing_has_been_recorded(tmp_path):
    assert build.load_quality(tmp_path / "missing.json", tmp_path / "eval") == {}


def test_latency_is_drawn_once_per_rung_with_only_that_rungs_sweeps(run):
    latency = [(name, data) for kind, name, data in run()
               if kind == "latency_vs_load_chart"]
    assert [name for name, _ in latency] == [
        "latency-vs-load-fp16.png", "latency-vs-load-int8.png", "latency-vs-load-int4.png",
    ]
    for name, sweeps in latency:
        rung = name.removeprefix("latency-vs-load-").removesuffix(".png")
        assert {s["precision"] for s in sweeps} == {rung}


def test_the_ladder_chart_gets_every_rung_at_once(run):
    ladder = [data for kind, _, data in run() if kind == "throughput_vs_precision_chart"]
    assert len(ladder) == 1
    assert {s["precision"] for s in ladder[0]} == {"fp16", "int8", "int4"}


def test_one_crossover_per_priceable_rung_and_a_named_skip_for_the_rest(run, capsys):
    names = [name for kind, name, _ in run() if kind == "crossover_chart"]
    # int8 has no knee on long_in: skipped by name, never drawn at zero capacity.
    assert names == ["crossover-fp16.png", "crossover-int4.png"]
    assert "SKIPPED  crossover int8" in capsys.readouterr().out


def test_quality_is_skipped_and_said_so_when_nothing_is_recorded(run, capsys):
    assert "quality_vs_precision_chart" not in [kind for kind, _, _ in run()]
    assert "SKIPPED  quality" in capsys.readouterr().out


def test_quality_is_drawn_from_whichever_rungs_have_run(repo, run):
    (repo / "baseline.json").write_text(json.dumps(_record("fp16", 0.414)),
                                        encoding="utf-8")
    quality = [data for kind, _, data in run() if kind == "quality_vs_precision_chart"]
    assert len(quality) == 1
    assert sorted(quality[0]) == ["fp16"]
