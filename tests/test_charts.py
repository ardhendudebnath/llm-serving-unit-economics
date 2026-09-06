from __future__ import annotations

import pytest

from bench.config import GpuSpec
from bench.cost import ApiPricing, Capacity, UnpricedError

pytest.importorskip("matplotlib")

from bench.report.charts import crossover_chart, latency_vs_load_chart

GPU = GpuSpec(
    key="test", name="Test GPU", vram_gb=12, market_usd_per_hour=1.0,
    rate_read_on="2026-09-07", rate_source="fixture",
)
UNPRICED = GpuSpec(key="u", name="Unpriced", vram_gb=12, market_usd_per_hour=0.0)

CAP = Capacity(
    gpu=GPU, profile="long_in", precision="fp16", knee_rps=2.0, slo_p95_s=5.0,
    mean_tokens_in=1000.0, mean_tokens_out=100.0,
)
API = ApiPricing("test-api", usd_in_per_m=2.0, usd_out_per_m=10.0, read_on="2026-06-24")


def _sweep(profile: str = "long_in", *, throttled_at: float | None = None) -> dict:
    points = []
    for rate, p95 in ((0.25, 0.8), (0.5, 0.9), (1.0, 1.2), (2.0, 2.4), (4.0, 11.0)):
        gpu = None
        if throttled_at is not None and rate >= throttled_at:
            gpu = {"throttled_fraction": 0.6, "mean_utilisation_pct": 88.0}
        points.append({
            "profile": profile,
            "target_rate_rps": rate,
            "achieved_rps": rate * 0.97,
            "duration_s": 120.0,
            "error_rate": 0.0,
            "total_latency_s": {"p50": p95 * 0.5, "p95": p95, "p99": p95 * 1.3},
            "gpu": gpu,
        })
    return {
        "profile": profile, "precision": "fp16", "model": "test",
        "slo_p95_s": 5.0, "knee_rps": 2.0, "points": points,
    }


# ------------------------------------------------------------- crossover ----

def test_crossover_chart_renders(tmp_path):
    out = crossover_chart(CAP, API, tmp_path / "crossover.png")
    assert out.exists()
    # A PNG that is only a few hundred bytes is an empty axis.
    assert out.stat().st_size > 20_000


def test_crossover_chart_refuses_an_unpriced_gpu(tmp_path):
    # The chart cannot be drawn from an invented hourly rate. That refusal is
    # the point, not an inconvenience.
    cap = Capacity(
        gpu=UNPRICED, profile="long_in", precision="fp16", knee_rps=2.0,
        slo_p95_s=5.0, mean_tokens_in=1000.0, mean_tokens_out=100.0,
    )
    with pytest.raises(UnpricedError):
        crossover_chart(cap, API, tmp_path / "nope.png")


def test_crossover_chart_handles_no_crossover_without_pretending(tmp_path):
    # A near-free API means self-hosting never wins. The chart must still
    # render and say so, rather than drawing a crossover off the right edge.
    cheap = ApiPricing("cheap", usd_in_per_m=1e-5, usd_out_per_m=1e-5,
                       read_on="2026-06-24")
    out = crossover_chart(CAP, cheap, tmp_path / "none.png")
    assert out.exists() and out.stat().st_size > 20_000


def test_crossover_chart_accepts_both_currencies(tmp_path):
    inr = crossover_chart(CAP, API, tmp_path / "inr.png", currency="INR")
    usd = crossover_chart(CAP, API, tmp_path / "usd.png", currency="USD")
    assert inr.exists() and usd.exists()


def test_crossover_chart_creates_missing_directories(tmp_path):
    out = crossover_chart(CAP, API, tmp_path / "deep" / "nested" / "c.png")
    assert out.exists()


# --------------------------------------------------------- latency vs load --

def test_latency_chart_renders_one_panel_per_profile(tmp_path):
    out = latency_vs_load_chart(
        [_sweep("short"), _sweep("long_in"), _sweep("long_out")],
        tmp_path / "latency.png",
    )
    assert out.exists()
    assert out.stat().st_size > 20_000


def test_latency_chart_rings_throttled_points(tmp_path):
    # Throttled points must be visually distinguishable: their throughput is a
    # floor, not the card's sustained rate.
    plain = latency_vs_load_chart([_sweep()], tmp_path / "plain.png")
    ringed = latency_vs_load_chart(
        [_sweep(throttled_at=2.0)], tmp_path / "ringed.png"
    )
    # The ringed chart carries an extra legend entry and extra marks, so it is
    # strictly larger than the same chart without them.
    assert ringed.stat().st_size > plain.stat().st_size


def test_latency_chart_rejects_an_empty_sweep_list(tmp_path):
    # Silently rendering an empty figure would look like a measurement.
    with pytest.raises(ValueError):
        latency_vs_load_chart([], tmp_path / "empty.png")
