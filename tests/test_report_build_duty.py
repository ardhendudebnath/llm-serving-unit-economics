"""The report prices owned hardware once, around the clock -- not per duty cycle.

An earlier version drew one crossover chart per duty cycle, each priced at that
duty cycle's cost per serving hour. The crossover already bills the card for
every hour of the month and shows idle time as utilisation, so those charts
counted the idle hours twice and would have put the 2 h/day crossover about 11x
too high. These tests hold the corrected behaviour.
"""

from __future__ import annotations

import json
import sys

import pytest

from bench.config import DUTY_CYCLES, LAPTOP, priced
from bench.cost import ApiPricing
from bench.report import build

API = ApiPricing("test-api", usd_in_per_m=1.0, usd_out_per_m=5.0, read_on="2026-06-24")


def _long_in_sweep() -> dict:
    return {
        "profile": "long_in", "precision": "fp16", "model": "gst-4b",
        "slo_p95_s": 10.0, "knee_rps": 2.0,
        "points": [{
            "profile": "long_in", "target_rate_rps": 2.0, "achieved_rps": 1.95,
            "duration_s": 150.0, "completed": 295, "error_rate": 0.0,
            "mean_tokens_in": 1620.0, "mean_tokens_out": 48.0,
            "total_latency_s": {"p50": 3.0, "p95": 4.8, "p99": 5.2, "n": 295},
        }],
    }


@pytest.fixture
def drawn(tmp_path, monkeypatch):
    """Run build.main() against one synthetic sweep, capturing crossover calls."""
    sweeps = tmp_path / "sweeps"
    sweeps.mkdir()
    (sweeps / "sweep_long_in_fp16.json").write_text(
        json.dumps(_long_in_sweep()), encoding="utf-8"
    )

    calls: list[tuple[str, object]] = []

    def fake_crossover(cap, api, out_path, **kw):
        calls.append((out_path.name, cap))
        return out_path

    monkeypatch.setattr(build, "crossover_chart", fake_crossover)
    monkeypatch.setattr(build, "latency_vs_load_chart", lambda sweeps, out, **kw: out)
    monkeypatch.setattr(build, "throughput_vs_precision_chart",
                        lambda sweeps, out, **kw: out)
    monkeypatch.setattr(build, "quality_vs_precision_chart",
                        lambda quality, out, **kw: out)
    monkeypatch.setattr(build, "api_pricing_from_harness", lambda key: API)

    def run(gpu: str) -> list:
        monkeypatch.setattr(sys, "argv", [
            "build", "--sweeps", str(sweeps), "--out", str(tmp_path / "charts"),
            "--baseline", str(tmp_path / "no-baseline.json"),
            "--eval", str(tmp_path / "eval"), "--gpu", gpu,
        ])
        assert build.main() == 0
        return calls

    return run


def test_owned_hardware_gets_exactly_one_crossover_chart_per_rung(drawn):
    assert [name for name, _ in drawn(LAPTOP.gpu_key)] == ["crossover-fp16.png"]


def test_the_chart_is_priced_around_the_clock_with_the_measured_knee(drawn):
    (_, cap), = drawn(LAPTOP.gpu_key)
    assert priced(cap.gpu)
    assert cap.gpu.market_usd_per_hour == pytest.approx(LAPTOP.usd_per_serving_hour(1.0))
    # The measured knee, unscaled: idle time is expressed through utilisation,
    # not by shrinking capacity or inflating the rate.
    assert cap.knee_rps == 2.0


def test_no_partial_duty_cycle_rate_reaches_the_chart(drawn):
    (_, cap), = drawn(LAPTOP.gpu_key)
    for label, duty in DUTY_CYCLES.items():
        if duty < 1.0:
            assert cap.gpu.market_usd_per_hour != pytest.approx(
                LAPTOP.usd_per_serving_hour(duty)
            ), f"crossover priced at the {label} serving-hour rate"


def test_an_unpriced_rentable_gpu_draws_no_crossover(drawn):
    assert drawn("t4") == []
