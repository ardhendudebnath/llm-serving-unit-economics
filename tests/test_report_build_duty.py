"""The report draws one crossover chart per duty cycle for owned hardware.

An owned laptop has no single hourly price -- the rate moves about 11x across
the duty cycles -- so a report that drew one chart would be publishing one
silently chosen assumption as the result.
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
    """Run build.main() against one synthetic sweep, capturing chart calls."""
    sweeps = tmp_path / "sweeps"
    sweeps.mkdir()
    (sweeps / "sweep_long_in_fp16.json").write_text(
        json.dumps(_long_in_sweep()), encoding="utf-8"
    )

    calls: list[tuple[str, object, str | None]] = []

    def fake_crossover(cap, api, out_path, **kw):
        calls.append((out_path.name, cap, kw.get("title")))
        return out_path

    monkeypatch.setattr(build, "crossover_chart", fake_crossover)
    monkeypatch.setattr(build, "latency_vs_load_chart", lambda sweeps, out, **kw: out)
    monkeypatch.setattr(build, "api_pricing_from_harness", lambda key: API)

    def run(gpu: str) -> list:
        monkeypatch.setattr(sys, "argv", [
            "build", "--sweeps", str(sweeps), "--out", str(tmp_path / "charts"),
            "--gpu", gpu,
        ])
        assert build.main() == 0
        return calls

    return run


def test_owned_hardware_gets_one_chart_per_duty_cycle(drawn):
    calls = drawn(LAPTOP.gpu_key)
    assert [name for name, _, _ in calls] == [
        "crossover-24h.png", "crossover-8h.png", "crossover-2h.png"
    ]


def test_each_chart_is_priced_at_its_own_duty_cycle(drawn):
    calls = drawn(LAPTOP.gpu_key)
    for (_, cap, title), (label, duty) in zip(calls, DUTY_CYCLES.items(), strict=True):
        assert priced(cap.gpu)
        assert cap.gpu.market_usd_per_hour == pytest.approx(LAPTOP.usd_per_hour(duty))
        # The duty cycle is on the chart, not only in the filename.
        assert label in title


def test_an_unpriced_rentable_gpu_draws_no_crossover(drawn):
    # A rentable card with no rate read from a provider is refused rather than
    # drawn from an invented number.
    assert drawn("t4") == []
