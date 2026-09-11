"""The owned-laptop cost: sourced inputs, stated duty cycles, pinned figures."""

from __future__ import annotations

from dataclasses import replace

import pytest

from bench.config import DUTY_CYCLES, LAPTOP, get_gpu, priced


@pytest.mark.parametrize(
    ("label", "expected_inr"),
    [("24 h/day", 10.63), ("8 h/day", 29.66), ("2 h/day", 115.27)],
)
def test_hourly_cost_matches_the_published_figures(label, expected_inr):
    # Rs 2,49,990 over 3 years, 140 W at Rs 8/kWh. Pinned so a change to any
    # input shows up as a failing number rather than a silently moved chart.
    assert LAPTOP.inr_per_hour(DUTY_CYCLES[label]) == pytest.approx(expected_inr, abs=0.01)


def test_cost_per_hour_rises_as_the_card_serves_less():
    costs = [LAPTOP.inr_per_hour(d) for d in sorted(DUTY_CYCLES.values(), reverse=True)]
    assert costs == sorted(costs)
    # About 11x between serving all day and two hours a day -- the reason no
    # single duty cycle is allowed to stand in for the answer.
    assert 10 < costs[-1] / costs[0] < 12


def test_the_whole_state_tariff_range_barely_moves_the_answer():
    # Electricity is about a tenth of the hourly cost at 24 h/day, so the
    # unverifiable national average is not worth guessing at.
    cheap = replace(LAPTOP, tariff_inr_per_kwh=3.0).inr_per_hour(1.0)
    dear = replace(LAPTOP, tariff_inr_per_kwh=14.0).inr_per_hour(1.0)
    assert dear - cheap == pytest.approx(1.54, abs=0.01)
    assert (dear - cheap) / LAPTOP.inr_per_hour(1.0) < 0.15


def test_the_laptop_is_unpriced_until_a_duty_cycle_is_chosen():
    # The base spec must stay unpriced so no cost curve can be drawn from one
    # silently chosen number.
    assert not priced(get_gpu(LAPTOP.gpu_key))

    spec = LAPTOP.priced_gpu(DUTY_CYCLES["8 h/day"])
    assert priced(spec)
    assert spec.rate_read_on == "2026-09-11"
    assert "33% duty cycle" in spec.rate_source
    assert spec.market_usd_per_hour == pytest.approx(LAPTOP.usd_per_hour(8 / 24))


def test_every_input_carries_its_basis():
    for field in ("price_source", "price_read_on", "life_basis",
                  "power_basis", "tariff_basis"):
        assert getattr(LAPTOP, field).strip(), f"{field} is empty"
    # Assumptions must say they are assumptions.
    assert "assumption" in LAPTOP.life_basis
    assert "assumption" in LAPTOP.tariff_basis
