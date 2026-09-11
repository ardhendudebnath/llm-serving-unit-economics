"""The owned-laptop cost: sourced inputs, and two figures that must not be mixed."""

from __future__ import annotations

import inspect
from dataclasses import replace

import pytest

from bench.config import DUTY_CYCLES, LAPTOP, get_gpu, priced


@pytest.mark.parametrize(
    ("label", "expected_inr"),
    [("24 h/day", 10.63), ("8 h/day", 29.66), ("2 h/day", 115.27)],
)
def test_cost_per_serving_hour_matches_the_published_figures(label, expected_inr):
    # Rs 2,49,990 over 3 years, 140 W at Rs 8/kWh. Pinned so a change to any
    # input shows up as a failing number rather than a silently moved table.
    assert LAPTOP.inr_per_serving_hour(DUTY_CYCLES[label]) == pytest.approx(
        expected_inr, abs=0.01
    )


def test_cost_per_serving_hour_rises_as_the_card_serves_less():
    costs = [LAPTOP.inr_per_serving_hour(d)
             for d in sorted(DUTY_CYCLES.values(), reverse=True)]
    assert costs == sorted(costs)
    assert 10 < costs[-1] / costs[0] < 12


def test_the_whole_state_tariff_range_barely_moves_the_answer():
    # Electricity is about a tenth of the hourly cost at 24 h/day, so the
    # unverifiable national average is not worth guessing at.
    cheap = replace(LAPTOP, tariff_inr_per_kwh=3.0).inr_per_serving_hour(1.0)
    dear = replace(LAPTOP, tariff_inr_per_kwh=14.0).inr_per_serving_hour(1.0)
    assert dear - cheap == pytest.approx(1.54, abs=0.01)
    assert (dear - cheap) / LAPTOP.inr_per_serving_hour(1.0) < 0.15


def test_the_laptop_is_unpriced_until_priced_for_the_crossover():
    assert not priced(get_gpu(LAPTOP.gpu_key))
    spec = LAPTOP.priced_gpu()
    assert priced(spec)
    assert spec.rate_read_on == "2026-09-11"
    assert "around the clock" in spec.rate_source


def test_the_crossover_price_is_the_around_the_clock_rate():
    """Idle time must be counted once.

    monthly() bills a card for every hour and shows idleness as utilisation. A
    rate already inflated by a partial duty cycle would count the idle hours a
    second time -- at 2 h/day that overstates monthly cost, and the crossover,
    by about 11x.
    """
    assert LAPTOP.priced_gpu().market_usd_per_hour == pytest.approx(
        LAPTOP.usd_per_serving_hour(1.0)
    )
    inflated = LAPTOP.usd_per_serving_hour(DUTY_CYCLES["2 h/day"])
    assert inflated / LAPTOP.priced_gpu().market_usd_per_hour > 10


def test_priced_gpu_cannot_be_given_a_duty_cycle():
    # The signature is the guard: with no parameter, the double count cannot
    # come back by passing one.
    assert not inspect.signature(LAPTOP.priced_gpu).parameters


def test_every_input_carries_its_basis():
    for field in ("price_source", "price_read_on", "life_basis",
                  "power_basis", "tariff_basis"):
        assert getattr(LAPTOP, field).strip(), f"{field} is empty"
    assert "assumption" in LAPTOP.life_basis
    assert "assumption" in LAPTOP.tariff_basis
