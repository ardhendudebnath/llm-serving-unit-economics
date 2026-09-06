from __future__ import annotations

import pytest

from bench.config import (
    GPUS,
    LADDER,
    amortised_usd_per_hour,
    get_gpu,
    priced,
    vram_estimate_gb,
)


def test_no_gpu_ships_with_an_invented_rate():
    # Every rate stays at zero until read from a provider and dated. A cost
    # curve built on a remembered price is fiction that looks like a result.
    for spec in GPUS.values():
        assert not priced(spec), f"{spec.key} carries an unsourced rate"


def test_the_local_card_is_recorded_as_measured_not_as_assumed():
    # 12 GB, not the desktop 5070 Ti's 16. Read from nvidia-smi rather than
    # from the model name, which is exactly the sort of thing that silently
    # invalidates a VRAM plan.
    spec = get_gpu("rtx5070ti-laptop")
    assert spec.vram_gb == 12


def test_an_8b_model_does_not_fit_the_local_card_at_fp16():
    # The constraint that drives the model choice: weights alone exceed the
    # card before any KV cache exists.
    assert vram_estimate_gb(8.0, "fp16") == 16.0
    assert vram_estimate_gb(8.0, "fp16") > get_gpu("rtx5070ti-laptop").vram_gb


def test_a_4b_model_fits_at_every_rung_of_the_ladder():
    vram = get_gpu("rtx5070ti-laptop").vram_gb
    # Leaving real headroom for KV cache, activations and CUDA context.
    assert vram_estimate_gb(4.0, "fp16") == 8.0
    assert vram_estimate_gb(4.0, "int8") == 4.0
    assert vram_estimate_gb(4.0, "int4") == 2.0
    assert all(vram_estimate_gb(4.0, rung) < vram for rung in LADDER)


# ------------------------------------------------------------- amortised ----

def test_amortised_rate_splits_capital_and_energy():
    # $1000 card, 3 years, serving 100% of the time -> 26,280 serving hours.
    # Capital: 1000 / 26280 = $0.03805/hr.
    # Energy: 100W at $0.10/kWh = $0.01/hr.
    rate = amortised_usd_per_hour(
        hardware_usd=1000.0,
        useful_life_years=3.0,
        duty_cycle=1.0,
        mean_power_w=100.0,
        electricity_usd_per_kwh=0.10,
    )
    assert rate == pytest.approx(1000.0 / 26_280 + 0.01, rel=1e-6)


def test_duty_cycle_dominates_the_answer():
    # The input that moves the result most, and the one most often left
    # unstated: a card serving 10% of the time costs far more per serving-hour.
    kw = {
        "hardware_usd": 1000.0, "useful_life_years": 3.0,
        "mean_power_w": 100.0, "electricity_usd_per_kwh": 0.10,
    }
    full = amortised_usd_per_hour(duty_cycle=1.0, **kw)
    tenth = amortised_usd_per_hour(duty_cycle=0.1, **kw)

    # Capital scales exactly 10x -- $0.0381/hr becomes $0.3805/hr...
    assert (tenth - 0.01) == pytest.approx((full - 0.01) * 10.0)
    # ...but energy per serving-hour is unchanged at $0.01, so it dilutes the
    # total to a little over 8x rather than 10x. Asserted precisely, because
    # "roughly 10x" was wrong and the arithmetic is the point of this test.
    assert (tenth / full) == pytest.approx(8.127, rel=1e-3)


def test_pue_only_touches_energy():
    kw = {
        "hardware_usd": 1000.0, "useful_life_years": 3.0, "duty_cycle": 1.0,
        "mean_power_w": 100.0, "electricity_usd_per_kwh": 0.10,
    }
    assert amortised_usd_per_hour(pue=2.0, **kw) == pytest.approx(
        amortised_usd_per_hour(pue=1.0, **kw) + 0.01
    )


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
def test_impossible_duty_cycle_is_rejected(bad):
    with pytest.raises(ValueError):
        amortised_usd_per_hour(
            hardware_usd=1000.0, useful_life_years=3.0, duty_cycle=bad,
            mean_power_w=100.0, electricity_usd_per_kwh=0.10,
        )


def test_zero_life_is_rejected_rather_than_dividing_by_zero():
    with pytest.raises(ValueError):
        amortised_usd_per_hour(
            hardware_usd=1000.0, useful_life_years=0.0, duty_cycle=1.0,
            mean_power_w=100.0, electricity_usd_per_kwh=0.10,
        )
