from __future__ import annotations

import pytest

from bench.config import GpuSpec
from bench.cost import (
    HOURS_PER_MONTH,
    ApiPricing,
    Capacity,
    UnpricedError,
    crossover,
    monthly,
)

# $1.00/hr keeps the arithmetic checkable by hand: one GPU is $730/month.
GPU = GpuSpec(
    key="test",
    name="Test GPU",
    vram_gb=24,
    market_usd_per_hour=1.0,
    rate_read_on="2026-09-06",
    rate_source="fixture",
)

UNPRICED = GpuSpec(key="unpriced", name="Unpriced", vram_gb=24, market_usd_per_hour=0.0)

# 2 rps at the SLO -> 7,200 requests/hour on one GPU.
CAP = Capacity(
    gpu=GPU,
    profile="long_in",
    precision="fp16",
    knee_rps=2.0,
    slo_p95_s=5.0,
    mean_tokens_in=1000.0,
    mean_tokens_out=100.0,
)

# $0.003 per request: 1000 in at $2/M, 100 out at $10/M.
API = ApiPricing(
    model_id="test-api-model",
    usd_in_per_m=2.0,
    usd_out_per_m=10.0,
    read_on="2026-06-24",
)


def test_api_cost_per_request():
    assert API.usd_per_request(1000, 100) == pytest.approx(0.003)


def test_marginal_cost_per_1000_at_full_utilisation():
    # $1/hr over 7200 req/hr = $0.0001389/req = $0.1389 per 1000.
    assert CAP.usd_per_1000_requests() == pytest.approx(1.0 / 7200 * 1000)


def test_unpriced_gpu_refuses_a_cost_figure_rather_than_inventing_one():
    cap = Capacity(
        gpu=UNPRICED, profile="long_in", precision="fp16", knee_rps=2.0,
        slo_p95_s=5.0, mean_tokens_in=1000.0, mean_tokens_out=100.0,
    )
    with pytest.raises(UnpricedError):
        cap.usd_per_1000_requests()
    with pytest.raises(UnpricedError):
        monthly(cap, API, 1_000_000)


def test_self_host_cost_is_flat_below_one_gpu_of_capacity():
    # The defining property: a GPU costs the same whether it is busy or idle.
    low = monthly(CAP, API, 10_000)
    mid = monthly(CAP, API, 1_000_000)
    assert low.gpus_needed == mid.gpus_needed == 1
    assert low.self_host_usd == mid.self_host_usd == pytest.approx(730.0)
    # ...while the API side scales linearly and is far cheaper down here.
    assert low.api_usd == pytest.approx(30.0)
    assert not low.self_host_wins


def test_self_host_cost_steps_when_a_second_gpu_is_needed():
    # One GPU covers 2 rps = 2 * 730 * 3600 = 5,256,000 requests/month.
    per_gpu = CAP.knee_rps * HOURS_PER_MONTH * 3600.0
    assert per_gpu == pytest.approx(5_256_000)

    just_under = monthly(CAP, API, per_gpu * 0.99)
    just_over = monthly(CAP, API, per_gpu * 1.01)
    assert just_under.gpus_needed == 1
    assert just_over.gpus_needed == 2
    assert just_over.self_host_usd == pytest.approx(1460.0)


def test_utilisation_falls_out_of_the_step_and_is_reported():
    # Just past the step, the second GPU is almost entirely idle -- which is
    # exactly the situation that makes a per-request cost figure dishonest.
    per_gpu = CAP.knee_rps * HOURS_PER_MONTH * 3600.0
    point = monthly(CAP, API, per_gpu * 1.01)
    assert point.gpus_needed == 2
    assert point.utilisation == pytest.approx(0.505, abs=1e-3)


def test_capacity_is_sized_to_peak_not_mean():
    # A 3x peak-to-mean ratio needs 3x the capacity for the same monthly volume.
    volume = 5_000_000
    flat = monthly(CAP, API, volume, peak_to_mean=1.0)
    bursty = monthly(CAP, API, volume, peak_to_mean=3.0)
    assert flat.gpus_needed == 1
    assert bursty.gpus_needed == 3
    assert bursty.self_host_usd == pytest.approx(3 * 730.0)
    # Utilisation collapses correspondingly -- the honest cost of headroom.
    assert bursty.utilisation < flat.utilisation


def test_crossover_is_where_the_lines_actually_meet():
    # 730 = 0.003 * V  ->  V = 243,333.3
    v = crossover(CAP, API)
    assert v is not None
    assert v == pytest.approx(243_333.3, rel=1e-4)

    # Verified by evaluating either side of it.
    assert not monthly(CAP, API, v * 0.99).self_host_wins
    assert monthly(CAP, API, v * 1.01).self_host_wins


def test_crossover_returns_none_when_self_hosting_never_wins():
    # A near-free API. Self-hosting cannot beat it at any volume the search
    # covers, and that must not be reported as a very large crossover.
    cheap = ApiPricing(
        model_id="cheap", usd_in_per_m=0.0001, usd_out_per_m=0.0001, read_on="2026-06-24"
    )
    assert crossover(CAP, cheap, max_requests_per_month=1_000_000) is None


def test_burstiness_does_not_move_a_crossover_that_sits_inside_one_gpu():
    # Worth pinning because the intuition points the other way. The crossover
    # is at ~243k requests/month, which is 0.09 rps against a 2 rps card -- so
    # even a 4x burst still fits on the one GPU that was already being paid
    # for, and nothing about the comparison changes.
    #
    # This is the general case, not a quirk of the fixture: a crossover exists
    # only when API per-request price exceeds self-hosted marginal cost, and
    # when it does, it lands far inside the first GPU's capacity.
    flat = crossover(CAP, API, peak_to_mean=1.0)
    bursty = crossover(CAP, API, peak_to_mean=4.0)
    assert flat is not None and bursty is not None
    assert bursty == pytest.approx(flat)


def test_enough_burstiness_removes_the_crossover_entirely():
    # Peak-to-mean multiplies self-hosted marginal cost, because the headroom
    # is idle silicon that still bills. Marginal cost here is $0.0001389/req
    # flat; at 25x it becomes $0.003472, which is above the API's $0.003 -- so
    # self-hosting loses at every volume rather than winning eventually.
    assert crossover(CAP, API, peak_to_mean=1.0) is not None
    assert crossover(CAP, API, peak_to_mean=25.0) is None


def test_burstiness_still_changes_cost_and_utilisation_at_a_given_volume():
    # Even where the crossover is unmoved, the sizing is not: this is what
    # makes peak-to-mean worth carrying rather than assuming away.
    volume = 20_000_000
    flat = monthly(CAP, API, volume, peak_to_mean=1.0)
    bursty = monthly(CAP, API, volume, peak_to_mean=3.0)
    assert bursty.gpus_needed == 3 * flat.gpus_needed
    assert bursty.self_host_usd == pytest.approx(3 * flat.self_host_usd)
    assert bursty.utilisation == pytest.approx(flat.utilisation / 3)


def test_zero_knee_is_an_error_not_an_infinite_cost():
    cap = Capacity(
        gpu=GPU, profile="long_in", precision="fp16", knee_rps=0.0,
        slo_p95_s=5.0, mean_tokens_in=1000.0, mean_tokens_out=100.0,
    )
    with pytest.raises(ValueError):
        cap.usd_per_1000_requests()
