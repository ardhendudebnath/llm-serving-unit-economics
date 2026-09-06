from __future__ import annotations

import pytest

from bench.metrics import Distribution, LoadPoint, RequestRecord, find_knee, percentile


def test_percentile_matches_linear_interpolation():
    # numpy's default method on [1..5]: p50 lands exactly on an order
    # statistic, p95 and p99 interpolate. Pinned so a future rewrite cannot
    # silently switch to nearest-rank, which would move every published tail.
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(vals, 50) == 3.0
    assert percentile(vals, 95) == pytest.approx(4.8)
    assert percentile(vals, 99) == pytest.approx(4.96)
    assert percentile(vals, 0) == 1.0
    assert percentile(vals, 100) == 5.0


def test_percentile_is_order_independent():
    assert percentile([5.0, 1.0, 3.0, 2.0, 4.0], 50) == 3.0


def test_percentile_of_single_sample():
    assert percentile([7.0], 99) == 7.0


def test_percentile_rejects_empty():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_thin_tail_is_flagged():
    assert Distribution.of([float(i) for i in range(50)]).tail_is_thin
    assert not Distribution.of([float(i) for i in range(200)]).tail_is_thin


def _record(*, sent: float, done: float, first: float | None = None, ok: bool = True):
    return RequestRecord(
        profile="short",
        queued_at=sent,
        sent_at=sent,
        done_at=done,
        first_token_at=first,
        tokens_out=10,
        status=200 if ok else 500,
        error=None if ok else "boom",
    )


def test_client_lag_is_surfaced_not_absorbed():
    # sent late relative to schedule: the generator fell behind, and the
    # measured latency is therefore optimistic.
    r = RequestRecord(profile="short", queued_at=10.0, sent_at=12.5, done_at=13.0)
    assert r.coordinated_omission_s == pytest.approx(2.5)
    # total is measured from send, not from schedule
    assert r.total_s == pytest.approx(0.5)


def test_lag_never_negative_when_sent_early():
    r = RequestRecord(profile="short", queued_at=10.0, sent_at=9.8, done_at=10.0)
    assert r.coordinated_omission_s == 0.0


def test_error_rate_and_throughput_exclude_failures():
    point = LoadPoint(
        profile="short",
        target_rate_rps=4.0,
        duration_s=10.0,
        records=[_record(sent=0, done=1)] * 8 + [_record(sent=0, done=1, ok=False)] * 2,
    )
    assert point.error_rate == pytest.approx(0.2)
    # 8 successes over 10s, not 10 over 10s
    assert point.achieved_rps == pytest.approx(0.8)
    assert point.output_tokens_per_s == pytest.approx(8.0)


def _point(rate: float, p95: float, *, error_rate: float = 0.0) -> LoadPoint:
    n = 100
    n_bad = int(n * error_rate)
    # A flat sample whose p95 is exactly the value we want to test against.
    records = [_record(sent=0.0, done=p95) for _ in range(n - n_bad)]
    records += [_record(sent=0.0, done=p95, ok=False) for _ in range(n_bad)]
    return LoadPoint(profile="short", target_rate_rps=rate, duration_s=60.0, records=records)


def test_knee_is_last_passing_not_first_failing():
    # 3.0 passes, 4.0 fails, 5.0 "passes" again on a noisy curve. The knee is
    # 3.0 -- crediting the deployment with 5.0 would size it for a rate it
    # only met after already falling over.
    points = [_point(1.0, 1.0), _point(3.0, 1.8), _point(4.0, 9.0), _point(5.0, 1.5)]
    knee = find_knee(points, slo_p95_s=2.0)
    assert knee is not None
    assert knee.target_rate_rps == 3.0


def test_knee_rejects_a_rate_that_meets_slo_by_erroring():
    # Fast p95, but 20% of requests failed. Shedding load is not meeting an SLO.
    points = [_point(1.0, 1.0), _point(2.0, 0.5, error_rate=0.2)]
    knee = find_knee(points, slo_p95_s=2.0)
    assert knee is not None
    assert knee.target_rate_rps == 1.0


def test_knee_is_none_when_slo_never_met():
    assert find_knee([_point(1.0, 30.0)], slo_p95_s=2.0) is None


def test_knee_walks_in_rate_order_not_list_order():
    points = [_point(5.0, 9.0), _point(1.0, 1.0), _point(3.0, 1.5)]
    knee = find_knee(points, slo_p95_s=2.0)
    assert knee is not None
    assert knee.target_rate_rps == 3.0
