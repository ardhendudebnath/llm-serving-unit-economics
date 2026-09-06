"""End-to-end tests for the load generator, over real HTTP.

Everything else in this suite tests arithmetic. These drive the actual client
against a server that streams actual SSE, because the load generator is the
instrument every published latency number comes from, and an instrument that
has never been pointed at anything is not evidence.

The mock server's latency is *specified* rather than measured, so the
assertions can be tight: told to take 200 ms to first token, the generator must
report close to 200 ms. Any gap is the instrument's own error.
"""

from __future__ import annotations

import asyncio
import itertools
import statistics

import pytest

from bench.loadgen import RunConfig, discover_model, run_point
from bench.workloads import Request
from tests.mock_server import MODEL_ID, Behaviour, MockServer

pytest.importorskip("httpx")

# Every test in this module drives real HTTP with real sleeps. Marked so the
# fast local loop can skip them with `-m "not e2e"`; CI runs them all.
pytestmark = pytest.mark.e2e


def _corpus(n: int = 40) -> list[Request]:
    return [
        Request(
            id=f"e2e-{i:03d}",
            profile="short",
            system="you are a test",
            prompt=f"classify item {i}",
            max_tokens=16,
            source="mock",
            source_id=str(i),
        )
        for i in range(n)
    ]


def _drive(server: MockServer, *, rate: float, duration: float,
           warmup: float = 0.0, seed: int = 7, **kw):
    cfg = RunConfig(
        base_url=server.base_url,
        model=MODEL_ID,
        profile="short",
        rate_rps=rate,
        duration_s=duration,
        warmup_s=warmup,
        seed=seed,
        **kw,
    )
    return asyncio.run(run_point(cfg, _corpus()))


# ------------------------------------------------------------ basic loop ----

def test_the_generator_actually_completes_requests():
    with MockServer(Behaviour(ttft_ms=10, inter_token_ms=1, tokens_out=8)) as server:
        point = _drive(server, rate=20.0, duration=2.0)

    assert point.records, "no requests were sent at all"
    assert point.error_rate == 0.0
    assert len(point.completed) == len(point.records)
    assert server.served >= len(point.records)


def test_model_discovery_asks_the_server_what_it_serves():
    with MockServer() as server:
        assert discover_model(server.base_url) == MODEL_ID


# ------------------------------------------------------------------ TTFT ----

def test_ttft_measures_the_first_content_delta_not_the_role_delta():
    # vLLM sends a role-only delta before any content. Counting it as the first
    # token would report a TTFT that no user ever experiences -- here it would
    # read as near-zero instead of 250 ms.
    with MockServer(Behaviour(ttft_ms=250, inter_token_ms=1, tokens_out=4,
                              concurrency=32)) as server:
        point = _drive(server, rate=4.0, duration=3.0)

    ttft = point.ttft()
    assert ttft is not None
    assert ttft.p50 == pytest.approx(0.25, abs=0.12)
    # Emphatically not ~0, which is what counting the role delta would give.
    assert ttft.p50 > 0.15


def test_total_latency_covers_ttft_plus_decode():
    # 100ms to first token, then 10 tokens at 20ms each = ~0.1 + 0.18 = 0.28s.
    with MockServer(Behaviour(ttft_ms=100, inter_token_ms=20, tokens_out=10,
                              concurrency=32)) as server:
        point = _drive(server, rate=4.0, duration=3.0)

    total = point.total_latency()
    ttft = point.ttft()
    assert total is not None and ttft is not None
    assert total.p50 == pytest.approx(0.28, abs=0.12)
    # Decode time is real and separable -- the two must not collapse together.
    assert total.p50 > ttft.p50


def test_token_counts_come_from_the_usage_chunk():
    # Not from counting SSE chunks: a tokenizer emitting multi-token chunks
    # would make that wrong, silently and only for some models.
    with MockServer(Behaviour(ttft_ms=5, inter_token_ms=1, tokens_out=7,
                              prompt_tokens=321, concurrency=32)) as server:
        point = _drive(server, rate=10.0, duration=2.0)

    done = point.completed
    assert done
    assert all(r.tokens_in == 321 for r in done)
    assert all(r.tokens_out == 7 for r in done)


# ------------------------------------------------------------- open loop ----

def test_the_generator_is_open_loop_under_saturation():
    """The property the whole design exists for.

    The server is deliberately far too slow for the offered rate: one request
    at a time, 300 ms each, against 15 requests/second. A closed-loop generator
    would send roughly 3 requests in 3 seconds, because it waits for each reply
    -- and would then report a latency of 300 ms and call the server healthy.

    Open-loop, the generator must fire on schedule regardless, so it sends tens
    of requests and the latency distribution shows the queue building. That
    growing tail *is* the measurement.
    """
    with MockServer(Behaviour(ttft_ms=300, inter_token_ms=1, tokens_out=2,
                              concurrency=1)) as server:
        point = _drive(server, rate=15.0, duration=3.0)

    sent = len(point.records)
    # A closed-loop client could not have issued more than ~10 here.
    assert sent > 25, f"only {sent} requests sent -- generator is not open-loop"

    total = point.total_latency()
    assert total is not None
    # Queueing behind a 1-at-a-time server must show up as a tail far above
    # the 300 ms service time.
    assert total.p95 > 1.0
    assert total.p95 > total.p50


def test_client_lag_stays_near_zero_when_the_client_is_not_the_bottleneck():
    # If this rises, the generator failed to keep its own schedule and every
    # latency it reports is optimistic. It is reported per point rather than
    # silently absorbed, so it needs to be trustworthy when nothing is wrong.
    with MockServer(Behaviour(ttft_ms=5, inter_token_ms=1, tokens_out=4,
                              concurrency=64)) as server:
        point = _drive(server, rate=25.0, duration=2.0)

    assert point.worst_client_lag_s() < 0.5


def test_inflight_cap_shows_up_as_client_lag_rather_than_vanishing():
    # Capping concurrency turns the generator partially closed-loop. That must
    # be visible, because it makes the point's latency optimistic.
    with MockServer(Behaviour(ttft_ms=200, inter_token_ms=1, tokens_out=2,
                              concurrency=1)) as server:
        point = _drive(server, rate=20.0, duration=2.0, max_inflight=2)

    assert point.worst_client_lag_s() > 0.5


# ----------------------------------------------------------- arrivals -------

def test_arrivals_are_poisson_not_uniform():
    """Exponential gaps, not a fixed cadence.

    Real traffic is bursty and batches differently from evenly spaced load, so
    a generator that quietly ticked every 1/rate seconds would report a
    throughput no deployment reproduces.

    Tested on the *scheduled* times rather than the sent times, since the point
    is the arrival process itself. For an exponential distribution the standard
    deviation equals the mean, which a uniform schedule (sd = 0) fails
    decisively.
    """
    with MockServer(Behaviour(ttft_ms=2, inter_token_ms=0.5, tokens_out=2,
                              concurrency=64)) as server:
        point = _drive(server, rate=40.0, duration=4.0)

    scheduled = sorted(r.queued_at for r in point.records)
    assert len(scheduled) > 60
    gaps = [b - a for a, b in itertools.pairwise(scheduled)]

    mean = statistics.fmean(gaps)
    sd = statistics.stdev(gaps)
    # Exponential: sd/mean == 1. Uniform spacing would give ~0.
    assert 0.6 < (sd / mean) < 1.5, f"coefficient of variation {sd / mean:.2f}"


def test_the_offered_rate_is_honoured_when_the_server_can_keep_up():
    with MockServer(Behaviour(ttft_ms=5, inter_token_ms=1, tokens_out=4,
                              concurrency=64)) as server:
        point = _drive(server, rate=30.0, duration=3.0)

    # Poisson over a short window is noisy, so this is a loose band -- it is
    # checking the generator is not systematically off, not that it is exact.
    assert 20.0 < point.achieved_rps < 40.0


def test_the_same_seed_reproduces_the_same_schedule():
    with MockServer(Behaviour(ttft_ms=2, inter_token_ms=0.5, tokens_out=2,
                              concurrency=64)) as server:
        first = _drive(server, rate=20.0, duration=2.0, seed=99)
        second = _drive(server, rate=20.0, duration=2.0, seed=99)

    a = [round(r.queued_at - min(x.queued_at for x in first.records), 4)
         for r in first.records]
    b = [round(r.queued_at - min(x.queued_at for x in second.records), 4)
         for r in second.records]
    assert a == b


# --------------------------------------------------------------- errors -----

def test_failed_requests_are_recorded_as_data_not_raised():
    # A 500 is a measurement, not an exception: the error rate is what
    # disqualifies a load point from counting as meeting its SLO.
    with MockServer(Behaviour(ttft_ms=5, inter_token_ms=1, tokens_out=2,
                              concurrency=32, error_rate=0.25)) as server:
        point = _drive(server, rate=30.0, duration=2.0)

    assert point.records
    assert 0.1 < point.error_rate < 0.45
    failed = [r for r in point.records if not r.ok]
    assert failed and all(r.status == 500 for r in failed)
    assert all("http_500" in (r.error or "") for r in failed)


def test_a_dead_server_produces_records_rather_than_an_exception():
    # Nothing listening on this port. The sweep must survive it and report,
    # not crash halfway through a GPU block.
    cfg = RunConfig(
        base_url="http://127.0.0.1:1",
        model=MODEL_ID,
        profile="short",
        rate_rps=10.0,
        duration_s=1.0,
        warmup_s=0.0,
    )
    point = asyncio.run(run_point(cfg, _corpus()))

    assert point.records
    assert point.error_rate == 1.0
    assert point.total_latency() is None


# ---------------------------------------------------------------- warmup ----

def test_warmup_requests_are_excluded_from_the_measurement():
    with MockServer(Behaviour(ttft_ms=5, inter_token_ms=1, tokens_out=2,
                              concurrency=32)) as server:
        point = _drive(server, rate=20.0, duration=2.0, warmup=1.0)

    # The server saw warmup traffic...
    assert server.served > len(point.records)
    # ...but none of it is in the recorded distribution.
    assert point.duration_s == pytest.approx(2.0, abs=1.0)
