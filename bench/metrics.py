"""Latency distributions, throughput, and the arithmetic behind the knee.

Stdlib only, so the cost model and the CI gate can read a measurement without
installing anything.

**There is no `mean_latency` in this module, and that is deliberate.** An
average latency figure is the single clearest signal that a benchmark never ran
real load: it hides exactly the tail that decides whether a deployment meets an
SLO. Every summary here reports a distribution, and every distribution carries
the `n` it was computed from, because a p99 over 100 requests is very nearly
the maximum and a reader is entitled to know that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Percentiles reported everywhere. p50 for the typical request, p95 for the
#: SLO, p99 because that is where continuous batching either holds up or does
#: not.
REPORTED = (50.0, 95.0, 99.0)


@dataclass(slots=True)
class RequestRecord:
    """One request's timeline. All times are monotonic seconds.

    `queued_at` is when the arrival process said the request should be sent,
    not when it was. The gap between the two is client-side queueing, and if it
    is not near zero the load generator itself is the bottleneck -- which would
    silently understate the server's throughput. `coordinated_omission_s`
    exists so that failure is visible rather than inferred.
    """

    profile: str
    queued_at: float
    sent_at: float
    done_at: float
    #: None when the response was not streamed, or when the request failed
    #: before any token arrived.
    first_token_at: float | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    status: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and 200 <= self.status < 300

    @property
    def total_s(self) -> float:
        """Wall time from send to last token."""
        return self.done_at - self.sent_at

    @property
    def ttft_s(self) -> float | None:
        """Time to first token. None when the response was not streamed."""
        if self.first_token_at is None:
            return None
        return self.first_token_at - self.sent_at

    @property
    def coordinated_omission_s(self) -> float:
        """How late the client was in sending this request.

        Non-zero means the generator could not keep up with its own schedule,
        so measured latency is optimistic. Reported, never silently absorbed.
        """
        return max(0.0, self.sent_at - self.queued_at)


def percentile(values: list[float], p: float) -> float:
    """Linear interpolation between order statistics (numpy's default method).

    Chosen over nearest-rank so the numbers are directly comparable to anything
    computed with numpy or pandas, which is what a reader checking this work
    will reach for. Stated explicitly because the two methods disagree by a
    visible margin in the tail at the sample sizes used here.
    """
    if not values:
        raise ValueError("percentile of an empty sample is undefined")
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"percentile {p} out of range")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


@dataclass(slots=True)
class Distribution:
    """A latency distribution. Carries `n` so the tail can be judged."""

    n: int
    p50: float
    p95: float
    p99: float
    minimum: float
    maximum: float

    @classmethod
    def of(cls, values: list[float]) -> Distribution:
        ordered = sorted(values)
        return cls(
            n=len(ordered),
            p50=percentile(ordered, 50.0),
            p95=percentile(ordered, 95.0),
            p99=percentile(ordered, 99.0),
            minimum=ordered[0],
            maximum=ordered[-1],
        )

    @property
    def tail_is_thin(self) -> bool:
        """True when n is too small for p99 to mean much.

        At n < 100 the 99th percentile sits inside the top single observation,
        so it is reporting the maximum under another name.
        """
        return self.n < 100

    def as_row(self) -> dict:
        return {
            "n": self.n,
            "p50": round(self.p50, 4),
            "p95": round(self.p95, 4),
            "p99": round(self.p99, 4),
            "min": round(self.minimum, 4),
            "max": round(self.maximum, 4),
            "p99_from_thin_tail": self.tail_is_thin,
        }


@dataclass(slots=True)
class LoadPoint:
    """One point on the latency-vs-load curve: everything at one arrival rate."""

    profile: str
    target_rate_rps: float
    duration_s: float
    records: list[RequestRecord] = field(default_factory=list)

    @property
    def completed(self) -> list[RequestRecord]:
        return [r for r in self.records if r.ok]

    @property
    def error_rate(self) -> float:
        if not self.records:
            return 0.0
        return 1.0 - len(self.completed) / len(self.records)

    @property
    def achieved_rps(self) -> float:
        """Successful requests per second actually served.

        Divided by wall-clock duration, not by the sum of latencies: this is
        throughput as a operator would measure it, and it falls below the target
        rate once the server saturates. The gap is the interesting part.
        """
        if self.duration_s <= 0:
            return 0.0
        return len(self.completed) / self.duration_s

    @property
    def output_tokens_per_s(self) -> float:
        if self.duration_s <= 0:
            return 0.0
        return sum(r.tokens_out for r in self.completed) / self.duration_s

    def total_latency(self) -> Distribution | None:
        vals = [r.total_s for r in self.completed]
        return Distribution.of(vals) if vals else None

    def ttft(self) -> Distribution | None:
        vals = [r.ttft_s for r in self.completed if r.ttft_s is not None]
        return Distribution.of(vals) if vals else None

    def worst_client_lag_s(self) -> float:
        """Largest client-side send delay across the point.

        If this is not small relative to p50, the generator was the bottleneck
        and the whole point is suspect.
        """
        if not self.records:
            return 0.0
        return max(r.coordinated_omission_s for r in self.records)

    def as_row(self) -> dict:
        total = self.total_latency()
        ttft = self.ttft()
        return {
            "profile": self.profile,
            "target_rate_rps": self.target_rate_rps,
            "achieved_rps": round(self.achieved_rps, 4),
            "duration_s": round(self.duration_s, 2),
            "requests": len(self.records),
            "completed": len(self.completed),
            "error_rate": round(self.error_rate, 4),
            "output_tokens_per_s": round(self.output_tokens_per_s, 2),
            "total_latency_s": total.as_row() if total else None,
            "ttft_s": ttft.as_row() if ttft else None,
            "worst_client_lag_s": round(self.worst_client_lag_s(), 4),
        }


def find_knee(points: list[LoadPoint], slo_p95_s: float) -> LoadPoint | None:
    """The highest arrival rate whose p95 still meets the SLO.

    Returns the last point that passes, walking in rate order -- not the first
    that fails. Those differ when the curve is noisy, and taking the last pass
    is the conservative reading: it will not credit the deployment with a rate
    it only met once before falling over.

    Points with a non-trivial error rate cannot pass. A server that meets its
    latency target by refusing requests has not met its latency target.
    """
    passing: LoadPoint | None = None
    for point in sorted(points, key=lambda p: p.target_rate_rps):
        dist = point.total_latency()
        if dist is None or point.error_rate > 0.01 or dist.p95 > slo_p95_s:
            break
        passing = point
    return passing
