"""Cost per 1000 requests, and the volume at which self-hosting overtakes an API.

The plan's formula is the starting point, not the finish:

    cost_per_1000_requests = (gpu_hourly_rate / requests_per_hour_at_slo) * 1000

That is the *marginal* cost of a request on a GPU running flat out at its knee.
It is the right number for a saturated deployment and the wrong number for
almost every real one, because it silently assumes 100 % utilisation.

**Self-hosting cost is a step function, and drawing it as a line is the mistake
this module exists to avoid.** A GPU costs its hourly rate whether it serves one
request an hour or ten thousand. So monthly cost is:

    ceil(required_rps / knee_rps) * hourly_rate * hours_per_month

which is flat across each GPU's capacity and then jumps. API cost is linear
through the origin with no floor. The two therefore cross exactly once in the
region of interest, and the crossover is the deliverable.

Drawn honestly, the self-host curve is a sawtooth in cost-per-request: it falls
as a GPU fills up, then jumps when the next one is needed. A reader who sees a
smooth line knows the author divided one number by another and stopped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from bench.config import USD_TO_INR, GpuSpec, priced

#: Hours in an average month, 365*24/12. Stated rather than assumed, because
#: "730" and "720" give visibly different monthly figures and reports rarely
#: say which they used.
HOURS_PER_MONTH = 730.0


class UnpricedError(RuntimeError):
    """Raised rather than returning a figure nobody read from a provider."""


@dataclass(frozen=True, slots=True)
class ApiPricing:
    """Per-token prices for the API being compared against, with a date.

    Sourced from Project 01's registry so the two projects cannot drift into
    quoting different prices for the same model. `read_on` travels with the
    numbers into every report.
    """

    model_id: str
    usd_in_per_m: float
    usd_out_per_m: float
    read_on: str

    def usd_per_request(self, tokens_in: float, tokens_out: float) -> float:
        return (
            tokens_in / 1_000_000 * self.usd_in_per_m
            + tokens_out / 1_000_000 * self.usd_out_per_m
        )


def api_pricing_from_harness(model_key: str) -> ApiPricing:
    """Pull a dated price out of Project 01's registry.

    Imported lazily: the harness is an optional extra, and the rest of this
    module has to work without it.
    """
    try:
        from harness.runners.registry import PRICES_READ_ON, get
    except ImportError as exc:  # pragma: no cover - guarded in CI
        raise UnpricedError(
            "API prices come from Project 01's registry. Install it:\n"
            "    pip install -e '.[eval]'"
        ) from exc

    spec = get(model_key)
    if spec.usd_in_per_m <= 0 or spec.usd_out_per_m <= 0:
        raise UnpricedError(
            f"{model_key!r} carries no price in Project 01's registry. "
            "It is zero deliberately -- read the provider's price and date it "
            "before publishing a comparison against this model."
        )
    return ApiPricing(
        model_id=spec.model_id,
        usd_in_per_m=spec.usd_in_per_m,
        usd_out_per_m=spec.usd_out_per_m,
        read_on=PRICES_READ_ON,
    )


@dataclass(frozen=True, slots=True)
class Capacity:
    """What one GPU sustains at the SLO. Comes from a measured knee.

    `knee_rps` is not the peak throughput the server can reach -- it is the
    highest arrival rate at which p95 still met the SLO. Those differ a lot,
    and quoting peak throughput as capacity is how a deployment ends up
    missing its latency target at exactly the load it was sized for.
    """

    gpu: GpuSpec
    profile: str
    precision: str
    knee_rps: float
    slo_p95_s: float
    #: Mean tokens per request at the knee, for the API-side comparison. Both
    #: sides must be charged for the same work or the chart is meaningless.
    mean_tokens_in: float
    mean_tokens_out: float

    @property
    def knee_rph(self) -> float:
        return self.knee_rps * 3600.0

    def usd_per_1000_requests(self) -> float:
        """The plan's formula: marginal cost at full utilisation.

        This is the floor -- the best case a self-hosted request can achieve.
        Real cost is this divided by utilisation, which `monthly` computes.
        """
        if not priced(self.gpu):
            raise UnpricedError(
                f"{self.gpu.key!r} has no rate read from a provider. Set "
                "market_usd_per_hour and rate_read_on in bench/config.py "
                "before publishing any cost figure."
            )
        if self.knee_rph <= 0:
            raise ValueError("knee is zero -- the server met the SLO at no rate")
        return self.gpu.market_usd_per_hour / self.knee_rph * 1000.0

    def inr_per_1000_requests(self) -> float:
        return self.usd_per_1000_requests() * USD_TO_INR


@dataclass(frozen=True, slots=True)
class MonthlyPoint:
    """Both sides of the comparison at one monthly request volume."""

    requests_per_month: float
    gpus_needed: int
    self_host_usd: float
    api_usd: float
    utilisation: float

    @property
    def self_host_inr(self) -> float:
        return self.self_host_usd * USD_TO_INR

    @property
    def api_inr(self) -> float:
        return self.api_usd * USD_TO_INR

    @property
    def self_host_wins(self) -> bool:
        return self.self_host_usd < self.api_usd

    def as_row(self) -> dict:
        return {
            "requests_per_month": self.requests_per_month,
            "gpus_needed": self.gpus_needed,
            "self_host_usd": round(self.self_host_usd, 2),
            "self_host_inr": round(self.self_host_inr, 2),
            "api_usd": round(self.api_usd, 2),
            "api_inr": round(self.api_inr, 2),
            "gpu_utilisation": round(self.utilisation, 4),
            "self_host_wins": self.self_host_wins,
        }


def monthly(
    capacity: Capacity,
    api: ApiPricing,
    requests_per_month: float,
    *,
    peak_to_mean: float = 1.0,
) -> MonthlyPoint:
    """Cost of serving `requests_per_month` both ways.

    `peak_to_mean` is the ratio of peak to average traffic. It defaults to 1.0
    -- a perfectly flat month, which no real service has -- because a default
    of anything else would bury an assumption inside a headline number. Set it
    from measured traffic and say what you set it to. At 1.0 the self-hosted
    side is being flattered, and the report must say so.
    """
    if requests_per_month < 0:
        raise ValueError("negative volume")

    mean_rps = requests_per_month / (HOURS_PER_MONTH * 3600.0)
    peak_rps = mean_rps * peak_to_mean

    # Capacity has to cover the peak, not the mean. Sizing to the mean means
    # missing the SLO for half the month.
    gpus = max(1, math.ceil(peak_rps / capacity.knee_rps)) if peak_rps > 0 else 1

    if not priced(capacity.gpu):
        raise UnpricedError(
            f"{capacity.gpu.key!r} has no rate read from a provider -- "
            "refusing to produce a cost curve from a made-up hourly rate."
        )

    self_host = gpus * capacity.gpu.market_usd_per_hour * HOURS_PER_MONTH
    api_usd = requests_per_month * api.usd_per_request(
        capacity.mean_tokens_in, capacity.mean_tokens_out
    )

    # The number that decides whether the self-hosted figure is honest. If this
    # is low, the per-request cost above is dominated by idle time.
    installed_rps = gpus * capacity.knee_rps
    utilisation = (mean_rps / installed_rps) if installed_rps > 0 else 0.0

    return MonthlyPoint(
        requests_per_month=requests_per_month,
        gpus_needed=gpus,
        self_host_usd=self_host,
        api_usd=api_usd,
        utilisation=utilisation,
    )


def crossover(
    capacity: Capacity,
    api: ApiPricing,
    *,
    peak_to_mean: float = 1.0,
    max_requests_per_month: float = 100_000_000.0,
) -> float | None:
    """Monthly volume at which self-hosting first becomes cheaper.

    Solved by bisection rather than algebraically, because the self-hosted side
    is a step function and has no closed form. Returns None when self-hosting
    never wins below `max_requests_per_month` -- which is a real outcome and
    must not be reported as a very large crossover.
    """
    def wins(v: float) -> bool:
        return monthly(capacity, api, v, peak_to_mean=peak_to_mean).self_host_wins

    if wins(0.0):
        # An API that costs more than a GPU at zero volume would mean a
        # per-request price above the whole monthly rental. Treat as a bug in
        # the inputs rather than a finding.
        raise ValueError("self-hosting wins at zero volume -- check the inputs")
    if not wins(max_requests_per_month):
        return None

    low, high = 0.0, max_requests_per_month
    # 60 halvings resolves to well under one request/month across the range.
    for _ in range(60):
        mid = (low + high) / 2.0
        if wins(mid):
            high = mid
        else:
            low = mid
    return high


def curve(
    capacity: Capacity,
    api: ApiPricing,
    volumes: list[float],
    *,
    peak_to_mean: float = 1.0,
) -> list[MonthlyPoint]:
    """Both cost lines across a range of volumes -- the crossover chart's data."""
    return [monthly(capacity, api, v, peak_to_mean=peak_to_mean) for v in volumes]


def log_volumes(start: float = 1_000.0, stop: float = 100_000_000.0, per_decade: int = 24) -> list[float]:
    """Log-spaced volumes for the chart.

    Log spacing because the interesting range spans five orders of magnitude
    and a linear sweep would put almost every sample past the crossover.
    """
    decades = math.log10(stop / start)
    n = int(decades * per_decade)
    return [start * 10 ** (i * decades / n) for i in range(n + 1)]
