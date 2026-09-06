"""Hardware, precisions and prices, each carrying the date it was read.

Two rules are enforced here rather than trusted to discipline. Both are
inherited from Project 01, which learned them the same way.

**State prices with a date.** A cost curve quoting a GPU rate without saying
when it was read is not reproducible. `rate_read_on` is stamped into every
report, and `priced()` refuses to produce a figure from a rate nobody has read.

**Publish the market rate, not what you happened to pay.** This benchmark runs
on free-tier GPU hours. Reporting the resulting cost as zero would be a lie
that flatters the project -- it would put the self-host crossover at one
request a month and make the whole chart worthless. So every published figure
uses `market_usd_per_hour`: what an hour of this GPU class actually costs to
rent, read from a provider on a stated date.

What was actually spent is a *different* number, and it lives in
`docs/cost-log.md`. Keeping the two apart is the point. A reader who wants to
know what the curves would cost a company reads the market rate; a reader who
wants to know what this project cost its author reads the log. Collapsing them
would answer neither question honestly.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Rupees per US dollar. Must track Project 01's registry, which reports the
#: same conversion on its leaderboard; `tests/test_cross_repo.py` fails if the
#: two drift apart while the harness is installed.
USD_TO_INR = 88.0
FX_READ_ON = "2026-09-04"


@dataclass(frozen=True, slots=True)
class GpuSpec:
    """One GPU class, priced at what renting it costs -- not at what we paid."""

    key: str
    name: str
    vram_gb: int
    #: On-demand rate for this class, in USD/hour. Zero means unread, and
    #: `priced()` will refuse to build a cost figure from it.
    market_usd_per_hour: float
    #: When the rate above was read, and from where. Both are printed in the
    #: report; a rate without a source is an assertion, not a measurement.
    rate_read_on: str = ""
    rate_source: str = ""
    #: How this project actually obtained the hours. Recorded so the report can
    #: say plainly that the measurements were free and the pricing is not.
    obtained_via: str = ""

    def inr_per_hour(self) -> float:
        return self.market_usd_per_hour * USD_TO_INR


#: Rates are deliberately left at 0.0 until read from a provider and dated.
#: Filling one in from memory is how a cost curve becomes fiction.
GPUS: dict[str, GpuSpec] = {
    # The card these measurements actually run on.
    #
    # **It cannot be rented, and that is a genuine problem for the cost curve
    # rather than a detail.** No cloud offers a laptop 5070 Ti, so there is no
    # provider rate to read. Substituting a rentable card's price would be
    # worse than leaving this blank: throughput was measured on *this* silicon,
    # and pairing it with a different card's hourly rate produces a figure that
    # describes no machine that exists.
    #
    # The defensible route is `amortised_usd_per_hour()` below -- what an hour
    # on this card costs its owner, from the purchase price, an assumed useful
    # life and the measured power draw. It is a real number for a real machine,
    # and every input to it is stated.
    #
    # 12 GB, not the desktop card's 16. Read from nvidia-smi on 2026-09-07:
    # 12,227 MiB, driver 595.79, compute capability 12.0 (Blackwell, sm_120).
    "rtx5070ti-laptop": GpuSpec(
        key="rtx5070ti-laptop",
        name="NVIDIA GeForce RTX 5070 Ti Laptop GPU",
        vram_gb=12,
        market_usd_per_hour=0.0,
        obtained_via="owned hardware; see amortised_usd_per_hour()",
    ),
    # Rentable comparators. Used to answer "what would this cost in a cloud",
    # which needs its *own* throughput measurement -- these rows exist so a
    # future sweep on rented hardware has somewhere to record its rate, not so
    # local throughput can borrow a cloud price.
    "t4": GpuSpec(
        key="t4",
        name="NVIDIA T4",
        vram_gb=16,
        market_usd_per_hour=0.0,
        obtained_via="Colab / Kaggle free tier",
    ),
    "a10g": GpuSpec(
        key="a10g",
        name="NVIDIA A10G",
        vram_gb=24,
        market_usd_per_hour=0.0,
        obtained_via="rentable",
    ),
    "l4": GpuSpec(
        key="l4",
        name="NVIDIA L4",
        vram_gb=24,
        market_usd_per_hour=0.0,
        obtained_via="rentable",
    ),
}


def amortised_usd_per_hour(
    *,
    hardware_usd: float,
    useful_life_years: float,
    duty_cycle: float,
    mean_power_w: float,
    electricity_usd_per_kwh: float,
    pue: float = 1.0,
) -> float:
    """Hourly cost of owning a GPU, for hardware that cannot be rented.

    Capital cost spread over the hours it will actually be used, plus the
    electricity it draws while used. Every input is an argument rather than a
    constant, because every one of them is an assumption the reader is entitled
    to disagree with -- and the report prints them next to the result.

    `duty_cycle` is the fraction of its life the card spends serving. This is
    the input that moves the answer most: a card serving 10 % of the time costs
    ten times as much per serving-hour as one serving continuously, and quoting
    an amortised rate without stating it is meaningless.

    `pue` covers cooling and conversion overhead in a datacentre. Left at 1.0
    for a laptop on a desk, where there is no such overhead to attribute.
    """
    if not 0.0 < duty_cycle <= 1.0:
        raise ValueError("duty_cycle must be in (0, 1]")
    if useful_life_years <= 0:
        raise ValueError("useful_life_years must be positive")

    serving_hours = useful_life_years * 365.0 * 24.0 * duty_cycle
    capital_per_hour = hardware_usd / serving_hours
    energy_per_hour = (mean_power_w / 1000.0) * pue * electricity_usd_per_kwh
    return capital_per_hour + energy_per_hour


@dataclass(frozen=True, slots=True)
class Precision:
    """One rung of the quantisation ladder."""

    key: str
    #: What to pass vLLM as --quantization. None means native weights.
    vllm_quantization: str | None
    #: Bytes per weight, for the VRAM estimate only. The measured peak is what
    #: gets published; this exists to catch "that was never going to fit"
    #: before renting anything.
    bytes_per_weight: float
    note: str = ""


LADDER: dict[str, Precision] = {
    "fp16": Precision(
        key="fp16",
        vllm_quantization=None,
        bytes_per_weight=2.0,
        note="baseline; every quality delta is measured against this rung",
    ),
    "int8": Precision(
        key="int8",
        vllm_quantization="compressed-tensors",
        bytes_per_weight=1.0,
        note="W8A8; needs a pre-quantised checkpoint, not a runtime flag",
    ),
    "int4": Precision(
        key="int4",
        vllm_quantization="awq",
        bytes_per_weight=0.5,
        note="AWQ; the rung the extraction task is expected to suffer on",
    ),
}


def priced(gpu: GpuSpec) -> bool:
    """False when a cost figure for this GPU would be fabricated."""
    return gpu.market_usd_per_hour > 0 and bool(gpu.rate_read_on)


def get_gpu(key: str) -> GpuSpec:
    if key not in GPUS:
        raise KeyError(f"unknown gpu {key!r}; known: {sorted(GPUS)}")
    return GPUS[key]


def get_precision(key: str) -> Precision:
    if key not in LADDER:
        raise KeyError(f"unknown precision {key!r}; known: {sorted(LADDER)}")
    return LADDER[key]


def vram_estimate_gb(params_b: float, precision: str) -> float:
    """Weights only, in GB. Deliberately excludes KV cache and activations.

    This is a go/no-go check before renting, not a capacity plan. The KV cache
    is the thing that actually decides whether a context length fits, and it
    depends on batch size and sequence length -- so it is measured, not
    guessed. See `bench.metrics` for the measured figure.
    """
    return params_b * get_precision(precision).bytes_per_weight
