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
        obtained_via="Modal monthly free credit",
    ),
    "l4": GpuSpec(
        key="l4",
        name="NVIDIA L4",
        vram_gb=24,
        market_usd_per_hour=0.0,
        obtained_via="Lightning AI free tier",
    ),
}


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
