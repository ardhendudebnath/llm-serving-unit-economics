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

from dataclasses import dataclass, replace

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
    #: As nvidia-smi reports it, in MiB. Stored in the unit it was measured in
    #: because "12 GB" is ambiguous by 7 % -- this card is 12,227 MiB, which is
    #: 11.94 GiB or 12.82 GB depending on which you meant, and the difference
    #: is larger than the margin that made vLLM refuse to start.
    vram_mib: int
    #: On-demand rate for this class, in USD/hour. Zero means unread, and
    #: `priced()` will refuse to build a cost figure from it.
    market_usd_per_hour: float
    #: Fraction of total VRAM that is actually available to a process.
    #:
    #: **Not 1.0, and assuming it is will cost you a failed start.** On this
    #: laptop the GPU also drives the display under WDDM, and vLLM measures
    #: free memory only after creating its own CUDA context. Measured: vLLM saw
    #: 10.74 of 11.94 GiB free, so 0.90 utilisation -- 10.75 GiB -- failed by
    #: 10 MiB, while nvidia-smi had reported 11,737 MiB free moments earlier.
    usable_fraction: float = 1.0
    #: When the rate above was read, and from where. Both are printed in the
    #: report; a rate without a source is an assertion, not a measurement.
    rate_read_on: str = ""
    rate_source: str = ""
    #: How this project actually obtained the hours. Recorded so the report can
    #: say plainly that the measurements were free and the pricing is not.
    obtained_via: str = ""

    def inr_per_hour(self) -> float:
        return self.market_usd_per_hour * USD_TO_INR

    @property
    def vram_gb(self) -> float:
        """Total VRAM in decimal GB, to match the checkpoint sizes.

        Repo sizes are byte counts divided by 1e9, so comparing them against a
        GiB figure would understate what fits by 7 %.
        """
        return self.vram_mib * 1024 * 1024 / 1e9

    @property
    def usable_gb(self) -> float:
        """What a process can actually claim. This is the number that binds."""
        return self.vram_gb * self.usable_fraction


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
        vram_mib=12_227,
        # 10.74 of 11.94 GiB, measured by vLLM at startup on 2026-09-07 --
        # not read from nvidia-smi, which reported considerably more free.
        usable_fraction=0.90,
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
        vram_mib=16_384,
        market_usd_per_hour=0.0,
        obtained_via="Colab / Kaggle free tier",
    ),
    "a10g": GpuSpec(
        key="a10g",
        name="NVIDIA A10G",
        vram_mib=24_576,
        market_usd_per_hour=0.0,
        obtained_via="rentable",
    ),
    "l4": GpuSpec(
        key="l4",
        name="NVIDIA L4",
        vram_mib=24_576,
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


#: How much of its life the card spends serving -- for the cost per *serving
#: hour* only. Reported as a set because that figure moves about 11x across it.
#:
#: **Deliberately not used to price the crossover.** The crossover already
#: accounts for idle time: monthly cost is flat whatever the volume, and the
#: chart's utilisation panel shows how much of the card sits unused. Pricing the
#: card at a partial duty cycle as well counts those idle hours twice. An
#: earlier version did exactly that, and would have put the 2 h/day crossover
#: about 11x too high.
DUTY_CYCLES: dict[str, float] = {
    "24 h/day": 1.0,
    "8 h/day": 8 / 24,
    "2 h/day": 2 / 24,
}


@dataclass(frozen=True, slots=True)
class OwnedHardware:
    """What owning a card costs, with every input sourced.

    Two different figures, and mixing them up is the trap:

      inr_per_serving_hour(d)   what one hour of serving costs when the card
                                serves fraction d of its life. Moves about 11x
                                across DUTY_CYCLES. An explanatory figure.

      priced_gpu()              the card as a GpuSpec for the crossover, priced
                                as available around the clock -- the same basis
                                a rented card is billed on, which is what
                                monthly() assumes.

    The laptop's `GpuSpec` in GPUS stays unpriced, so a crossover can only be
    drawn from the second.
    """

    gpu_key: str
    price_inr: float
    price_source: str
    price_read_on: str
    useful_life_years: float
    life_basis: str
    power_w: float
    power_basis: str
    tariff_inr_per_kwh: float
    tariff_basis: str

    def usd_per_serving_hour(self, duty_cycle: float) -> float:
        return amortised_usd_per_hour(
            hardware_usd=self.price_inr / USD_TO_INR,
            useful_life_years=self.useful_life_years,
            duty_cycle=duty_cycle,
            mean_power_w=self.power_w,
            electricity_usd_per_kwh=self.tariff_inr_per_kwh / USD_TO_INR,
        )

    def inr_per_serving_hour(self, duty_cycle: float) -> float:
        return self.usd_per_serving_hour(duty_cycle) * USD_TO_INR

    def priced_gpu(self) -> GpuSpec:
        """The card as a `GpuSpec` for the crossover chart.

        Priced as available around the clock: capital spread over every hour of
        its life, electricity at the power limit for every hour. `monthly()`
        charges a card's rate for every hour of the month and lets utilisation
        show how much of it sat idle, so idleness must not be discounted here
        as well -- that would count the idle hours twice.

        Takes no duty cycle, on purpose, so the double count cannot be
        reintroduced by passing one.

        Charging the power limit around the clock overstates electricity for an
        idle card, which measured about 8 W. The overstatement is bounded by the
        electricity term, about 10 % of the rate, and errs against
        self-hosting rather than in its favour.
        """
        return replace(
            get_gpu(self.gpu_key),
            market_usd_per_hour=self.usd_per_serving_hour(1.0),
            rate_read_on=self.price_read_on,
            rate_source=(f"amortised over {self.useful_life_years:g} years, "
                         f"available around the clock; {self.price_source}"),
        )


#: The machine the measurements ran on: ASUS ROG Strix G16, G615LR-S5190WS,
#: Core Ultra 9 275HX, RTX 5070 Ti Laptop GPU 12 GB, 32 GB RAM, 1 TB -- read
#: from the system itself, not assumed.
#:
#: Price is today's street price, not what this particular unit cost its owner.
#: Read on 2026-09-11 from three listings that all confirm the same
#: configuration:
#:
#:   Flipkart   Rs 2,49,990 selling, Rs 2,75,990 MRP
#:     https://www.flipkart.com/asus-rog-strix-g16-2025-ai-pc-office-2024-m365-basic-intel-core-ultra-9-275hx-32-gb-1-tb-ssd-windows-11-home-12-gb-graphics-nvidia-geforce-rtx-5070-ti-240-hz-140-w-g615lr-s5190ws-gaming-laptop/p/itm59ca3b2dafe6c
#:   IndiaMART  Rs 2,59,990
#:     https://www.indiamart.com/proddetail/asus-rog-strix-g16-g615lr-s5190ws-gaming-laptop-2856670854191.html
#:   Pricekeeda Rs 2,75,990 (Amazon, out of stock)
#:     https://www.pricekeeda.com/asus-rog-strix-g16-g615lr-s5190ws-laptop-price-in-india/
#:
#: A search summary quoted Rs 3,59,990 for the Flipkart listing; the page
#: itself says Rs 2,49,990. Recorded because it is exactly how an unverified
#: figure would have entered the cost curve.
#:
#: Tariff: no national residential average could be verified -- a widely
#: repeated Rs 5.5/kWh figure traced to pages that refused to load, and
#: Wikipedia gives none. Rs 8/kWh is an assumption inside the Rs 3-14/kWh state
#: range NoBroker lists (updated 2026-07-26). It barely matters: electricity is
#: about 10 % of the hourly cost at 24 h/day, and the whole state range moves
#: that figure by about Rs 1.5.
LAPTOP = OwnedHardware(
    gpu_key="rtx5070ti-laptop",
    price_inr=249_990.0,
    price_source="Flipkart selling price Rs 2,49,990 for G615LR-S5190WS (MRP Rs 2,75,990)",
    price_read_on="2026-09-11",
    useful_life_years=3.0,
    life_basis="assumption: a common useful life for a laptop, not a measurement",
    power_w=140.0,
    power_basis=("the card's power limit as reported by nvidia-smi; GPU only, "
                 "so it excludes the CPU and the rest of the laptop"),
    tariff_inr_per_kwh=8.0,
    tariff_basis=("assumption inside the Rs 3-14/kWh state range listed by "
                  "NoBroker (updated 2026-07-26); no national average verifiable"),
)


#: The model under test.
#:
#: Qwen3-4B-Instruct-2507, chosen against the measured 12 GB constraint rather
#: than by reputation. `docs/models.md` records why, and what was rejected.
#:
#: **Non-thinking on purpose.** The hybrid-thinking Qwen3-4B emits its chain
#: into a separate field that still bills as output tokens, which would inflate
#: decode time and make the `long_in` profile measure reasoning rather than
#: extraction. Project 01's prompt wants four labelled lines.
MODEL_FAMILY = "Qwen3-4B-Instruct-2507"

#: Architecture, read from the published config.json on 2026-09-07. These drive
#: the KV-cache arithmetic below, so they are recorded rather than assumed.
N_LAYERS = 36
N_KV_HEADS = 8
HEAD_DIM = 128
#: The model supports 262,144 positions. Reserving for that would consume the
#: entire card in KV cache for context the workload never sends -- the
#: `long_in` corpus peaks near 1,800 tokens. See `kv_cache_gb()`.
MAX_POSITIONS = 262_144


@dataclass(frozen=True, slots=True)
class Precision:
    """One rung of the quantisation ladder."""

    key: str
    #: What to pass vLLM as --quantization. None means native weights.
    vllm_quantization: str | None
    #: The pre-quantised checkpoint this rung is served from. int8 and int4 are
    #: not runtime flags -- there is no converting an fp16 repo on the way in.
    repo: str
    #: Measured from the repo's safetensors on 2026-09-07, not derived from a
    #: parameter count. This is the number that decides what fits.
    weights_gb: float
    #: dtype vLLM is told to use. "auto" honours the checkpoint.
    dtype: str = "auto"
    note: str = ""


#: **int8 and int4 come from one publisher, in one format, on purpose.**
#: Mixing an AWQ community repo with a RedHatAI W8A8 would confound precision
#: with quantisation methodology -- the ladder would be measuring "whoever
#: quantised it" alongside the bit width, and the headline finding is supposed
#: to be about the bit width.
LADDER: dict[str, Precision] = {
    "fp16": Precision(
        key="fp16",
        vllm_quantization=None,
        repo="Qwen/Qwen3-4B-Instruct-2507",
        weights_gb=8.04,
        # **The checkpoint is bfloat16, not float16.** Forcing float16 on
        # BF16-trained weights narrows the exponent range and risks overflow in
        # attention -- a real numerical difference, not a naming quibble. The
        # rung is called "fp16" because the plan does, but what is served is
        # 16-bit native, and the report says which.
        dtype="bfloat16",
        note="16-bit baseline; every quality delta is measured against it",
    ),
    "int8": Precision(
        key="int8",
        vllm_quantization="compressed-tensors",
        repo="RedHatAI/Qwen3-4B-Instruct-2507-quantized.w8a8",
        weights_gb=5.19,
        note="W8A8, INT8 weights and activations",
    ),
    "int4": Precision(
        key="int4",
        vllm_quantization="compressed-tensors",
        repo="RedHatAI/Qwen3-4B-Instruct-2507-quantized.w4a16",
        weights_gb=3.43,
        # Not AWQ. Same compressed-tensors format as the int8 rung above, from
        # the same publisher, which is the whole point.
        note="W4A16, INT4 weights; the rung extraction is expected to suffer on",
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


def kv_cache_gb(tokens: int) -> float:
    """KV cache for `tokens` tokens, in GB, at 16-bit.

    2 (K and V) x kv_heads x head_dim x 2 bytes, per layer. For this model that
    is 144 KiB per token, so a single 4,096-token sequence costs 0.60 GB --
    which is why max sequence length and max batch size trade directly against
    each other on a 12 GB card rather than being independent knobs.
    """
    per_token = 2 * N_KV_HEADS * HEAD_DIM * 2 * N_LAYERS
    return per_token * tokens / 1e9


#: What deploy/k8s/configmap.yaml actually sets. Kept here so the estimate
#: below describes the deployed configuration rather than a hopeful one.
GPU_MEMORY_UTILISATION = 0.85


def max_safe_utilisation(gpu_key: str) -> float:
    """The highest `--gpu-memory-utilization` this card will actually accept.

    vLLM reads that flag as a fraction of *total* VRAM and then refuses to
    start if less than that is free, so the ceiling is `usable_fraction` --
    not 1.0, and not whatever nvidia-smi reports free.
    """
    return get_gpu(gpu_key).usable_fraction


def concurrent_sequences(precision: str, context_len: int, gpu_key: str,
                         *, gpu_memory_utilisation: float = GPU_MEMORY_UTILISATION
                         ) -> float:
    """How many sequences of `context_len` fit alongside the weights.

    An estimate, and labelled as one: it ignores activations, fragmentation and
    vLLM's own bookkeeping, so the real figure is somewhat lower. Its job is to
    catch a configuration that was never going to fit before a run starts, not
    to replace the measured `peak_vram_mib` the sweep records.

    Returns 0.0 for a utilisation the card cannot honour, because a
    configuration that will not start has no capacity -- reporting a cheerful
    sequence count for one would be worse than useless.
    """
    gpu = get_gpu(gpu_key)
    if gpu_memory_utilisation > gpu.usable_fraction:
        return 0.0

    budget = gpu.vram_gb * gpu_memory_utilisation
    free = budget - get_precision(precision).weights_gb
    if free <= 0:
        return 0.0
    return free / kv_cache_gb(context_len)
