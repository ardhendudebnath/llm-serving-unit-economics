"""The published charts.

The crossover chart is the deliverable -- the one thing from this project a
reader is likely to share -- so most of the care here goes into not lying with
it.

**Self-hosted cost is drawn as the step function it is.** A GPU bills the same
idle as saturated, so monthly cost is flat across each card's capacity and then
jumps when the next card is needed. Drawing that as a smooth line would be the
single most common way this chart gets faked, and it would move the apparent
crossover. `drawstyle="steps-post"` is not a style choice here; it is the
finding.

**The x-axis is logarithmic**, because the interesting range spans five orders
of magnitude and a linear axis puts every sample past the crossover.

**Utilisation is drawn on the same figure**, because a cost curve without it
invites exactly the question the plan says you will be asked. The region below
20 % utilisation is shaded: in there the per-request cost is dominated by idle
time and the comparison flatters nobody.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Non-interactive backend, chosen before pyplot is imported. Charts are
# rendered in CI and on headless boxes; without this, importing pyplot can try
# to open a display and fail.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from bench.config import GPU_MEMORY_UTILISATION, LADDER, USD_TO_INR
from bench.cost import (
    ApiPricing,
    Capacity,
    crossover,
    curve,
    log_volumes,
)

#: Utilisation below which a self-hosted cost figure is not worth quoting.
#: Same threshold as the GpuUnderutilisedCostWaste alert.
LOW_UTILISATION = 0.20

SELF_HOST_COLOUR = "#1f6feb"
API_COLOUR = "#d1242f"
GRID_COLOUR = "#d8dee4"
ABSENT_COLOUR = "#8c959f"

#: Ladder order, so every chart reads fp16 -> int8 -> int4 whatever order the
#: result files were found in.
PRECISION_ORDER: tuple[str, ...] = tuple(LADDER)
PRECISION_COLOURS = {"fp16": "#1f6feb", "int8": "#8250df", "int4": "#bf8700"}

#: The harness metrics on the quality chart, in the order they matter: the
#: slab is the task; heading and chapter say how close a miss came.
QUALITY_METRICS: tuple[tuple[str, str], ...] = (
    ("slab_acc", "GST slab"),
    ("hsn_acc", "HSN heading"),
    ("chapter_acc", "HSN chapter"),
)


def _thousands(value: float, _pos: int = 0) -> str:
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if value >= limit:
            trimmed = value / limit
            return f"{trimmed:.0f}{suffix}" if trimmed >= 10 else f"{trimmed:.1f}{suffix}"
    return f"{value:.0f}"


def crossover_chart(
    capacity: Capacity,
    api: ApiPricing,
    out_path: Path,
    *,
    peak_to_mean: float = 1.0,
    currency: str = "INR",
    title: str | None = None,
) -> Path:
    """Self-hosted vs API monthly cost against request volume.

    Returns the path written. Raises `UnpricedError` upstream if the GPU rate
    has not been read from a provider -- this chart cannot be drawn from an
    invented hourly rate, which is the point.
    """
    volumes = log_volumes()
    points = curve(capacity, api, volumes, peak_to_mean=peak_to_mean)
    cross = crossover(capacity, api, peak_to_mean=peak_to_mean)

    to_currency = (lambda usd: usd * USD_TO_INR) if currency == "INR" else (lambda u: u)
    symbol = "₹" if currency == "INR" else "$"

    fig, (ax, ax_util) = plt.subplots(
        2, 1, figsize=(11, 8), height_ratios=[3, 1], sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    self_host = [to_currency(p.self_host_usd) for p in points]
    api_cost = [to_currency(p.api_usd) for p in points]

    # steps-post: the cost holds flat until the next GPU is needed, then jumps.
    ax.plot(volumes, self_host, drawstyle="steps-post", color=SELF_HOST_COLOUR,
            linewidth=2.4, label=f"Self-hosted ({capacity.gpu.name})", zorder=3)
    ax.plot(volumes, api_cost, color=API_COLOUR, linewidth=2.4,
            label=f"API ({api.model_id})", zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(f"Monthly cost ({symbol})")
    ax.grid(True, which="major", color=GRID_COLOUR, linewidth=0.7)
    ax.grid(True, which="minor", color=GRID_COLOUR, linewidth=0.3, alpha=0.5)

    if cross is not None:
        at = next(p for p in points if p.requests_per_month >= cross)
        ax.axvline(cross, color="#57606a", linestyle="--", linewidth=1.2, zorder=2)
        ax.annotate(
            f"crossover\n{_thousands(cross)} req/month\n"
            f"({at.utilisation:.0%} GPU utilisation)",
            xy=(cross, to_currency(at.self_host_usd)),
            xytext=(12, -60), textcoords="offset points",
            fontsize=9, color="#24292f",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
                  "edgecolor": "#57606a", "alpha": 0.95},
            arrowprops={"arrowstyle": "->", "color": "#57606a"},
        )
        subtitle = (
            f"Self-hosting is cheaper above ~{_thousands(cross)} requests/month "
            f"at a p95 of {capacity.slo_p95_s:g}s"
        )
    else:
        # A real outcome, not a missing measurement. Said plainly rather than
        # drawn as a crossover somewhere off the right edge.
        subtitle = (
            "Self-hosting is never cheaper at this quality bar — the API's "
            "per-request price is below the self-hosted marginal cost"
        )

    # pad leaves room for the subtitle drawn just below it; at a smaller pad
    # the title's descenders sit on the subtitle's capitals.
    ax.set_title(
        title or f"Serving {capacity.profile} at {capacity.precision}",
        fontsize=13, fontweight="bold", loc="left", pad=28,
    )
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=10,
            color="#57606a", va="bottom")
    ax.legend(frameon=False, loc="upper left")

    # --- utilisation, sharing the x-axis -----------------------------------
    utilisation = [p.utilisation for p in points]
    ax_util.plot(volumes, utilisation, color="#8250df", linewidth=2.0,
                 drawstyle="steps-post")
    ax_util.axhspan(0, LOW_UTILISATION, color="#d1242f", alpha=0.08, zorder=0)
    ax_util.axhline(LOW_UTILISATION, color="#d1242f", linewidth=0.9,
                    linestyle=":", zorder=1)
    ax_util.text(
        volumes[0], LOW_UTILISATION * 0.45,
        "  below 20% — per-request cost is mostly idle time",
        fontsize=8, color="#d1242f", va="center",
    )

    ax_util.set_ylim(0, 1.05)
    ax_util.set_ylabel("GPU\nutilisation")
    ax_util.set_xlabel("Requests per month")
    ax_util.grid(True, which="major", color=GRID_COLOUR, linewidth=0.7)
    ax_util.xaxis.set_major_formatter(FuncFormatter(_thousands))
    ax_util.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))

    footnote = (
        f"GPU {symbol}{to_currency(capacity.gpu.market_usd_per_hour):,.2f}/hr "
        f"(read {capacity.gpu.rate_read_on or 'UNREAD'}) · "
        f"API prices read {api.read_on} · "
        f"peak-to-mean {peak_to_mean:g}× · "
        f"knee {capacity.knee_rps:g} rps at p95 ≤ {capacity.slo_p95_s:g}s\n"
        "Excludes engineering time, on-call, redundancy and idle capacity "
        "beyond the peak-to-mean ratio. One replica, no HA."
    )
    fig.text(0.01, -0.02, footnote, fontsize=7.5, color="#57606a", va="top")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def latency_vs_load_chart(
    sweeps: list[dict],
    out_path: Path,
    *,
    slo_p95_s: float | None = None,
    title: str = "Latency vs load, by workload profile",
) -> Path:
    """p50/p95/p99 against arrival rate, one panel per profile.

    Percentiles, never a mean. Points where the sweep recorded GPU throttling
    are ringed, because their throughput is a floor rather than the card's
    sustained rate and a reader should not have to take that on trust.
    """
    if not sweeps:
        raise ValueError("no sweeps to plot")

    fig, axes = plt.subplots(
        1, len(sweeps), figsize=(5.2 * len(sweeps), 4.6), sharey=True, squeeze=False,
    )

    for ax, sweep in zip(axes[0], sweeps, strict=True):
        points = sorted(sweep["points"], key=lambda p: p["target_rate_rps"])
        rates = [p["target_rate_rps"] for p in points]

        for key, style, label in (
            ("p50", {"linewidth": 1.6, "alpha": 0.75}, "p50"),
            ("p95", {"linewidth": 2.4}, "p95"),
            ("p99", {"linewidth": 1.6, "linestyle": "--"}, "p99"),
        ):
            ax.plot(rates, [p["total_latency_s"][key] for p in points],
                    marker="o", markersize=4, label=label, **style)

        throttled = [
            (p["target_rate_rps"], p["total_latency_s"]["p95"])
            for p in points
            # Parenthesised deliberately: `x or 0 > 0.1` parses as
            # `x or (0 > 0.1)`, which rings every point with any throttling at
            # all rather than the ones where it mattered.
            if ((p.get("gpu") or {}).get("throttled_fraction") or 0.0) > 0.1
        ]
        if throttled:
            ax.scatter(*zip(*throttled, strict=True), s=140, facecolors="none",
                       edgecolors="#bf8700", linewidths=1.8, zorder=5,
                       label="GPU throttled")

        if slo_p95_s or sweep.get("slo_p95_s"):
            slo = slo_p95_s or sweep["slo_p95_s"]
            ax.axhline(slo, color="#d1242f", linestyle=":", linewidth=1.4)
            ax.text(rates[0], slo * 1.08, f" p95 SLO {slo:g}s", fontsize=8,
                    color="#d1242f", va="bottom")

        if knee := sweep.get("knee_rps"):
            ax.axvline(knee, color="#57606a", linestyle="--", linewidth=1.0)
            # Anchored to the axes rather than the data, so the label cannot
            # land on top of a line whose height depends on the measurement.
            ax.annotate(
                f"knee {knee:g} rps",
                xy=(knee, 1.0), xycoords=("data", "axes fraction"),
                xytext=(4, -6), textcoords="offset points",
                fontsize=8, color="#57606a", va="top", ha="left",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white",
                      "edgecolor": "none", "alpha": 0.85},
            )

        ax.set_title(f"{sweep['profile']} · {sweep['precision']}", fontsize=11)
        ax.set_xlabel("Arrival rate (req/s, Poisson)")

        # Both axes logarithmic, with explicit ticks.
        #
        # Log y because the sweep spans sub-second to saturated: on a linear
        # axis the whole sub-SLO region -- the part any capacity decision
        # actually turns on -- compresses into the bottom few pixels.
        #
        # Explicit ticks because matplotlib's default log minor labels collide
        # into unreadable overlap at these ranges. The measured rates are the
        # only x values that exist, so they are the only ones worth labelling.
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(rates)
        ax.set_xticklabels([f"{r:g}" for r in rates], fontsize=9)
        ax.minorticks_off()
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _: f"{v:g}" if v >= 1 else f"{v:.1f}"
        ))
        ax.grid(True, which="major", color=GRID_COLOUR, linewidth=0.7)

    axes[0][0].set_ylabel("End-to-end latency (s)")
    axes[0][0].legend(frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def throughput_vs_precision_chart(
    sweeps: list[dict],
    out_path: Path,
    *,
    title: str = "Capacity by precision, at the p95 SLO",
) -> Path:
    """The knee for each workload profile, one bar per rung of the ladder.

    Two kinds of absence are drawn differently from each other and from a bar,
    because they mean different things:

      "no rate met the SLO"   measured: the server met the target at no rate tried
      "not measured"          no sweep exists for this rung yet

    A zero-height bar would blur the two, and make a rung that was never run
    look like one that failed.

    Each bar carries what was measured at its knee -- output tokens/s and p95
    time to first token -- plus the peak VRAM across the rung's sweep.
    """
    if not sweeps:
        raise ValueError("no sweeps to plot")

    profiles = list(dict.fromkeys(s["profile"] for s in sweeps))
    by_rung = {(s["profile"], s["precision"]): s for s in sweeps}
    throttled_knees = 0

    fig, axes = plt.subplots(
        1, len(profiles), figsize=(4.4 * len(profiles), 4.8), squeeze=False,
    )

    for ax, profile in zip(axes[0], profiles, strict=True):
        slo = next(s["slo_p95_s"] for s in sweeps if s["profile"] == profile)
        tallest = 0.0

        for x, precision in enumerate(PRECISION_ORDER):
            sweep = by_rung.get((profile, precision))
            knee = sweep.get("knee_rps") if sweep else None
            point = next(
                (p for p in sweep["points"] if p["target_rate_rps"] == knee), None
            ) if sweep and knee else None

            if sweep is None or point is None:
                # y in axes coordinates, so the label sits on the baseline
                # whatever scale the measured bars give the panel.
                ax.text(
                    x, 0.03, "not\nmeasured" if sweep is None else "no rate met\nthe SLO",
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=8, color=ABSENT_COLOUR if sweep is None else API_COLOUR,
                )
                continue

            ax.bar(x, knee, width=0.62, zorder=3,
                   color=PRECISION_COLOURS.get(precision, ABSENT_COLOUR))
            tallest = max(tallest, knee)

            label = [f"{knee:g} rps"]
            if (tok_s := point.get("output_tokens_per_s")) is not None:
                label.append(f"{tok_s:,.0f} tok/s out")
            if (ttft := (point.get("ttft_s") or {}).get("p95")) is not None:
                label.append(f"TTFT p95 {ttft:.2f}s")
            peak = max(((p.get("gpu") or {}).get("peak_vram_mib") or 0.0)
                       for p in sweep["points"])
            if peak:
                label.append(f"{peak / 1024:.1f} GiB peak")
            if ((point.get("gpu") or {}).get("throttled_fraction") or 0.0) > 0.1:
                throttled_knees += 1

            ax.annotate("\n".join(label), xy=(x, knee), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7.5, color="#24292f")

        ax.set_xticks(range(len(PRECISION_ORDER)))
        ax.set_xticklabels(PRECISION_ORDER)
        ax.set_xlim(-0.6, len(PRECISION_ORDER) - 0.4)
        # Headroom for four lines of label above the tallest bar.
        ax.set_ylim(0, tallest * 1.7 if tallest else 1.0)
        ax.set_title(f"{profile} · p95 ≤ {slo:g}s", fontsize=11)
        ax.grid(True, axis="y", color=GRID_COLOUR, linewidth=0.7)
        ax.set_axisbelow(True)

    axes[0][0].set_ylabel("Knee (req/s)")
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()

    footnote = [
        ("Knee: the highest Poisson arrival rate whose p95 end-to-end latency met "
         "the SLO. Labels are measured at the knee; peak VRAM is across the sweep."),
        ("Peak VRAM is mostly vLLM's preallocation (gpu_memory_utilization "
         f"{GPU_MEMORY_UTILISATION:g}): smaller weights buy KV cache, not free memory."),
    ]
    if throttled_knees:
        footnote.append(
            f"The GPU was throttled for over 10 % of busy samples at {throttled_knees} "
            "of these knees, so those rates are floors."
        )
    fig.text(0.01, -0.01, "\n".join(footnote), fontsize=7.5, color="#57606a", va="top")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def quality_vs_precision_chart(
    quality: dict[str, dict],
    out_path: Path,
    *,
    title: str = "Quality by precision, on Project 01's harness",
) -> Path:
    """Mean accuracy per rung, with whiskers spanning every repeat run.

    `quality` maps precision to a record written by `gate/record.py`: the fp16
    baseline, or a rung's spread.json. The whisker is the full min-max of the
    repeats rather than a standard deviation. Five runs barely estimate a
    standard deviation, and the range is what the quality gate's tolerance is
    built from, so the chart shows a reader the same thing the gate acts on.

    A rung with no record is labelled "not measured", never drawn at zero.
    """
    if not quality:
        raise ValueError("no quality records to plot")

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    slot = 0.8 / len(PRECISION_ORDER)
    handles = []

    for j, precision in enumerate(PRECISION_ORDER):
        record = quality.get(precision) or {}
        scores = record.get("scores") or {}
        colour = PRECISION_COLOURS.get(precision, ABSENT_COLOUR)
        offset = (j - (len(PRECISION_ORDER) - 1) / 2) * slot

        for i, (key, _) in enumerate(QUALITY_METRICS):
            x = i + offset
            if key not in scores:
                ax.text(x, 0.02, "not measured", transform=ax.get_xaxis_transform(),
                        rotation=90, ha="center", va="bottom", fontsize=7,
                        color=ABSENT_COLOUR)
                continue

            mean = scores[key] * 100.0
            runs = (record.get("observed_per_run") or {}).get(key) or [mean]
            ax.bar(x, mean, width=slot * 0.9, color=colour, zorder=3)
            # Clamped at zero: the mean is stored unrounded and the runs to two
            # decimals, so when every run agrees one side can come out a hair
            # negative -- and matplotlib rejects a negative error bar.
            ax.errorbar(
                x, mean,
                yerr=[[max(0.0, mean - min(runs))], [max(0.0, max(runs) - mean)]],
                color="#24292f", capsize=3, linewidth=1.1, zorder=4,
            )
            ax.text(x, max(*runs, mean) + 1.5, f"{mean:.1f}", ha="center",
                    va="bottom", fontsize=7.5)

        if scores:
            runs_n = record.get("n_repeats") or "?"
            handles.append(Patch(color=colour, label=f"{precision} ({runs_n} runs)"))

    ax.set_xticks(range(len(QUALITY_METRICS)))
    ax.set_xticklabels([label for _, label in QUALITY_METRICS])
    ax.set_xlim(-0.6, len(QUALITY_METRICS) - 0.4)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, axis="y", color=GRID_COLOUR, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.text(
        0.01, -0.01,
        "Bars: mean over repeat runs. Whiskers: lowest to highest run. A rung whose "
        "bar sits inside fp16's whisker is within the noise the quality gate measured.",
        fontsize=7.5, color="#57606a", va="top",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
