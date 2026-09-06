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
from matplotlib.ticker import FuncFormatter

from bench.config import USD_TO_INR
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
