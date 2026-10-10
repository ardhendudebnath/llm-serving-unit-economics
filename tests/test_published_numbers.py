"""The numbers in the README are the numbers in results/, and this says so.

Every headline in this repo is a measurement: a knee from a sweep file, a mean
from five eval runs, a crossover the cost model solves. They are then *typed
into prose*, and prose has no compiler. A sweep re-run at a different SLO, a
sixth eval run, a changed USD rate -- any of them moves the data and leaves the
README asserting the old figure, in bold, above a chart that now disagrees with
it. Nothing else in the build notices.

That is the same failure `test_doc_links.py` exists to stop, one level deeper:
there the link resolved to a file, here the sentence resolves to a number. For
a repo whose whole claim is "measured, not estimated", the claim itself should
be under test.

So these tests parse the published tables out of README.md and recompute each
cell from the committed data. They deliberately recompute rather than compare
against stored expectations: a stored expectation is another copy of the prose
and drifts with it.

Most of this file needs nothing but the standard library, so it runs in the
dependency-free CI job alongside the gate. Only the crossover test reaches for
`bench.report.build`, which pulls in matplotlib; it skips where that is absent
and runs in the chart job. See .github/workflows/ci.yml.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

#: The same text with every run of whitespace flattened to one space. The
#: README is hard-wrapped, so a sentence quoting a number is regularly split
#: across two lines -- and reflowing a paragraph must not fail a test about
#: arithmetic.
PROSE = re.sub(r"\s+", " ", README)

SWEEPS = ROOT / "results" / "sweeps"
EVAL = ROOT / "results" / "eval"
BASELINE = ROOT / "gate" / "baseline.json"

PROFILES = ("short", "long_in", "long_out")
PRECISIONS = ("fp16", "int8", "int4")

#: README metric label -> the key the harness records it under.
QUALITY_METRICS = {
    "Slab accuracy": "slab_acc",
    "HSN accuracy": "hsn_acc",
    "Chapter accuracy": "chapter_acc",
    "Abstention accuracy": "abstention_acc",
    "Stale-slab rate": "stale_slab_rate",
    "Unparseable": "unparseable",
}

#: A capacity cell: "**8 rps** · p95 1.78 s", or "**none**" where no arrival
#: rate met the SLO. The second form carries no p95 by design -- there is no
#: passing point to quote one from.
CAPACITY_CELL = re.compile(
    r"\*\*(?P<knee>none|[\d.]+)(?:\s*rps)?\*\*"
    r"(?:\s*·\s*p95\s*(?P<p95>[\d.]+)\s*s)?"
)

#: A quality cell: "41.4 % (39.3–42.9)", or "25.0 %" where every run scored
#: the same and a range would be noise.
QUALITY_CELL = re.compile(
    r"(?P<mean>[\d.]+)\s*%"
    r"(?:\s*\((?P<lo>[\d.]+)\s*[–—-]\s*(?P<hi>[\d.]+)\))?"
)


def _table(header: str) -> list[list[str]]:
    """Rows of the markdown table whose header line starts with `header`.

    Matched on the header rather than on position, so inserting a paragraph
    above a table does not silently point this at a different one.
    """
    lines = README.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.startswith(header)), None
    )
    assert start is not None, f"no table in README.md with header {header!r}"

    rows = []
    for line in lines[start + 2:]:  # +2 skips the header and its underline
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def _sweep(profile: str, precision: str) -> dict:
    path = SWEEPS / f"sweep_{profile}_{precision}.json"
    assert path.exists(), f"README publishes a figure with no sweep behind it: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _quality_record(precision: str) -> dict:
    """Where each rung's five runs were banked.

    fp16's live in the gate baseline because fp16 *is* the baseline; the other
    rungs carry their own spread file next to their runs.
    """
    path = BASELINE if precision == "fp16" else EVAL / precision / "spread.json"
    assert path.exists(), f"README publishes quality for {precision} with no record: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _number(pattern: str) -> float:
    """The single number matching `pattern` in the README, commas removed."""
    found = re.search(pattern, PROSE)
    assert found, f"the README no longer states the number matched by {pattern!r}"
    return float(found.group(1).replace(",", ""))


# ------------------------------------------------------------------ guards --

def test_the_readme_tables_are_actually_being_parsed():
    # Without this, a renamed heading or a reformatted table would empty the
    # tests below and they would pass by checking nothing at all.
    capacity = _table("| Profile | fp16 | int8 | int4 |")
    quality = _table("| Metric | fp16 | int8 | int4 |")
    assert len([r for r in capacity if r[0].strip("`") in PROFILES]) == len(PROFILES)
    assert len(quality) >= len(QUALITY_METRICS)


# --------------------------------------------------------------- capacity --

@pytest.mark.parametrize("profile", PROFILES)
def test_the_capacity_table_matches_the_sweeps(profile: str):
    row = next(r for r in _table("| Profile | fp16 | int8 | int4 |")
               if r[0].strip("`") == profile)

    for precision, cell in zip(PRECISIONS, row[1:]):
        parsed = CAPACITY_CELL.search(cell)
        assert parsed, f"unreadable capacity cell for {profile}/{precision}: {cell!r}"

        sweep = _sweep(profile, precision)
        published = parsed.group("knee")

        if published == "none":
            assert not sweep.get("knee_rps"), (
                f"README says {profile}/{precision} never met the SLO, but the "
                f"sweep records a knee at {sweep.get('knee_rps')} rps"
            )
            continue

        knee = sweep.get("knee_rps")
        assert knee == pytest.approx(float(published)), (
            f"README publishes a {published} rps knee for {profile}/{precision}; "
            f"the sweep measured {knee}"
        )

        point = next(p for p in sweep["points"] if p["target_rate_rps"] == knee)
        measured = round(point["total_latency_s"]["p95"], 2)
        assert measured == pytest.approx(float(parsed.group("p95"))), (
            f"README publishes p95 {parsed.group('p95')} s at the "
            f"{profile}/{precision} knee; the sweep measured {measured} s"
        )


def test_every_published_knee_was_measured_against_the_slo_in_the_heading():
    """The table is titled "at a p95 of 10 s". The sweeps must agree.

    A knee only means something next to the target it was judged against, and
    that target lives in three places at once -- the heading here, the Makefile
    and the alert rules. This holds the first two together; test_observability
    holds the third.
    """
    slo = _number(r"### Capacity at a p95 of ([\d.]+) s")
    for profile in PROFILES:
        for precision in PRECISIONS:
            assert _sweep(profile, precision)["slo_p95_s"] == pytest.approx(slo)


# ---------------------------------------------------------------- quality --

@pytest.mark.parametrize("metric", sorted(QUALITY_METRICS))
def test_the_quality_table_matches_the_recorded_runs(metric: str):
    key = QUALITY_METRICS[metric]
    row = next(r for r in _table("| Metric | fp16 | int8 | int4 |")
               if r[0] == metric)

    for precision, cell in zip(PRECISIONS, row[1:]):
        parsed = QUALITY_CELL.search(cell)
        assert parsed, f"unreadable quality cell for {metric}/{precision}: {cell!r}"

        record = _quality_record(precision)
        runs = record["observed_per_run"][key]
        mean = round(record["scores"][key] * 100, 1)

        assert mean == pytest.approx(float(parsed.group("mean"))), (
            f"README publishes {metric} {parsed.group('mean')} % for "
            f"{precision}; the {len(runs)} recorded runs mean {mean} %"
        )

        if parsed.group("lo") is None:
            # No bracket is itself a claim: every run scored the same.
            assert round(min(runs), 1) == round(max(runs), 1), (
                f"README prints {metric}/{precision} with no range, but its "
                f"runs span {min(runs)}–{max(runs)}"
            )
            continue

        assert round(min(runs), 1) == pytest.approx(float(parsed.group("lo")))
        assert round(max(runs), 1) == pytest.approx(float(parsed.group("hi")))


def test_the_published_run_count_is_the_run_count_recorded():
    """"Five runs per rung" is a claim about the files, not a turn of phrase.

    The spread in the table, and the gate tolerance derived from it, are only
    worth what the repeat count behind them is worth.
    """
    assert "Five runs per rung" in README
    for precision in PRECISIONS:
        record = _quality_record(precision)
        assert record["n_repeats"] == 5, f"{precision} banked {record['n_repeats']} runs"
        assert len(record["source_runs"]) == 5


# -------------------------------------------------------------- crossover --

def test_the_headline_crossover_follows_from_the_committed_sweep():
    """The whole cost argument, recomputed from the sweep and the quoted price.

    The API price is read back out of the README rather than from Project 01's
    registry, on purpose. It makes this test self-contained -- it runs where
    the harness is not installed, which is everywhere CI runs -- and it checks
    the thing most likely to rot: that the published volume still follows from
    the price printed beside it.
    """
    pytest.importorskip("matplotlib")  # bench.report.build draws

    from dataclasses import replace

    from bench.config import LAPTOP, USD_TO_INR
    from bench.cost import ApiPricing, crossover, monthly
    from bench.report.build import capacity_from

    usd_in = _number(r"list price of \$([\d.]+) and \$[\d.]+ per million")
    usd_out = _number(r"list price of \$[\d.]+ and \$([\d.]+) per million")
    api = ApiPricing(
        model_id="published-in-readme",
        usd_in_per_m=usd_in,
        usd_out_per_m=usd_out,
        read_on=re.search(r"\(read (\d{4}-\d{2}-\d{2})\)", PROSE).group(1),
    )

    capacity = capacity_from(_sweep("long_in", "int8"), "rtx5070ti-laptop")
    assert capacity is not None, "the int8 long_in sweep no longer carries a priceable knee"
    capacity = replace(capacity, gpu=LAPTOP.priced_gpu())

    volume = crossover(capacity, api)
    assert volume is not None, "self-hosting no longer wins at any volume"
    point = monthly(capacity, api, volume)

    published = {
        "crossover requests/month": (
            _number(r"would charge for ([\d,]+) `long_in` requests"), round(volume)
        ),
        "headline, to the nearest thousand": (
            _number(r"above \*\*about ([\d,]+) `long_in` requests a month\*\*"),
            round(volume, -3),
        ),
        "self-host ₹/month": (
            _number(r"₹([\d,]+) a month"),
            round(point.self_host_usd * USD_TO_INR),
        ),
        "API ₹/request": (
            _number(r"₹([\d.]+) each"),
            round(api.usd_per_request(capacity.mean_tokens_in,
                                      capacity.mean_tokens_out) * USD_TO_INR, 3),
        ),
        "self-host floor ₹/request": (
            _number(r"self-hosted floor is ₹([\d.]+) a request"),
            round(capacity.inr_per_1000_requests() / 1000, 4),
        ),
        "utilisation at the break-even, %": (
            _number(r"break-even sits at ([\d.]+) % GPU utilisation"),
            round(point.utilisation * 100, 1),
        ),
    }
    wrong = {
        name: (says, computes)
        for name, (says, computes) in published.items()
        if says != pytest.approx(computes)
    }
    assert not wrong, f"README publishes figures the cost model no longer produces: {wrong}"


def test_the_crossover_really_is_the_same_at_every_precision():
    """The README claims quantisation cannot move the crossover. Check it.

    It holds only because one card covers the crossover volume many times over,
    so the step function never steps and the rungs share a knee. If a future
    sweep breaks that, the claim becomes false quietly -- the chart would still
    render, with a different curve and the same sentence beside it.
    """
    pytest.importorskip("matplotlib")

    from dataclasses import replace

    from bench.config import LAPTOP
    from bench.cost import ApiPricing, crossover
    from bench.report.build import capacity_from

    api = ApiPricing(model_id="published-in-readme", usd_in_per_m=1.0,
                     usd_out_per_m=5.0, read_on="2026-06-24")

    volumes = {}
    for precision in PRECISIONS:
        capacity = capacity_from(_sweep("long_in", precision), "rtx5070ti-laptop")
        if capacity is None:
            continue
        volumes[precision] = crossover(
            replace(capacity, gpu=LAPTOP.priced_gpu()), api
        )

    assert len(volumes) == len(PRECISIONS), f"only priced {sorted(volumes)}"
    assert len({round(v) for v in volumes.values()}) == 1, (
        f"the README says the crossover is the same at every precision: {volumes}"
    )
