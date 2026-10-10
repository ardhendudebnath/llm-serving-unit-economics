"""The numbers in the docs are the numbers in results/, and this says so.

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

The headline figures are then checked across every document that restates
them -- the README, the decision doc and the write-up. A figure written down
in three places drifts in two of them, and the decision doc is the one a
reader acts on.

Most of this file needs nothing but the standard library, so it runs in the
dependency-free CI job alongside the gate. Only the cost tests reach for
`bench.report.build`, which pulls in matplotlib; they skip where that is absent
and run in the chart job. See .github/workflows/ci.yml.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

def _flatten(text: str) -> str:
    """Markdown prose as one long line, so a sentence can be matched whole.

    Two things get in the way of reading a figure out of a sentence. These
    files are hard-wrapped, so a sentence quoting a number is regularly split
    across lines -- and reflowing a paragraph must never fail a test about
    arithmetic. And the decision doc states most of its figures inside
    blockquotes, where the wrap drops a "> " into the middle of the sentence.
    Both come out here.
    """
    return re.sub(r"\s+", " ", re.sub(r"(?m)^\s*>\s?", "", text))


#: The README, flattened. Line-based parsing of its tables uses README itself.
PROSE = _flatten(README)

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


# --------------------------------------------------- across the documents --

#: Everything a reader is handed. The decision doc and the write-up restate the
#: headline figures in their own words, and the decision doc is the one someone
#: acts on -- so it is the worst place for a stale number to survive.
DOCUMENTS = {
    name: _flatten((ROOT / name).read_text(encoding="utf-8"))
    for name in ("README.md", "docs/decision.md", "docs/post.md")
}


def _stated(*patterns: str) -> list[tuple[str, float]]:
    """Every value any document states for one figure, tagged with its file.

    Several patterns per figure because each document says it in its own
    words. Matching the phrasing rather than the digits is the point: a
    pattern containing the expected number would pass by tautology.
    """
    return [
        (name, float(match.replace(",", "")))
        for name, prose in DOCUMENTS.items()
        for pattern in patterns
        for match in re.findall(pattern, prose)
    ]


def _agree(figure: str, stated: list[tuple[str, float]], *accepted: float) -> None:
    """Every document's version of `figure` is one of the computed values.

    More than one is accepted where the docs legitimately round -- "about
    47,000" beside the exact 47,423 is good writing, not drift.
    """
    assert stated, (
        f"no document states {figure} any more. Either it was dropped, or it "
        "was reworded and this check is now watching nothing."
    )
    wrong = [
        (name, value) for name, value in stated
        if not any(value == pytest.approx(a) for a in accepted)
    ]
    assert not wrong, f"{figure}: the data gives {accepted}, the docs say {wrong}"


def _consistent(figure: str, stated: list) -> None:
    """The documents agree with each other, where there is no local source.

    The API's list price is read off a provider's page, so nothing in this
    repo can confirm it. What a test can still catch is one document updated
    and the other two left behind.
    """
    assert stated, f"no document states {figure} any more"
    values = {value for _, value in stated}
    assert len(values) == 1, f"{figure} is written three ways: {stated}"


def test_every_document_agrees_on_the_measured_quality():
    expected = {p: round(_quality_record(p)["scores"]["slab_acc"] * 100, 1)
                for p in PRECISIONS}

    _agree(
        "fp16 slab accuracy",
        _stated(r"gets ([\d.]+) % of slabs right",
                r"scores \*{0,2}([\d.]+) %\*{0,2} slab accuracy"),
        expected["fp16"],
    )
    _agree("int8 slab accuracy",
           _stated(r"\*{0,2}([\d.]+) %\*{0,2} at int8"), expected["int8"])
    _agree("int4 slab accuracy",
           _stated(r"\*{0,2}([\d.]+) %\*{0,2} at int4"), expected["int4"])


def test_every_document_agrees_on_the_measured_capacity():
    knee = _sweep("long_in", "fp16")["knee_rps"]
    _agree(
        "the long_in knee",
        _stated(r"`long_in` at \*{0,2}([\d.]+) rps",
                r"knees land at ([\d.]+) rps",
                r"same ([\d.]+) rps"),
        knee,
    )


def test_every_document_agrees_on_the_cost_figures():
    """The cost chain, recomputed once and checked wherever it is quoted."""
    pytest.importorskip("matplotlib")

    from dataclasses import replace

    from bench.config import LAPTOP, USD_TO_INR
    from bench.cost import HOURS_PER_MONTH, ApiPricing, crossover, monthly
    from bench.report.build import capacity_from

    _consistent("the API's input price",
                _stated(r"\$([\d.]+) and \$[\d.]+ per million"))
    _consistent("the API's output price",
                _stated(r"\$[\d.]+ and \$([\d.]+) per million"))
    _consistent(
        "the date the API price was read",
        [(name, re.search(r"read (\d{4}-\d{2}-\d{2})", prose).group(1))
         for name, prose in DOCUMENTS.items()
         if re.search(r"read (\d{4}-\d{2}-\d{2})", prose)],
    )

    usd_in = _stated(r"\$([\d.]+) and \$[\d.]+ per million")[0][1]
    usd_out = _stated(r"\$[\d.]+ and \$([\d.]+) per million")[0][1]
    api = ApiPricing(model_id="published-in-the-docs", usd_in_per_m=usd_in,
                     usd_out_per_m=usd_out, read_on="quoted above")

    capacity = capacity_from(_sweep("long_in", "int8"), "rtx5070ti-laptop")
    capacity = replace(capacity, gpu=LAPTOP.priced_gpu())
    volume = crossover(capacity, api)
    point = monthly(capacity, api, volume)

    _agree("the crossover volume",
           _stated(r"([\d,]+) `long_in` requests"),
           round(volume), round(volume, -3))
    # The write-up hedges once -- "about ₹7,760 a month" -- and gives the exact
    # figure where it does the arithmetic. Both are roundings of the same
    # number, and a stale figure is still not one of them.
    inr_per_month = point.self_host_usd * USD_TO_INR
    _agree("the laptop's monthly cost",
           _stated(r"₹([\d,]+) a month"),
           round(inr_per_month), round(inr_per_month, -1))
    _agree("the API's cost per request",
           _stated(r"₹([\d.]+) each"),
           round(api.usd_per_request(capacity.mean_tokens_in,
                                     capacity.mean_tokens_out) * USD_TO_INR, 3))
    # Anchored to the break-even, not to the unit: the README quotes a 93 %
    # utilisation elsewhere, from a contaminated sweep point, and a pattern
    # that matched any percentage beside the words "GPU utilisation" would
    # hold that unrelated number to this one's value.
    _agree("utilisation at the break-even",
           _stated(r"break-even sits at ([\d.]+) % GPU utilisation",
                   r"a month\*{0,2}, at ([\d.]+) % GPU utilisation",
                   r"busy ([\d.]+) % of the time"),
           round(point.utilisation * 100, 1))
    _agree("the self-hosted floor",
           _stated(r"floor is ₹([\d.]+) a request"),
           round(capacity.inr_per_1000_requests() / 1000, 4))
    _agree("the self-hosted floor per 1000",
           _stated(r"₹([\d.]+) per 1000 requests"),
           round(capacity.inr_per_1000_requests(), 2))
    _agree("the laptop's amortised hour",
           _stated(r"₹([\d.]+) amortised hour"),
           round(LAPTOP.inr_per_serving_hour(1.0), 2))
    _agree("capacity if traffic were flat",
           _stated(r"([\d.]+) M requests a month"),
           round(capacity.knee_rps * 3600 * HOURS_PER_MONTH / 1e6, 2))
