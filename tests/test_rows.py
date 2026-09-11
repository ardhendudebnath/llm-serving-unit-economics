"""Row-by-row comparison: how a rung got worse, not just by how much."""

from __future__ import annotations

import pytest

from bench.report.rows import ABSTAIN, across_runs, compare, to_markdown


def _run(rows: list[tuple[str, str, str | None, bool]], *, name: str = "m",
         sha: str = "abc123", prompt: str = "v1") -> dict:
    return {
        "served_model_id": name, "dataset_sha": sha, "prompt_version": prompt,
        "rows": [{"id": rid, "gold_slab": gold, "predicted_slab": pred, "correct": ok}
                 for rid, gold, pred, ok in rows],
    }


BASELINE = _run([
    ("r1", "18", "18", True),      # same answer, both right
    ("r2", "18", "18", True),      # candidate abstains: lost
    ("r3", "5", "18", False),      # different answer, candidate right: gained
    ("r4", "18", ABSTAIN, False),  # candidate answers
    ("r5", "18", ABSTAIN, False),  # both abstain
    ("r6", "18", None, False),     # unparsed on both sides
], name="gst-4b")

CANDIDATE = _run([
    ("r1", "18", "18", True),
    ("r2", "18", ABSTAIN, False),
    ("r3", "5", "5", True),
    ("r4", "18", "12", False),
    ("r5", "18", ABSTAIN, False),
    ("r6", "18", None, False),
], name="gst-4b-int4")


def test_every_row_lands_in_exactly_one_bucket():
    diff = compare(BASELINE, CANDIDATE)
    assert diff.buckets == {
        "same_answer": ["r1"],
        "different_answer": ["r3"],
        "newly_abstained": ["r2"],
        "newly_answered": ["r4"],
        "both_abstained": ["r5"],
        "unreadable": ["r6"],
    }
    assert sum(len(ids) for ids in diff.buckets.values()) == diff.n == 6


def test_correctness_is_counted_apart_from_the_answer_changing():
    diff = compare(BASELINE, CANDIDATE)
    assert diff.lost == ["r2"]
    # r3's answer changed and got better; r4's changed and stayed wrong.
    assert diff.gained == ["r3"]


def test_two_unparsed_rows_are_not_the_same_answer():
    # Both sides carry None; None == None must not read as agreement.
    diff = compare(_run([("r", "18", None, False)]), _run([("r", "18", None, False)]))
    assert diff.buckets["unreadable"] == ["r"]
    assert diff.buckets["same_answer"] == []


def test_a_model_that_only_abstains_more_shows_as_exactly_that():
    # The int4 shape: every committed answer unchanged, the loss entirely from
    # abstaining on rows the baseline got right.
    base = _run([("a", "18", "18", True), ("b", "18", "18", True), ("c", "5", "9", False)])
    cand = _run([("a", "18", "18", True), ("b", "18", ABSTAIN, False),
                 ("c", "5", ABSTAIN, False)])
    diff = compare(base, cand)
    assert diff.count("different_answer") == 0
    assert diff.buckets["newly_abstained"] == ["b", "c"]
    assert diff.lost == ["b"]


@pytest.mark.parametrize("field", ["dataset_sha", "prompt_version"])
def test_refuses_runs_over_different_inputs(field):
    other = dict(CANDIDATE, **{field: "something-else"})
    with pytest.raises(ValueError, match=field):
        compare(BASELINE, other)


def test_refuses_runs_over_different_rows():
    fewer = dict(CANDIDATE, rows=CANDIDATE["rows"][:-1])
    with pytest.raises(ValueError, match="different rows"):
        compare(BASELINE, fewer)


def test_markdown_counts_every_bucket_and_lists_only_changed_rows():
    report = to_markdown(compare(BASELINE, CANDIDATE), BASELINE, CANDIDATE)
    assert "`gst-4b-int4` against `gst-4b`" in report
    assert "| newly abstained (`gst-4b-int4` said UNANSWERABLE) | 1 |" in report
    assert "**Lost** (`gst-4b` right, `gst-4b-int4` wrong): 1 — r2" in report
    table = report.split("## Rows whose answer changed", 1)[1]
    # Changed rows are listed; unchanged and jointly abstained ones are not.
    for listed in ("| r2 |", "| r3 |", "| r4 |", "| r6 |"):
        assert listed in table
    for omitted in ("| r1 |", "| r5 |"):
        assert omitted not in table


def test_across_runs_gives_each_baseline_run_its_own_line():
    # A second baseline run that got r2 wrong: against it, the candidate's
    # abstention on r2 costs nothing. The summary must show both readings.
    rows_b = [dict(row) for row in BASELINE["rows"]]
    rows_b[1]["correct"] = False
    table = across_runs(CANDIDATE, [dict(BASELINE, run_id="run-a"),
                                    dict(BASELINE, run_id="run-b", rows=rows_b)])
    assert "| run-a | 1 | 1 | 1 | 1 | 1 | 1 |" in table
    assert "| run-b | 1 | 1 | 1 | 1 | 0 | 1 |" in table
