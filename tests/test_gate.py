from __future__ import annotations

import json

import pytest

from gate import record
from gate.compare import Baseline, GateError, compare, markdown

# Project 01's five repeat runs of one model over the same 28 examples with
# identical inputs: 53.6, 50.0, 53.6, 50.0, 64.3. A 14.3-point spread from
# sampling alone. Used as the fixture here so these tests exercise the real
# noise problem rather than a tidy invented one.
REPEAT_SLAB = [0.536, 0.500, 0.536, 0.500, 0.643]


def _run(slab: float, **over) -> dict:
    summary = {
        "n": 28,
        "slab_acc": slab,
        "hsn_acc": 0.579,
        "chapter_acc": 0.630,
        "abstention_acc": 0.779,
        "stale_slab_rate": 0.021,
        "stale_cited_rate": 0.114,
        "unparseable": 1,
        "errored": 1,
    }
    summary.update(over.pop("summary", {}))
    run = {
        "model_key": "open-weight-local",
        "served_model_id": "test-model",
        "dataset_sha": "8c32ff29e970",
        "dataset_n": 28,
        "prompt_version": "v1",
        "summary": summary,
    }
    run.update(over)
    return run


@pytest.fixture
def baseline() -> Baseline:
    built = record.build([_run(s) for s in REPEAT_SLAB], precision="fp16")
    return Baseline(
        model=built["model"], precision=built["precision"],
        dataset_sha=built["dataset_sha"], prompt_version=built["prompt_version"],
        recorded_at=built["recorded_at"], scores=built["scores"],
        noise=built["noise"], n_repeats=built["n_repeats"],
    )


# --------------------------------------------------------------- recording --

def test_noise_floor_is_the_measured_spread():
    built = record.build([_run(s) for s in REPEAT_SLAB], precision="fp16")
    assert built["n_repeats"] == 5
    assert built["scores"]["slab_acc"] == pytest.approx(0.543, abs=1e-3)
    # 64.3 - 50.0 = 14.3 points, taken from the runs and not chosen.
    assert built["noise"]["slab_acc"] == pytest.approx(14.3, abs=1e-6)
    # Metrics that did not move record no noise, and fall back to their floor.
    assert built["noise"]["hsn_acc"] == pytest.approx(0.0)


def test_recording_refuses_too_few_repeats():
    with pytest.raises(GateError, match="not a spread"):
        record.build([_run(0.5), _run(0.6)], precision="fp16")


def test_recording_refuses_runs_from_different_configurations():
    # Mixing precisions would bank their quality difference as "noise", and
    # the gate would then wave through exactly that regression.
    runs = [_run(s) for s in REPEAT_SLAB]
    runs[2]["dataset_sha"] = "deadbeefcafe"
    with pytest.raises(GateError, match="dataset_sha"):
        record.build(runs, precision="fp16")


def test_recording_keeps_the_per_run_numbers_for_audit():
    built = record.build([_run(s) for s in REPEAT_SLAB], precision="fp16")
    assert built["observed_per_run"]["slab_acc"] == [53.6, 50.0, 53.6, 50.0, 64.3]


# ----------------------------------------------------------------- gating --

def test_a_drop_smaller_than_the_noise_floor_passes(baseline):
    # 54.3 -> 45.0 is a 9.3-point fall and it PASSES, which looks wrong until
    # you remember the same configuration produced a 14.3-point spread on its
    # own. Blocking here would block roughly half of all no-op reruns.
    result = compare(baseline, _run(0.45))
    assert result.passed
    slab = next(f for f in result.findings if f.metric.key == "slab_acc")
    assert slab.verdict == "within noise"
    assert slab.delta_pp == pytest.approx(-9.3, abs=0.05)


def test_a_drop_larger_than_the_noise_floor_blocks(baseline):
    result = compare(baseline, _run(0.38))
    assert not result.passed
    assert [f.metric.key for f in result.regressions] == ["slab_acc"]


def test_a_real_improvement_is_recognised_and_does_not_block(baseline):
    result = compare(baseline, _run(0.85))
    assert result.passed
    slab = next(f for f in result.findings if f.metric.key == "slab_acc")
    assert slab.verdict == "improved"


def test_lower_is_better_metrics_have_their_sign_the_right_way_round(baseline):
    # Reciting an abolished GST rate more often is a regression even though
    # the number went up. Getting this backwards would build a gate that
    # blocks improvements and passes regressions while looking like it worked.
    worse = compare(baseline, _run(0.543, summary={"stale_slab_rate": 0.30}))
    finding = next(f for f in worse.findings if f.metric.key == "stale_slab_rate")
    assert finding.delta_pp < 0
    assert finding.verdict == "regressed"
    assert not worse.passed

    better = compare(baseline, _run(0.543, summary={"stale_slab_rate": 0.0}))
    assert next(
        f for f in better.findings if f.metric.key == "stale_slab_rate"
    ).verdict in {"improved", "within noise"}
    assert better.passed


def test_errors_and_unparseable_gate_tightly(baseline):
    # These should sit near zero, so their floor is 0 and any rise blocks.
    # A serving change that starts mangling the output format is precisely
    # what this gate is for.
    result = compare(baseline, _run(0.543, summary={"unparseable": 6}))
    assert not result.passed
    assert "unparseable" in {f.metric.key for f in result.regressions}


# ------------------------------------------------------------- refusals ----

def test_a_changed_dataset_is_refused_not_compared(baseline):
    with pytest.raises(GateError, match="dataset changed"):
        compare(baseline, _run(0.9, dataset_sha="ffffffffffff"))


def test_a_changed_prompt_is_refused_not_compared(baseline):
    with pytest.raises(GateError, match="prompt changed"):
        compare(baseline, _run(0.9, prompt_version="v2"))


def test_an_unmeasured_noise_floor_is_refused(baseline):
    baseline.n_repeats = 1
    with pytest.raises(GateError, match="run-to-run noise"):
        compare(baseline, _run(0.5))


def test_a_run_with_no_summary_is_refused(baseline):
    with pytest.raises(GateError, match="no `summary`"):
        compare(baseline, {"dataset_sha": "8c32ff29e970", "prompt_version": "v1"})


# -------------------------------------------------------------- reporting --

def test_pr_comment_shows_the_tolerance_and_where_it_came_from(baseline):
    text = markdown(compare(baseline, _run(0.38)))
    assert "blocked" in text
    assert "±14.3" in text
    # A reviewer who cannot see why a 9-point drop passed will not trust the
    # gate, so the provenance of the tolerance is on the page.
    assert "5 repeat runs" in text
    assert "not a chosen" in text


def test_pr_comment_says_passed_when_it_passed(baseline):
    assert "passed" in markdown(compare(baseline, _run(0.543)))


def test_baseline_load_reports_a_missing_file_usefully(tmp_path):
    with pytest.raises(GateError, match="no baseline at"):
        Baseline.load(tmp_path / "nope.json")


def test_baseline_round_trips_through_json(tmp_path, baseline):
    built = record.build([_run(s) for s in REPEAT_SLAB], precision="fp16")
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(built), encoding="utf-8")
    loaded = Baseline.load(path)
    assert loaded.n_repeats == 5
    assert loaded.tolerance(
        next(m for m in __import__("gate.compare", fromlist=["METRICS"]).METRICS
             if m.key == "slab_acc")
    ) == pytest.approx(14.3)
