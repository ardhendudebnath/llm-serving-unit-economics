"""The PR-comment header must not misattribute precision.

The quantisation ladder uses the gate to compare an int8 or int4 run against
the fp16 baseline. The header used to print the candidate's served name beside
the baseline's precision, rendering an int8 result as "gst-4b-int8 at fp16" --
a false statement on the one page a reviewer reads.
"""

from __future__ import annotations

from gate import record
from gate.compare import Baseline, compare, markdown
from tests.test_gate import REPEAT_SLAB, _run

LONG_SHA = "8c32ff29e970115f61c4da5d778dc9e38379ed7502e9f4aa50b35e487ddb95ee"


def _baseline(**over) -> Baseline:
    built = record.build([_run(s, **over) for s in REPEAT_SLAB], precision="fp16")
    return Baseline(
        model=built["model"], precision=built["precision"],
        dataset_sha=built["dataset_sha"], prompt_version=built["prompt_version"],
        recorded_at=built["recorded_at"], scores=built["scores"],
        noise=built["noise"], n_repeats=built["n_repeats"],
    )


def test_a_quantised_candidate_is_not_labelled_with_the_baseline_precision():
    text = markdown(compare(_baseline(), _run(0.5, served_model_id="gst-4b-int8")))
    assert "`gst-4b-int8` at **fp16**" not in text
    assert "`gst-4b-int8` against the **fp16** baseline (`test-model`)" in text


def test_a_rerun_of_the_baseline_configuration_says_so():
    text = markdown(compare(_baseline(), _run(0.5)))
    assert "`test-model` at **fp16**, against its own baseline" in text


def test_a_missing_served_id_is_not_attributed_to_the_baseline_model():
    # Falling back to the baseline's name would claim an unknown run was the
    # baseline configuration.
    text = markdown(compare(_baseline(), _run(0.5, served_model_id="")))
    assert "unrecorded model" in text
    assert "`test-model` at" not in text


def test_the_dataset_sha_is_shortened_in_the_header():
    base = _baseline(dataset_sha=LONG_SHA)
    text = markdown(compare(base, _run(0.5, dataset_sha=LONG_SHA)))
    assert f"dataset `{LONG_SHA[:12]}`" in text
    assert LONG_SHA not in text
