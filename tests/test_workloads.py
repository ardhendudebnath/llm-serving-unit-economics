from __future__ import annotations

import pytest

from bench import workloads


@pytest.mark.parametrize("profile", sorted(workloads.PROFILES))
def test_every_profile_has_a_built_corpus(profile):
    rows = workloads.load(profile)
    assert rows, f"{profile} corpus is empty"
    assert all(r.prompt.strip() for r in rows)
    assert all(r.profile == profile for r in rows)


@pytest.mark.parametrize("profile", sorted(workloads.PROFILES))
def test_provenance_survives_into_the_corpus(profile):
    # Every replayed request must trace back to a real collected document.
    # A corpus that lost its source ids cannot be audited.
    rows = workloads.load(profile)
    assert all(r.source_id for r in rows)


def _median_prompt_chars(profile: str) -> int:
    lengths = sorted(len(r.prompt) for r in workloads.load(profile))
    return lengths[len(lengths) // 2]


def test_the_three_profiles_are_actually_different_shapes():
    # The point of three profiles is that they stress different parts of the
    # server. If they converged in shape the sweep would measure one thing
    # three times, so the separation is asserted rather than assumed.
    short = _median_prompt_chars("short")
    long_in = _median_prompt_chars("long_in")
    long_out = _median_prompt_chars("long_out")

    # Prefill: long_in must dominate by an order of magnitude.
    assert long_in > 10 * short
    # long_out is a brief prompt, comparable to short, not to long_in.
    assert long_out < short * 2
    # Decode: long_out is the only profile that generates at length.
    assert workloads.get("long_out").max_tokens > 10 * workloads.get("short").max_tokens
    assert workloads.get("long_in").max_tokens == workloads.get("short").max_tokens


def test_short_and_long_out_draw_on_disjoint_products():
    # Sampled disjointly on purpose: running one profile after the other must
    # not let vLLM's prefix cache serve prompts warmed by the previous sweep.
    short_ids = {r.source_id for r in workloads.load("short")}
    long_out_ids = {r.source_id for r in workloads.load("long_out")}
    assert not (short_ids & long_out_ids)


def test_long_in_replays_project_01s_exact_prompt():
    # The extraction profile is Project 01's task. If this template drifts, the
    # load numbers stop describing the workload the quality gate scores.
    row = workloads.load("long_in")[0]
    assert "SLAB:" in row.prompt
    assert "HSN:" in row.prompt
    assert "ANSWERABLE:" in row.prompt
    assert row.system.startswith("You classify goods for Indian GST")


def test_long_out_carries_no_classification_system_prompt():
    # Borrowing Project 01's system prompt here would push the model toward a
    # short answer, which is the opposite of what this profile measures.
    assert all(r.system == "" for r in workloads.load("long_out"))


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        workloads.load("does-not-exist")
