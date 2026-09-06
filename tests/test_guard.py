from __future__ import annotations

from gate.guard import classify


def test_a_precision_change_demands_a_scored_run():
    reasons, attached, _ = classify(["deploy/k8s/configmap.yaml"])
    assert reasons and not attached


def test_a_precision_change_with_a_run_attached_is_allowed_through():
    reasons, attached, _ = classify([
        "deploy/k8s/configmap.yaml",
        "results/eval/20260906T101500Z_open-weight-local_shared.json",
    ])
    assert reasons and attached


def test_documentation_changes_do_not_need_a_run():
    reasons, _, _ = classify(["README.md", "docs/decision.md"])
    assert not reasons


def test_every_serving_surface_is_covered():
    # Each of these changes what comes out of the server, so a score taken
    # before the change no longer describes it.
    for path in (
        "deploy/k8s/configmap.yaml",
        "serving/entrypoint.sh",
        "serving/Dockerfile",
        "deploy/k8s/deployment.yaml",
    ):
        reasons, _, _ = classify([path])
        assert reasons, f"{path} should require a fresh eval"


def test_windows_path_separators_are_normalised():
    # git on Windows can report backslashes; a guard that missed them would
    # silently let an unmeasured serving change through.
    reasons, _, _ = classify(["deploy\\k8s\\configmap.yaml"])
    assert reasons


def test_editing_the_baseline_is_flagged_for_human_review():
    _, _, baseline_edited = classify(["gate/baseline.json"])
    assert baseline_edited


def test_blank_lines_are_ignored():
    reasons, attached, _ = classify(["", "  ", "deploy/k8s/configmap.yaml", ""])
    assert reasons and not attached


def test_a_byte_order_mark_does_not_disable_the_gate():
    # Found the hard way: a BOM on the first line renders invisibly and made
    # classify() report "no serving change" for a precision change. That is
    # failing open, which is the one direction a gate must never fail.
    reasons, _, _ = classify(["﻿deploy/k8s/configmap.yaml"])
    assert reasons


def test_guard_blocks_when_the_diff_produced_nothing(tmp_path, capsys):
    # An empty changed-file list means the diff failed, not that nothing
    # changed -- a pull request always changes something.
    from gate.guard import main

    empty = tmp_path / "changed.txt"
    empty.write_text("\n  \n", encoding="utf-8")

    sys_argv = ["gate.guard", "--changed", str(empty)]
    import sys

    original, sys.argv = sys.argv, sys_argv
    try:
        assert main() == 1
    finally:
        sys.argv = original
    assert "did not run" in capsys.readouterr().out
