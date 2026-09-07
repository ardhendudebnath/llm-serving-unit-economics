"""Guards against this repo and Project 01 drifting apart.

Two repositories that describe the same deployment can disagree silently:
different conversion rates on the same leaderboard, a registry row this project
depends on that the pinned commit does not contain, a pin updated in one file
and not the other. Nothing fails at the time; it fails later, on a GPU, in the
middle of the one run the whole project turns on.

The pin checks need only this repo and always run. The registry checks need
Project 01 importable and skip individually without it -- CI installs it
through the `eval` extra, where they do run.

`importorskip` is called inside each test rather than at module level on
purpose: at module level it would skip this entire file, including the pin
checks that need nothing installed, and a skipped guard is indistinguishable
from a passing one in a CI summary.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from bench.config import USD_TO_INR

PYPROJECT = Path("pyproject.toml")
GATE_WORKFLOW = Path(".github/workflows/quality-gate.yml")

#: The row this project adds to Project 01's registry and then depends on.
REQUIRED_KEY = "open-weight-vllm"

_SKIP_REASON = "Project 01 not installed; CI installs it via the `eval` extra"


def _registry():
    return pytest.importorskip("harness.runners.registry", reason=_SKIP_REASON)


def _pinned_in_pyproject() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r"gst-eval-harness @ git\+\S+?@([0-9a-f]{7,40})", text)
    assert match, "pyproject no longer pins gst-eval-harness to a commit"
    return match.group(1)


def _pinned_in_workflow() -> str:
    text = GATE_WORKFLOW.read_text(encoding="utf-8")
    match = re.search(r"^\s*ref:\s*([0-9a-f]{7,40})\s*$", text, re.MULTILINE)
    assert match, "the gate workflow no longer pins a harness commit"
    return match.group(1)


# ------------------------------------------------------------------ pins ---

def test_the_harness_is_pinned_to_a_commit_not_a_branch():
    # A gate whose baseline can move underneath it is not a gate.
    for pin in (_pinned_in_pyproject(), _pinned_in_workflow()):
        assert re.fullmatch(r"[0-9a-f]{7,40}", pin), f"{pin!r} is not a commit sha"


def test_both_files_pin_the_same_harness_commit():
    """The failure this catches is entirely silent.

    Updating the pin in one place installs one harness for local work and
    checks out another in CI, so the gate scores against a different harness
    than the developer ran -- and both look fine.
    """
    assert _pinned_in_pyproject() == _pinned_in_workflow()


# -------------------------------------------------------------- registry ---

def test_the_registry_row_this_project_depends_on_exists():
    # Without it, `harness.run --model open-weight-vllm` fails with an unknown
    # model key -- correctly, but only once a GPU is already running.
    assert REQUIRED_KEY in _registry().MODELS


def test_the_vllm_row_takes_its_model_id_from_the_environment():
    """The point of the row, and the thing that must not regress.

    A pinned model_id would be sent to a vLLM server that does not recognise
    it, while the result file recorded the pinned id as what was measured.
    Empty means the runner reads NIM_MODEL and raises when it is unset: it can
    fail loudly, but it cannot silently score the wrong model.
    """
    spec = _registry().get(REQUIRED_KEY)
    assert spec.model_id == ""
    assert spec.provider == "nim"


def test_the_self_hosted_row_carries_no_per_token_price():
    # Self-hosted cost is GPU time. A per-token price here would be fabricated,
    # and `priced()` is what stops it reaching a leaderboard.
    registry = _registry()
    assert not registry.priced(registry.get(REQUIRED_KEY))


def test_the_currency_conversion_matches_project_01():
    """Both repos report rupee figures for the same deployment.

    If these drift, the leaderboard's cost column and this project's cost
    curves quote different rupees for the same dollar, and nothing anywhere
    says so.
    """
    assert USD_TO_INR == _registry().USD_TO_INR
