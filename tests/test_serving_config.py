"""The configuration the docs describe is the configuration the cluster runs.

`test_published_numbers.py` holds the measurements to the data. This holds the
*settings* to the manifests, which is the other half of the same promise: a
number is only reproducible next to the configuration it was measured at, and
that configuration is written down in five places -- the ConfigMap the cluster
applies, the runbook a reader follows, the caption under the published table,
the Makefile, and the image's own defaults.

Nothing checked them against each other, and they had already come apart. The
runbook told a reader to serve with `GPU_MEMORY_UTILIZATION=0.90`, which is the
exact value the ConfigMap records as refusing to start on this card, and to
sweep at `--slo 5.0`, against which no published knee means anything. Both read
as instructions, not as history.

Needs pyyaml, which is in the dev extra; CI runs this file in its own step.
See .github/workflows/ci.yml and the note in pyproject.toml.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from bench.config import LADDER

ROOT = Path(__file__).resolve().parent.parent

CONFIG = yaml.safe_load(
    (ROOT / "deploy" / "k8s" / "configmap.yaml").read_text(encoding="utf-8")
)["data"]

README = (ROOT / "README.md").read_text(encoding="utf-8")
SETUP = (ROOT / "docs" / "setup.md").read_text(encoding="utf-8")
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")
GATE_WORKFLOW = (
    ROOT / ".github" / "workflows" / "quality-gate.yml"
).read_text(encoding="utf-8")

PRECISIONS = ("fp16", "int8", "int4")


def _deployment() -> dict:
    for doc in yaml.safe_load_all(
        (ROOT / "deploy" / "k8s" / "deployment.yaml").read_text(encoding="utf-8")
    ):
        if doc and doc.get("kind") == "Deployment":
            return doc
    pytest.fail("no Deployment in deploy/k8s/deployment.yaml")


def _container() -> dict:
    return _deployment()["spec"]["template"]["spec"]["containers"][0]


def _runbook_serve_flags() -> dict[str, str]:
    """The `-e KEY=VALUE` flags from the runbook's own serve command."""
    block = re.search(r"docker run -d --name vllm.*?llm-serving:local", SETUP, re.DOTALL)
    assert block, "the runbook no longer shows a serve command to check"
    return dict(re.findall(r"-e (\w+)=([^\s\\]+)", block.group(0)))


# ------------------------------------------------- what the docs describe --

def test_the_capacity_caption_matches_the_configmap():
    """The caption under the published table states the serving settings.

    It is the sentence that makes the knees mean something, so it has to be
    the settings that were served, not the ones that were served once.
    """
    caption = re.search(
        r"`max_model_len` (\d+), (\d+) sequences, prefix caching (\w+)",
        re.sub(r"\s+", " ", README),
    )
    assert caption, "the README no longer captions the table with its settings"
    max_len, max_seqs, prefix = caption.groups()

    assert CONFIG["MAX_MODEL_LEN"] == max_len
    assert CONFIG["MAX_NUM_SEQS"] == max_seqs
    assert prefix == ("on" if CONFIG["ENABLE_PREFIX_CACHING"] == "1" else "off")


@pytest.mark.parametrize(
    "key", ["MAX_MODEL_LEN", "MAX_NUM_SEQS", "GPU_MEMORY_UTILIZATION", "PRECISION"]
)
def test_the_runbook_serves_what_the_cluster_deploys(key: str):
    """A reader following docs/setup.md gets the deployed configuration.

    This is the check that was missing when the runbook sat on 0.90 while the
    ConfigMap sat on 0.85 and explained, at length, why 0.90 fails here.
    """
    flags = _runbook_serve_flags()
    assert key in flags, f"the runbook's serve command no longer sets {key}"
    assert flags[key] == CONFIG[key], (
        f"docs/setup.md serves {key}={flags[key]}, the cluster deploys "
        f"{key}={CONFIG[key]}"
    )


def test_the_runbook_measures_against_the_published_slo():
    """One SLO, in the runbook, the Makefile and the heading above the knees.

    The mock-server example in the README is deliberately not included: its
    latency is specified rather than measured, so it runs against its own
    target and comparing it to a real knee would be the mistake.
    """
    published = float(
        re.search(r"### Capacity at a p95 of ([\d.]+) s", README).group(1)
    )
    runbook = float(
        re.search(r"bench\.sweep .*?--slo ([\d.]+)", SETUP).group(1)
    )
    makefile = float(re.search(r"(?m)^SLO \?= ([\d.]+)", MAKEFILE).group(1))
    assert runbook == published == makefile, (
        f"README heading {published}, runbook {runbook}, Makefile {makefile}"
    )


# ------------------------------------------- what the manifests deploy ----

def test_the_configmap_serves_a_rung_of_the_ladder():
    precision = CONFIG["PRECISION"]
    assert precision in LADDER, f"PRECISION {precision!r} is not a rung"

    repo = LADDER[precision].repo
    model_id = CONFIG["MODEL_ID"]
    # int8 is built here and served from a path; the hub rungs are served by
    # repo id. Both are the same rung.
    assert model_id in (repo, f"/models/{repo}"), (
        f"the ConfigMap serves {model_id!r} as the {precision!r} rung, but the "
        f"ladder says that rung is {repo!r}"
    )


def test_the_startup_budget_is_the_one_the_gate_waits_out():
    """The gate's readiness loop says it matches the startup probe. Check it.

    Two numbers in two languages, with a comment in the workflow asserting
    they agree. If the probe budget grows and the loop does not, CI gives up
    on a server that was still starting and reports it as a failed gate.
    """
    probe = _container()["startupProbe"]
    budget = probe["periodSeconds"] * probe["failureThreshold"]

    step = re.search(r"Wait for readiness.*?exit 1", GATE_WORKFLOW, re.DOTALL)
    assert step, "the gate no longer has a readiness wait to check"
    tries = int(re.search(r"seq 1 (\d+)", step.group(0)).group(1))
    sleep = int(re.search(r"sleep (\d+)", step.group(0)).group(1))

    assert tries * sleep == budget, (
        f"the startup probe allows {budget} s, the gate waits {tries * sleep} s"
    )


# --------------------------------------- what the measurements recorded ----

def _recorded_names(precision: str) -> set[str]:
    """Every served name the artefacts for one rung carry.

    The served name is the only part of the serving configuration that travels
    into the results: a sweep records it as `model`, a harness run as
    `served_model_id`, and the gate prints it as the candidate.
    """
    names = set()
    for path in (ROOT / "results" / "sweeps").glob(f"sweep_*_{precision}.json"):
        names.add(json.loads(path.read_text(encoding="utf-8"))["model"])

    eval_dir = ROOT / "results" / "eval" / ("" if precision == "fp16" else precision)
    for path in eval_dir.glob("2026*.json"):
        names.add(json.loads(path.read_text(encoding="utf-8"))["served_model_id"])
    return names


@pytest.mark.parametrize("precision", PRECISIONS)
def test_every_artefact_of_one_rung_agrees_on_the_served_name(precision: str):
    names = _recorded_names(precision)
    assert names, f"no artefacts found for {precision}"
    assert len(names) == 1, (
        f"{precision}'s sweeps and eval runs disagree on the served name: {names}"
    )


def test_the_served_name_carries_the_rung():
    """Each rung was served under its own name, and must go on being.

    Serve int8 under fp16's name and the two rungs' results stop being
    distinguishable by name after the fact -- in the sweep files, in the gate
    table, in the charts. The ConfigMap's comment says this; this checks it.
    """
    names = {p: _recorded_names(p).pop() for p in PRECISIONS}
    assert len(set(names.values())) == len(PRECISIONS), (
        f"two rungs were measured under one name: {names}"
    )
    assert CONFIG["SERVED_MODEL_NAME"] == names[CONFIG["PRECISION"]], (
        f"the ConfigMap serves PRECISION {CONFIG['PRECISION']} as "
        f"{CONFIG['SERVED_MODEL_NAME']!r}, but that rung was measured as "
        f"{names[CONFIG['PRECISION']]!r}"
    )
