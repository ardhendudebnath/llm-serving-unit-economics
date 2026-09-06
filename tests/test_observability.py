"""Checks on the dashboard and alert rules as committed artifacts.

Dashboards and alert rules drift apart from the code they describe silently:
nothing fails, the panel just quietly plots a metric nobody records any more,
or the alert fires at a threshold the benchmark no longer uses. These tests
make that drift a build failure.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from bench.metrics import ERROR_CEILING

DASHBOARD = Path("deploy/observability/grafana-dashboard.json")
RULES = Path("deploy/observability/rules.yml")

yaml = pytest.importorskip("yaml")


@pytest.fixture(scope="module")
def dashboard() -> dict:
    return json.loads(DASHBOARD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rules() -> dict:
    return yaml.safe_load(RULES.read_text(encoding="utf-8"))


def _panels(dashboard: dict) -> list[dict]:
    return [p for p in dashboard["panels"] if p.get("type") != "row"]


def _expressions(dashboard: dict) -> list[str]:
    return [
        t["expr"]
        for panel in _panels(dashboard)
        for t in panel.get("targets", [])
        if t.get("expr")
    ]


def _recorded_names(rules: dict) -> set[str]:
    return {
        rule["record"]
        for group in rules["groups"]
        for rule in group["rules"]
        if "record" in rule
    }


def _alerts(rules: dict) -> dict[str, dict]:
    return {
        rule["alert"]: rule
        for group in rules["groups"]
        for rule in group["rules"]
        if "alert" in rule
    }


# --------------------------------------------------------------- structure --

def test_dashboard_is_valid_json_with_a_stable_uid(dashboard):
    # A stable uid is what lets the dashboard be re-provisioned over itself
    # rather than accumulating copies.
    assert dashboard["uid"] == "llm-serving-unit-economics"
    assert dashboard["title"]
    assert _panels(dashboard)


def test_every_panel_has_a_title_and_at_least_one_query(dashboard):
    for panel in _panels(dashboard):
        assert panel.get("title"), f"panel {panel.get('id')} has no title"
        assert panel.get("targets"), f"panel {panel['title']!r} queries nothing"


def test_panels_do_not_overlap_on_the_grid(dashboard):
    # Overlapping gridPos silently reflows the whole dashboard in the browser.
    seen: dict[tuple[int, int], str] = {}
    for panel in dashboard["panels"]:
        pos = panel["gridPos"]
        for x in range(pos["x"], pos["x"] + pos["w"]):
            for y in range(pos["y"], pos["y"] + pos["h"]):
                cell = (x, y)
                assert cell not in seen, (
                    f"{panel['title']!r} overlaps {seen[cell]!r} at {cell}"
                )
                seen[cell] = panel["title"]


def test_no_panel_runs_off_the_24_column_grid(dashboard):
    for panel in dashboard["panels"]:
        pos = panel["gridPos"]
        assert pos["x"] + pos["w"] <= 24, f"{panel['title']!r} exceeds the grid"


# ------------------------------------------------------- required coverage --

@pytest.mark.parametrize(
    "needle",
    [
        # RED
        "vllm:request_success_total",
        "vllm:request_failure_total",
        "vllm:e2e_request_latency_seconds_bucket",
        # GPU — the figure that decides whether the cost number is honest
        "DCGM_FI_DEV_GPU_UTIL",
        "DCGM_FI_DEV_FB_USED",
        # Serving internals
        "vllm:num_requests_waiting",
        "vllm:num_requests_running",
        "vllm:gpu_cache_usage_perc",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:time_to_first_token_seconds_bucket",
        # Drift
        "vllm:request_prompt_tokens_bucket",
        # Cost
        "serving:cost_per_1000_requests_usd",
        "serving:cost_per_hour_usd",
    ],
)
def test_the_plan_s_required_metrics_are_all_dashboarded(dashboard, needle):
    joined = " ".join(_expressions(dashboard))
    assert needle in joined, f"{needle} is not plotted anywhere"


def test_latency_is_shown_as_percentiles_never_as_a_mean(dashboard):
    exprs = " ".join(_expressions(dashboard))
    for q in ("0.50", "0.95", "0.99"):
        assert f"histogram_quantile({q}" in exprs
    # A _sum/_count ratio is how a mean latency sneaks onto a dashboard.
    assert "_sum" not in exprs or "_count" not in exprs


# ------------------------------------------------- cross-checks with rules --

def test_dashboard_recording_rules_actually_exist(dashboard, rules):
    """Catches a panel plotting a recording rule nobody records.

    Grafana shows an empty panel for this, not an error, so it can sit broken
    for months.
    """
    recorded = _recorded_names(rules)
    used = {
        name
        for expr in _expressions(dashboard)
        for name in re.findall(r"\b(serving:[a-z0-9_:]+)", expr)
    }
    missing = used - recorded
    assert not missing, f"dashboard plots undefined recording rules: {sorted(missing)}"


def test_the_error_rate_threshold_matches_the_benchmark(dashboard, rules):
    """One definition of "working", shared by three places.

    bench.metrics.ERROR_CEILING disqualifies a load point; the alert pages on
    the same number; the dashboard draws its threshold line there. If they
    drift, the benchmark and the alert disagree about the same deployment.
    """
    assert ERROR_CEILING == 0.01

    panel = next(p for p in _panels(dashboard) if p["title"] == "Error rate")
    steps = panel["fieldConfig"]["defaults"]["thresholds"]["steps"]
    assert any(s.get("value") == ERROR_CEILING for s in steps)

    alert = _alerts(rules)["ErrorRateAboveOnePercent"]
    assert f"> {ERROR_CEILING}" in alert["expr"]


def test_the_gpu_utilisation_threshold_matches_the_cost_alert(dashboard, rules):
    # The gauge turns amber at exactly the level the cost alert fires, so the
    # dashboard and the alert tell the same story.
    panel = next(p for p in _panels(dashboard) if p["title"] == "GPU utilisation")
    steps = panel["fieldConfig"]["defaults"]["thresholds"]["steps"]
    assert any(s.get("value") == 20 for s in steps)

    alert = _alerts(rules)["GpuUnderutilisedCostWaste"]
    assert "< 20" in alert["expr"]


def test_the_kv_cache_threshold_matches_its_alert(dashboard, rules):
    panel = next(p for p in _panels(dashboard) if p["title"] == "KV cache usage")
    steps = panel["fieldConfig"]["defaults"]["thresholds"]["steps"]
    assert any(s.get("value") == 0.95 for s in steps)
    assert "> 0.95" in _alerts(rules)["KVCacheNearlyExhausted"]["expr"]


# ------------------------------------------------------------ alert rules --

def test_every_alert_explains_itself(rules):
    # An alert that fires at 3am with no summary is a page nobody can action.
    for name, alert in _alerts(rules).items():
        assert alert.get("annotations", {}).get("summary"), f"{name} has no summary"
        assert alert.get("labels", {}).get("severity"), f"{name} has no severity"


def test_every_alert_waits_before_firing(rules):
    # Without `for`, a single scrape blip pages someone.
    for name, alert in _alerts(rules).items():
        assert alert.get("for"), f"{name} fires on a single evaluation"


def test_the_cost_alerts_are_labelled_as_cost(rules):
    # The GPU-underutilisation alert is the one the plan calls a maturity
    # signal. It is a cost problem, not a reliability one, and is labelled so
    # it routes somewhere different from a page.
    cost = {n for n, a in _alerts(rules).items() if a["labels"].get("kind") == "cost"}
    assert "GpuUnderutilisedCostWaste" in cost
    assert _alerts(rules)["GpuUnderutilisedCostWaste"]["labels"]["severity"] == "ticket"


def test_the_gpu_rate_is_not_silently_set_to_something_invented(rules):
    # Same discipline as bench/config.py: the hourly rate stays at zero, and
    # visibly labelled as unset, until someone reads one from a provider.
    rate = next(
        r for g in rules["groups"] for r in g["rules"]
        if r.get("record") == "gpu:hourly_rate_usd"
    )
    assert rate["expr"].strip() == "vector(0)"
    assert "unset" in rate["labels"]["note"]
