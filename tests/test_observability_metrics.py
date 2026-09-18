"""The dashboard and alert rules must read metrics that exist, at the rates measured.

A PromQL query over a metric that does not exist returns nothing, not an
error, so its panel is blank and its alert can never fire -- silently. Two
names in this repo did exactly that: vllm:gpu_cache_usage_perc and
vllm:request_failure_total, both from older vLLM releases. The fixture holds
the metric families a live vLLM 0.11.0 server exported on 2026-09-18.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "vllm-0.11.0-metrics.txt"
OBS = ROOT / "deploy" / "observability"
DASHBOARD = OBS / "grafana-dashboard.json"
RULES = OBS / "rules.yml"

#: Metric names the vLLM server itself exports. DCGM_* comes from a separate
#: exporter and is not in the fixture.
NAME = re.compile(r"\b((?:vllm:|http_)[A-Za-z0-9_:]+)")
SERIES_SUFFIXES = ("_bucket", "_sum", "_count")


def _exported() -> set[str]:
    return {line.split()[0] for line in FIXTURE.read_text(encoding="utf-8").splitlines()
            if line.strip()}


def _family(series: str) -> str:
    for suffix in SERIES_SUFFIXES:
        if series.endswith(suffix):
            return series.removesuffix(suffix)
    return series


def _dashboard_exprs() -> list[str]:
    panels = json.loads(DASHBOARD.read_text(encoding="utf-8"))["panels"]
    return [t["expr"] for p in panels for t in p.get("targets", []) if "expr" in t]


def _rule_exprs() -> list[str]:
    """The `expr:` values only, so comments naming an old metric do not count."""
    lines = RULES.read_text(encoding="utf-8").splitlines()
    exprs = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("expr:"):
            continue
        rest = stripped.removeprefix("expr:").strip()
        if rest and rest != "|":
            exprs.append(rest)
            continue
        indent = len(line) - len(line.lstrip())
        block = []
        for following in lines[i + 1:]:
            if following.strip() and len(following) - len(following.lstrip()) <= indent:
                break
            block.append(following)
        exprs.append("\n".join(block))
    return exprs


@pytest.mark.parametrize("source", ["dashboard", "rules"])
def test_every_vllm_metric_read_is_one_vllm_exports(source):
    exprs = _dashboard_exprs() if source == "dashboard" else _rule_exprs()
    assert exprs, f"no expressions found in the {source}"
    referenced = {_family(name) for expr in exprs for name in NAME.findall(expr)}
    missing = sorted(referenced - _exported())
    assert not missing, f"the {source} reads metrics vLLM 0.11.0 does not export: {missing}"


def test_the_extractor_sees_the_multiline_rules():
    # Guards the test above: a parser that silently found no multi-line
    # expressions would make it pass vacuously.
    joined = "\n".join(_rule_exprs())
    assert "http_requests_total" in joined
    assert "vllm:request_prompt_tokens_bucket" in joined


def _measured_slo() -> float:
    slos = {json.loads(p.read_text(encoding="utf-8"))["slo_p95_s"]
            for p in (ROOT / "results" / "sweeps").glob("sweep_*.json")}
    assert len(slos) == 1, f"the published sweeps disagree on the SLO: {slos}"
    return slos.pop()


def test_the_latency_alert_uses_the_slo_the_sweeps_measured_against():
    match = re.search(r"serving:latency_p95_seconds > ([0-9.]+)",
                      RULES.read_text(encoding="utf-8"))
    assert match
    assert float(match.group(1)) == _measured_slo()


def test_the_dashboard_draws_the_same_slo():
    # The slo constant and the latency panel's red line must both be the SLO
    # the knees were measured at, or the dashboard shows a different
    # deployment from the one the charts describe.
    dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    slo_var = next(v for v in dashboard["templating"]["list"] if v["name"] == "slo")
    assert float(slo_var["query"]) == _measured_slo()
    latency = next(p for p in dashboard["panels"]
                   if p.get("title") == "End-to-end latency percentiles")
    red = [s["value"] for s in latency["fieldConfig"]["defaults"]["thresholds"]["steps"]
           if s["color"] == "red"]
    assert red == [_measured_slo()]
