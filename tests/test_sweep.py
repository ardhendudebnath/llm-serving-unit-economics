"""The sweep: cooled points, resumable checkpoints, a knee re-derived from what was kept."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from bench import sweep
from bench.metrics import LoadPoint, RequestRecord


def _point(profile: str, rate: float, latency_s: float, n: int = 20) -> LoadPoint:
    records = [
        RequestRecord(profile=profile, queued_at=0.0, sent_at=0.0, done_at=latency_s,
                      first_token_at=min(0.1, latency_s), tokens_in=1600,
                      tokens_out=48, status=200)
        for _ in range(n)
    ]
    return LoadPoint(profile=profile, target_rate_rps=rate, duration_s=10.0,
                     records=records)


class _NoGpu:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def summary(self) -> dict:
        return {}


@pytest.fixture
def h(monkeypatch, tmp_path):
    """sweep_one with the load generator, GPU and clock replaced.

    `latency` maps each rate to what every request at that rate takes. Tests
    change it between runs to model a point measured hot and then re-measured.
    """
    latency: dict[float, float] = {}
    sleeps: list[float] = []
    measured: list[float] = []

    async def fake_run_point(cfg, corpus):
        measured.append(cfg.rate_rps)
        return _point(cfg.profile, cfg.rate_rps, latency[cfg.rate_rps])

    monkeypatch.setattr(sweep.workloads, "load", lambda profile: [])
    monkeypatch.setattr(sweep, "run_point", fake_run_point)
    monkeypatch.setattr(sweep.gpu, "Monitor", _NoGpu)
    monkeypatch.setattr(sweep.gpu, "sample_once", lambda: None)
    monkeypatch.setattr(sweep.time, "sleep", sleeps.append)

    def run(rates, *, cooldown_s: float = 60.0) -> sweep.SweepState:
        return sweep.sweep_one(
            "long_in", base_url="http://unused", model="m", precision="int4",
            slo_p95_s=10.0, rates=tuple(rates), duration_s=10.0, warmup_s=0.0,
            out=tmp_path, cooldown_s=cooldown_s,
        )

    return SimpleNamespace(run=run, latency=latency, sleeps=sleeps, measured=measured,
                           checkpoint=tmp_path / "sweep_long_in_int4.json")


def test_the_first_point_is_cooled_too(h):
    # Whatever ran before this sweep -- in a chain, another profile's saturated
    # last point -- has left the card hot, so the first point needs the same
    # cooldown as every other.
    h.latency.update({0.25: 1.0, 0.5: 1.2})
    h.run([0.25, 0.5])
    assert h.sleeps == [60.0, 60.0]


def test_no_cooldown_is_taken_when_none_is_asked_for(h):
    h.latency.update({0.25: 1.0, 0.5: 1.2})
    h.run([0.25, 0.5], cooldown_s=0.0)
    assert h.sleeps == []


def test_resuming_skips_rates_already_recorded(h):
    h.latency.update({0.25: 1.0, 0.5: 1.2})
    h.run([0.25, 0.5])
    h.run([0.25, 0.5])
    assert h.measured == [0.25, 0.5]


def test_a_sweep_that_never_saturates_says_so(h):
    h.latency.update({0.25: 1.0, 0.5: 1.2})
    assert h.run([0.25, 0.5]).stopped_because == "swept every requested rate"


def test_one_hot_point_hides_the_knee_and_remeasuring_it_restores_it(h):
    """The int4 long_in case, end to end.

    Measured hot, 0.25 rps fails the SLO, and because the knee walks up from
    the lowest rate the sweep reads as meeting the SLO at no rate at all.
    Re-measured cool, the knee is the 2 rps the other points always showed, and
    the sweep still says it stopped at saturation, because it did.
    """
    h.latency.update({0.25: 18.2, 0.5: 1.3, 1.0: 1.7, 2.0: 3.8, 4.0: 41.1})
    hot = h.run([0.25, 0.5, 1.0, 2.0, 4.0])
    assert hot.knee() is None
    assert "exceeded 3x" in hot.stopped_because

    recorded = json.loads(h.checkpoint.read_text(encoding="utf-8"))
    recorded["points"] = [p for p in recorded["points"] if p["target_rate_rps"] != 0.25]
    h.checkpoint.write_text(json.dumps(recorded), encoding="utf-8")

    h.latency[0.25] = 0.9
    cool = h.run([0.25])

    assert h.measured.count(0.25) == 2
    assert cool.knee()["target_rate_rps"] == 2.0
    assert "exceeded 3x" in cool.stopped_because
    saved = json.loads(h.checkpoint.read_text(encoding="utf-8"))
    assert saved["knee_rps"] == 2.0
    assert [p["target_rate_rps"] for p in saved["points"]] == [0.25, 0.5, 1.0, 2.0, 4.0]
