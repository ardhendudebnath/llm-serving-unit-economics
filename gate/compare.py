"""The deploy gate: block a change that makes the served model worse.

    python -m gate.compare --candidate results/eval/<run>.json

Exit 0 to allow the deploy, 1 to block it.

**The hard part of this gate is not comparing two numbers -- it is knowing
which differences mean anything.** Project 01 ran the same model over the same
28 examples five times with identical inputs and got slab accuracies of 53.6,
50.0, 53.6, 50.0 and 64.3: a 14.3-point spread from nothing but sampling. A
gate with a "fail on any 1-point regression" rule would have blocked three of
those five runs against any of the others, and would have been switched off
inside a week for crying wolf. A gate nobody trusts is worse than no gate,
because it launders a rubber stamp as a control.

So tolerance here is **measured, not chosen**. The baseline records the spread
observed across repeated runs of the *same* configuration, and a candidate has
to move a metric further than that noise floor before the gate calls it a
regression. If the noise floor has not been measured, the gate refuses to run
rather than inventing a threshold -- an ungrounded tolerance is how a gate ends
up either blocking everything or nothing.

Two further refusals, both because the comparison would otherwise be
meaningless rather than merely wrong:

  - a different dataset SHA. Scores over different examples are not comparable,
    and a gate that quietly compared them would pass a regression whenever the
    dataset happened to get easier.
  - a different prompt version. Same argument: it would measure the prompt
    change, and attribute the result to the serving change under review.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BASELINE = Path("gate/baseline.json")


@dataclass(frozen=True, slots=True)
class Metric:
    key: str
    label: str
    higher_is_better: bool
    #: Minimum tolerance in points, applied when the measured noise floor is
    #: smaller. Guards against a baseline that happened to record identical
    #: repeats and would otherwise gate on the last bit of floating point.
    floor_pp: float
    #: False for metrics that are reported but never block. `n` and the
    #: gradable counts describe the run, they do not grade it.
    blocking: bool = True


#: Sign conventions are explicit per metric, because half of these get better
#: as they go down. Getting one backwards would build a gate that blocks
#: improvements and passes regressions, and it would look like it was working.
METRICS: tuple[Metric, ...] = (
    Metric("slab_acc", "Slab accuracy", higher_is_better=True, floor_pp=1.0),
    Metric("hsn_acc", "HSN-4 accuracy", higher_is_better=True, floor_pp=1.0),
    Metric("chapter_acc", "Chapter accuracy", higher_is_better=True, floor_pp=1.0),
    Metric("abstention_acc", "Abstention accuracy", higher_is_better=True, floor_pp=1.0),
    # Lower is better: these count answers drawn from an abolished rate table,
    # which is the failure Project 01 exists to measure.
    Metric("stale_slab_rate", "Abolished slab answered", higher_is_better=False, floor_pp=1.0),
    Metric("stale_cited_rate", "Abolished slab recited", higher_is_better=False, floor_pp=1.0),
    # Not quality, but a serving change that starts mangling the output format
    # or erroring is exactly what this gate is for -- and unlike accuracy,
    # these should be near zero, so the floor is tight.
    Metric("unparseable", "Unparseable responses", higher_is_better=False, floor_pp=0.0),
    Metric("errored", "Errored calls", higher_is_better=False, floor_pp=0.0),
)

METRICS_BY_KEY = {m.key: m for m in METRICS}

#: Fewer repeats than this and the recorded spread is not a spread. Two runs
#: give one gap, which says nothing about how wide the distribution is.
MIN_REPEATS = 3


class GateError(RuntimeError):
    """The comparison cannot be made. Distinct from the comparison failing."""


@dataclass(slots=True)
class Baseline:
    """The committed scores a candidate is judged against.

    `noise` is the observed spread (max - min) per metric across `n_repeats`
    runs of this same configuration. It is the whole basis for tolerance.
    """

    model: str
    precision: str
    dataset_sha: str
    prompt_version: str
    recorded_at: str
    scores: dict[str, float]
    noise: dict[str, float] = field(default_factory=dict)
    n_repeats: int = 0
    note: str = ""

    @classmethod
    def load(cls, path: Path) -> Baseline:
        if not path.exists():
            raise GateError(
                f"no baseline at {path}. Record one before gating:\n"
                f"    python -m gate.record --runs results/eval/*.json"
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            model=raw["model"],
            precision=raw["precision"],
            dataset_sha=raw["dataset_sha"],
            prompt_version=raw["prompt_version"],
            recorded_at=raw["recorded_at"],
            scores=raw["scores"],
            noise=raw.get("noise", {}),
            n_repeats=raw.get("n_repeats", 0),
            note=raw.get("note", ""),
        )

    def tolerance(self, metric: Metric) -> float:
        """How far this metric may move before it counts as a regression.

        The measured spread, or the metric's floor, whichever is larger. Never
        a number chosen to make a particular run pass.
        """
        return max(self.noise.get(metric.key, 0.0), metric.floor_pp)


@dataclass(slots=True)
class Finding:
    metric: Metric
    baseline: float
    candidate: float
    tolerance_pp: float

    @property
    def delta_pp(self) -> float:
        """Change in points. Positive always means *better*, whatever the sign
        convention of the underlying metric -- so the verdict logic below does
        not have to think about direction twice."""
        raw = (self.candidate - self.baseline) * self._scale
        delta = raw if self.metric.higher_is_better else -raw
        # Negating an exact zero yields -0.0, which formats as "-0.0" and
        # reads on a pull request as a regression that rounded away.
        return delta + 0.0 if delta else 0.0

    @property
    def _scale(self) -> float:
        # Rates arrive as fractions, counts as integers. Both are reported in
        # "points" so one tolerance concept covers them.
        return 100.0 if self.metric.key.endswith(("_acc", "_rate", "_f1")) else 1.0

    @property
    def verdict(self) -> str:
        if not self.metric.blocking:
            return "info"
        if self.delta_pp < -self.tolerance_pp:
            return "regressed"
        if self.delta_pp > self.tolerance_pp:
            return "improved"
        return "within noise"

    @property
    def blocks(self) -> bool:
        return self.verdict == "regressed"


@dataclass(slots=True)
class GateResult:
    baseline: Baseline
    findings: list[Finding]
    candidate_meta: dict

    @property
    def passed(self) -> bool:
        return not any(f.blocks for f in self.findings)

    @property
    def regressions(self) -> list[Finding]:
        return [f for f in self.findings if f.blocks]


def _check_comparable(baseline: Baseline, candidate: dict) -> None:
    """Refuse comparisons that would produce a number without a meaning."""
    if baseline.n_repeats < MIN_REPEATS:
        raise GateError(
            f"baseline records {baseline.n_repeats} repeat(s); at least "
            f"{MIN_REPEATS} are needed to know what run-to-run noise looks "
            "like. Without it this gate would be picking a tolerance out of "
            "the air.\n"
            "    python -m gate.record --runs results/eval/*.json"
        )

    cand_sha = candidate.get("dataset_sha", "")
    if cand_sha and cand_sha != baseline.dataset_sha:
        raise GateError(
            f"dataset changed: baseline scored {baseline.dataset_sha}, "
            f"candidate scored {cand_sha}. Scores over different examples are "
            "not comparable -- re-record the baseline against the new dataset "
            "instead of comparing across it."
        )

    cand_prompt = candidate.get("prompt_version", "")
    if cand_prompt and cand_prompt != baseline.prompt_version:
        raise GateError(
            f"prompt changed: baseline used {baseline.prompt_version!r}, "
            f"candidate used {cand_prompt!r}. This gate would measure the "
            "prompt change and blame the serving change."
        )


def compare(baseline: Baseline, candidate: dict) -> GateResult:
    """Judge one candidate run against the committed baseline."""
    _check_comparable(baseline, candidate)

    summary = candidate.get("summary") or {}
    if not summary:
        raise GateError("candidate run has no `summary` block -- did the eval run finish?")

    findings: list[Finding] = []
    for metric in METRICS:
        if metric.key not in summary or metric.key not in baseline.scores:
            continue
        findings.append(
            Finding(
                metric=metric,
                baseline=float(baseline.scores[metric.key]),
                candidate=float(summary[metric.key]),
                tolerance_pp=baseline.tolerance(metric),
            )
        )

    if not findings:
        raise GateError("no metrics in common between baseline and candidate")

    return GateResult(
        baseline=baseline,
        findings=findings,
        candidate_meta={
            "model_key": candidate.get("model_key", ""),
            "served_model_id": candidate.get("served_model_id", ""),
            "dataset_sha": candidate.get("dataset_sha", ""),
            "dataset_n": candidate.get("dataset_n", 0),
            "prompt_version": candidate.get("prompt_version", ""),
            "finished_at": candidate.get("finished_at", ""),
        },
    )


_ICON = {"regressed": "🔴", "improved": "🟢", "within noise": "⚪", "info": "·"}


def markdown(result: GateResult) -> str:
    """The table posted onto the pull request.

    Shows the tolerance in its own column. A reviewer who cannot see why a
    2-point drop passed will not trust the gate, and the answer -- that the
    same configuration moves that much on its own -- is the most important
    thing on the page.
    """
    b = result.baseline
    head = "### Quality gate: " + ("**passed**" if result.passed else "**blocked**")

    lines = [
        head,
        "",
        (
            f"`{result.candidate_meta.get('served_model_id') or b.model}` at "
            f"**{b.precision}** · dataset `{b.dataset_sha}` "
            f"({result.candidate_meta.get('dataset_n', '?')} rows) · "
            f"prompt `{b.prompt_version}`"
        ),
        "",
        "| Metric | Baseline | This run | Δ | Tolerance | |",
        "|---|---:|---:|---:|---:|:--:|",
    ]

    for f in result.findings:
        scale = f._scale
        # Rates read as percentages; unparseable and errored are counts of
        # responses and are shown as whole numbers.
        if scale == 100.0:
            base, cand = f"{f.baseline * 100:.1f}%", f"{f.candidate * 100:.1f}%"
        else:
            base, cand = f"{f.baseline:.0f}", f"{f.candidate:.0f}"
        lines.append(
            f"| {f.metric.label} | {base} | {cand} "
            f"| {f.delta_pp:+.1f} "
            f"| ±{f.tolerance_pp:.1f} "
            f"| {_ICON[f.verdict]} |"
        )

    lines += [
        "",
        (
            f"Tolerance is the spread measured across **{b.n_repeats} repeat "
            f"runs** of this same configuration, recorded {b.recorded_at} — "
            "not a chosen threshold. A change has to move a metric further "
            "than the configuration moves on its own before it counts."
        ),
    ]

    if result.regressions:
        lines += ["", "**Blocking regressions:**"]
        for f in result.regressions:
            lines.append(
                f"- {f.metric.label}: {f.delta_pp:+.1f} points, "
                f"outside the ±{f.tolerance_pp:.1f} noise floor."
            )
    if b.note:
        lines += ["", f"> {b.note}"]

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", type=Path, required=True,
                    help="a run result produced by Project 01's harness")
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--markdown", type=Path, default=None,
                    help="write the PR comment table here")
    args = ap.parse_args()

    try:
        baseline = Baseline.load(args.baseline)
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        result = compare(baseline, candidate)
    except GateError as exc:
        # Exit 2, not 1: "cannot judge" is a different outcome from "judged and
        # failed", and CI should not report an unmeasurable baseline as a
        # quality regression.
        print(f"\n  gate cannot run: {exc}\n")
        return 2

    text = markdown(result)
    print("\n" + text + "\n")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text + "\n", encoding="utf-8")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
