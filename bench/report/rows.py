"""Compare two rungs answer by answer, not just by score.

    python -m bench.report.rows \
        --baseline results/eval/*_open-weight-vllm_shared.json \
        --candidate results/eval/int4/<int4 run>.json \
        --out results/eval/int4/rows-vs-fp16.md

(The glob is expanded by a POSIX shell; from PowerShell, list the files.)

`--baseline` takes several runs because a baseline is not one fixed set of
answers: fp16's five runs differ from each other by a row. The detail table is
drawn against the first, and a summary line per run shows what holds against
all of them.

A score says how much worse a rung is, not *how* it got worse. A quantised
model scoring 16 points below fp16 could have forgotten which slab goods fall
into, or it could still know and have stopped committing to an answer. Those
are different findings with different fixes, and only the rows tell them
apart.

Every row lands in exactly one bucket:

    same answer        both committed to the same slab
    different answer   both committed, to different slabs
    newly abstained    the baseline answered; the candidate said UNANSWERABLE
    newly answered     the reverse
    both abstained
    unreadable         either side has no parsed slab (error or unparseable)

Correctness is reported separately, as the rows the candidate lost (baseline
right, candidate wrong) and gained (the reverse), because a changed answer is
not necessarily a worse one.

Refuses to compare runs over different rows, datasets or prompt versions, for
the same reason gate/compare.py does: the difference would then be partly the
inputs, and nothing would say how much.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

ABSTAIN = "UNANSWERABLE"

#: Bucket names in the order they are reported.
BUCKETS = ("same_answer", "different_answer", "newly_abstained",
           "newly_answered", "both_abstained", "unreadable")


@dataclass(frozen=True, slots=True)
class RowDiff:
    n: int
    buckets: dict[str, list[str]] = field(default_factory=dict)
    lost: list[str] = field(default_factory=list)
    gained: list[str] = field(default_factory=list)

    def count(self, bucket: str) -> int:
        return len(self.buckets[bucket])


def _rows(run: dict) -> dict[str, dict]:
    return {row["id"]: row for row in run["rows"]}


def _bucket(base_slab: str | None, cand_slab: str | None) -> str:
    # Checked first: two unparsed rows both carry None, and None == None must
    # not be counted as the same answer.
    if base_slab is None or cand_slab is None:
        return "unreadable"
    base_abstains, cand_abstains = base_slab == ABSTAIN, cand_slab == ABSTAIN
    if base_abstains and cand_abstains:
        return "both_abstained"
    if cand_abstains:
        return "newly_abstained"
    if base_abstains:
        return "newly_answered"
    return "same_answer" if base_slab == cand_slab else "different_answer"


def compare(baseline: dict, candidate: dict) -> RowDiff:
    for key in ("dataset_sha", "prompt_version"):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(
                f"runs disagree on {key} ({baseline.get(key)!r} vs "
                f"{candidate.get(key)!r}); a row comparison would mix a change "
                "of inputs into a change of model"
            )
    base, cand = _rows(baseline), _rows(candidate)
    if base.keys() != cand.keys():
        raise ValueError(
            f"runs cover different rows: {len(base.keys() - cand.keys())} only in "
            f"the baseline, {len(cand.keys() - base.keys())} only in the candidate"
        )

    buckets: dict[str, list[str]] = {name: [] for name in BUCKETS}
    lost: list[str] = []
    gained: list[str] = []
    for row_id in sorted(base):
        b, c = base[row_id], cand[row_id]
        buckets[_bucket(b.get("predicted_slab"), c.get("predicted_slab"))].append(row_id)
        if b.get("correct") and not c.get("correct"):
            lost.append(row_id)
        elif c.get("correct") and not b.get("correct"):
            gained.append(row_id)

    return RowDiff(n=len(base), buckets=buckets, lost=lost, gained=gained)


def to_markdown(diff: RowDiff, baseline: dict, candidate: dict) -> str:
    b_name = baseline.get("served_model_id") or "baseline"
    c_name = candidate.get("served_model_id") or "candidate"
    labels = {
        "same_answer": "same answer",
        "different_answer": "different answer",
        "newly_abstained": f"newly abstained (`{c_name}` said {ABSTAIN})",
        "newly_answered": f"newly answered (`{b_name}` said {ABSTAIN})",
        "both_abstained": "both abstained",
        "unreadable": "unreadable on either side",
    }

    lines = [
        f"# `{c_name}` against `{b_name}`, row by row",
        "",
        (f"{diff.n} rows · dataset `{(baseline.get('dataset_sha') or '')[:12]}` · "
         f"prompt `{baseline.get('prompt_version', '')}`"),
        "",
        "| | rows |",
        "|---|---:|",
        *(f"| {labels[name]} | {diff.count(name)} |" for name in BUCKETS),
        "",
        f"**Lost** (`{b_name}` right, `{c_name}` wrong): {len(diff.lost)}"
        + (f" — {', '.join(diff.lost)}" if diff.lost else ""),
        "",
        f"**Gained** (`{b_name}` wrong, `{c_name}` right): {len(diff.gained)}"
        + (f" — {', '.join(diff.gained)}" if diff.gained else ""),
    ]

    changed = [row_id for name in BUCKETS if name not in ("same_answer", "both_abstained")
               for row_id in diff.buckets[name]]
    if changed:
        base, cand = _rows(baseline), _rows(candidate)
        lines += [
            "",
            "## Rows whose answer changed",
            "",
            f"| id | gold | `{b_name}` | `{c_name}` |",
            "|---|---|---|---|",
        ]
        for row_id in sorted(changed):
            b, c = base[row_id], cand[row_id]
            lines.append(
                f"| {row_id} | {b.get('gold_slab')} | {_mark(b)} | {_mark(c)} |"
            )
    return "\n".join(lines) + "\n"


def across_runs(candidate: dict, baselines: list[dict]) -> str:
    """One summary line per baseline run.

    Compared against a single run of a baseline that varies, a candidate can
    look a row better or worse than it is. A line per run keeps the claim to
    what holds against every one of them.
    """
    lines = [
        "## Against every baseline run",
        "",
        ("| baseline run | same | different | newly abstained | newly answered "
         "| lost | gained |"),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for base in baselines:
        diff = compare(base, candidate)
        label = base.get("run_id") or base.get("served_model_id") or "baseline"
        lines.append(
            f"| {label} | {diff.count('same_answer')} | {diff.count('different_answer')} "
            f"| {diff.count('newly_abstained')} | {diff.count('newly_answered')} "
            f"| {len(diff.lost)} | {len(diff.gained)} |"
        )
    return "\n".join(lines) + "\n"


def _mark(row: dict) -> str:
    slab = row.get("predicted_slab")
    return f"{slab if slab is not None else '—'} {'✓' if row.get('correct') else '✗'}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", type=Path, nargs="+", required=True,
                    help="one or more runs of the baseline; the detail table uses "
                         "the first")
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    baselines = [json.loads(p.read_text(encoding="utf-8")) for p in args.baseline]
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    try:
        report = to_markdown(compare(baselines[0], candidate), baselines[0], candidate)
        if len(baselines) > 1:
            report += "\n" + across_runs(candidate, baselines)
    except ValueError as exc:
        print(f"\n  cannot compare: {exc}\n")
        return 2

    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
