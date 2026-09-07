"""Decide whether a change needs a fresh eval run attached to it.

    python -m gate.guard --changed changed.txt

CI runs on GitHub-hosted runners, which have no GPU. The eval therefore cannot
be executed inside the pipeline on this project's budget, and pretending
otherwise would produce a gate that is green because it never ran.

So the enforceable rule is the one a free runner *can* check: **a change that
alters what the model outputs must arrive with a scored run attached.** The
measurement happens on borrowed GPU time, by hand, and its result is committed
to the pull request; CI's job is to refuse changes that skipped that step, and
then to compare the attached run against the baseline.

That is a weaker gate than running the eval in CI, and the README says so
plainly rather than implying a pipeline that does not exist. It is not a weak
gate: it makes an unmeasured serving change impossible to merge, which is the
failure mode that actually matters.
"""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path

#: Changing any of these changes what comes out of the server, so a score
#: recorded before the change no longer describes it.
SERVING_PATHS: tuple[tuple[str, str], ...] = (
    ("deploy/k8s/configmap.yaml", "serving configuration (precision, context, batch size)"),
    ("serving/entrypoint.sh", "how vLLM is launched"),
    ("serving/Dockerfile", "the serving image, including the vLLM version"),
    ("deploy/k8s/deployment.yaml", "the deployed pod spec"),
)

#: Where a scored run must land. Project 01's harness writes this shape.
RESULTS_GLOB = "results/eval/*.json"

#: Changing the baseline itself is allowed, but it is the one change a gate
#: cannot check -- so it is called out for human review rather than passed.
BASELINE_PATH = "gate/baseline.json"


def _normalise(path: str) -> str:
    """Strip the things that make a path silently fail to match.

    A BOM on the first line is the dangerous one: it renders invisibly, and it
    would make this guard report "no serving change" for a pull request that
    changes the precision -- failing open, which is the one direction a gate
    must never fail. Backslashes because git on Windows reports them.
    """
    return path.replace("﻿", "").strip().replace("\\", "/")


def classify(changed: list[str]) -> tuple[list[str], bool, bool]:
    """Return (reasons a run is needed, a run was attached, baseline was edited)."""
    normalised = [n for c in changed if (n := _normalise(c))]

    reasons = [
        f"`{path}` — {why}"
        for path, why in SERVING_PATHS
        if path in normalised
    ]
    attached = any(fnmatch.fnmatch(c, RESULTS_GLOB) for c in normalised)
    baseline_edited = BASELINE_PATH in normalised
    return reasons, attached, baseline_edited


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--changed", type=Path, required=True,
                    help="file listing changed paths, one per line")
    args = ap.parse_args()

    # utf-8-sig so a BOM is stripped by the decoder as well as by _normalise.
    changed = args.changed.read_text(encoding="utf-8-sig").splitlines()

    # An empty list is not "nothing changed" -- a pull request always changes
    # something. It means the diff failed, and treating that as a pass would
    # disable the gate exactly when CI is misconfigured.
    if not [c for c in changed if c.strip()]:
        print(
            f"\n  BLOCKED: {args.changed} lists no changed files.\n\n"
            "  A pull request always changes something, so this means the diff\n"
            "  did not run -- most likely a shallow checkout. The gate refuses\n"
            "  to pass rather than reporting nothing to do.\n"
        )
        return 1

    reasons, attached, baseline_edited = classify(changed)

    if baseline_edited:
        print(
            "\n  NOTE: gate/baseline.json is modified by this change.\n"
            "  A gate cannot check its own baseline -- this needs a human to\n"
            "  confirm the new baseline came from repeat runs of the new\n"
            "  configuration and not from the run being gated.\n"
        )

    if not reasons:
        print("  no serving-behaviour change; nothing to gate")
        # Exit 0 with no eval required. Written to stdout for the workflow to
        # branch on rather than inferred from an exit code alone.
        print("::notice::gate skipped -- no serving change in this pull request")
        return 0

    print("\n  This change alters what the model outputs:")
    for reason in reasons:
        print(f"    - {reason}")

    if not attached:
        print(
            "\n  BLOCKED: no scored run attached.\n\n"
            "  CI has no GPU, so the eval cannot run here. Run it against the\n"
            "  new configuration on your own hardware and commit the result:\n\n"
            "      NIM_BASE_URL=http://localhost:8000 NIM_MODEL=<served id> \\\n"
            "        python -m harness.run --model open-weight-vllm\n"
            "      cp <harness>/results/<run>.json results/eval/\n\n"
            "  Then push. The gate will compare it against gate/baseline.json.\n"
        )
        return 1

    print("\n  scored run attached; handing over to gate.compare\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
