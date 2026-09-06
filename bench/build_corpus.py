"""Build the three workload corpora from Project 01's collected traffic.

    python -m bench.build_corpus --harness ../domain-eval-harness

Prompts are rendered here, at build time, by importing Project 01's own
`harness.prompt` rather than re-implementing its template. Two consequences,
both wanted: the load generator replays the exact bytes the quality run sends,
and it stays stdlib-only because no harness import happens at request time.

The rendered corpora are committed. A load sweep has to be replayable months
later against a repo that may no longer have Project 01 checked out beside it,
and re-deriving the prompts from a moved dependency is how a "reproducible"
benchmark stops reproducing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from bench.workloads import CORPUS_DIR, PROFILES

#: Sample sizes. Large enough that the replay does not loop often within a
#: run -- a corpus short enough to cycle would let vLLM's automatic prefix
#: cache serve repeats, inflating throughput for reasons that have nothing to
#: do with the deployment.
SIZES = {"short": 200, "long_in": 60, "long_out": 60}

#: `short` and `long_out` both draw on product listings. They are sampled
#: disjointly so a sweep that runs one after the other cannot benefit from a
#: prefix cache warmed by the other.
SEED = 20260906

#: Sanity bounds, in characters. Open Food Facts carries some near-empty
#: titles and some pathological ones; an AAR that survived PDF extraction
#: badly is not representative traffic.
SHORT_MIN, SHORT_MAX = 8, 400
LONG_MIN, LONG_MAX = 2_000, 40_000

LONG_OUT_TEMPLATE = """\
Write a GST classification note for the goods below, of the kind a tax \
practitioner would put on file for a client.

Goods: {description}

Cover, in prose:
- the HSN heading you would claim, and why that heading rather than the \
nearest competing one
- the schedule entry that fixes the rate, and the combined rate itself
- which facts about the goods would change the answer
- what you would ask the client to confirm before filing"""


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_harness_prompt(harness_dir: Path):
    """Import Project 01's prompt module from a sibling checkout."""
    if not (harness_dir / "harness" / "prompt.py").exists():
        raise SystemExit(
            f"no harness/prompt.py under {harness_dir}. Point --harness at a "
            "checkout of gst-eval-harness."
        )
    sys.path.insert(0, str(harness_dir.resolve()))
    from harness import prompt as prompt_mod  # noqa: PLC0415

    return prompt_mod


def build(harness_dir: Path, out_dir: Path) -> dict[str, int]:
    prompt_mod = _load_harness_prompt(harness_dir)
    data = harness_dir / "data" / "raw"

    off = [
        r for r in _read_jsonl(data / "off.jsonl")
        if SHORT_MIN <= len(r.get("input", "").strip()) <= SHORT_MAX
    ]
    aar = [
        r for r in _read_jsonl(data / "aar.jsonl")
        if LONG_MIN <= len(r.get("input", "").strip()) <= LONG_MAX
    ]
    if not off or not aar:
        raise SystemExit("source corpora are empty after filtering -- check --harness")

    rng = random.Random(SEED)
    rng.shuffle(off)
    rng.shuffle(aar)

    n_short = min(SIZES["short"], len(off))
    n_long_out = min(SIZES["long_out"], max(0, len(off) - n_short))
    short_rows = off[:n_short]
    long_out_rows = off[n_short : n_short + n_long_out]
    long_in_rows = aar[: min(SIZES["long_in"], len(aar))]

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}

    written["short"] = _write(
        out_dir / "short.jsonl",
        [
            _request(f"short-{i:04d}", "short", prompt_mod.SYSTEM,
                     prompt_mod.build(r["input"]), r)
            for i, r in enumerate(short_rows)
        ],
    )
    written["long_in"] = _write(
        out_dir / "long_in.jsonl",
        [
            _request(f"longin-{i:04d}", "long_in", prompt_mod.SYSTEM,
                     prompt_mod.build(r["input"]), r)
            for i, r in enumerate(long_in_rows)
        ],
    )
    written["long_out"] = _write(
        out_dir / "long_out.jsonl",
        [
            # No system prompt: this profile is not the classification task and
            # borrowing its "answer with the rate in force" instruction would
            # push the model toward a short answer, which is the opposite of
            # what this profile is for.
            _request(f"longout-{i:04d}", "long_out", "",
                     LONG_OUT_TEMPLATE.format(description=r["input"].strip()), r)
            for i, r in enumerate(long_out_rows)
        ],
    )
    return written


def _request(rid: str, profile: str, system: str, prompt: str, src: dict) -> dict:
    return {
        "id": rid,
        "profile": profile,
        "system": system,
        "prompt": prompt,
        "max_tokens": PROFILES[profile].max_tokens,
        "source": src.get("source", ""),
        "source_id": str(src.get("source_id", "")),
        # Characters, not tokens. The token count is whatever the served
        # model's tokenizer says, and that is recorded per request at run time
        # from the server's own usage field rather than estimated here.
        "prompt_chars": len(prompt),
    }


def _write(path: Path, rows: list[dict]) -> int:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--harness",
        type=Path,
        default=Path("../domain-eval-harness"),
        help="checkout of gst-eval-harness (Project 01)",
    )
    ap.add_argument("--out", type=Path, default=CORPUS_DIR)
    args = ap.parse_args()

    counts = build(args.harness, args.out)
    print(f"\n  workload corpora -> {args.out}\n")
    for profile, n in counts.items():
        path = args.out / f"{profile}.jsonl"
        chars = [json.loads(line)["prompt_chars"] for line in
                 path.read_text(encoding="utf-8").splitlines() if line.strip()]
        median = sorted(chars)[len(chars) // 2] if chars else 0
        print(f"    {profile:<9} {n:>4} requests · median prompt {median:>6,} chars "
              f"· sha {_sha(path)}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
