"""The three request shapes, drawn from real domain traffic.

Uniform synthetic load is the third pitfall in the project plan, and it is the
one that quietly invalidates everything downstream: prompts of identical length
batch far more neatly than real ones, so a benchmark built on them reports a
throughput the deployment will never see. Every prompt here is a real Indian
GST classification request, taken from the corpora Project 01 collected.

The three profiles behave completely differently under continuous batching,
which is the point of measuring all three:

    short     brief prompt, brief completion   -- classification-shaped.
              Prefill is trivial, so throughput is bounded by decode and the
              batch fills with many small sequences.

    long_in   long document, brief completion  -- extraction-shaped. This is
              Project 01's actual task. Prefill dominates, the KV cache is
              large per sequence, and max batch size is bounded by VRAM rather
              than by compute.

    long_out  brief prompt, long completion    -- summarisation-shaped. Decode
              dominates, sequences stay resident for a long time, and this is
              where continuous batching earns its keep against static batching.

Prompts are stored fully rendered, exactly as Project 01's `harness.prompt`
builds them, rather than as raw inputs re-templated here. Two reasons: the load
test then exercises the same bytes the quality run does, and `bench.loadgen`
stays stdlib-only instead of importing the harness at request time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CORPUS_DIR = Path("data/workloads")


@dataclass(frozen=True, slots=True)
class Profile:
    """One request shape, with the output cap that defines it."""

    key: str
    shape: str
    #: Cap on generated tokens. This is part of the profile's identity, not a
    #: safety limit -- `long_out` is only long because this is large, and
    #: changing it changes what is being measured.
    max_tokens: int
    #: What the profile is meant to represent, printed in reports.
    stands_for: str
    note: str = ""


PROFILES: dict[str, Profile] = {
    "short": Profile(
        key="short",
        shape="brief in, brief out",
        max_tokens=48,
        stands_for="classifying a packaged retail product into a GST slab",
        note="from Open Food Facts listings, e.g. 'Parle-G Biscuit, 45gm'",
    ),
    "long_in": Profile(
        key="long_in",
        shape="long in, brief out",
        max_tokens=48,
        stands_for="Project 01's task: reading an advance ruling and "
                   "returning the slab and heading",
        note="the rung the quantisation ladder is expected to hurt most, "
             "because exact-match scoring has no partial credit",
    ),
    "long_out": Profile(
        key="long_out",
        shape="brief in, long out",
        max_tokens=768,
        stands_for="writing a classification note a practitioner could file",
        note="decode-bound; the profile that separates continuous batching "
             "from static batching most clearly",
    ),
}


@dataclass(frozen=True, slots=True)
class Request:
    """One replayable request. `system` and `prompt` are already rendered."""

    id: str
    profile: str
    system: str
    prompt: str
    max_tokens: int
    #: Provenance, so any request in a load run traces back to a real document.
    source: str = ""
    source_id: str = ""
    #: Token count at build time, from the served model's own tokenizer where
    #: one was available. Zero means uncounted -- reports say so rather than
    #: substituting a character-length guess.
    prompt_tokens: int = 0


def corpus_path(profile: str) -> Path:
    return CORPUS_DIR / f"{profile}.jsonl"


def load(profile: str, limit: int | None = None) -> list[Request]:
    """Read one profile's corpus.

    Raises rather than returning an empty list: a load run against zero
    requests reports a throughput of zero and an error rate of zero, which
    looks like a result.
    """
    if profile not in PROFILES:
        raise KeyError(f"unknown profile {profile!r}; known: {sorted(PROFILES)}")

    path = corpus_path(profile)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Build it first:\n"
            f"    python -m bench.build_corpus --harness ../domain-eval-harness"
        )

    rows: list[Request] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw = json.loads(line)
            rows.append(
                Request(
                    id=raw["id"],
                    profile=raw.get("profile", profile),
                    system=raw.get("system", ""),
                    prompt=raw["prompt"],
                    max_tokens=raw.get("max_tokens", PROFILES[profile].max_tokens),
                    source=raw.get("source", ""),
                    source_id=raw.get("source_id", ""),
                    prompt_tokens=raw.get("prompt_tokens", 0),
                )
            )

    if not rows:
        raise ValueError(f"{path} is empty -- nothing to replay")
    return rows[:limit] if limit else rows


def get(profile: str) -> Profile:
    if profile not in PROFILES:
        raise KeyError(f"unknown profile {profile!r}; known: {sorted(PROFILES)}")
    return PROFILES[profile]
