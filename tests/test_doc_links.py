"""Every local link and image in the docs points at something that exists.

A portfolio repo is read by following its links. A dead one costs the reader
more trust than the paragraph around it earns, and nothing else in the build
notices: Markdown has no compiler, and GitHub renders a broken image as a
silent box. These are the links committed in this repo, checked against the
files committed beside them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Markdown links and images: [text](target) and ![alt](target).
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

#: Files whose links are checked: the repo's own prose, not anything vendored.
DOCS = sorted(
    [ROOT / "README.md"]
    + [p for p in (ROOT / "docs").rglob("*.md")]
    + [p for p in (ROOT / "results").rglob("*.md")]
)


def _targets(path: Path) -> list[str]:
    return LINK.findall(path.read_text(encoding="utf-8"))


def test_the_docs_are_actually_being_scanned():
    # Guards the tests below: a broken regex or an empty file list would make
    # them pass by finding nothing.
    assert len(DOCS) >= 5, f"expected several docs, found {[p.name for p in DOCS]}"
    assert sum(len(_targets(p)) for p in DOCS) >= 20


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_local_link_resolves(doc: Path):
    missing = []
    for target in _targets(doc):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        # Strip an anchor: README.md#section points at the file.
        relative = target.split("#", 1)[0]
        if not relative:
            continue
        if not (doc.parent / relative).resolve().exists():
            missing.append(target)
    assert not missing, f"{doc.relative_to(ROOT)} links to missing paths: {missing}"


def test_every_committed_chart_and_screenshot_is_referenced():
    """The other direction: an image nobody links to is usually a stale one.

    It also catches a chart renamed in the report but left behind in the repo,
    which would otherwise sit there looking current.
    """
    linked = {
        (doc.parent / target.split("#", 1)[0]).resolve()
        for doc in DOCS
        for target in _targets(doc)
        if not target.startswith(("http://", "https://", "mailto:", "#"))
    }
    images = [p for folder in ("charts", "screenshots")
              for p in (ROOT / "docs" / folder).glob("*.png")]
    orphans = sorted(p.name for p in images if p.resolve() not in linked)
    assert not orphans, f"committed images nothing links to: {orphans}"
