"""Keep a table of contents at the top of RESEARCH_LOG.md.

The log is 75 sections long and grows every session; scrolling to find "which
section covered the 12% target" is a real cost. The TOC is generated, not
hand-maintained, so it cannot drift: rerun this script after appending.

Usage::

    python scripts/build_research_toc.py
    python scripts/build_research_toc.py --check     # fail if stale (CI/use in tests)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START = "<!-- TOC -->"
END = "<!-- /TOC -->"
HEADING = re.compile(r"^### (23\.\d+)\s+(.+)$")


def build_toc(text: str) -> str:
    entries = []
    for line in text.splitlines():
        match = HEADING.match(line.strip())
        if match:
            number, title = match.groups()
            entries.append(f"- {number} {title}")
    body = "\n".join(["## 目录", "", *entries, ""])
    return f"{START}\n{body}{END}"


def apply_toc(text: str) -> str:
    block = build_toc(text)
    if START in text and END in text:
        start = text.index(START)
        end = text.index(END) + len(END)
        return text[:start] + block + text[end:]
    # Insert before the first "## 1." heading so the intro stays on top.
    anchor = text.find("\n## 1.")
    if anchor == -1:
        return text.rstrip("\n") + "\n\n" + block + "\n"
    return text[:anchor] + "\n" + block + "\n" + text[anchor:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=str(ROOT / "RESEARCH_LOG.md"))
    parser.add_argument("--check", action="store_true", help="exit 1 when the TOC is stale")
    args = parser.parse_args(argv)

    path = Path(args.path)
    text = path.read_text(encoding="utf-8")
    updated = apply_toc(text)
    if args.check:
        if updated != text:
            print("TOC is stale; run scripts/build_research_toc.py")
            return 1
        print("TOC up to date")
        return 0
    if updated == text:
        print("TOC already up to date")
        return 0
    path.write_text(updated, encoding="utf-8")
    count = updated.count("\n- 23.") + (1 if updated.startswith("\n- 23.") else 0)
    print(f"updated TOC ({count} sections)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
