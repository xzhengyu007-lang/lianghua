"""Every claim on the decision page must trace back to an artifact.

RESEARCH_LOG 23.88 made the memo generated and byte-compared, which guarantees it
matches the generator. It does not guarantee the *generator* reads the right
things: a number could be pasted into build_decision_memo.py as a literal and stay
there for years.

This audit closes that loop from the other side. For each §-reference and report
path on the page it checks the target exists, and for each quoted figure -- every
"+-x.xx%" and every three-decimal Sharpe -- it checks the string occurs somewhere
under reports/. A figure that appears nowhere is either derived on the fly (then
it belongs in DERIVED below, with a reason) or it is a transcription.

Coverage limit: it only sees two-decimal percentages and three-decimal ratios. A
derived value written with one decimal, an integer, or a bare ratio is not
machine-checked. A wrong figure that happens to coincide with some other file's
number also passes -- this is a transcription smoke test, not a proof.

Usage::

    python scripts/audit_decision_claims.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEMO = ROOT / "DECISION.md"
LOG = ROOT / "RESEARCH_LOG.md"
REPORTS = ROOT / "reports"

# Figures the memo computes rather than quotes: pattern -> reason.
#
# Currently empty, and that is a finding rather than an omission. The scanner only
# produces two-decimal percentages and three-decimal ratios, and every such figure
# on the page does come from an artifact. The derived values on the page (6.5
# years, 1.54x, +13.4%) are written with one decimal or as bare ratios, so the
# scanner never sees them -- which is exactly the coverage limit documented above.
# If a derived figure ever lands in the scanner's net, it goes here with a reason.
DERIVED: dict[str, str] = {}

SCAN_SUFFIXES = (".json", ".csv", ".md")
MAX_BYTES = 2_000_000


def _corpus() -> str:
    chunks = []
    for path in REPORTS.rglob("*"):
        if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
            continue
        try:
            if path.stat().st_size > MAX_BYTES:
                continue
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(chunks)


def audit(memo_text: str, log_text: str, corpus: str, reports_dir: Path) -> dict:
    sections = sorted({m for m in re.findall(r"§(\d+\.\d+)", memo_text)})
    missing_sections = [s for s in sections if ("### " + s) not in log_text]

    paths = sorted({m for m in re.findall(r"reports/([A-Za-z0-9_\-]+)", memo_text)})
    missing_reports = [p for p in paths if not (reports_dir / p).exists()]

    figures = sorted(set(re.findall(r"[+-]\d+\.\d{2}%", memo_text) + re.findall(r"\b\d\.\d{3}\b", memo_text)))
    orphans = []
    for figure in figures:
        if figure in DERIVED:
            continue
        if figure not in corpus:
            orphans.append(figure)

    return {
        "sections_cited": sections,
        "missing_sections": missing_sections,
        "reports_cited": paths,
        "missing_reports": missing_reports,
        "figures_checked": len(figures),
        "orphan_figures": orphans,
        "derived_allowed": sorted(DERIVED),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memo", default=str(MEMO))
    parser.add_argument("--output", default=str(REPORTS / "audit_decision_claims"))
    args = parser.parse_args(argv)

    memo_text = Path(args.memo).read_text(encoding="utf-8")
    log_text = LOG.read_text(encoding="utf-8")
    result = audit(memo_text, log_text, _corpus(), REPORTS)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    problems = (
        len(result["missing_sections"]) + len(result["missing_reports"]) + len(result["orphan_figures"])
    )
    print("sections cited: %d (missing %d)" % (len(result["sections_cited"]), len(result["missing_sections"])))
    print("reports cited : %d (missing %d)" % (len(result["reports_cited"]), len(result["missing_reports"])))
    print("figures checked: %d (orphans %d)" % (result["figures_checked"], len(result["orphan_figures"])))
    for item in result["missing_sections"]:
        print("  MISSING SECTION §%s" % item)
    for item in result["missing_reports"]:
        print("  MISSING REPORT reports/%s" % item)
    for item in result["orphan_figures"]:
        print("  ORPHAN FIGURE %s" % item)
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
