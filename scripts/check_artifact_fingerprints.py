"""Fingerprint the upstream research artifacts so a silent change cannot hide.

RESEARCH_LOG 23.98 made the gap explicit: a research script can be re-run and
produce slightly different numbers, and nothing anywhere would notice. The
publish-config check (23.87) covers the deployable configuration; this covers the
five upstream studies the rest of the analysis stands on.

The baseline is a content hash per artifact file, stored in
data/artifact_fingerprints.json (tracked; data/cache is not). Any change shows up
as a per-file mismatch with the old and new digest, so the question is never
"did something change" but "which file changed".

Updating the baseline is deliberate: re-run the study, read the new numbers,
then --update and say in RESEARCH_LOG why they moved.

Usage::

    python scripts/check_artifact_fingerprints.py            # verify
    python scripts/check_artifact_fingerprints.py --update   # re-baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
BASELINE = ROOT / "data" / "artifact_fingerprints.json"

# The upstream studies: everything downstream reads these numbers.
TARGETS = {
    "factor_cohorts": ("summary.json", "ic_by_cohort.csv"),
    "breadth_ic": ("summary.json", "ic_by_horizon.csv"),
    "pnl_cohorts": ("summary.json", "summary.csv"),
    "selection_slack": ("summary.json", "yearly.csv"),
    "phase_recheck": ("spread.csv", "phases.csv"),
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan() -> dict:
    out = {}
    for report, files in TARGETS.items():
        entry = {}
        for name in files:
            path = REPORTS / report / name
            entry[name] = digest(path) if path.exists() else None
        out[report] = entry
    return out


def compare(current: dict, baseline: dict) -> list[dict]:
    problems = []
    for report, files in current.items():
        for name, value in files.items():
            old = (baseline.get(report) or {}).get(name)
            if value is None:
                problems.append({"report": report, "file": name, "kind": "missing_artifact"})
            elif old is None:
                problems.append({"report": report, "file": name, "kind": "not_in_baseline"})
            elif old != value:
                problems.append(
                    {
                        "report": report,
                        "file": name,
                        "kind": "changed",
                        "baseline": old[:12],
                        "current": value[:12],
                    }
                )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="rewrite the baseline")
    parser.add_argument("--baseline", default=str(BASELINE))
    args = parser.parse_args(argv)

    baseline_path = Path(args.baseline)
    current = scan()

    if args.update:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps({"updated": datetime.now().strftime("%Y-%m-%d %H:%M"), "hashes": current},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("baseline written: %s" % baseline_path)
        return 0

    if not baseline_path.exists():
        print("no baseline at %s; run with --update" % baseline_path)
        return 2

    stored = json.loads(baseline_path.read_text(encoding="utf-8")).get("hashes", {})
    problems = compare(current, stored)

    for problem in problems:
        if problem["kind"] == "changed":
            print("  CHANGED  %s/%s  %s -> %s"
                  % (problem["report"], problem["file"], problem["baseline"], problem["current"]))
        else:
            print("  %-18s %s/%s" % (problem["kind"].upper(), problem["report"], problem["file"]))
    print("artifacts checked: %d files across %d reports, %d problems"
          % (sum(len(v) for v in current.values()), len(current), len(problems)))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
