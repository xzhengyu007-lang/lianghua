"""Rebuild the reports the decision rests on, in one command.

RESEARCH_LOG 23.95 found 201 report directories that predate the current engine,
and made the point that a 201-item alarm is not actionable. The actionable set is
the twelve reports DECISION.md and the README authority table quote. This script
re-runs exactly those, in dependency order, and then re-checks freshness so the
answer is not "it exited 0" but "the critical set is fresh".

Nothing here is clever: it is a registry plus a subprocess loop. The value is that
the registry is cross-checked against audit_staleness.CRITICAL by a test, so the
two cannot drift apart.

Usage::

    python scripts/refresh_critical.py --dry-run
    python scripts/refresh_critical.py --only factor_cohorts pnl_cohorts
    python scripts/refresh_critical.py            # everything, ~25 minutes
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

# (report directory, script, extra args). Order matters only for readability;
# none of these reads another's output.
PLAN = [
    ("factor_cohorts", "run_factor_cohorts.py", []),
    ("breadth_ic", "run_breadth_ic.py", []),
    ("pnl_cohorts", "run_pnl_cohorts.py", []),
    ("selection_slack", "run_selection_slack.py", []),
    ("phase_recheck", "run_phase_recheck.py", []),
    ("phase_blend", "run_phase_blend.py", []),
    ("voltarget_blend", "run_voltarget_blend.py", []),
    ("capacity_blend", "run_capacity_blend.py", []),
    ("year_table_blend", "run_year_table_blend.py", []),
    ("power_recheck", "run_power_recheck.py", []),
    ("cadence_blend", "run_cadence_blend.py", []),
    ("cross_pool_random", "run_cross_pool_random.py", []),
    ("execution_timing", "run_execution_timing.py", []),
    ("quarter_consistency", "run_quarter_consistency.py", []),
    ("random_control", "run_random_control.py", []),
    ("naive_benchmark", "run_naive_benchmark.py", []),
    ("filter_blend_ablation", "run_filter_blend_ablation.py", []),
    ("rank_overlap", "run_rank_overlap.py", []),
]

POST = [
    ("publish_check", "run_publish_check.py", []),
    ("verification", "run_verification.py", []),
]


def plan_entries():
    return [(name, script, list(args)) for name, script, args in PLAN + POST]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--only", nargs="+", default=None, help="run only these report names")
    parser.add_argument("--output", default=str(ROOT / "reports" / "refresh_critical"))
    args = parser.parse_args(argv)

    entries = plan_entries()
    if args.only:
        wanted = set(args.only)
        entries = [e for e in entries if e[0] in wanted]
        missing = wanted - {e[0] for e in entries}
        if missing:
            print("unknown targets: %s" % sorted(missing))
            return 2

    if args.dry_run:
        print("plan (%d steps):" % len(entries))
        for name, script, extra in entries:
            print("  %-18s scripts/%s %s" % (name, script, " ".join(extra)))
        return 0

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    started = time.time()
    for name, script, extra in entries:
        print("\n=== %s (%s) ===" % (name, script), flush=True)
        step_started = time.time()
        completed = subprocess.run(
            [PYTHON, str(ROOT / "scripts" / script)] + extra,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        duration = time.time() - step_started
        tail = (completed.stdout or "").strip().splitlines()[-3:]
        results.append(
            {
                "name": name,
                "script": script,
                "exit_code": completed.returncode,
                "seconds": round(duration, 1),
                "tail": tail,
            }
        )
        print("  exit=%d  %.1fs" % (completed.returncode, duration), flush=True)
        if completed.returncode != 0:
            print((completed.stderr or "")[-800:], flush=True)

    total = time.time() - started
    failed = [r for r in results if r["exit_code"] != 0]

    fresh = None
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "audit_staleness", ROOT / "scripts" / "audit_staleness.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        code_mtime = module.newest_mtime(module.ENGINE.glob("*.py"))
        data_mtime = module.newest_mtime(module.CACHE.glob("*.parquet"))
        rows = {r["report"]: r for r in module.scan_reports(code_mtime, data_mtime)}
        fresh = {
            name: bool(
                name in rows
                and not rows[name]["stale_by_code"]
                and not rows[name]["stale_by_data"]
            )
            for name, _script, _extra in plan_entries()
        }
    except Exception as exc:  # freshness is a report, not a hard dependency
        print("freshness check skipped: %s" % exc)

    payload = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_seconds": round(total, 1),
        "steps": results,
        "failed": [r["name"] for r in failed],
        "fresh_after": fresh,
    }
    (out / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )

    print("\n---- summary ----")
    for row in results:
        print("  %-18s exit=%d %7.1fs" % (row["name"], row["exit_code"], row["seconds"]))
    print("  total %.1fs, %d failed" % (total, len(failed)))
    if fresh is not None:
        stale = [k for k, v in fresh.items() if not v]
        print("  fresh after run: %d/%d%s" % (
            len(fresh) - len(stale), len(fresh), "" if not stale else "  stale: " + ", ".join(stale)))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
