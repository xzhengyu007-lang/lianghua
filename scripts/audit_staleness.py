"""Which published results are now stale, and which citations point at nothing?

The repository has 100+ report directories, ~50 scripts and 66 findings. Nothing
tells a reader whether a given table was produced by the engine that exists today,
or whether a FINDINGS row still points at something that is there.

Two checks, both cheap enough to run after every change:

* staleness -- a report is stale if the engine source or the cached data has been
  modified after the report was written. That does not mean it is wrong, but it
  does mean it has not been re-derived since the last change, so it must not be
  quoted without re-running it.
* citation integrity -- every FINDINGS row that names a report directory or a
  research-log section must point at something that exists.

Usage::

    python scripts/audit_staleness.py
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
ENGINE = ROOT / "quant_system"
CACHE = ROOT / "data" / "cache"

# Directories that are regenerated on demand rather than produced by a study.
TRANSIENT = {"dashboard", "verification", "publish_check", "runs"}

# The reports the decision actually rests on: DECISION.md and the README
# authority table quote these, so they must be derived from today's engine.
CRITICAL = (
    # Every report DECISION.md reads. tests/test_audit_staleness.py asserts this
    # set covers the generator's dependencies, so the page cannot silently rest
    # on an artifact nobody is watching.
    "random_control",
    "naive_benchmark",
    "quarter_consistency",
    "execution_timing",
    "year_table_blend",
    "voltarget_blend",
    "phase_recheck",
    "phase_blend",
    "power_recheck",
    "capacity_blend",
    "cadence_blend",
    "cross_pool_random",
    "selection_slack",
    "breadth_ic",
    "factor_cohorts",
    "pnl_cohorts",
)


def newest_mtime(paths) -> float:
    newest = 0.0
    for path in paths:
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            continue
    return newest


def scan_reports(code_mtime: float, data_mtime: float) -> list[dict]:
    rows = []
    now = time.time()
    for directory in sorted(REPORTS.iterdir()):
        if not directory.is_dir():
            continue
        files = [p for p in directory.rglob("*") if p.is_file()]
        if not files:
            continue
        written = newest_mtime(files)
        rows.append(
            {
                "report": directory.name,
                "written": datetime.fromtimestamp(written).strftime("%Y-%m-%d %H:%M"),
                "age_days": round((now - written) / 86400.0, 1),
                "has_metrics": (directory / "metrics.json").exists(),
                "has_summary": (directory / "summary.json").exists(),
                "transient": directory.name in TRANSIENT,
                "stale_by_code": bool(code_mtime > written),
                "stale_by_data": bool(data_mtime > written),
            }
        )
    return rows


def check_citations() -> list[dict]:
    findings = ROOT / "FINDINGS.md"
    log = ROOT / "RESEARCH_LOG.md"
    if not findings.exists():
        return []

    log_text = log.read_text(encoding="utf-8") if log.exists() else ""
    problems = []
    for line in findings.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 5 or not cells[0].isdigit():
            continue
        reference = cells[4]
        for section in re.findall(r"§(\d+\.\d+)", reference):
            if ("### " + section) not in log_text:
                problems.append({"row": int(cells[0]), "kind": "missing_section", "target": section})
        for report in re.findall(r"reports/([A-Za-z0-9_\-]+)", reference):
            if not (REPORTS / report).exists():
                problems.append({"row": int(cells[0]), "kind": "missing_report", "target": report})
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(REPORTS / "audit_staleness"))
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    code_mtime = newest_mtime(ENGINE.glob("*.py"))
    cache_files = list(CACHE.glob("*.parquet"))
    data_mtime = newest_mtime(cache_files)
    print("engine newest: %s" % datetime.fromtimestamp(code_mtime).strftime("%Y-%m-%d %H:%M"), flush=True)
    print("cache newest : %s (%d files)" % (
        datetime.fromtimestamp(data_mtime).strftime("%Y-%m-%d %H:%M"), len(cache_files)), flush=True)

    rows = scan_reports(code_mtime, data_mtime)
    problems = check_citations()

    studies = [r for r in rows if not r["transient"]]
    stale_code = [r for r in studies if r["stale_by_code"]]
    stale_data = [r for r in studies if r["stale_by_data"]]
    fresh = [r for r in studies if not r["stale_by_code"] and not r["stale_by_data"]]

    by_name = {r["report"]: r for r in rows}
    critical_rows = [
        {
            "report": name,
            "present": name in by_name,
            "fresh": bool(name in by_name and not by_name[name]["stale_by_code"] and not by_name[name]["stale_by_data"]),
        }
        for name in CRITICAL
    ]
    critical_stale = [r for r in critical_rows if not r["fresh"]]

    summary = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "critical": critical_rows,
        "critical_stale": len(critical_stale),
        "engine_newest": datetime.fromtimestamp(code_mtime).strftime("%Y-%m-%d %H:%M"),
        "cache_newest": datetime.fromtimestamp(data_mtime).strftime("%Y-%m-%d %H:%M"),
        "counts": {
            "reports_total": len(rows),
            "studies": len(studies),
            "fresh": len(fresh),
            "stale_by_code": len(stale_code),
            "stale_by_data": len(stale_data),
            "citation_problems": len(problems),
        },
        "stale_examples": sorted(
            [{"report": r["report"], "written": r["written"], "age_days": r["age_days"]} for r in stale_code],
            key=lambda r: r["age_days"],
        )[:15],
        "citation_problems": problems,
        "rows": rows,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )

    lines = ["# 新鲜度审计", ""]
    lines.append("引擎最新改动：**%s**；缓存最新写入：**%s**。" % (summary["engine_newest"], summary["cache_newest"]))
    lines.append("")
    lines.append("| 项 | 数量 |")
    lines.append("|---|---:|")
    lines.append("| 报告目录 | %d |" % len(rows))
    lines.append("| 其中研究类（排除 %s） | %d |" % ("/".join(sorted(TRANSIENT)), len(studies)))
    lines.append("| 未被引擎改动追上的 | **%d** |" % len(fresh))
    lines.append("| 引擎改动后未重跑 | **%d** |" % len(stale_code))
    lines.append("| 数据更新后未重跑 | **%d** |" % len(stale_data))
    lines.append("| 台账引用失效 | **%d** |" % len(problems))
    lines.append("")
    lines.append("## 关键报告（决策所依赖的，必须新鲜）")
    lines.append("")
    lines.append("| 报告 | 存在 | 新鲜 |")
    lines.append("|---|---|---|")
    for row in critical_rows:
        lines.append("| `%s` | %s | %s |" % (
            row["report"],
            "是" if row["present"] else "**否**",
            "是" if row["fresh"] else "**否**",
        ))
    lines.append("")
    lines.append("上面这一组是 `DECISION.md` 与 README 权威表实际引用的报告。**它们全部为「是」时，")
    lines.append("决策页才是可信的**；其余历史产物只需要在「引用到时」重跑。")
    lines.append("")
    if stale_code:
        lines.append("## 最久未重跑的研究（引擎已变）")
        lines.append("")
        lines.append("| 报告 | 生成时间 | 已过天数 |")
        lines.append("|---|---|---:|")
        for row in summary["stale_examples"]:
            lines.append("| `%s` | %s | %.1f |" % (row["report"], row["written"], row["age_days"]))
        lines.append("")
    if problems:
        lines.append("## 台账引用失效")
        lines.append("")
        lines.append("| 条目 | 类型 | 目标 |")
        lines.append("|---:|---|---|")
        for row in problems:
            lines.append("| %d | %s | `%s` |" % (row["row"], row["kind"], row["target"]))
        lines.append("")
    lines.append("> 说明：**「未重跑」不等于「错」**。它只说明这份报告不是用今天的引擎和数据重新推导的，")
    lines.append("> 因此引用前必须重跑。`DECISION.md` 与 README 权威表所依赖的报告都在本表里可查。")
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\ncounts: %s" % json.dumps(summary["counts"], ensure_ascii=False))
    print("report: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
