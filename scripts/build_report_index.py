"""Build reports/INDEX.md: one line per report directory with its headline metrics.

There are well over a hundred experiment directories by now, and finding "which
run produced the +43.68%" was a manual grep. This script is read-only over the
artifacts: it never recomputes anything, it just indexes what is on disk.

Usage::

    python scripts/build_report_index.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
SKIP = {"runs"}


def read_metrics(directory: Path) -> dict | None:
    path = directory / "metrics.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload.get("metrics", payload) if isinstance(payload, dict) else None


def list_analysis_reports(reports: Path) -> list[dict]:
    """Directories that hold a markdown report but no single ``metrics.json``.

    Cross-window studies (attribution, bootstrap CI, ablation panels) write one
    table per window instead of one metric set, so they cannot carry
    ``metrics.json``. They still deserve an entry, otherwise the index quietly
    hides half the research.
    """
    found = []
    for directory in sorted(reports.iterdir()):
        if not directory.is_dir() or directory.name in SKIP:
            continue
        report = directory / "report.md"
        if not report.exists() or (directory / "metrics.json").exists():
            continue
        found.append(
            {
                "name": directory.name,
                "generated": datetime.fromtimestamp(report.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            }
        )
    return found


def build(reports: Path = REPORTS, output: Path | None = None) -> tuple[Path, int]:
    reports = Path(reports)
    rows = []
    for directory in sorted(reports.iterdir()):
        if not directory.is_dir() or directory.name in SKIP:
            continue
        metrics = read_metrics(directory)
        if not metrics:
            continue
        stat = (directory / "metrics.json").stat()
        rows.append(
            {
                "name": directory.name,
                "generated": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "return": metrics.get("total_return"),
                "sharpe": metrics.get("sharpe"),
                "mdd": metrics.get("max_drawdown"),
                "calmar": metrics.get("calmar"),
            }
        )

    rows.sort(key=lambda row: row["generated"], reverse=True)
    lines = [
        "# 报告索引",
        "",
        f"自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}，共 {len(rows)} 个目录。",
        "由 `scripts/build_report_index.py` 从各目录的 `metrics.json` 读取，未做任何重算。",
        "",
        "| 目录 | 生成时间 | 累计收益 | Sharpe | 最大回撤 | 卡玛 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(value, pattern, scale=1.0):
            return pattern.format(value * scale) if isinstance(value, (int, float)) else "-"

        lines.append(
            f"| `{row['name']}` | {row['generated']} | {fmt(row['return'], '{:+.2f}%', 100)} | "
            f"{fmt(row['sharpe'], '{:.2f}')} | {fmt(row['mdd'], '{:.2f}%', 100)} | {fmt(row['calmar'], '{:.2f}')} |"
        )

    analysis = list_analysis_reports(reports)
    if analysis:
        lines.extend(
            [
                "",
                f"## 多窗口研究目录（{len(analysis)} 个）",
                "",
                "这些目录没有单一 `metrics.json`（一次产出多窗口/多变体表格），"
                "从 `report.md` 看结论：",
                "",
                "| 目录 | 生成时间 |",
                "|---|---|",
            ]
        )
        for row in sorted(analysis, key=lambda item: item["generated"], reverse=True):
            lines.append(f"| `{row['name']}` | {row['generated']} |")

    output = Path(output) if output else reports / "INDEX.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output, len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", default=str(REPORTS))
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    output, count = build(Path(args.reports), Path(args.output) if args.output else None)
    print(f"indexed {count} reports -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
