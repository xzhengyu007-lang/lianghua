"""Calendar-year breakdown of the published out-of-sample record.

Readers ask "how did it do each year", and the three-window table answers a
different question. This script stitches the three published walk-forward
reports into one daily return series (they cover disjoint, adjacent windows)
and reports per calendar year: return, volatility, Sharpe, drawdown, hit rate.

The stitched series is *not* a single continuous backtest: each report restarts
its book at the window boundary, so a year that straddles a boundary inherits
that restart. Only 2021-12 -> 2022-01 and 2023-12 -> 2024-01 fall on boundaries,
and both are treated as ordinary month ends in the table (the effect is a few
days of re-entry cost, not a regime change).

Usage::

    python scripts/run_year_table.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.equity_io import daily_returns_from_equity  # noqa: E402


REPORTS = [
    "reports/wf_pit267_2019_2021",
    "reports/wf_pit267_2022_trend120",
    "reports/wf_pit_full267_year",
]


def year_stats(returns: pd.Series) -> dict:
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    vol = float(returns.std(ddof=0) * np.sqrt(252))
    sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(252)) if returns.std(ddof=0) else float("nan")
    return {
        "return": float(equity.iloc[-1] - 1.0),
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((returns > 0).mean()),
        "best_day": float(returns.max()),
        "worst_day": float(returns.min()),
        "days": int(len(returns)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "reports" / "year_table"))
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    series = {}
    for report in REPORTS:
        path = ROOT / report
        if not (path / "equity.csv").exists():
            print(f"skip missing {report}")
            continue
        series[report] = daily_returns_from_equity(path)
    if not series:
        print("no reports found")
        return 1

    combined = pd.concat(series.values()).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    rows = []
    for year, block in combined.groupby(combined.index.year):
        stats = year_stats(block)
        stats["year"] = int(year)
        rows.append(stats)

    table = pd.DataFrame(rows).set_index("year")
    table.to_csv(out / "years.csv", encoding="utf-8-sig")
    combined.to_csv(out / "daily_returns.csv", encoding="utf-8-sig")
    (out / "summary.json").write_text(
        json.dumps(
            {"reports": REPORTS, "start": str(combined.index.min().date()),
             "end": str(combined.index.max().date()), "years": rows},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    lines = ["# 年度分解", "",
             f"拼接三段已发布的样本外报告（{combined.index.min().date()} .. {combined.index.max().date()}），"
             "按自然年统计。", "",
             "| 年份 | 收益 | 波动 | Sharpe | 最大回撤 | 日胜率 | 最好单日 | 最差单日 | 天数 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for year, row in table.iterrows():
        lines.append(
            f"| {year} | {row['return']:+.2%} | {row['volatility']:.2%} | {row['sharpe']:.2f} | "
            f"{row['max_drawdown']:.2%} | {row['hit_rate']:.1%} | {row['best_day']:+.2%} | "
            f"{row['worst_day']:+.2%} | {row['days']} |"
        )
    lines.append("")
    lines.append("注意：拼接点（2021-12→2022-01、2023-12→2024-01）两侧各自重启账本，"
                 "那一两天的再建仓成本属于协议差异，不是策略行为。")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(table.round(4).to_string())
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
