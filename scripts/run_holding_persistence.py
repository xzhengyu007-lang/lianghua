"""Holding persistence: is the low-vol book a "core" or a monthly rotation?

Every walk-forward report writes ``positions.csv`` with daily share counts. That
is enough to measure, causally and without refitting anything:

* how many names are actually held each day (the target is 8, the cap is 8),
* how often the holding set changes and how much of the previous basket survives
  the change (retention),
* how long a name stays in the book (holding spells, in trading days),
* how many distinct names the book touches over the window.

The point is not to judge the strategy but to nail down what it is: a 60-day
volatility estimate with a 20-day rebalance that keeps ~60% of its names is a
monthly rotation with a persistence halo, not a buy-and-hold core. That matters
for cost drag, for the "long-term holding" narrative, and for capacity.

Usage::

    python scripts/run_holding_persistence.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

REPORTS = {
    "2019-07..2021-12": ROOT / "reports" / "attrib_1921",
    "2022-01..2023-12": ROOT / "reports" / "attrib_2022",
    "2024-01..2026-11": ROOT / "reports" / "attrib_2024",
    "2025 standalone": ROOT / "reports" / "attrib_2025",
}


def load_holdings(report_dir: Path) -> pd.DataFrame:
    """Daily boolean holdings matrix (rows = dates, columns = symbols)."""
    frame = pd.read_csv(report_dir / "positions.csv", parse_dates=["date"])
    matrix = frame.pivot_table(
        index="date", columns="symbol", values="shares", aggfunc="sum"
    )
    matrix = matrix.sort_index()
    return (matrix.fillna(0.0) > 0)


def holding_sets(flags: pd.DataFrame) -> pd.Series:
    return flags.apply(lambda row: frozenset(row.index[row.to_numpy()]), axis=1)


def rebalance_table(sets: pd.Series) -> pd.DataFrame:
    """One row per day the holding set changes, with retention vs the prior set."""
    rows = []
    previous: frozenset | None = None
    previous_day = None
    for day, current in sets.items():
        if previous is None:
            previous, previous_day = current, day
            rows.append({"date": day, "names": len(current), "retention": np.nan,
                         "added": len(current), "removed": 0,
                         "gap_days": np.nan, "initial": True})
            continue
        if current == previous:
            continue
        base = len(previous) or 1
        rows.append({
            "date": day,
            "names": len(current),
            "retention": len(current & previous) / base,
            "added": len(current - previous),
            "removed": len(previous - current),
            "gap_days": int(np.busday_count(previous_day.date(), day.date())),
            "initial": False,
        })
        previous, previous_day = current, day
    return pd.DataFrame(rows)


def holding_spells(flags: pd.DataFrame) -> pd.DataFrame:
    """Contiguous runs of a symbol being held, in trading days."""
    rows = []
    dates = flags.index
    for symbol in flags.columns:
        values = flags[symbol].to_numpy()
        start = None
        for i, held in enumerate(values):
            if held and start is None:
                start = i
            elif not held and start is not None:
                rows.append({"symbol": symbol, "start": dates[start], "end": dates[i - 1],
                             "days": i - start})
                start = None
        if start is not None:
            rows.append({"symbol": symbol, "start": dates[start], "end": dates[-1],
                         "days": len(values) - start})
    return pd.DataFrame(rows)


def summarise(label: str, flags: pd.DataFrame) -> dict:
    sets = holding_sets(flags)
    sizes = sets.apply(len)
    table = rebalance_table(sets)
    spells = holding_spells(flags)
    retention = table["retention"].dropna()
    return {
        "label": label,
        "days": int(len(sets)),
        "avg_names": float(sizes.mean()),
        "min_names": int(sizes.min()),
        "max_names": int(sizes.max()),
        "changes": int((~table["initial"]).sum()),
        "mean_retention": float(retention.mean()) if len(retention) else float("nan"),
        "median_retention": float(retention.median()) if len(retention) else float("nan"),
        "spells": int(len(spells)),
        "mean_spell_days": float(spells["days"].mean()) if len(spells) else float("nan"),
        "median_spell_days": float(spells["days"].median()) if len(spells) else float("nan"),
        "p90_spell_days": float(spells["days"].quantile(0.9)) if len(spells) else float("nan"),
        "max_spell_days": int(spells["days"].max()) if len(spells) else 0,
        "share_spells_ge_250d": float((spells["days"] >= 250).mean()) if len(spells) else float("nan"),
        "distinct_names": int(flags.any(axis=0).sum()),
        "rebalances_per_year": float((~table["initial"]).sum() / max(len(sets) / 243.0, 1e-9)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "reports" / "holding_persistence"))
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    summaries = []
    for label, report_dir in REPORTS.items():
        flags = load_holdings(report_dir)
        sets = holding_sets(flags)
        table = rebalance_table(sets)
        spells = holding_spells(flags)
        summary = summarise(label, flags)
        summaries.append(summary)
        slug = label.replace(" ", "_").replace("..", "_").replace(".", "")
        table.to_csv(out / f"rebalances_{slug}.csv", index=False, encoding="utf-8-sig")
        spells.to_csv(out / f"spells_{slug}.csv", index=False, encoding="utf-8-sig")
        print(f"{label}: {summary['changes']} changes, retention "
              f"{summary['mean_retention']:.3f}, median spell "
              f"{summary['median_spell_days']:.0f} days", flush=True)

    frame = pd.DataFrame(summaries)
    frame.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = ["# 持仓存续率（8 只等权核心）", ""]
    lines.append("| 窗口 | 交易日 | 平均持仓 | 变动次数 | 平均留存 | 中位留存 | 中位持有时长 | 平均持有时长 | p90 时长 | 最长 | ≥250 日 | 涉及个股 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summaries:
        lines.append(
            f"| {row['label']} | {row['days']} | {row['avg_names']:.2f} | {row['changes']} | "
            f"{row['mean_retention']:.3f} | {row['median_retention']:.3f} | "
            f"{row['median_spell_days']:.0f} | {row['mean_spell_days']:.0f} | "
            f"{row['p90_spell_days']:.0f} | {row['max_spell_days']} | "
            f"{row['share_spells_ge_250d']:.0%} | {row['distinct_names']} |"
        )
    lines.append("")
    lines.append("留存 = 新一期持仓中与上一期重合的名字数 / 上一期持仓数；变动次数 = 持仓集合发生变化的天数；")
    lines.append("持有时长 = 某只股票连续在组合中的交易日数（含变动当日）。")
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(frame.round(3).to_string(index=False))
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
