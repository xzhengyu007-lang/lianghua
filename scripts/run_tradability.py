"""Tradability: how often is a name we want (or hold) actually tradeable?

Every execution assumption in this project rests on "the bar exists and volume
is positive". This script measures that assumption instead of asserting it:

* breadth: how many of the 267 PIT names have a real bar on each day, by year;
* thin days: share of days where less than 90% of the pool traded;
* holdings: share of held name-days with no bar, i.e. a position we could not
  have adjusted that day;
* episodes: the longest unbroken suspension hit by a name while it was in the
  book.

A suspended holding is not a modelling error -- the backtester marks it at the
last close and resumes when the bar returns -- but it does mean the strategy
cannot rebalance on the calendar it advertises for those names, and that a
"20-day" holding period is sometimes longer.

Usage::

    python scripts/run_tradability.py
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

from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE  # noqa: E402
from quant_system.data import align_panel, pit_universe  # noqa: E402

REPORTS = {
    "2019-07..2021-12": ROOT / "reports" / "attrib_1921",
    "2022-01..2023-12": ROOT / "reports" / "attrib_2022",
    "2024-01..2026-11": ROOT / "reports" / "attrib_2024",
    "2025 段": ROOT / "reports" / "attrib_2025_year",
}


def load_tradable(cache_dir: Path) -> pd.DataFrame:
    panel = _rebuilt_panel(list(pit_universe()), cache_dir, "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _ = align_panel(panel)
    close = pd.DataFrame({s: f["close"] for s, f in aligned.items()})
    volume = pd.DataFrame({s: f["volume"] for s, f in aligned.items()})
    return close.notna() & volume.notna() & (volume > 0)


def breadth_table(tradable: pd.DataFrame) -> pd.DataFrame:
    counts = tradable.sum(axis=1)
    rows = []
    for year, group in counts.groupby(counts.index.year):
        rows.append({
            "year": int(year),
            "days": int(len(group)),
            "median_tradable": float(group.median()),
            "min_tradable": int(group.min()),
            "share_days_below_90pct": float((group < 0.9 * tradable.shape[1]).mean()),
        })
    return pd.DataFrame(rows)


def holdings_exposure(report_dir: Path, tradable: pd.DataFrame) -> dict:
    frame = pd.read_csv(report_dir / "positions.csv", parse_dates=["date"])
    held = frame[frame["shares"] > 0]
    if not len(held):
        return {"label": report_dir.name, "held_days": 0}
    bars = tradable.reindex(index=held["date"].unique(), columns=sorted(tradable.columns))
    values = bars.to_numpy()
    index = {sym: j for j, sym in enumerate(bars.columns)}
    dates = {d: i for i, d in enumerate(bars.index)}
    missing = 0
    for row in held.itertuples(index=False):
        j = index.get(row.symbol)
        i = dates.get(row.date)
        if j is None or i is None:
            continue
        if not values[i, j]:
            missing += 1
    return {
        "label": report_dir.name,
        "held_name_days": int(len(held)),
        "suspended_name_days": int(missing),
        "suspended_share": float(missing / max(len(held), 1)),
    }


def longest_holding_suspensions(report_dir: Path, tradable: pd.DataFrame, top: int = 5) -> pd.DataFrame:
    frame = pd.read_csv(report_dir / "positions.csv", parse_dates=["date"])
    held = frame[frame["shares"] > 0]
    rows = []
    for symbol, group in held.groupby("symbol"):
        days = group["date"].sort_values().to_numpy()
        column = tradable.get(symbol)
        if column is None:
            continue
        flags = column.reindex(pd.DatetimeIndex(days)).fillna(False).to_numpy()
        run = 0
        best = 0
        for ok in flags:
            if not ok:
                run += 1
                best = max(best, run)
            else:
                run = 0
        if best:
            rows.append({"symbol": symbol, "longest_suspension_days_held": int(best)})
    if not rows:
        return pd.DataFrame(columns=["symbol", "longest_suspension_days_held"])
    return pd.DataFrame(rows).sort_values("longest_suspension_days_held", ascending=False).head(top)


def frozen_exposure(report_dir: Path, tradable: pd.DataFrame) -> dict:
    """Weight actually parked in names that had no bar that day.

    A suspended holding is marked at its last close, so it contributes zero
    variance for as long as the suspension lasts. That is a defensible mark, but
    it means the book is not the "fully invested 8 names" its weights claim: on
    those days the strategy holds a riskless sleeve it cannot trade out of. This
    measures the size of that sleeve instead of assuming it away.
    """
    frame = pd.read_csv(report_dir / "positions.csv", parse_dates=["date"])
    held = frame[frame["shares"] > 0]
    if not len(held):
        return {"label": report_dir.name, "days": 0, "share_days_frozen": 0.0}

    bars = tradable.reindex(
        index=sorted(held["date"].unique()), columns=sorted(tradable.columns)
    ).fillna(False)
    values = bars.to_numpy()
    column = {sym: j for j, sym in enumerate(bars.columns)}
    row = {d: i for i, d in enumerate(bars.index)}

    tradable_today = np.fromiter(
        (
            False if column.get(sym) is None or row.get(day) is None
            else bool(values[row[day], column[sym]])
            for sym, day in zip(held["symbol"], held["date"])
        ),
        dtype=bool,
        count=len(held),
    )
    frozen = ~tradable_today
    weights = held["weight"].fillna(0.0).to_numpy()
    codes, days = pd.factorize(held["date"], sort=True)
    n_days = len(days)
    frozen_names = np.bincount(codes, weights=frozen.astype(float), minlength=n_days)
    frozen_weights = np.bincount(
        codes, weights=np.where(frozen, weights, 0.0), minlength=n_days
    )
    return {
        "label": report_dir.name,
        "days": int(n_days),
        "days_with_frozen": int((frozen_names > 0).sum()),
        "share_days_frozen": float((frozen_names > 0).mean()),
        "mean_frozen_weight": float(frozen_weights.mean()),
        "p95_frozen_weight": float(np.quantile(frozen_weights, 0.95)),
        "max_frozen_weight": float(frozen_weights.max()),
        "frozen_name_days": int(frozen.sum()),
        "frozen_share": float(frozen.mean()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "tradability"))
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    tradable = load_tradable(Path(args.cache_dir))
    breadth = breadth_table(tradable)
    breadth.to_csv(out / "breadth.csv", index=False, encoding="utf-8-sig")

    exposures = []
    episodes = []
    frozen_rows = []
    for label, report_dir in REPORTS.items():
        if not (report_dir / "positions.csv").exists():
            print(f"skip missing {report_dir}")
            continue
        row = holdings_exposure(report_dir, tradable)
        row["label"] = label
        exposures.append(row)
        top = longest_holding_suspensions(report_dir, tradable)
        top.insert(0, "window", label)
        episodes.append(top)
        frozen = frozen_exposure(report_dir, tradable)
        frozen["label"] = label
        frozen_rows.append(frozen)

    exposure_frame = pd.DataFrame(exposures)
    exposure_frame.to_csv(out / "holdings.csv", index=False, encoding="utf-8-sig")

    frozen_frame = pd.DataFrame(frozen_rows)
    if len(frozen_frame):
        frozen_frame.to_csv(out / "frozen.csv", index=False, encoding="utf-8-sig")
    episode_frame = pd.concat(episodes, ignore_index=True) if episodes else pd.DataFrame()
    episode_frame.to_csv(out / "episodes.csv", index=False, encoding="utf-8-sig")

    summary = {
        "symbols": int(tradable.shape[1]),
        "days": int(tradable.shape[0]),
        "start": str(tradable.index.min().date()),
        "end": str(tradable.index.max().date()),
        "overall_tradable_share": float(tradable.to_numpy().mean()),
        "breadth": breadth.to_dict("records"),
        "holdings": exposures,
        "episodes": episode_frame.to_dict("records"),
        "frozen": frozen_rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 可交易性（PIT 267）", ""]
    lines.append(f"面板 {summary['start']} .. {summary['end']}，{summary['symbols']} 只，"
                 f"全样本可交易占比 {summary['overall_tradable_share']:.2%}。")
    lines.append("")
    lines.append("## 每日可交易只数（按年）")
    lines.append("")
    lines.append("| 年份 | 交易日 | 中位可交易 | 最少 | <90% 池可交易的天数占比 |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in summary["breadth"]:
        lines.append(f"| {row['year']} | {row['days']} | {row['median_tradable']:.0f} | "
                     f"{row['min_tradable']} | {row['share_days_below_90pct']:.1%} |")
    lines.append("")
    lines.append("## 持仓遇到停牌")
    lines.append("")
    lines.append("| 窗口 | 持仓股-日 | 停牌股-日 | 占比 |")
    lines.append("|---|---:|---:|---:|")
    for row in exposures:
        lines.append(f"| {row['label']} | {row['held_name_days']} | {row['suspended_name_days']} | "
                     f"{row['suspended_share']:.2%} |")
    lines.append("")
    if len(frozen_frame):
        lines.append("## 冻结槽位（持仓当日无 K 线，按最后收盘计价）")
        lines.append("")
        lines.append("| 窗口 | 交易日 | 有冻结仓位的天数 | 占比 | 平均冻结权重 | p95 | 最高 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in frozen_rows:
            lines.append(
                f"| {row['label']} | {row['days']} | {row['days_with_frozen']} | "
                f"{row['share_days_frozen']:.1%} | {row['mean_frozen_weight']:.2%} | "
                f"{row['p95_frozen_weight']:.2%} | {row['max_frozen_weight']:.2%} |"
            )
        lines.append("")
    if len(episode_frame):
        lines.append("## 持仓期最长的停牌片段")
        lines.append("")
        lines.append("| 窗口 | 代码 | 持仓期间停牌交易日 |")
        lines.append("|---|---|---:|")
        for row in episode_frame.itertuples(index=False):
            lines.append(f"| {row.window} | {row.symbol} | {row.longest_suspension_days_held} |")
        lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(breadth.to_string(index=False))
    print(exposure_frame.to_string(index=False))
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
