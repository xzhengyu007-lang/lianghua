"""What are the eight names, really? Selection character across periods.

RESEARCH_LOG 23.80 ruled out breadth as a constraint: the book filled eight slots
on all 91 rebalances and the sector cap never bound. That moves the question
from "can we still pick eight" to "is it the same eight as before".

This measures the picks instead of the picks' returns: where do the selected
names sit in the cross-section of volatility, traded value and price, how many
sectors do they span, and how much of the book is in names that later leave the
pool (23.79). Same source as the strategy -- score() and generate() are called
directly, so there is no second implementation to drift.

Usage::

    python scripts/run_selection_character.py
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
from quant_system.indicators import realised_volatility  # noqa: E402
from quant_system.sectors import sector_of  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402

PERIODS = {
    "2019-2021": ("2019-07-01", "2021-12-31"),
    "2022-2023": ("2022-01-01", "2023-12-31"),
    "2024-2026": ("2024-01-01", "2026-11-06"),
}


def percentile_of(row: pd.Series, values: pd.Series) -> float:
    """Where does the median picked name sit inside the whole tradable pool?"""
    pool = values.dropna()
    if pool.empty or not np.isfinite(row):
        return float("nan")
    return float((pool <= row).mean())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "selection_character"))
    parser.add_argument("--vol-window", type=int, default=60)
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)

    close = pd.DataFrame({s: f["close"] for s, f in aligned.items()})
    volume = pd.DataFrame({s: f["volume"] for s, f in aligned.items()})
    traded = close.notna() & volume.notna() & (volume > 0)
    amount = (close * volume * 100.0).where(traded)
    amount_ma = amount.rolling(20).mean()
    vol = realised_volatility(aligned, args.vol_window).reindex_like(close)

    codes = set(BROAD_UNIVERSE)
    strategy = build_strategy("low_vol_core", rebalance_reset="year")
    weights = strategy.generate(aligned)
    held_matrix = weights.fillna(0.0) > 0
    changed = held_matrix.ne(held_matrix.shift(1)).any(axis=1)
    first = held_matrix.index[held_matrix.any(axis=1)][0]

    rows = []
    for date in [d for d in held_matrix.index[changed] if d >= first]:
        names = list(held_matrix.columns[held_matrix.loc[date]])
        if not names:
            continue
        pool_vol = vol.loc[date].where(traded.loc[date])
        pool_amount = amount_ma.loc[date].where(traded.loc[date])
        rows.append(
            {
                "date": date,
                "n_held": len(names),
                "vol_median": float(vol.loc[date, names].median()),
                "vol_pct_in_pool": percentile_of(
                    -float(vol.loc[date, names].median()), -pool_vol
                ),
                "amount_median_yuan": float(amount_ma.loc[date, names].median()),
                "amount_pct_in_pool": percentile_of(
                    float(amount_ma.loc[date, names].median()), pool_amount
                ),
                "price_median": float(close.loc[date, names].median()),
                "sectors": len({sector_of(n) for n in names}),
                "delisted_cohort_share": float(
                    np.mean([n[-6:] not in codes for n in names])
                ),
            }
        )

    frame = pd.DataFrame(rows)
    frame["period"] = "other"
    for label, (start, end) in PERIODS.items():
        mask = (frame["date"] >= start) & (frame["date"] <= end)
        frame.loc[mask, "period"] = label
    frame = frame[frame["period"] != "other"]
    frame.to_csv(out / "rebalances.csv", index=False, encoding="utf-8-sig")

    grouped = frame.groupby("period").agg(
        rebalances=("date", "count"),
        vol_median=("vol_median", "median"),
        vol_pct_in_pool=("vol_pct_in_pool", "median"),
        amount_median_yuan=("amount_median_yuan", "median"),
        amount_pct_in_pool=("amount_pct_in_pool", "median"),
        price_median=("price_median", "median"),
        sectors=("sectors", "median"),
        delisted_cohort_share=("delisted_cohort_share", "mean"),
    ).reset_index()
    grouped.to_csv(out / "periods.csv", index=False, encoding="utf-8-sig")

    summary = {"periods": grouped.to_dict("records"), "vol_window": args.vol_window}
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )

    lines = ["# 选出来的 8 只是什么东西", ""]
    lines.append("按三段窗口统计每次调仓选中标的的横截面特征。百分位是「被选中的中位标的")
    lines.append("在整个当日可交易池里的位置」，0 表示池子最左端、1 表示最右端。")
    lines.append("")
    lines.append("| 窗口 | 调仓次数 | 选中波动中位 | 波动百分位 | 日成交额中位 | 成交额百分位 | 价格中位 | 覆盖行业 | 退市口径占比 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in grouped.itertuples(index=False):
        lines.append(
            "| %s | %d | %.2f%% | %.0f%% | %.0f 万 | %.0f%% | %.1f | %.0f | %.0f%% |"
            % (
                row.period,
                row.rebalances,
                100 * row.vol_median,
                row.vol_pct_in_pool * 100,
                row.amount_median_yuan / 1e4,
                row.amount_pct_in_pool * 100,
                row.price_median,
                row.sectors,
                100 * row.delisted_cohort_share,
            )
        )
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(grouped.to_string(index=False))
    print("\nreport: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
