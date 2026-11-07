"""Re-check the calendar-anchor claim under the current (rebuilt) data.

RESEARCH_LOG 23.21 found the 20-day rebalance phase alone moved the same strategy
from +47.5% to +166.5%, and 23.23 proposed re-anchoring the cadence every January
as the fix. Both were measured before the adjustment rebuild and before "year"
became the default. Those two changes are exactly the kind that invalidate a
calibration, so the claim gets re-measured rather than inherited.

For each anchor mode and each of the 20 possible phases, run the full history once
and slice the three independent windows. The question is narrow: does the yearly
re-anchor still shrink the phase spread, and by how much, on the data we ship.

Usage::

    python scripts/run_phase_recheck.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.backtest import Backtester, slice_result  # noqa: E402
from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, RiskConfig  # noqa: E402
from quant_system.data import align_panel, pit_universe  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402

WINDOWS = {
    "2019-2021": ("2019-07-01", "2021-12-31"),
    "2022-2023": ("2022-01-01", "2023-12-31"),
    "2024-2026": ("2024-01-01", "2026-11-06"),
}
MODES = ("start", "year")
PHASES = tuple(range(20))


def run_once(aligned, reset: str, offset: int):
    strategy = build_strategy(
        "low_vol_core",
        top_n=8,
        vol_window=60,
        trend_window=120,
        max_per_sector=2,
        rebalance_days=20,
        rebalance_reset=reset,
        rebalance_offset=offset,
    )
    weights = build_portfolio(
        strategy.generate(aligned),
        aligned,
        risk=RiskConfig(),
        use_regime=False,
        use_vol_target=True,
    )
    return Backtester(BacktestConfig()).run(aligned, weights)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "phase_recheck"))
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)
    print("panel: %d symbols" % len(aligned), flush=True)

    rows = []
    for reset in MODES:
        for offset in PHASES:
            result = run_once(aligned, reset, offset)
            for window, (start, end) in WINDOWS.items():
                sliced = slice_result(result, start=start, end=end)
                metrics = sliced.metrics
                rows.append(
                    {
                        "mode": reset,
                        "offset": offset,
                        "window": window,
                        "total_return": float(metrics.total_return),
                        "sharpe": float(metrics.sharpe),
                        "max_drawdown": float(metrics.max_drawdown),
                    }
                )
            print("  %s offset=%d done" % (reset, offset), flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "phases.csv", index=False, encoding="utf-8-sig")

    agg = frame.groupby(["window", "mode"]).agg(
        ret_median=("total_return", "median"),
        ret_min=("total_return", "min"),
        ret_max=("total_return", "max"),
        sharpe_median=("sharpe", "median"),
        sharpe_min=("sharpe", "min"),
        sharpe_max=("sharpe", "max"),
        mdd_min=("max_drawdown", "min"),
        mdd_max=("max_drawdown", "max"),
    ).reset_index()
    agg["ret_spread_pp"] = (agg["ret_max"] - agg["ret_min"]) * 100
    agg["sharpe_spread"] = agg["sharpe_max"] - agg["sharpe_min"]
    agg["mdd_spread_pp"] = (agg["mdd_max"] - agg["mdd_min"]) * 100
    agg.to_csv(out / "spread.csv", index=False, encoding="utf-8-sig")

    summary = {"windows": {k: list(v) for k, v in WINDOWS.items()}, "phases": list(PHASES), "spread": agg.to_dict("records")}
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )

    lines = ["# 调仓相位复核（重建口径 + 当前默认配置）", ""]
    lines.append("同一策略、同一数据、同一窗口，只改 20 日调仓日历的起始相位（0–19），")
    lines.append("两种锚点各跑一遍。§23.21/§23.23 的原始结论是在加法复权口径下得到的，")
    lines.append("这里在重建口径上重测。")
    lines.append("")
    lines.append("| 窗口 | 锚点 | 收益中位 | 最差 | 最好 | 收益跨度 | Sharpe 中位 | Sharpe 跨度 | 回撤跨度 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in agg.itertuples(index=False):
        lines.append(
            "| %s | %s | %+.2f%% | %+.2f%% | %+.2f%% | %.1fpp | %.3f | %.3f | %.1fpp |"
            % (
                row.window,
                row.mode,
                row.ret_median * 100,
                row.ret_min * 100,
                row.ret_max * 100,
                row.ret_spread_pp,
                row.sharpe_median,
                row.sharpe_spread,
                row.mdd_spread_pp,
            )
        )
    lines.append("")
    lines.append("## 年度重置把跨度收缩了多少")
    lines.append("")
    lines.append("| 窗口 | 收益跨度 start | 收益跨度 year | 收缩 | Sharpe 跨度 start | Sharpe 跨度 year |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for window in WINDOWS:
        sub = agg[agg["window"] == window].set_index("mode")
        if "start" not in sub.index or "year" not in sub.index:
            continue
        s = sub.loc["start"]
        y = sub.loc["year"]
        shrink = (1 - y["ret_spread_pp"] / s["ret_spread_pp"]) * 100 if s["ret_spread_pp"] else float("nan")
        lines.append(
            "| %s | %.1fpp | %.1fpp | %.0f%% | %.3f | %.3f |"
            % (window, s["ret_spread_pp"], y["ret_spread_pp"], shrink, s["sharpe_spread"], y["sharpe_spread"])
        )
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + agg[["window", "mode", "ret_median", "ret_spread_pp", "sharpe_median", "sharpe_spread"]].to_string(index=False))
    print("\nreport: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
