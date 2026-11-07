"""Capacity curve: how much capital before the book stops being buildable?

``reports/capacity_v2/`` sweeps *costs* on the broad universe. That answers
"how expensive is trading", not "how much money can this strategy carry". The
binding constraint for capacity is the ADV participation cap: when an intended
order exceeds ``max_adv_participation`` of trailing volume it is truncated, and
the book drifts away from its target (lower exposure, wrong names, worse
tracking).

This script runs the identical 18% configuration at several initial cash levels
and reports, for each:

* realised performance (CAGR / Sharpe / drawdown),
* average gross exposure vs the 1M baseline (fill loss),
* half-L1 distance between realised and target weights (execution drift).

Usage::

    python scripts/run_capacity_curve.py
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

from quant_system.backtest import Backtester, slice_result  # noqa: E402
from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, CostConfig, RiskConfig  # noqa: E402
from quant_system.data import align_panel, pit_universe  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import LowVolCoreStrategy  # noqa: E402


PARAMS = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)
WINDOW = ("2024-01-01", "2026-11-05")
CASH_LEVELS = (1e6, 1e7, 5e7, 1e8, 2e8, 5e8, 1e9, 2e9)


def weight_drift(realized: pd.DataFrame, wanted: pd.DataFrame) -> pd.Series:
    """Half-L1 distance between realised and target weights, per day.

    Missing names on either side count as a zero weight, so an unfilled order
    shows up as drift instead of being silently dropped.
    """
    columns = [c for c in wanted.columns if c in realized.columns]
    return (realized[columns] - wanted[columns]).abs().sum(axis=1) / 2.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "capacity_curve"))
    parser.add_argument("--participation", type=float, default=0.20)
    parser.add_argument("--impact", type=float, default=0.0,
                        help="square-root impact coefficient in bps (0 = pure volume cap)")
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _ = align_panel(panel)

    strategy = LowVolCoreStrategy(rebalance_reset="year", **PARAMS)
    target = build_portfolio(
        strategy.generate(aligned), aligned, risk=RiskConfig(), use_regime=False, use_vol_target=True
    )

    rows = []
    for cash in CASH_LEVELS:
        print(f"running cash={cash:,.0f} ...", flush=True)
        config = BacktestConfig(
            initial_cash=cash,
            max_adv_participation=args.participation,
            costs=CostConfig(impact_coefficient_bps=args.impact),
        )
        result = Backtester(config).run(aligned, target)
        sliced = slice_result(result, start=WINDOW[0], end=WINDOW[1])
        realized = sliced.weights.reindex(sliced.equity.index).fillna(0.0)
        wanted = target.reindex(realized.index).fillna(0.0)
        drift = weight_drift(realized, wanted)
        m = sliced.metrics
        rows.append(
            {
                "cash": cash,
                "cagr": float(m.cagr),
                "total_return": float(m.total_return),
                "annual_volatility": float(m.annual_volatility),
                "sharpe": float(m.sharpe),
                "max_drawdown": float(m.max_drawdown),
                "avg_gross": float(realized.sum(axis=1).mean()),
                "target_gross": float(wanted[columns].sum(axis=1).mean()),
                "weight_drift": float(drift.mean()),
                "max_weight_drift": float(drift.max()),
                "turnover": float(m.turnover),
                "impact_bps": args.impact,
            }
        )

    frame = pd.DataFrame(rows)
    base = frame.iloc[0]
    frame["fill_ratio"] = frame["avg_gross"] / base["avg_gross"]
    frame.to_csv(out / "capacity.csv", index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(
        json.dumps(
            {"window": WINDOW, "participation": args.participation,
             "params": PARAMS, "levels": rows},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    lines = ["# 容量曲线（PIT 267，2024-01 .. 2026-11）", ""]
    lines.append(f"ADV 参与上限 {args.participation:.0%}；冲击系数 {args.impact:.0f}bp；「仓位漂移」= 0.5 × L1(实现权重 − 目标权重) 的日均值。")
    lines.append("")
    lines.append("| 资金 | CAGR | 波动 | Sharpe | 最大回撤 | 平均仓位 | 目标仓位 | 仓位漂移 | 换手 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['cash'] / 1e6:,.0f} 百万 | {row['cagr']:+.2%} | {row['annual_volatility']:.2%} | "
            f"{row['sharpe']:.2f} | {row['max_drawdown']:.2%} | {row['avg_gross']:.3f} | "
            f"{row['target_gross']:.3f} | {row['weight_drift']:.3f} | {row['turnover']:.1f}x |"
        )
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(frame.round(4).to_string(index=False))
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
