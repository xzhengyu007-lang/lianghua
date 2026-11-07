"""Concentration study: how much of the headline comes from a few names?

Drops the top-N realised-PnL contributors and reruns the full strategy. If the
edge is structural, removing a handful of winners should not collapse it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.backtest import Backtester  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, RiskConfig  # noqa: E402
from quant_system.data import (  # noqa: E402
    load_panel,
    load_symbol,
    trim_panel,
    usable_window,
)
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402

PARAMS = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)
DROPS = [1, 3, 5, 10]


def run(panel, benchmark):
    raw = build_strategy("low_vol_core", **PARAMS).generate(panel)
    weights = build_portfolio(
        raw, panel, benchmark=benchmark, risk=RiskConfig(), use_regime=False
    )
    return Backtester(BacktestConfig(initial_cash=1_000_000)).run(
        panel, weights, benchmark=benchmark
    )


def main() -> int:
    cache = ROOT / "data" / "cache"
    panel = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    start, end = usable_window(panel, min_active_ratio=0.95)
    panel = {symbol: frame.loc[start:end] for symbol, frame in panel.items()}
    benchmark = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]

    base = run(panel, benchmark)
    trades = base.trades
    realised = (
        trades[trades["side"] == "sell"]
        .groupby("symbol")["realized_pnl"]
        .sum()
        .sort_values(ascending=False)
    )
    total = float(realised.sum())

    rows = [
        {
            "case": "baseline",
            "total_return": base.metrics.total_return,
            "sharpe": base.metrics.sharpe,
            "max_drawdown": base.metrics.max_drawdown,
        }
    ]
    for n in DROPS:
        dropped = list(realised.head(n).index)
        subset = {s: f for s, f in panel.items() if s not in dropped}
        result = run(subset, benchmark)
        rows.append(
            {
                "case": f"drop_top_{n}",
                "total_return": result.metrics.total_return,
                "sharpe": result.metrics.sharpe,
                "max_drawdown": result.metrics.max_drawdown,
            }
        )
        print(
            f"drop top {n:2d}: total={result.metrics.total_return:.3%} "
            f"sharpe={result.metrics.sharpe:.3f}",
            flush=True,
        )

    out = ROOT / "reports" / "concentration"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "cases.csv", index=False, encoding="utf-8-sig")
    realised.rename("realized_pnl").to_csv(out / "realized_by_symbol.csv", encoding="utf-8-sig")
    summary = {
        "window": [str(start.date()), str(end.date())],
        "symbols": len(panel),
        "params": PARAMS,
        "total_realised_pnl": total,
        "top10_share_of_realised_pnl": float(realised.head(10).sum() / total) if total else None,
        "cases": rows,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"top10 share: {summary['top10_share_of_realised_pnl']:.1%}")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
