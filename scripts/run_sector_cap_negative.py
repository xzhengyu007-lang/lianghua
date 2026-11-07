"""Sector-cap alternatives, kept as a negative-result record.

The low-vol book's return is concentrated in financials and energy. A natural
fix is to hold one name per sector and widen the book. The full-sample results
reject that: every cap=1 variant is far below cap=2 and has a deeper drawdown.
This script writes the table so the rejected direction is reproducible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.backtest import Backtester  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, RiskConfig  # noqa: E402
from quant_system.data import load_panel, load_symbol, trim_panel, usable_window  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.report import write_json  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402

CASES = [(6, 1), (8, 1), (12, 1), (16, 1), (20, 1), (10, 2), (12, 2), (16, 2)]
OUT = ROOT / "reports" / "sector_cap_negative"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "data" / "cache"
    panel = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    start, end = usable_window(panel, min_active_ratio=0.95)
    panel = {symbol: frame.loc[start:end] for symbol, frame in panel.items()}
    benchmark = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
    rows = []
    for top_n, cap in CASES:
        params = dict(
            top_n=top_n, max_per_sector=cap, vol_window=60, trend_window=120, rebalance_days=20
        )
        raw = build_strategy("low_vol_core", **params).generate(panel)
        weights = build_portfolio(
            raw, panel, benchmark=benchmark, risk=RiskConfig(), use_regime=False
        )
        result = Backtester(BacktestConfig(initial_cash=1_000_000)).run(
            panel, weights, benchmark=benchmark
        )
        metrics = result.metrics.to_dict()
        rows.append(
            {
                "top_n": top_n,
                "max_per_sector": cap,
                "total_return": metrics["total_return"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "exposure": metrics["exposure"],
            }
        )
        print(
            f"top_n={top_n:2d} cap={cap}: {metrics['total_return']:+.2%} "
            f"sharpe={metrics['sharpe']:.3f} mdd={metrics['max_drawdown']:.2%}",
            flush=True,
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "cases.csv", index=False, encoding="utf-8-sig")
    write_json(
        {
            "window": [str(start.date()), str(end.date())],
            "rows": rows,
            "note": (
                "Negative result: cap=1 forces the low-vol book into higher-vol "
                "sectors and is worse on return, Sharpe and drawdown."
            ),
        },
        OUT / "summary.json",
    )
    print(f"report: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
