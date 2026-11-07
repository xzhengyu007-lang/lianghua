"""Verify the paper ledger reconciles with the backtester.

Runs a small backtest, replays every fill into the ledger, and checks that the
ledger's final cash plus mark-to-market equals the backtest equity within
rounding. This is the property that makes the daily ledger trustworthy: the
research engine and the operating loop must agree on the same trades.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.backtest import Backtester  # noqa: E402
from quant_system.config import BacktestConfig, CostConfig, RiskConfig  # noqa: E402
from quant_system.data import load_panel, load_symbol  # noqa: E402
from quant_system.paper import Ledger, apply_fills, summarize  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402


def main() -> int:
    cache = ROOT / "data" / "cache"
    symbols = ["sh600036", "sh600887", "sh600900", "sh601088", "sh601668", "sh601939"]
    panel = {s: f for s, f in load_panel(symbols, cache).items()}
    bench = load_symbol("sh000300", cache, adjust="")["close"]
    raw = build_strategy("low_vol_core", top_n=4, vol_window=60, trend_window=120,
                         max_per_sector=2, rebalance_days=40).generate(panel)
    weights = build_portfolio(raw, panel, benchmark=bench, risk=RiskConfig(), use_regime=False)
    costs = CostConfig()
    result = Backtester(
        BacktestConfig(initial_cash=1_000_000, costs=costs, max_adv_participation=1.0)
    ).run(panel, weights, benchmark=bench)

    trades = result.trades.sort_values(["date", "side"], ascending=[True, False])
    print(f"trades to replay: {len(trades)}")
    ledger = Ledger(cash=1_000_000.0)
    for date, group in trades.groupby("date"):
        fills = group[["symbol", "side", "shares", "price"]].copy()
        ledger = apply_fills(ledger, fills, date=date, costs=costs)

    last = result.equity.index[-1]
    prices = pd.Series(
        {s: float(panel[s]["close"].loc[:last].ffill().iloc[-1]) for s in panel}
    )
    summary = summarize(ledger, prices)
    expected = float(result.equity.iloc[-1])
    print(f"backtest equity : {expected:,.2f}")
    print(f"ledger equity   : {summary['equity']:,.2f}")
    print(f"difference      : {summary['equity'] - expected:,.2f}")
    print(f"backtest trades : {len(result.trades)}  ledger trades: {summary['trades']}")
    print(f"backtest fees   : {result.trades['fees'].sum():,.2f}")
    ledger_fees = sum(float(t["commission"]) + float(t["stamp_duty"]) + float(t["transfer_fee"]) for t in ledger.trades)
    print(f"ledger fees     : {ledger_fees:,.2f}")
    ok = abs(summary["equity"] - expected) < max(50.0, expected * 1e-5)
    print("RECONCILES" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
