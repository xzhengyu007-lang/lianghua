"""Does the ADV participation cap read volume the order is about to consume? (23.17)"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from quant_system.backtest import Backtester
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, RiskConfig
from quant_system.data import load_panel, load_symbol, trim_panel, usable_window
from quant_system.portfolio import build_portfolio
from quant_system.report import write_json
from quant_system.strategies import build_strategy

OUT = ROOT / "reports" / "adv_lookahead"
OUT.mkdir(parents=True, exist_ok=True)
cache = ROOT / "data" / "cache"
panel = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
start, end = usable_window(panel, min_active_ratio=0.95)
panel = {s: f.loc[start:end] for s, f in panel.items()}
benchmark = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
params = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)
raw = build_strategy("low_vol_core", **params).generate(panel)
weights = build_portfolio(raw, panel, benchmark=benchmark, risk=RiskConfig(), use_regime=False)

rows = []
for cash in [1e6, 5e7, 1e8, 3e8, 1e9]:
    for inclusive in (False, True):
        cfg = BacktestConfig(initial_cash=cash, adv_includes_today=inclusive)
        res = Backtester(cfg).run(panel, weights, benchmark=benchmark)
        m = res.metrics.to_dict()
        rows.append({
            "initial_cash": cash,
            "adv_includes_today": inclusive,
            "total_return": round(m["total_return"], 6),
            "cagr": round(m["cagr"], 6),
            "sharpe": round(m["sharpe"], 6),
            "max_drawdown": round(m["max_drawdown"], 6),
            "trades": int(res.metadata["trades"]),
        })
        print(rows[-1], flush=True)

frame = pd.DataFrame(rows)
frame["lookahead_gain_pp"] = frame.groupby("initial_cash")["total_return"].transform(
    lambda s: (s - s.iloc[0]) * 100
)
frame.to_csv(OUT / "adv_lookahead_compare.csv", index=False, encoding="utf-8-sig")
write_json(
    {
        "results": rows,
        "note": (
            "adv_includes_today=True reproduces the pre-fix capacity model, which sized "
            "open orders with the volume of the bar being traded. lookahead_gain_pp is "
            "the total-return inflation relative to the causal (lagged) model."
        ),
    },
    OUT / "adv_lookahead.json",
)
print(frame.to_string(index=False))
