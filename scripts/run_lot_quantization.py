"""How much of the capacity conclusion is only 100-share lot rounding? (23.17)"""
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

OUT = ROOT / "reports" / "lot_quantization"
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
for cash in [1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8]:
    cfg = BacktestConfig(initial_cash=cash)
    res = Backtester(cfg).run(panel, weights, benchmark=benchmark)
    m = res.metrics.to_dict()
    rows.append({
        "initial_cash": cash,
        "total_return": round(m["total_return"], 6),
        "cagr": round(m["cagr"], 6),
        "sharpe": round(m["sharpe"], 6),
        "max_drawdown": round(m["max_drawdown"], 6),
        "exposure": round(m["exposure"], 6),
        "trades": int(res.metadata["trades"]),
    })
    print(rows[-1], flush=True)

frame = pd.DataFrame(rows)
frame.to_csv(OUT / "by_capital.csv", index=False, encoding="utf-8-sig")
write_json({"results": rows, "note": "100-share lot quantization only; ADV cap active but not binding below 50m"}, OUT / "lot_quantization.json")
print()
print(frame.to_string(index=False))
