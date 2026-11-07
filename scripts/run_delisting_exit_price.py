from __future__ import annotations

import json
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

OUT = ROOT / "reports" / "delisting_exit_price"
OUT.mkdir(parents=True, exist_ok=True)
cache = ROOT / "data" / "cache"
survivors = list(BROAD_UNIVERSE)
survivor_panel = trim_panel(load_panel(survivors, cache), min_coverage=0.95)
start, end = usable_window(survivor_panel, min_active_ratio=0.95)
benchmark = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
manifest = json.loads((ROOT / "data" / "delisted_manifest.json").read_text(encoding="utf-8"))
candidates = [
    s for s, info in manifest.items()
    if pd.Timestamp(info["last_bar"]) >= pd.Timestamp(start)
    and pd.Timestamp(info["first_bar"]) <= pd.Timestamp(end)
]
extra = load_panel(candidates, cache)
extra = {s: f for s, f in extra.items() if len(f) >= 120}
pit_panel = {
    s: f.loc[start:end] for s, f in {**survivor_panel, **extra}.items()
}
print(f"window {start.date()}..{end.date()} pit={len(pit_panel)} delisted={len(extra)}", flush=True)

params = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)
raw = build_strategy("low_vol_core", **params).generate(pit_panel)
weights = build_portfolio(raw, pit_panel, benchmark=benchmark, risk=RiskConfig(), use_regime=False)
res = Backtester(BacktestConfig(initial_cash=1_000_000.0)).run(pit_panel, weights, benchmark=benchmark)
m = res.metrics.to_dict()
tr = res.trades[res.trades["reason"] == "delisting"]
rows = []
for _, t in tr.iterrows():
    s, d = t["symbol"], t["date"]
    frame = pit_panel[s]
    o = float(frame.loc[d, "open"])
    c = float(frame.loc[d, "close"])
    rows.append({
        "date": str(d.date()),
        "symbol": s,
        "shares": int(t["shares"]),
        "exec_price_open": round(float(t["price"]), 4),
        "open": round(o, 4),
        "close": round(c, 4),
        "close_vs_open_pct": round((c / o - 1) * 100, 3) if o else None,
        "notional": round(float(t["notional"]), 2),
    })
frame = pd.DataFrame(rows)
frame.to_csv(OUT / "delisting_exits.csv", index=False, encoding="utf-8-sig")
print(frame.to_string(index=False) if len(frame) else "no delisting exits")
summary = {
    "window": [str(start.date()), str(end.date())],
    "symbols": len(pit_panel),
    "metrics": {k: round(float(v), 6) for k, v in m.items()},
    "delisting_exits": int(len(frame)),
    "exit_day_close_vs_open_mean_pct": round(float(frame["close_vs_open_pct"].mean()), 4) if len(frame) else None,
    "exit_notional": round(float(frame["notional"].sum()), 2) if len(frame) else 0.0,
    "note": "exits fill at the final bar's open; the pre-fix model filled at that bar's close",
}
write_json(summary, OUT / "delisting_exit_price.json")
print()
print(summary)
