from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from quant_system.backtest import Backtester
from quant_system.config import BacktestConfig, RiskConfig
from quant_system.data import load_panel, load_symbol, trim_panel, usable_window
from quant_system.portfolio import build_portfolio
from quant_system.report import write_json
from quant_system.strategies import build_strategy

OUT = ROOT/"reports"/"adjustment_audit"
OUT.mkdir(parents=True, exist_ok=True)
cache = ROOT/"data"/"cache"
raw_dir = ROOT/"data"/"cache_rawcheck"
symbols = sorted(p.name.replace("_raw.parquet","") for p in raw_dir.glob("*.parquet"))
print("symbols with raw:", len(symbols), flush=True)

qfq_panel = trim_panel(load_panel(symbols, cache), min_coverage=0.95)
start, end = usable_window(qfq_panel, min_active_ratio=0.95)
benchmark = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]

def load_raw(symbols, start, end):
    out = {}
    for s in symbols:
        p = raw_dir/f"{s}_raw.parquet"
        f = pd.read_parquet(p)
        f.index = pd.to_datetime(f.index)
        out[s] = f.loc[start:end]
    return {s: f for s, f in out.items() if len(f)}

raw_panel = load_raw(symbols, start, end)
qfq_panel = {s: f.loc[start:end] for s, f in qfq_panel.items() if s in raw_panel}
raw_panel = {s: f for s, f in raw_panel.items() if s in qfq_panel}
print(f"common symbols={len(qfq_panel)} window={start.date()}..{end.date()}", flush=True)

params = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)
rows = []
results = {}
for label, panel in (("qfq_subtraction", qfq_panel), ("raw_no_adjust", raw_panel)):
    raw = build_strategy("low_vol_core", **params).generate(panel)
    weights = build_portfolio(raw, panel, benchmark=benchmark, risk=RiskConfig(), use_regime=False)
    res = Backtester(BacktestConfig(initial_cash=1_000_000.0)).run(panel, weights, benchmark=benchmark)
    m = res.metrics.to_dict()
    results[label] = res
    rows.append({"basis": label, **{k: round(float(v), 6) for k, v in m.items()}})
    print(label, {k: round(float(v), 4) for k, v in m.items() if k in ("total_return","cagr","sharpe","max_drawdown","annual_volatility")}, flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT/"qfq_vs_raw.csv", index=False, encoding="utf-8-sig")
# Compare daily returns of the two equity curves
a = results["qfq_subtraction"].returns
b = results["raw_no_adjust"].returns
common = a.index.intersection(b.index)
corr = float(a.loc[common].corr(b.loc[common]))
print("return correlation between bases:", round(corr, 4))
# Also compare the two price series directly for one symbol
s = "sh600000"
qr = qfq_panel[s]["close"].pct_change()
rr = raw_panel[s]["close"].pct_change()
print(f"{s}: price-level return corr={float(qr.corr(rr)):.4f}, "
      f"annual vol qfq={float(qr.std()*252**0.5):.4f}, raw={float(rr.std()*252**0.5):.4f}")
write_json({"rows": rows, "return_correlation": corr, "symbols": len(qfq_panel),
            "window": [str(start.date()), str(end.date())]}, OUT/"qfq_vs_raw.json")
print(df[["basis","total_return","cagr","sharpe","max_drawdown","annual_volatility"]].to_string(index=False))
