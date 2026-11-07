"""Archived full-period baseline: report, dashboard and factor diagnostics (superseded, kept for history)."""
import json, sys
from pathlib import Path
import pandas as pd

ROOT = Path.cwd(); sys.path.insert(0, str(ROOT))
from quant_system.backtest import Backtester
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, RiskConfig
from quant_system.data import load_panel, load_symbol, trim_panel, usable_window
from quant_system.html_report import write_dashboard
from quant_system.metrics import performance_metrics
from quant_system.portfolio import build_portfolio
from quant_system.report import save_backtest, write_json
from quant_system.strategies import build_strategy

PARAMS = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)
cache = ROOT/"data"/"cache"
panel = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
start, end = usable_window(panel, min_active_ratio=0.95)
bench = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
raw = build_strategy("low_vol_core", **PARAMS).generate(panel)
w = build_portfolio(raw, panel, benchmark=bench, risk=RiskConfig(), use_regime=False)
res = Backtester(BacktestConfig(initial_cash=1_000_000)).run(panel, w, benchmark=bench)
out = ROOT/"reports"/"final_low_vol_long"
out.mkdir(parents=True, exist_ok=True)
save_backtest(res, out, title="低波核心 · 全历史（2019-06 起）", benchmark=bench,
    caveats=[
        "生存者偏差已实测：加入 140 只窗口内真实退市标的（point-in-time 池，262 只），"
        "总收益由 +145.9% 降至 +133.2%，Sharpe 0.907→0.895，最大回撤 -16.7%→-19.9%。",
        "合成压力测试给出更悲观上界：16 只低波退市陷阱下总收益 +85%（中性）~ +45%（保守）。",
        "行业分类为静态研究映射，不是历史时点分类。",
        "容量：50bp 冲击假设下 100 万规模总收益 -1.7pp，5000 万规模 -10.3pp。",
    ],
    notes=["收盘信号、次日开盘成交，整手与 T+1。",
           "费用：佣金 0.025%（最低 5 元）、印花税 0.05%、过户费 0.001%、滑点 5bp。",
           "单日单票成交 ≤ 20 日均量 10%；退市按最后真实 K 线清算。",
           "默认关闭基准趋势开关：样本外证伪，它把收益从 +74% 压到 +4%。",
           "默认口径仍为当前上市样本（128 只、122 只有效），point-in-time 结果见 reports/point_in_time。"])
bench_eq = bench.reindex(res.equity.index).ffill(); bench_eq = bench_eq/bench_eq.dropna().iloc[0]*res.equity.iloc[0]
write_json({"metrics": res.metrics.to_dict(), "benchmark": performance_metrics(bench.pct_change(fill_method=None).dropna(), equity=bench).to_dict(), "params": PARAMS, "metadata": res.metadata, "window": [str(start.date()), str(end.date())]}, out/"summary.json")
print({k: round(v,4) for k,v in res.metrics.to_dict().items() if isinstance(v,(int,float))})
print(out)