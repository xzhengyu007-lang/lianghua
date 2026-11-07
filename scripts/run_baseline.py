"""Full-period baseline: report, dashboard, and factor diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analytics import factor_decay, factor_report  # noqa: E402
from quant_system.backtest import Backtester, simulate_buy_and_hold  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, RiskConfig  # noqa: E402
from quant_system.data import (  # noqa: E402
    load_panel,
    load_symbol,
    price_frame,
    trim_panel,
    usable_window,
    validate_panel,
)
from quant_system.html_report import write_dashboard  # noqa: E402
from quant_system.metrics import performance_metrics  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.report import save_backtest, write_json  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402

PARAMS = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)


def main() -> int:
    cache = ROOT / "data" / "cache"
    panel = load_panel(list(BROAD_UNIVERSE), cache)
    quality = validate_panel(panel)
    panel = trim_panel(panel, min_coverage=0.95)
    start, end = usable_window(panel, min_active_ratio=0.95)
    benchmark = load_symbol("sh000300", cache, adjust="")["close"]
    benchmark = benchmark.loc[start:end]
    print(f"symbols={len(panel)} window={start.date()}..{end.date()} bars={len(benchmark)}", flush=True)
    print(json.dumps(quality.to_dict(), ensure_ascii=False)[:400], flush=True)

    strategy = build_strategy("low_vol_core", **PARAMS)
    raw = strategy.generate(panel)
    # Overlay ablation showed the benchmark-trend switch destroyed most of the
    # return; the long-history baseline therefore uses volatility targeting only.
    weights = build_portfolio(
        raw, panel, benchmark=benchmark, risk=RiskConfig(), use_regime=False
    )
    cfg = BacktestConfig(initial_cash=1_000_000)
    result = Backtester(cfg).run(panel, weights, benchmark=benchmark)

    output = ROOT / "reports" / "baseline_long"
    output.mkdir(parents=True, exist_ok=True)
    save_backtest(
        result,
        output,
        title="Low-volatility core · broad long-history panel",
        benchmark=benchmark,
        notes=[
            "信号收盘生成、次日开盘成交；股数在调仓日冻结。",
            "计入佣金/印花税/过户费/滑点，并限制单日成交量为 20 日均量的 10%。",
            "退市标的按最后真实 K 线强制清算，避免用冻结价格估值。",
            "未启用基准趋势开关：滚动样本外显示它把收益从 +74% 压到 +4%。",
            "波动率目标仅按选股切换时点锁定一次缩放，避免每日碎单。",
            "股票池为当前仍上市的宽基样本，存在幸存者偏差，结果偏乐观。",
        ],
    )

    bench_aligned = benchmark.reindex(result.equity.index).ffill()
    bench_returns = bench_aligned.pct_change(fill_method=None).fillna(0.0)
    bench_metrics = performance_metrics(bench_returns, equity=bench_aligned, risk_free=0.0)

    equal = simulate_buy_and_hold(panel, config=cfg, benchmark=benchmark)
    equal_metrics = equal.metrics
    # Buy-and-hold on a survivorship-biased panel compounds extreme winners, so
    # report the median member and the cap-weighted-ish benchmark as context.
    member_returns = []
    for symbol, frame in panel.items():
        series = frame["close"].loc[start:end]
        if len(series) > 1 and series.iloc[0] > 0:
            member_returns.append(float(series.iloc[-1] / series.iloc[0] - 1.0))
    member_returns = pd.Series(member_returns)
    survivorship_note = {
        "members": int(len(member_returns)),
        "median_total_return": float(member_returns.median()),
        "mean_total_return": float(member_returns.mean()),
        "share_above_benchmark": float((member_returns > 0.1435).mean()),
        "note": "current-listed pool: survivorship-biased; median is the honest member view",
    }

    close = price_frame(panel, "close")
    returns = close.pct_change(fill_method=None)
    factors = {
        "low_vol_20": -returns.rolling(20).std(ddof=0),
        "low_vol_60": -returns.rolling(60).std(ddof=0),
        "momentum_20": close.shift(5) / close.shift(25) - 1.0,
        "momentum_60": close.shift(5) / close.shift(65) - 1.0,
        "trend_120": close / close.rolling(120).mean() - 1.0,
    }
    decay = {
        "horizon_5": factor_report(panel, factors, horizon=5),
        "horizon_20": factor_report(panel, factors, horizon=20),
    }

    summary = {
        "window": {"start": str(start.date()), "end": str(end.date()), "symbols": len(panel)},
        "params": PARAMS,
        "strategy": result.metrics.to_dict(),
        "benchmark": bench_metrics.to_dict(),
        "equal_weight": equal_metrics.to_dict(),
        "metadata": result.metadata,
        "factor_decay": decay,
        "data_quality": quality.to_dict(),
        "survivorship": survivorship_note,
    }
    write_json(summary, output / "summary.json")

    bench_equity = bench_aligned / bench_aligned.dropna().iloc[0] * result.equity.iloc[0]
    write_dashboard(
        output / "dashboard.html",
        title="Low-volatility core · broad long-history panel",
        equity=result.equity,
        metrics=result.metrics.to_dict(),
        benchmark_equity=bench_equity,
        trades=result.trades,
        positions=result.positions,
        analytics=decay["horizon_20"],
        notes=[
            "基准：沪深 300；等权组合为同一股票池买入持有。",
            "股票池为当前上市样本，未做历史时点成分，存在幸存者偏差。",
        ],
    )

    table = pd.DataFrame(
        {
            "strategy": result.metrics.to_dict(),
            "benchmark": bench_metrics.to_dict(),
            "equal_weight": equal_metrics.to_dict(),
        }
    ).T
    print(table[["total_return", "cagr", "annual_volatility", "sharpe", "max_drawdown", "exposure"]].to_string())
    print(json.dumps(decay["horizon_20"], indent=2))
    print("survivorship:", json.dumps(survivorship_note, indent=2))
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())