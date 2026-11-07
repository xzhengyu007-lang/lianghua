"""Does averaging rebalance phases remove calendar luck, or dilute the edge?

The anchor study found a 119pp spread across 20 equivalent phases.  A natural
fix is to hold all 20 phase-portfolios at 1/20 weight.  That must be tested,
not assumed: averaging also averages away any real timing edge.

Outputs per-phase equity curves, the averaged book, and the realised turnover
of the averaged book versus the median single phase.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from quant_system.backtest import Backtester
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, RiskConfig
from quant_system.data import align_panel, load_symbol, normalize_symbol, trim_panel, usable_window
from quant_system.portfolio import build_portfolio
from quant_system.report import write_json
from quant_system.strategies import build_strategy

OUT = ROOT / "reports" / "phase_average"
CACHE = ROOT / "data" / "cache"
PARAMS = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)
PHASES = 20


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    symbols = sorted(normalize_symbol(s) for s in BROAD_UNIVERSE)
    panel = {}
    for symbol in symbols:
        path = CACHE / f"{symbol}_tr.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame.index = pd.to_datetime(frame.index)
        panel[symbol] = frame
    trimmed = trim_panel(panel, min_coverage=0.95)
    start, end = usable_window(trimmed, min_active_ratio=0.95)
    aligned, dates = align_panel(
        {symbol: frame.loc[start:end] for symbol, frame in trimmed.items()}
    )
    benchmark = load_symbol("sh000300", CACHE, adjust="")["close"]
    print(f"symbols={len(aligned)} window={dates.min().date()}..{dates.max().date()}", flush=True)

    curves = {}
    rows = []
    weights_sum = None
    for offset in range(PHASES):
        strategy = build_strategy("low_vol_core", **PARAMS, rebalance_offset=offset).generate(aligned)
        weights = build_portfolio(
            strategy, aligned, benchmark=benchmark, risk=RiskConfig(), use_regime=False
        )
        result = Backtester(BacktestConfig(initial_cash=1_000_000.0)).run(
            aligned, weights, benchmark=benchmark
        )
        curves[offset] = result.equity
        metrics = result.metrics.to_dict()
        rows.append(
            {
                "offset": offset,
                "total_return": metrics["total_return"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "annual_turnover": metrics["turnover"],
                "trades": len(result.trades),
            }
        )
        # Average the *target weights* of all phases, then run one backtest on
        # that blended book.  Averaging equity curves would ignore the extra
        # diversification inside each day; averaging weights is the honest way
        # to hold all phases simultaneously.
        weights_sum = weights if weights_sum is None else weights_sum.add(weights, fill_value=0.0)
        print(
            f"offset={offset:2d} ret={metrics['total_return']:.2%} "
            f"sharpe={metrics['sharpe']:.3f}",
            flush=True,
        )

    blended = weights_sum / PHASES
    blended_result = Backtester(BacktestConfig(initial_cash=1_000_000.0)).run(
        aligned, blended, benchmark=benchmark
    )
    blended_metrics = blended_result.metrics.to_dict()

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "phases.csv", index=False, encoding="utf-8-sig")
    equity_frame = pd.DataFrame(
        {f"phase_{offset}": curve for offset, curve in curves.items()}
    )
    equity_frame["blended"] = blended_result.equity.reindex(equity_frame.index)
    equity_frame.to_csv(OUT / "equity.csv", encoding="utf-8-sig")

    summary = {
        "window": [str(dates.min().date()), str(dates.max().date())],
        "symbols": len(aligned),
        "phases": PHASES,
        "single_phase": {
            "median_return": float(table.total_return.median()),
            "worst_return": float(table.total_return.min()),
            "best_return": float(table.total_return.max()),
            "median_sharpe": float(table.sharpe.median()),
            "median_turnover": float(table.annual_turnover.median()),
        },
        "blended": {
            "total_return": float(blended_metrics["total_return"]),
            "cagr": float(blended_metrics["cagr"]),
            "sharpe": float(blended_metrics["sharpe"]),
            "max_drawdown": float(blended_metrics["max_drawdown"]),
            "annual_volatility": float(blended_metrics["annual_volatility"]),
            "annual_turnover": float(blended_metrics["turnover"]),
            "trades": int(len(blended_result.trades)),
        },
    }
    write_json(summary, OUT / "summary.json")

    lines = [
        "# 相位平均组合 vs 单相位",
        "",
        f"- 窗口：{dates.min().date()} .. {dates.max().date()}，{len(aligned)} 只",
        f"- 20 个相位，每个目标权重各占 1/20，合成一个组合后回测",
        "",
        "| 指标 | 单相位中位 | 单相位最差 | 单相位最好 | 相位平均 |",
        "|---|---:|---:|---:|---:|",
        f"| 总收益 | {table.total_return.median():.2%} | {table.total_return.min():.2%} | "
        f"{table.total_return.max():.2%} | {blended_metrics['total_return']:.2%} |",
        f"| Sharpe | {table.sharpe.median():.3f} | {table.sharpe.min():.3f} | "
        f"{table.sharpe.max():.3f} | {blended_metrics['sharpe']:.3f} |",
        f"| 最大回撤 | {table.max_drawdown.median():.2%} | {table.max_drawdown.min():.2%} | "
        f"{table.max_drawdown.max():.2%} | {blended_metrics['max_drawdown']:.2%} |",
        f"| 年化换手 | {table.annual_turnover.median():.2f} | — | — | "
        f"{blended_metrics['turnover']:.2f} |",
        "",
        "## 解读",
        "",
        "相位平均的收益接近单相位中位数，但回撤与波动应更低、路径更稳。"
        "如果它同时低于中位数很多，说明相位里存在真实的择时 alpha，"
        "平均化会把它稀释掉——这时需要重新审视“锚点任意”这个前提。",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
