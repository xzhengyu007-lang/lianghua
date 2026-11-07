"""Robustness study for the default low-vol core configuration.

Four questions, all asked on the repaired full-history panel:

1. Parameter plateau: is the baseline sitting on a cliff, or on a broad plateau?
   One-at-a-time perturbations around the default (24 runs).
2. Year-by-year stability: does the edge appear in every regime, or is it one
   lucky stretch?
3. Concentration: how much of the total PnL comes from the best few names?
4. Sampling uncertainty: block-bootstrap the daily returns to get a distribution
   for Sharpe/CAGR rather than a single point estimate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.backtest import Backtester, slice_result  # noqa: E402
from quant_system.config import (  # noqa: E402
    BROAD_UNIVERSE,
    BacktestConfig,
    RiskConfig,
)
from quant_system.data import load_panel, load_symbol, trim_panel, usable_window  # noqa: E402
from quant_system.metrics import performance_metrics  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402

BASE = dict(
    top_n=8,
    vol_window=60,
    trend_window=120,
    max_per_sector=2,
    rebalance_days=20,
    buffer_rank=1.5,
)
SWEEPS = {
    "top_n": [5, 6, 8, 10, 12],
    "vol_window": [40, 60, 80, 120],
    "trend_window": [100, 120, 150, 200],
    "rebalance_days": [10, 20, 30, 40],
    "max_per_sector": [1, 2, 3, 5],
    "buffer_rank": [1.0, 1.5, 2.0],
}


def run(panel, benchmark, params, cash=1_000_000.0):
    raw = build_strategy("low_vol_core", **params).generate(panel)
    weights = build_portfolio(
        raw,
        panel,
        benchmark=benchmark,
        risk=RiskConfig(),
        use_regime=False,
        use_vol_target=True,
    )
    return Backtester(BacktestConfig(initial_cash=cash)).run(
        panel, weights, benchmark=benchmark
    )


def block_bootstrap(returns: pd.Series, *, block: int = 20, draws: int = 2000, seed: int = 11):
    rng = np.random.default_rng(seed)
    values = returns.to_numpy(dtype=float)
    n = len(values)
    if n < block * 2:
        return pd.DataFrame()
    starts = rng.integers(0, n - block, size=(draws, n // block + 1))
    samples = np.concatenate([values[s : s + block] for s in starts.ravel()]).reshape(
        draws, -1
    )
    rows = []
    for path in samples:
        series = pd.Series(path)
        metrics = performance_metrics(series, equity=(1 + series).cumprod())
        rows.append({"total_return": metrics.total_return, "cagr": metrics.cagr, "sharpe": metrics.sharpe, "max_drawdown": metrics.max_drawdown})
    return pd.DataFrame(rows)


def main() -> int:
    cache = ROOT / "data" / "cache"
    panel = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    start, end = usable_window(panel, min_active_ratio=0.95)
    benchmark = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
    print(f"window {start.date()}..{end.date()} symbols={len(panel)}", flush=True)
    output = ROOT / "reports" / "robustness"
    output.mkdir(parents=True, exist_ok=True)

    base_result = run(panel, benchmark, BASE)
    base_metrics = base_result.metrics.to_dict()
    print("base", {k: round(v, 4) for k, v in base_metrics.items() if isinstance(v, (int, float))}, flush=True)

    rows = [{"parameter": "baseline", "value": "-", "params": BASE, **base_metrics}]
    for name, values in SWEEPS.items():
        for value in values:
            if value == BASE[name]:
                continue
            params = dict(BASE)
            params[name] = value
            metrics = run(panel, benchmark, params).metrics.to_dict()
            rows.append({"parameter": name, "value": value, "params": params, **metrics})
            print(f"{name}={value} total={metrics['total_return']:+.3f} sharpe={metrics['sharpe']:.3f}", flush=True)
    plateau = pd.DataFrame(rows)
    plateau.to_csv(output / "parameter_plateau.csv", index=False, encoding="utf-8-sig")

    yearly = []
    y0, y1 = pd.Timestamp(start).year, pd.Timestamp(end).year
    for year in range(y0, y1 + 1):
        lo = max(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(start))
        hi = min(pd.Timestamp(f"{year}-12-31"), pd.Timestamp(end))
        if lo >= hi:
            continue
        sliced = slice_result(base_result, start=lo, end=hi)
        bench_slice = benchmark.loc[lo:hi]
        bench_metrics = performance_metrics(
            bench_slice.pct_change(fill_method=None).dropna(), equity=bench_slice
        )
        yearly.append(
            {
                "year": year,
                "total_return": sliced.metrics.total_return,
                "sharpe": sliced.metrics.sharpe,
                "max_drawdown": sliced.metrics.max_drawdown,
                "exposure": sliced.metrics.exposure,
                "trades": len(sliced.trades),
                "benchmark_return": bench_metrics.total_return,
                "benchmark_drawdown": bench_metrics.max_drawdown,
            }
        )
    yearly_frame = pd.DataFrame(yearly)
    yearly_frame.to_csv(output / "yearly.csv", index=False, encoding="utf-8-sig")

    trades = base_result.trades
    sells = trades[trades["side"] == "sell"] if len(trades) else trades
    by_symbol = (
        sells.groupby("symbol")["realized_pnl"].sum().sort_values(ascending=False)
        if len(sells)
        else pd.Series(dtype=float)
    )
    total_pnl = float(by_symbol.sum()) if len(by_symbol) else 0.0
    contribution = pd.DataFrame(
        {
            "symbol": by_symbol.index,
            "realized_pnl": by_symbol.values,
            "share_of_total": (by_symbol.values / total_pnl) if total_pnl else 0.0,
        }
    )
    contribution.to_csv(output / "contribution.csv", index=False, encoding="utf-8-sig")
    top5_share = float(contribution["share_of_total"].head(5).sum()) if len(contribution) else float("nan")
    positive_names = int((contribution["realized_pnl"] > 0).sum())
    print(f"contribution: top5={top5_share:.1%} positive_names={positive_names}/{len(contribution)}", flush=True)

    boot = block_bootstrap(base_result.returns, seed=11)
    boot.to_csv(output / "bootstrap.csv", index=False, encoding="utf-8-sig")
    quantiles = {
        col: {
            "p05": float(boot[col].quantile(0.05)),
            "p25": float(boot[col].quantile(0.25)),
            "median": float(boot[col].median()),
            "p75": float(boot[col].quantile(0.75)),
            "p95": float(boot[col].quantile(0.95)),
        }
        for col in ("total_return", "cagr", "sharpe", "max_drawdown")
    }

    summary = {
        "params": BASE,
        "window": [str(start.date()), str(end.date())],
        "symbols": len(panel),
        "base": base_metrics,
        "plateau": {
            "total_return_min": float(plateau["total_return"].min()),
            "total_return_max": float(plateau["total_return"].max()),
            "sharpe_min": float(plateau["sharpe"].min()),
            "sharpe_max": float(plateau["sharpe"].max()),
        },
        "yearly": yearly,
        "concentration": {
            "top5_share_of_realized_pnl": top5_share,
            "positive_names": positive_names,
            "traded_names": int(len(contribution)),
        },
        "bootstrap": quantiles,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with pd.option_context("display.width", 200):
        print(plateau[["parameter", "value", "total_return", "cagr", "sharpe", "max_drawdown"]].to_string(index=False))
        print(yearly_frame.to_string(index=False))
        print(contribution.head(10).to_string(index=False))
        print(json.dumps(quantiles, indent=2))
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
