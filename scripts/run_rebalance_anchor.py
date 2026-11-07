"""Rebalance-anchor sensitivity: how much of the headline is calendar luck?

A 20-trading-day cadence has an arbitrary phase.  This script reruns the exact
same strategy and prices with the cadence shifted by 0..19 bars, plus month-end
as a calendar-defined alternative.  If the distribution is wide, a single
backtest is not evidence of an edge.
"""

from __future__ import annotations

import sys
from pathlib import Path

import json

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

CACHE = ROOT / "data" / "cache"
PARAMS = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)

# Command line: --universe pit  --reset year  --output reports/xxx
UNIVERSE = "broad"
RESET = "start"
OUT = ROOT / "reports" / "rebalance_anchor"
for flag, name in (("--universe", "universe"), ("--reset", "reset"), ("--output", "output")):
    if flag in sys.argv:
        value = sys.argv[sys.argv.index(flag) + 1]
        if name == "universe":
            UNIVERSE = value
        elif name == "reset":
            RESET = value
        else:
            OUT = Path(value)
if UNIVERSE not in {"broad", "pit"}:
    raise SystemExit("--universe must be broad or pit")
if RESET not in {"start", "year"}:
    raise SystemExit("--reset must be start or year")


def run(panel, benchmark, **overrides):
    params = {**PARAMS, **overrides}
    strategy = build_strategy("low_vol_core", **params).generate(panel)
    weights = build_portfolio(
        strategy, panel, benchmark=benchmark, risk=RiskConfig(), use_regime=False
    )
    result = Backtester(BacktestConfig(initial_cash=1_000_000.0)).run(
        panel, weights, benchmark=benchmark
    )
    return result.metrics.to_dict(), result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if UNIVERSE == "pit":
        from quant_system.data import pit_universe

        names = pit_universe()
    else:
        names = BROAD_UNIVERSE
    symbols = sorted(normalize_symbol(s) for s in names)
    tr = {}
    for symbol in symbols:
        path = CACHE / f"{symbol}_tr.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame.index = pd.to_datetime(frame.index)
        tr[symbol] = frame
    if UNIVERSE == "pit":
        # Survivors define the window; delisted names are clipped to it, the
        # same rule the CLI uses.  A 95% cross-section filter would drop every
        # delisted name by construction.
        from quant_system.data import normalize_symbol as _normalize

        broad = {_normalize(s) for s in BROAD_UNIVERSE}
        anchor = trim_panel({s: f for s, f in tr.items() if s in broad}, min_coverage=0.95)
        start, end = usable_window(anchor, min_active_ratio=0.95)
        panel, dates = align_panel(
            {symbol: frame.loc[start:end] for symbol, frame in tr.items() if len(frame.loc[start:end])}
        )
    else:
        trimmed = trim_panel(tr, min_coverage=0.95)
        start, end = usable_window(trimmed, min_active_ratio=0.95)
        panel, dates = align_panel(
            {symbol: frame.loc[start:end] for symbol, frame in trimmed.items()}
        )
    benchmark = load_symbol("sh000300", CACHE, adjust="")["close"]
    print(f"symbols={len(panel)} window={dates.min().date()}..{dates.max().date()}", flush=True)

    rows = []
    for offset in range(20):
        metrics, result = run(
            panel, benchmark, rebalance_offset=offset, rebalance_reset=RESET
        )
        rows.append(
            {
                "anchor": f"interval_offset_{offset}",
                "offset": offset,
                "total_return": metrics["total_return"],
                "cagr": metrics["cagr"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "annual_volatility": metrics["annual_volatility"],
                "exposure": metrics["exposure"],
                "trades": len(result.trades),
            }
        )
        print(
            f"offset={offset:2d} ret={metrics['total_return']:.2%} "
            f"sharpe={metrics['sharpe']:.3f} mdd={metrics['max_drawdown']:.2%}",
            flush=True,
        )

    metrics, result = run(
        panel, benchmark, rebalance_on="month_end", rebalance_reset=RESET
    )
    rows.append(
        {
            "anchor": "month_end",
            "offset": -1,
            "total_return": metrics["total_return"],
            "cagr": metrics["cagr"],
            "sharpe": metrics["sharpe"],
            "max_drawdown": metrics["max_drawdown"],
            "annual_volatility": metrics["annual_volatility"],
            "exposure": metrics["exposure"],
            "trades": len(result.trades),
        }
    )
    print(
        f"month_end ret={metrics['total_return']:.2%} sharpe={metrics['sharpe']:.3f}",
        flush=True,
    )

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "anchors.csv", index=False, encoding="utf-8-sig")
    interval = table[table.offset >= 0]
    summary = {
        "universe": UNIVERSE,
        "reset": RESET,
        "window": [str(dates.min().date()), str(dates.max().date())],
        "symbols": len(panel),
        "parameters": PARAMS,
        "interval": {
            "min_return": float(interval.total_return.min()),
            "median_return": float(interval.total_return.median()),
            "max_return": float(interval.total_return.max()),
            "spread_pp": float((interval.total_return.max() - interval.total_return.min()) * 100),
            "min_sharpe": float(interval.sharpe.min()),
            "median_sharpe": float(interval.sharpe.median()),
            "max_sharpe": float(interval.sharpe.max()),
            "worst_mdd": float(interval.max_drawdown.min()),
            "best_mdd": float(interval.max_drawdown.max()),
        },
        "month_end": {
            "total_return": float(table.loc[table.anchor == "month_end", "total_return"].iloc[0]),
            "sharpe": float(table.loc[table.anchor == "month_end", "sharpe"].iloc[0]),
        },
    }
    write_json(summary, OUT / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
