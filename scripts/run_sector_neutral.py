"""Sector-neutral control for the low-volatility core strategy.

The baseline's +145.9% may be an industry bet: a low-vol screen mechanically
tilts toward banks and utilities. This script ranks volatility *within* each
static sector, then applies the same selection engine, overlays and costs.

If the edge survives, it is stock selection. If it collapses toward the
equal-weight panel, the headline is a sector allocation effect.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.backtest import Backtester  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, RiskConfig  # noqa: E402
from quant_system.data import (  # noqa: E402
    availability,
    load_panel,
    load_symbol,
    price_frame,
    trim_panel,
    usable_window,
)
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.sectors import sector_of  # noqa: E402
from quant_system.strategies import _select_weights, build_strategy  # noqa: E402

PARAMS = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)


def evaluate(panel, benchmark, raw):
    weights = build_portfolio(
        raw, panel, benchmark=benchmark, risk=RiskConfig(), use_regime=False
    )
    result = Backtester(BacktestConfig(initial_cash=1_000_000)).run(
        panel, weights, benchmark=benchmark
    )
    return {
        "total_return": result.metrics.total_return,
        "cagr": result.metrics.cagr,
        "sharpe": result.metrics.sharpe,
        "max_drawdown": result.metrics.max_drawdown,
        "exposure": result.metrics.exposure,
    }


def main() -> int:
    cache = ROOT / "data" / "cache"
    panel = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    start, end = usable_window(panel, min_active_ratio=0.95)
    panel = {symbol: frame.loc[start:end] for symbol, frame in panel.items()}
    benchmark = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
    print(f"window {start.date()}..{end.date()} symbols={len(panel)}", flush=True)

    base_raw = build_strategy("low_vol_core", **PARAMS).generate(panel)
    base = evaluate(panel, benchmark, base_raw)

    close = price_frame(panel, "close")
    returns = close.pct_change(fill_method=None)
    volatility = returns.rolling(60).std(ddof=0)
    trend = close / close.rolling(120).mean() - 1.0
    raw_scores = -volatility.where(trend > 0)

    sectors = pd.Series({column: sector_of(column) for column in raw_scores.columns})
    neutral = raw_scores.copy()
    for sector in sectors.unique():
        columns = [c for c in raw_scores.columns if sectors[c] == sector]
        if len(columns) < 2:
            continue
        neutral[columns] = raw_scores[columns].rank(axis=1, pct=True)

    available = availability(panel)
    neutral_raw = _select_weights(
        neutral,
        PARAMS["top_n"],
        available=available,
        max_per_sector=None,
        rebalance_days=PARAMS["rebalance_days"],
    )
    neutral_result = evaluate(panel, benchmark, neutral_raw)

    _, eligible = build_strategy("low_vol_core", **PARAMS).score(panel)
    equal = eligible.astype(float)
    equal = equal.div(equal.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    equal_result = evaluate(panel, benchmark, equal)

    out = ROOT / "reports" / "sector_neutral"
    out.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(
        {
            "baseline": base,
            "sector_neutral": neutral_result,
            "equal_weight": equal_result,
        }
    )
    comparison.to_csv(out / "comparison.csv", encoding="utf-8-sig")
    summary = {
        "window": [str(start.date()), str(end.date())],
        "symbols": len(panel),
        "params": PARAMS,
        "results": comparison.to_dict(),
        "note": (
            "Sector-neutral ranks volatility within each static sector. "
            "A collapse toward equal weight means the headline is a sector bet, "
            "not intra-sector stock selection."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(comparison.round(4).to_string())
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
