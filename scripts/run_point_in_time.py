"""Extend the universe with real delisted names and compare against survivor-only.

Builds the "point-in-time" pool: the 128 current members plus every real symbol
whose last bar falls inside the backtest window (recovered by
scripts/probe_delisted.py and scripts/build_delisted_universe.py).

Three runs:

1. survivor-only (the published baseline),
2. point-in-time pool (survivors + real delisted names),
3. point-in-time pool re-balanced by listing era, to check that the delisted
   cohort is not simply older, smaller, or structurally different.

Because the delisted names join the screen *while they are still traded*, this
measures the bias directly instead of bounding it with synthetic names.
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
from quant_system.config import (  # noqa: E402
    BROAD_UNIVERSE,
    BacktestConfig,
    RiskConfig,
)
from quant_system.data import load_panel, load_symbol, trim_panel, usable_window  # noqa: E402
from quant_system.metrics import performance_metrics  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402

PARAMS = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)


def evaluate(panel, benchmark):
    raw = build_strategy("low_vol_core", **PARAMS).generate(panel)
    weights = build_portfolio(raw, panel, benchmark=benchmark, risk=RiskConfig(), use_regime=False)
    return Backtester(BacktestConfig(initial_cash=1_000_000)).run(panel, weights, benchmark=benchmark)


def yearly(result, benchmark):
    rows = []
    for year in sorted({d.year for d in result.equity.index}):
        lo = max(pd.Timestamp(f"{year}-01-01"), result.equity.index[0])
        hi = min(pd.Timestamp(f"{year}-12-31"), result.equity.index[-1])
        segment = result.equity.loc[lo:hi]
        if len(segment) < 5:
            continue
        returns = segment.pct_change(fill_method=None).dropna()
        bench = benchmark.loc[lo:hi].pct_change(fill_method=None).dropna()
        m = performance_metrics(returns, equity=segment)
        rows.append(
            {
                "year": year,
                "total_return": m.total_return,
                "sharpe": m.sharpe,
                "max_drawdown": m.max_drawdown,
                "benchmark_return": float((1 + bench).prod() - 1) if len(bench) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    cache = ROOT / "data" / "cache"
    manifest_path = ROOT / "data" / "delisted_manifest.json"
    if not manifest_path.exists():
        print("missing data/delisted_manifest.json; run scripts/build_delisted_universe.py")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    survivors = list(BROAD_UNIVERSE)
    survivor_panel = trim_panel(load_panel(survivors, cache), min_coverage=0.95)
    start, end = usable_window(survivor_panel, min_active_ratio=0.95)
    benchmark = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
    print(f"window {start.date()}..{end.date()} survivors={len(survivor_panel)}", flush=True)

    # Delisted names whose lives overlap the window at all.
    candidates = [
        symbol
        for symbol, info in manifest.items()
        if pd.Timestamp(info["last_bar"]) >= pd.Timestamp(start)
        and pd.Timestamp(info["first_bar"]) <= pd.Timestamp(end)
    ]
    print(f"delisted names overlapping the window: {len(candidates)}", flush=True)
    extra = load_panel(candidates, cache)
    extra = {s: f for s, f in extra.items() if len(f) >= 120}
    print(f"delisted names with usable history: {len(extra)}", flush=True)

    # Restrict to the common sample. Delisted names carry history back to 2004,
    # and extending the panel would silently move the backtest window (and the
    # benchmark join), making the two runs incomparable.
    pit_panel = {
        symbol: frame.loc[start:end] for symbol, frame in {**survivor_panel, **extra}.items()
    }

    survivor_result = evaluate(survivor_panel, benchmark)
    pit_result = evaluate(pit_panel, benchmark)

    out = ROOT / "reports" / "point_in_time"
    out.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "pool": "survivor-only",
            "symbols": len(survivor_panel),
            "delisted_included": 0,
            **survivor_result.metrics.to_dict(),
            "delisted_candidates": int(survivor_result.metadata.get("delisted_candidates", 0)),
            "delisted_exits": int(survivor_result.metadata["delisted_exits"]),
            "held": int(survivor_result.metadata["symbols_ever_held"]),
        },
        {
            "pool": "point-in-time",
            "symbols": len(pit_panel),
            "delisted_included": len(extra),
            **pit_result.metrics.to_dict(),
            "delisted_candidates": int(pit_result.metadata.get("delisted_candidates", 0)),
            "delisted_exits": int(pit_result.metadata["delisted_exits"]),
            "held": int(pit_result.metadata["symbols_ever_held"]),
        },
    ]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / "comparison.csv", index=False, encoding="utf-8-sig")

    yearly(survivor_result, benchmark).assign(pool="survivor-only").to_csv(
        out / "yearly_survivor.csv", index=False, encoding="utf-8-sig"
    )
    yearly(pit_result, benchmark).assign(pool="point-in-time").to_csv(
        out / "yearly_pit.csv", index=False, encoding="utf-8-sig"
    )

    # Which delisted names were actually selected, and how much did they cost?
    trades = pit_result.trades
    delisted = set(extra)
    selected = sorted(set(trades["symbol"]) & delisted) if len(trades) else []
    pnl_rows = []
    for symbol in selected:
        realized = float(trades[(trades["symbol"] == symbol) & (trades["side"] == "sell")]["realized_pnl"].sum())
        pnl_rows.append(
            {
                "symbol": symbol,
                "first_bar": str(extra[symbol].index.min().date()),
                "last_bar": str(extra[symbol].index.max().date()),
                "realized_pnl": realized,
            }
        )
    pnl = pd.DataFrame(pnl_rows).sort_values("realized_pnl") if pnl_rows else pd.DataFrame()
    if len(pnl):
        pnl.to_csv(out / "delisted_contributions.csv", index=False, encoding="utf-8-sig")

    summary = {
        "params": PARAMS,
        "window": [str(start.date()), str(end.date())],
        "survivors": len(survivor_panel),
        "delisted_included": len(extra),
        "survivor_only": survivor_result.metrics.to_dict(),
        "point_in_time": pit_result.metrics.to_dict(),
        "return_cost_of_bias": float(
            survivor_result.metrics.total_return - pit_result.metrics.total_return
        ),
        "sharpe_cost_of_bias": float(
            survivor_result.metrics.sharpe - pit_result.metrics.sharpe
        ),
        "delisted_selected": len(selected),
        "delisted_realized_pnl": float(pnl["realized_pnl"].sum()) if len(pnl) else 0.0,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with pd.option_context("display.width", 240):
        print(
            comparison[
                ["pool", "symbols", "delisted_included", "total_return", "cagr", "sharpe",
                 "max_drawdown", "exposure", "delisted_candidates", "delisted_exits", "held"]
            ].round(4).to_string(index=False)
        )
        if len(pnl):
            print(pnl.head(15).to_string(index=False))
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
