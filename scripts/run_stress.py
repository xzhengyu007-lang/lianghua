"""Stress tests that attack the two most likely sources of fake alpha.

1. Survivorship / winner-picking: the universe is *current* members, so drop the
   best-performing names and see whether the edge survives. If the result
   collapses, the headline was mostly "we already know which stocks won".
2. Implementation lag: delay the signal by 1/3/5/10 trading days. Genuine
   cross-sectional edges survive a few days of delay; a within-a-day artifact
   does not.
3. Universe sampling: draw many random 80% subsets of the pool and look at the
   distribution of outcomes, rather than trusting one realisation.
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
from quant_system.data import load_panel, load_symbol, price_frame, trim_panel, usable_window  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402

PARAMS = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)


def evaluate(panel, benchmark, *, lag=0, params=None):
    raw = build_strategy("low_vol_core", **(params or PARAMS)).generate(panel)
    if lag:
        raw = raw.copy()
        raw.iloc[lag:] = raw.iloc[:-lag].to_numpy()
        raw.iloc[:lag] = 0.0
    weights = build_portfolio(
        raw, panel, benchmark=benchmark, risk=RiskConfig(), use_regime=False
    )
    result = Backtester(BacktestConfig(initial_cash=1_000_000)).run(
        panel, weights, benchmark=benchmark
    )
    return result.metrics.to_dict()


def main() -> int:
    cache = ROOT / "data" / "cache"
    panel = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    start, end = usable_window(panel, min_active_ratio=0.95)
    benchmark = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
    output = ROOT / "reports" / "stress"
    output.mkdir(parents=True, exist_ok=True)
    print(f"window {start.date()}..{end.date()} symbols={len(panel)}", flush=True)

    base = evaluate(panel, benchmark)
    print("base", {k: round(v, 4) for k, v in base.items() if isinstance(v, (int, float))}, flush=True)

    # ---- 1. drop the winners / losers ------------------------------------
    close = price_frame(panel, "close")
    total_return = (close.ffill().iloc[-1] / close.bfill().iloc[0] - 1.0).sort_values()
    drop_rows = [{"dropped": 0, **base}]
    for count in (5, 10, 20, 30):
        for label, symbols in (
            ("best", total_return.tail(count).index),
            ("worst", total_return.head(count).index),
        ):
            subset = {s: f for s, f in panel.items() if s not in set(symbols)}
            metrics = evaluate(subset, benchmark, params=PARAMS)
            drop_rows.append({"dropped": count, "which": label, **metrics})
            print(
                f"drop {count} {label}: total={metrics['total_return']:+.3f} "
                f"sharpe={metrics['sharpe']:.3f}",
                flush=True,
            )
    drops = pd.DataFrame(drop_rows)
    drops.to_csv(output / "drop_winners.csv", index=False, encoding="utf-8-sig")

    # ---- 2. implementation lag -------------------------------------------
    lag_rows = [{"lag_days": 0, **base}]
    for lag in (1, 3, 5, 10, 20):
        metrics = evaluate(panel, benchmark, lag=lag)
        lag_rows.append({"lag_days": lag, **metrics})
        print(
            f"lag {lag}: total={metrics['total_return']:+.3f} sharpe={metrics['sharpe']:.3f}",
            flush=True,
        )
        lags = pd.DataFrame(lag_rows)
    lags.to_csv(output / "lag.csv", index=False, encoding="utf-8-sig")

    # ---- 3. random universe subsets --------------------------------------
    rng = np.random.default_rng(20261106)
    symbols = sorted(panel)
    subset_rows = []
    for draw in range(40):
        size = int(len(symbols) * 0.8)
        chosen = set(rng.choice(symbols, size=size, replace=False).tolist())
        subset = {s: f for s, f in panel.items() if s in chosen}
        metrics = evaluate(subset, benchmark)
        subset_rows.append({"draw": draw, **metrics})
        if (draw + 1) % 10 == 0:
            print(f"subsets {draw + 1}/40", flush=True)
    subsets = pd.DataFrame(subset_rows)
    subsets.to_csv(output / "subsets.csv", index=False, encoding="utf-8-sig")

    def quantiles(frame, column):
        return {
            "p05": float(frame[column].quantile(0.05)),
            "median": float(frame[column].median()),
            "p95": float(frame[column].quantile(0.95)),
            "min": float(frame[column].min()),
            "max": float(frame[column].max()),
        }

    summary = {
        "params": PARAMS,
        "window": [str(start.date()), str(end.date())],
        "symbols": len(panel),
        "base": base,
        "drop_winners": drops.to_dict(orient="records"),
        "lag": lags.to_dict(orient="records"),
        "subset_bootstrap": {
            "draws": len(subsets),
            "size": int(len(symbols) * 0.8),
            "total_return": quantiles(subsets, "total_return"),
            "sharpe": quantiles(subsets, "sharpe"),
            "max_drawdown": quantiles(subsets, "max_drawdown"),
            "fraction_beating_index": float(
                (subsets["alpha"] > 0).mean()
            ),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with pd.option_context("display.width", 200):
        print(drops[["dropped", "which", "total_return", "sharpe", "max_drawdown"]].round(4).to_string(index=False))
        print(lags[["lag_days", "total_return", "sharpe", "max_drawdown"]].round(4).to_string(index=False))
        print(json.dumps(summary["subset_bootstrap"], indent=2))
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
