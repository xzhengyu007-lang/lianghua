"""Does the book actually diversify, or only look diversified?

A 6-8 name equal-weight book looks diversified by count. The question that
matters for risk is whether the *effective* number of independent bets is close
to 8, and whether the book loses that diversification exactly when it is needed
(when the market falls).

This script is diagnostic, not a strategy. It reports:

* effective breadth ``1 / sum(w_i^2)`` on each rebalance date,
* average pairwise correlation of held names, in calm and stressed months,
* portfolio beta and the share of variance explained by the benchmark,
* the correlation of the strategy's returns with the benchmark on the worst
  decile of benchmark days (a "diversification in a crisis" check).

All numbers come from the default 8/2 configuration so they describe the
published baseline, not a candidate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.backtest import Backtester  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, RiskConfig  # noqa: E402
from quant_system.data import (  # noqa: E402
    load_panel,
    load_symbol,
    price_frame,
    trim_panel,
    usable_window,
)
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.report import write_json  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402

PARAMS = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)
OUT = ROOT / "reports" / "diversification_audit"


def effective_breadth(weights: pd.DataFrame) -> pd.Series:
    """Effective number of names, ``1 / sum(w_i^2)`` on *normalised* weights.

    Dividing by gross exposure first is essential: the volatility target scales
    the whole book down, and a raw ``1/sum(w^2)`` would then report more names
    simply because every weight got smaller (8 equal names at 75% gross look
    like 14.2 names). That is a units error, not diversification.
    """
    gross = weights.sum(axis=1).replace(0.0, np.nan)
    normalised = weights.div(gross, axis=0)
    squared = (normalised**2).sum(axis=1)
    return (1.0 / squared.replace(0.0, np.nan)).dropna()


def pairwise_correlation(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    """Mean pairwise correlation of the held names, using trailing returns."""
    values = {}
    for date in returns.index:
        frame = returns.loc[:date].tail(window)
        if len(frame) < 20:
            continue
        corr = frame.corr(min_periods=20)
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        pairs = corr.to_numpy()[mask]
        pairs = pairs[np.isfinite(pairs)]
        if pairs.size:
            values[date] = float(pairs.mean())
    return pd.Series(values, dtype=float)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "data" / "cache"
    panel = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    start, end = usable_window(panel, min_active_ratio=0.95)
    panel = {s: f.loc[start:end] for s, f in panel.items()}
    benchmark = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
    print(f"window {start.date()}..{end.date()} symbols={len(panel)}", flush=True)

    raw = build_strategy("low_vol_core", **PARAMS).generate(panel)
    weights = build_portfolio(raw, panel, benchmark=benchmark, risk=RiskConfig(), use_regime=False)
    result = Backtester(BacktestConfig(initial_cash=1_000_000)).run(panel, weights, benchmark=benchmark)

    close = price_frame(panel, "close")
    returns = close.pct_change(fill_method=None)
    breadth = effective_breadth(weights)
    # Rebalance dates only: between them the basket is frozen and the breadth
    # number would just repeat the same value.
    held = weights.gt(0)
    changed = held.ne(held.shift(1)).any(axis=1)
    rebalance_breadth = breadth.reindex(weights.index[changed]).dropna()

    # Correlation of held names, measured on trailing 60 bars at each rebalance.
    held_returns = returns.reindex(columns=weights.columns)
    corr_values = {}
    for date in weights.index[changed]:
        frame = held_returns.loc[:date].tail(60)
        active = held.loc[date][held.loc[date]].index
        if len(active) < 2 or len(frame) < 20:
            continue
        corr = frame[active].corr(min_periods=20)
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        pairs = corr.to_numpy()[mask]
        pairs = pairs[np.isfinite(pairs)]
        if pairs.size:
            corr_values[date] = float(pairs.mean())
    corr_series = pd.Series(corr_values, dtype=float)

    strategy_returns = result.equity.pct_change(fill_method=None).dropna()
    bench_returns = benchmark.pct_change(fill_method=None).reindex(strategy_returns.index)
    joined = pd.concat([strategy_returns.rename("strategy"), bench_returns.rename("bench")], axis=1).dropna()

    beta = float(np.polyfit(joined["bench"], joined["strategy"], 1)[0])
    correlation = float(joined["strategy"].corr(joined["bench"]))
    r_squared = correlation**2

    # Stress: the worst 10% of benchmark days. A diversified book should have a
    # low beta there; a hidden concentrated bet will not.
    threshold = joined["bench"].quantile(0.10)
    stress = joined[joined["bench"] <= threshold]
    stress_beta = float(np.polyfit(stress["bench"], stress["strategy"], 1)[0]) if len(stress) > 5 else None
    stress_corr = float(stress["strategy"].corr(stress["bench"])) if len(stress) > 5 else None

    # Split correlation by benchmark state at the rebalance date.
    bench_state = bench_returns.rolling(20).mean()
    calm = corr_series[corr_series.index.map(lambda d: bench_state.get(d, 0.0) >= 0)]
    stressed = corr_series[corr_series.index.map(lambda d: bench_state.get(d, 0.0) < 0)]

    payload = {
        "window": [str(start.date()), str(end.date())],
        "params": PARAMS,
        "rebalances": int(len(rebalance_breadth)),
        "effective_breadth": {
            "mean": float(rebalance_breadth.mean()),
            "median": float(rebalance_breadth.median()),
            "min": float(rebalance_breadth.min()),
            "max": float(rebalance_breadth.max()),
            "target_names": PARAMS["top_n"],
        },
        "pairwise_correlation_60d": {
            "mean": float(corr_series.mean()),
            "median": float(corr_series.median()),
            "min": float(corr_series.min()),
            "max": float(corr_series.max()),
            "calm_mean": float(calm.mean()) if len(calm) else None,
            "stressed_mean": float(stressed.mean()) if len(stressed) else None,
            "calm_n": int(len(calm)),
            "stressed_n": int(len(stressed)),
        },
        "benchmark_exposure": {
            "beta": beta,
            "correlation": correlation,
            "r_squared": r_squared,
            "stress_beta": stress_beta,
            "stress_correlation": stress_corr,
            "stress_days": int(len(stress)),
            "stress_threshold": float(threshold),
        },
        "note": (
            "Effective breadth is 1/sum(w^2) at rebalance dates. Correlations "
            "use trailing 60 bars of held names only. Stress = worst decile of "
            "benchmark days. Descriptive only; no default changes."
        ),
    }
    write_json(payload, OUT / "summary.json")
    rebalance_breadth.to_frame("effective_breadth").to_csv(
        OUT / "breadth_by_rebalance.csv", encoding="utf-8-sig"
    )
    corr_series.to_frame("mean_pairwise_correlation").to_csv(
        OUT / "correlation_by_rebalance.csv", encoding="utf-8-sig"
    )
    print(
        f"breadth mean={payload['effective_breadth']['mean']:.2f} "
        f"min={payload['effective_breadth']['min']:.2f} of {PARAMS['top_n']}",
        flush=True,
    )
    print(
        f"pairwise corr mean={payload['pairwise_correlation_60d']['mean']:.3f} "
        f"calm={payload['pairwise_correlation_60d']['calm_mean']} "
        f"stressed={payload['pairwise_correlation_60d']['stressed_mean']}",
        flush=True,
    )
    print(
        f"beta={beta:.3f} corr={correlation:.3f} r2={r_squared:.3f} "
        f"stress_beta={stress_beta}",
        flush=True,
    )
    print(f"report: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
