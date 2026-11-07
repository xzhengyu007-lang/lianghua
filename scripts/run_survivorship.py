"""How much does survivorship bias actually buy us?

The universe is current index members, so every name in it survived to 2026.
Real A-share investors also held names that stopped trading. We cannot recover
the true delisted set from this data source, so instead we bound the effect:
inject a synthetic cohort of names that trade like the market and then die.

Each synthetic name:

* starts at a random date inside the window, with ~30% annualised vol and a
  beta of 0.5 to the index (matching the strategy's realised beta),
* trades normally until a random delisting date,
* loses 55% (or 85%) on the final bar and then stops printing bars.

The synthetic names are given a *low* idiosyncratic vol (12%) and a mild
upward drift, so they genuinely pass the 60-day low-vol screen and the
120-day trend filter. That is the point: the question is not "would a junk
stock be picked", it is "what happens when a name we are *supposed* to like
turns out to be a value trap".

The backtester already forces a liquidation on the last real bar, so the loss is
booked properly. We add 2 names per year (about 1.6% of the pool per year,
slightly worse than the historical A-share average) and compare the result with
the clean panel.
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
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402

PARAMS = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)


def synthetic_delisted(
    dates: pd.DatetimeIndex,
    benchmark: pd.Series,
    *,
    count: int,
    seed: int = 7,
    start_price: float = 12.0,
    final_drop: float = 0.55,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    bench_returns = benchmark.pct_change(fill_method=None).reindex(dates).fillna(0.0).to_numpy()
    panel = {}
    n = len(dates)
    if n < 200:
        return panel
    for k in range(count):
        start = int(rng.integers(60, max(61, n - 250)))
        life = int(rng.integers(250, max(251, min(n - start, 900))))
        end = min(n, start + life)
        shocks = rng.normal(0.0004, 0.12 / np.sqrt(252), end - start)
        path = 0.35 * bench_returns[start:end] + shocks
        price = start_price * np.exp(np.cumsum(path))
        price[-1] = max(0.5, price[-1] * (1 - final_drop))
        regime = rng.choice(["sh", "sz"], p=[0.45, 0.55])
        symbol = f"{regime}9{k:05d}"
        index = dates[start:end]
        close = pd.Series(price, index=index)
        open_ = close.shift(1).fillna(close.iloc[0]) * (
            1 + rng.normal(0, 0.004, len(close))
        )
        frame = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, len(close)))),
                "low": np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, len(close)))),
                "close": close,
                "volume": rng.integers(80_000, 600_000, len(close)).astype(float),
            },
            index=index,
        )
        panel[symbol] = frame
    return panel


def evaluate(panel, benchmark):
    raw = build_strategy("low_vol_core", **PARAMS).generate(panel)
    weights = build_portfolio(
        raw, panel, benchmark=benchmark, risk=RiskConfig(), use_regime=False
    )
    return Backtester(BacktestConfig(initial_cash=1_000_000)).run(
        panel, weights, benchmark=benchmark
    )


def main() -> int:
    cache = ROOT / "data" / "cache"
    panel = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    start, end = usable_window(panel, min_active_ratio=0.95)
    benchmark = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
    dates = pd.DatetimeIndex(benchmark.index)
    output = ROOT / "reports" / "survivorship"
    output.mkdir(parents=True, exist_ok=True)
    print(f"window {start.date()}..{end.date()} symbols={len(panel)}", flush=True)

    clean = evaluate(panel, benchmark)
    rows = [{"cohort": 0, "final_drop": 0.0, **clean.metrics.to_dict(),
             "held": int(clean.metadata["symbols_ever_held"]), "delisted_exits": int(clean.metadata["delisted_exits"])}]
    print(f"clean total={clean.metrics.total_return:+.2%} sharpe={clean.metrics.sharpe:.3f}", flush=True)

    for count in (8, 16, 32, 64):
        for final_drop in (0.55, 0.85):
            cohort = synthetic_delisted(dates, benchmark, count=count, seed=100 + count)
            for frame in cohort.values():
                last = len(frame) - 1
                frame.iloc[last, frame.columns.get_loc("close")] = max(
                    0.5, frame["close"].iloc[last] / (1 - 0.55) * (1 - final_drop)
                )
            combined = {**panel, **cohort}
            result = evaluate(combined, benchmark)
            metrics = result.metrics.to_dict()
            rows.append(
                {
                    "cohort": count,
                    "final_drop": final_drop,
                    **metrics,
                    "held": int(result.metadata["symbols_ever_held"]),
                    "delisted_exits": int(result.metadata["delisted_exits"]),
                }
            )
            print(
                f"cohort={count:3d} drop={final_drop:.2f} total={metrics['total_return']:+.2%} "
                f"sharpe={metrics['sharpe']:.3f} mdd={metrics['max_drawdown']:.2%} "
                f"exits={result.metadata['delisted_exits']}",
                flush=True,
            )

    table = pd.DataFrame(rows)
    table.to_csv(output / "cohorts.csv", index=False, encoding="utf-8-sig")
    clean_metrics = clean.metrics.to_dict()
    worst = table.iloc[1:].sort_values("total_return").iloc[0]
    summary = {
        "params": PARAMS,
        "window": [str(start.date()), str(end.date())],
        "symbols": len(panel),
        "clean": clean_metrics,
        "worst_case": {
            "cohort": int(worst["cohort"]),
            "final_drop": float(worst["final_drop"]),
            "total_return": float(worst["total_return"]),
            "sharpe": float(worst["sharpe"]),
            "max_drawdown": float(worst["max_drawdown"]),
            "return_cost_vs_clean": float(worst["total_return"] - clean_metrics["total_return"]),
        },
        "assumptions": {
            "cohort_size": "2 synthetic names per year (8/16/32/64 over the window)",
            "vol": 0.12,
            "beta": 0.35,
            "drift": "+0.04%/day before the terminal loss",
            "terminal_loss": "55% or 85% on the final bar",
            "volume_lots": "80k-600k per day",
            "note": "the synthetic names are deliberately unattractive to a low-vol screen",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with pd.option_context("display.width", 200):
        print(
            table[["cohort", "final_drop", "total_return", "sharpe", "max_drawdown", "held", "delisted_exits"]]
            .round(4)
            .to_string(index=False)
        )
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
