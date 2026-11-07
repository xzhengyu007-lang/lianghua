"""Random-selection control for the low-volatility core strategy.

The baseline reports +145.9% on 122 current-listed large caps. That number is
not evidence that the low-volatility *ranking* adds value: a random basket of
the same size, with the same rebalance dates and sector cap, benefits from the
same panel drift. This script measures that null distribution.

Each draw replaces the strategy scores with seeded uniform noise and reuses the
same selection engine, portfolio overlays, execution model and costs.
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
    trim_panel,
    usable_window,
)
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import _select_weights, build_strategy  # noqa: E402

PARAMS = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)
DRAWS = 100
SEED = 20261106


def run(panel, benchmark, raw):
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
    panel = {symbol: frame.loc[start:end] for symbol, frame in panel.items()}
    benchmark = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
    print(f"window {start.date()}..{end.date()} symbols={len(panel)}", flush=True)

    base_raw = build_strategy("low_vol_core", **PARAMS).generate(panel)
    base = run(panel, benchmark, base_raw)
    print(
        f"baseline total={base.metrics.total_return:.4f} sharpe={base.metrics.sharpe:.4f}",
        flush=True,
    )

    avail = availability(panel)
    rng = np.random.default_rng(SEED)
    rows = []
    for draw in range(DRAWS):
        scores = pd.DataFrame(
            rng.random(base_raw.shape),
            index=base_raw.index,
            columns=base_raw.columns,
        )
        raw = _select_weights(
            scores,
            PARAMS["top_n"],
            available=avail,
            max_per_sector=PARAMS["max_per_sector"],
            rebalance_days=PARAMS["rebalance_days"],
        )
        result = run(panel, benchmark, raw)
        rows.append(
            {
                "draw": draw,
                "total_return": result.metrics.total_return,
                "sharpe": result.metrics.sharpe,
                "max_drawdown": result.metrics.max_drawdown,
            }
        )
        print(
            f"draw {draw:02d}: total={result.metrics.total_return:.3%} "
            f"sharpe={result.metrics.sharpe:.3f}",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    out = ROOT / "reports" / "random_control"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "draws.csv", index=False, encoding="utf-8-sig")
    returns = frame["total_return"]
    summary = {
        "window": [str(start.date()), str(end.date())],
        "symbols": len(panel),
        "params": PARAMS,
        "draws": DRAWS,
        "seed": SEED,
        "baseline": {
            "total_return": base.metrics.total_return,
            "sharpe": base.metrics.sharpe,
            "max_drawdown": base.metrics.max_drawdown,
        },
        "random": {
            "mean_return": float(returns.mean()),
            "median_return": float(returns.median()),
            "p05_return": float(returns.quantile(0.05)),
            "p95_return": float(returns.quantile(0.95)),
            "max_return": float(returns.max()),
            "share_below_baseline": float((returns < base.metrics.total_return).mean()),
            "one_sided_p": float((returns >= base.metrics.total_return).mean()),
        },
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["random"], ensure_ascii=False, indent=2))
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
