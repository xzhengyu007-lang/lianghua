"""Random-selection control on the point-in-time pool.

The survivor-only null put the 6/2 book at p≈0.03, but that pool excludes real
delisted names and is therefore optimistic about the cross-section. This script
repeats the same null on the PIT pool with the same seed, cadence, sector cap,
overlays and costs, so the low-vol ranking can be compared with a random basket
of the same width on the more honest membership set.
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
from quant_system.data import (  # noqa: E402
    align_panel,
    availability,
    load_panel,
    load_symbol,
    pit_universe,
    trim_panel,
    usable_window,
)
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.report import write_json  # noqa: E402
from quant_system.strategies import _select_weights, build_strategy  # noqa: E402

PARAMS = dict(top_n=6, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)
DRAWS = 100
SEED = 20261107
OUT = ROOT / "reports" / "random_control_pit"


def run(panel, benchmark, raw):
    weights = build_portfolio(raw, panel, benchmark=benchmark, risk=RiskConfig(), use_regime=False)
    return Backtester(BacktestConfig(initial_cash=1_000_000)).run(panel, weights, benchmark=benchmark)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "data" / "cache"
    survivors = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    start, end = usable_window(survivors, min_active_ratio=0.95)
    benchmark = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
    universe = pit_universe()
    delisted = {symbol: name for symbol, name in universe.items() if symbol not in BROAD_UNIVERSE}
    panel = load_panel(list(survivors) + list(delisted), cache)
    panel = {
        symbol: frame.loc[start:end]
        for symbol, frame in panel.items()
        if len(frame.loc[start:end])
    }
    aligned, dates = align_panel(panel)
    print(f"window {start.date()}..{end.date()} symbols={len(aligned)}", flush=True)

    base_raw = build_strategy("low_vol_core", **PARAMS).generate(aligned)
    base = run(aligned, benchmark, base_raw)
    print(
        f"baseline total={base.metrics.total_return:.4f} sharpe={base.metrics.sharpe:.4f}",
        flush=True,
    )

    avail = availability(aligned)
    rng = np.random.default_rng(SEED)
    rows = []
    for draw in range(DRAWS):
        scores = pd.DataFrame(rng.random(base_raw.shape), index=base_raw.index, columns=base_raw.columns)
        raw = _select_weights(
            scores,
            PARAMS["top_n"],
            available=avail,
            max_per_sector=PARAMS["max_per_sector"],
            rebalance_days=PARAMS["rebalance_days"],
        )
        result = run(aligned, benchmark, raw)
        rows.append(
            {
                "draw": draw,
                "total_return": result.metrics.total_return,
                "sharpe": result.metrics.sharpe,
                "max_drawdown": result.metrics.max_drawdown,
            }
        )
        if draw % 20 == 0:
            print(f"draw {draw:03d}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "draws.csv", index=False, encoding="utf-8-sig")
    returns = frame["total_return"]
    summary = {
        "window": [str(start.date()), str(end.date())],
        "symbols": len(aligned),
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
            "mean_sharpe": float(frame["sharpe"].mean()),
            "median_sharpe": float(frame["sharpe"].median()),
            "p95_sharpe": float(frame["sharpe"].quantile(0.95)),
        },
    }
    write_json(summary, OUT / "summary.json")
    print(json.dumps(summary["random"], ensure_ascii=False, indent=2))
    print(f"report: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
