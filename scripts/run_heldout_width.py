"""Fully held-out final-year check for portfolio width.

Every walk-forward window overlaps the 2019-2026 sample that motivated the
top_n=6 candidate. This script isolates the last twelve months and evaluates
the two fixed widths using only information available before the cut. It is a
single market state, so it is reported as a counter-example, not a new tuning
target.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.backtest import Backtester  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, RiskConfig  # noqa: E402
from quant_system.data import load_panel, load_symbol, trim_panel, usable_window  # noqa: E402
from quant_system.metrics import performance_metrics  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.report import write_json  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402

FIXED = {"vol_window": 60, "trend_window": 120, "max_per_sector": 2, "rebalance_days": 20}
CUT = "2025-11-05"
OUT = ROOT / "reports" / "heldout_width"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "data" / "cache"
    panel = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    start, end = usable_window(panel, min_active_ratio=0.95)
    panel = {symbol: frame.loc[start:end] for symbol, frame in panel.items()}
    benchmark = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
    cut = pd.Timestamp(CUT)
    print(f"window {start.date()}..{end.date()} cut={cut.date()} symbols={len(panel)}", flush=True)

    rows = []
    for top_n in (6, 8):
        params = {**FIXED, "top_n": top_n}
        raw = build_strategy("low_vol_core", **params).generate(panel)
        weights = build_portfolio(
            raw, panel, benchmark=benchmark, risk=RiskConfig(), use_regime=False
        )
        result = Backtester(BacktestConfig(initial_cash=1_000_000)).run(
            panel, weights, benchmark=benchmark
        )
        segment = result.equity.loc[cut:]
        returns = segment.pct_change(fill_method=None).dropna()
        metrics = performance_metrics(returns, equity=segment)
        rows.append(
            {
                "top_n": top_n,
                "total_return": metrics.total_return,
                "sharpe": metrics.sharpe,
                "max_drawdown": metrics.max_drawdown,
                "bars": len(segment),
                "start": str(segment.index[0].date()),
                "end": str(segment.index[-1].date()),
            }
        )
        print(
            f"top_n={top_n}: {metrics.total_return:+.2%} "
            f"sharpe={metrics.sharpe:.3f} mdd={metrics.max_drawdown:.2%}",
            flush=True,
        )

    bench = benchmark.loc[cut:]
    bench_return = float(bench.iloc[-1] / bench.iloc[0] - 1)
    payload = {
        "window": [str(start.date()), str(end.date())],
        "cut": CUT,
        "symbols": len(panel),
        "fixed_params": FIXED,
        "rows": rows,
        "benchmark_return": bench_return,
        "note": (
            "Weights are causal; the equity slice after the cut is evaluated "
            "without using any post-cut information for selection. One year of "
            "one market state is a weak test, kept because it contradicts the "
            "candidate rather than supporting it."
        ),
    }
    write_json(payload, OUT / "summary.json")
    pd.DataFrame(rows).to_csv(OUT / "comparison.csv", index=False, encoding="utf-8-sig")
    print(f"benchmark {bench_return:+.2%}")
    print(f"report: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
