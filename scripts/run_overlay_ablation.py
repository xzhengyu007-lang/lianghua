"""Benchmark-relative robustness: is the edge real after removing the
survivorship-friendly equal-weight comparison?"""

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
from quant_system.data import load_panel, load_symbol, price_frame, trim_panel, usable_window  # noqa: E402
from quant_system.metrics import performance_metrics  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.report import write_json  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402

PARAMS = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)


def monthly_alpha(returns: pd.Series, bench: pd.Series) -> pd.DataFrame:
    df = pd.concat([returns.rename("s"), bench.rename("b")], axis=1).dropna()
    grouped = df.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    grouped["active"] = grouped["s"] - grouped["b"]
    grouped["hit"] = grouped["active"] > 0
    return grouped


def main() -> int:
    cache = ROOT / "data" / "cache"
    panel = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    start, end = usable_window(panel, min_active_ratio=0.95)
    bench = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
    bench_ret = bench.pct_change(fill_method=None)

    variants = {
        "with_regime": dict(use_regime=True, use_vol_target=True),
        "no_regime": dict(use_regime=False, use_vol_target=True),
        "no_overlay": dict(use_regime=False, use_vol_target=False),
    }
    rows = {}
    for name, flags in variants.items():
        raw = build_strategy("low_vol_core", **PARAMS).generate(panel)
        weights = build_portfolio(raw, panel, benchmark=bench, risk=RiskConfig(), **flags)
        result = Backtester(BacktestConfig(initial_cash=1_000_000)).run(panel, weights, benchmark=bench)
        metrics = result.metrics.to_dict()
        alpha = monthly_alpha(result.returns, bench_ret)
        metrics["monthly_hit_rate"] = float(alpha["hit"].mean())
        metrics["monthly_active_mean"] = float(alpha["active"].mean())
        metrics["information_ratio"] = metrics["information_ratio"]
        rows[name] = metrics
        print(
            f"{name:12s} ret={metrics['total_return']:+.2%} cagr={metrics['cagr']:+.2%} "
            f"sharpe={metrics['sharpe']:.2f} mdd={metrics['max_drawdown']:.2%} "
            f"exp={metrics['exposure']:.1%} ir={metrics['information_ratio']:.2f} "
            f"monthly_hit={metrics['monthly_hit_rate']:.1%}"
        )
    output = ROOT / "reports" / "overlay_ablation"
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        {
            "window": {"start": str(start.date()), "end": str(end.date())},
            "params": PARAMS,
            "variants": rows,
            "benchmark": performance_metrics(bench_ret.dropna(), equity=bench).to_dict(),
        },
        output / "summary.json",
    )
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())