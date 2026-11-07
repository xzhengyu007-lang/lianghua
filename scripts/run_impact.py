"""How much does a nonlinear impact model cost the strategy?

Impact is
    impact_bps = coefficient * sqrt(shares / ADV)
so it grows with the square root of participation, which is the shape the
empirical market-impact literature reports. The coefficient is the only free
parameter; 40-60 bps at 100% of ADV is in the published range for A-share
large caps. Realised participation in this strategy is far below 100%, so the
model adds only a few bps per order at 1M-50M book size.
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
    CostConfig,
    RiskConfig,
)
from quant_system.data import load_panel, load_symbol, trim_panel, usable_window  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402

PARAMS = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)


def main() -> int:
    cache = ROOT / "data" / "cache"
    panel = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    start, end = usable_window(panel, min_active_ratio=0.95)
    bench = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
    raw = build_strategy("low_vol_core", **PARAMS).generate(panel)
    weights = build_portfolio(raw, panel, benchmark=bench, risk=RiskConfig(), use_regime=False)
    print(f"window {start.date()}..{end.date()} symbols={len(panel)}", flush=True)

    rows = []
    for coefficient in (0.0, 25.0, 50.0, 100.0, 200.0):
        for cash in (1_000_000, 50_000_000):
            cfg = BacktestConfig(
                initial_cash=cash,
                costs=CostConfig(impact_coefficient_bps=coefficient),
            )
            result = Backtester(cfg).run(panel, weights, benchmark=bench)
            metrics = result.metrics.to_dict()
            rows.append(
                {
                    "impact_coefficient_bps": coefficient,
                    "cash": cash,
                    **{k: metrics[k] for k in ("total_return", "cagr", "sharpe", "max_drawdown", "turnover")},
                    "trades": result.metadata["trades"],
                }
            )
            print(
                f"impact={coefficient:5.1f} cash={cash:,.0f} total={metrics['total_return']:+.3f} "
                f"sharpe={metrics['sharpe']:.3f}",
                flush=True,
            )

    table = pd.DataFrame(rows)
    out = ROOT / "reports" / "impact"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "impact_sensitivity.csv", index=False, encoding="utf-8-sig")

    (out / "summary.json").write_text(
        json.dumps(
            {
                "params": PARAMS,
                "window": [str(start.date()), str(end.date())],
                "model": "impact_bps = coefficient * sqrt(shares / ADV)",
                "rows": rows,
                "note": "coefficient 0 is the published baseline; 50bps is the central estimate",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with pd.option_context("display.width", 200):
        print(table.round(4).to_string(index=False))
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
