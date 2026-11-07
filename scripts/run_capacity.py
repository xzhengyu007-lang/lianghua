"""Cost and capacity sensitivity on the full-history broad universe.

Rebuilt after the hfq data repair and after the regime overlay was switched off
by default, so the numbers here match the reported baseline.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

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
INITIAL_CASH = 1_000_000.0


def build_weights(panel, benchmark):
    raw = build_strategy("low_vol_core", **PARAMS).generate(panel)
    return build_portfolio(
        raw,
        panel,
        benchmark=benchmark,
        risk=RiskConfig(),
        use_regime=False,
        use_vol_target=True,
    )


def main() -> int:
    cache = ROOT / "data" / "cache"
    panel = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    start, end = usable_window(panel, min_active_ratio=0.95)
    benchmark = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
    weights = build_weights(panel, benchmark)
    print(f"window {start.date()}..{end.date()} symbols={len(panel)}", flush=True)

    base_costs = CostConfig()
    rows = []
    for slippage in (0.0, 5.0, 10.0, 20.0, 40.0):
        for commission in (0.0, 0.00025, 0.0005):
            for stamp in (base_costs.stamp_duty_rate, base_costs.stamp_duty_rate * 2):
                cfg = BacktestConfig(
                    initial_cash=INITIAL_CASH,
                    costs=CostConfig(
                        commission_rate=commission,
                        min_commission=5.0 if commission else 0.0,
                        stamp_duty_rate=stamp,
                        transfer_fee_rate=base_costs.transfer_fee_rate,
                        slippage_bps=slippage,
                    ),
                )
                metrics = Backtester(cfg).run(panel, weights, benchmark=benchmark).metrics.to_dict()
                rows.append(
                    {
                        "slippage_bps": slippage,
                        "commission_rate": commission,
                        "stamp_duty_rate": stamp,
                        "total_return": metrics["total_return"],
                        "cagr": metrics["cagr"],
                        "sharpe": metrics["sharpe"],
                        "max_drawdown": metrics["max_drawdown"],
                        "turnover": metrics["turnover"],
                    }
                )
            print(f"costs slippage={slippage} commission={commission}", flush=True)

    capacity = []
    for participation in (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0):
        for cash in (1_000_000, 10_000_000, 50_000_000, 200_000_000):
            cfg = BacktestConfig(initial_cash=cash, max_adv_participation=participation)
            result = Backtester(cfg).run(panel, weights, benchmark=benchmark)
            metrics = result.metrics.to_dict()
            capacity.append(
                {
                    "participation": participation,
                    "cash": cash,
                    "total_return": metrics["total_return"],
                    "cagr": metrics["cagr"],
                    "sharpe": metrics["sharpe"],
                    "max_drawdown": metrics["max_drawdown"],
                    "exposure": metrics["exposure"],
                    "trades": result.metadata["trades"],
                }
            )
            print(f"capacity cash={cash:,.0f} participation={participation}", flush=True)

    output = ROOT / "reports" / "capacity_v2"
    output.mkdir(parents=True, exist_ok=True)
    costs = pd.DataFrame(rows)
    cap = pd.DataFrame(capacity)
    costs.to_csv(output / "costs.csv", index=False, encoding="utf-8-sig")
    cap.to_csv(output / "capacity.csv", index=False, encoding="utf-8-sig")
    baseline = Backtester(BacktestConfig(initial_cash=INITIAL_CASH)).run(
        panel, weights, benchmark=benchmark
    )
    summary = {
        "params": PARAMS,
        "symbols": len(panel),
        "window": [str(start.date()), str(end.date())],
        "regime_overlay": False,
        "base": baseline.metrics.to_dict(),
        "costs": rows,
        "capacity": capacity,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(costs.to_string(index=False))
    print(cap.to_string(index=False))
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
