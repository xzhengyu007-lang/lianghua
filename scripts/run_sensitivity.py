"""Portfolio-construction sensitivity: risk budget, sizing, and turnover caps.

Complements `run_capacity.py` (frictions and market impact) and
`run_robustness.py` (signal parameter plateau). This one asks how much the
default configuration depends on the risk overlay settings themselves.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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


def main() -> int:
    cache = ROOT / "data" / "cache"
    panel = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    start, end = usable_window(panel, min_active_ratio=0.95)
    bench = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
    raw = build_strategy("low_vol_core", **PARAMS).generate(panel)
    print(f"window {start.date()}..{end.date()} symbols={len(panel)}", flush=True)

    # turnover_cap=False matches the published default, so `target_vol` and
    # `vol_lookback` sweeps move around the real baseline rather than around a
    # configuration nobody runs.
    def evaluate(risk, *, vol_target=True, turnover_cap=False, cash=1_000_000.0, **bt):
        weights = build_portfolio(
            raw,
            panel,
            benchmark=bench,
            risk=risk,
            use_regime=False,
            use_vol_target=vol_target,
            use_turnover_cap=turnover_cap,
        )
        return Backtester(BacktestConfig(initial_cash=cash, **bt)).run(
            panel, weights, benchmark=bench
        ).metrics.to_dict()

    rows = []
    for target_vol in (0.10, 0.12, 0.15, 0.18, 0.22, 0.30):
        metrics = evaluate(RiskConfig(target_annual_vol=target_vol))
        rows.append({"experiment": "target_vol", "value": target_vol, **metrics})
        print(f"target_vol={target_vol}", flush=True)

    for lookback in (20, 40, 60, 90):
        metrics = evaluate(RiskConfig(vol_lookback=lookback))
        rows.append({"experiment": "vol_lookback", "value": lookback, **metrics})
        print(f"vol_lookback={lookback}", flush=True)

    # The guard lives in the execution layer (BacktestConfig.drawdown_stop), not in
    # RiskConfig: RiskConfig.max_drawdown_stop only feeds the advisory warning text.
    # Sweeping the latter used to produce five identical rows, which is how the
    # dead knob was discovered.
    for stop in (0.10, 0.15, 0.20, 0.25, 0.40):
        metrics = evaluate(RiskConfig(), drawdown_stop=stop)
        rows.append({"experiment": "drawdown_stop", "value": stop, **metrics})
        print(f"drawdown_stop={stop}", flush=True)

    # The daily-turnover threshold only binds inside the (default-off) turnover
    # cap, so these rows are measured with the cap explicitly enabled.
    for cap in (0.15, 0.25, 0.35, 0.50, 1.00):
        metrics = evaluate(RiskConfig(max_daily_turnover=cap), turnover_cap=True)
        rows.append({"experiment": "max_daily_turnover_capped", "value": cap, **metrics})
        print(f"max_daily_turnover={cap}", flush=True)

    metrics = evaluate(RiskConfig(), vol_target=False)
    rows.append({"experiment": "vol_target_off", "value": 0, **metrics})
    metrics = evaluate(RiskConfig(), turnover_cap=False)
    rows.append({"experiment": "turnover_cap_off", "value": 0, **metrics})
    metrics = evaluate(RiskConfig(), turnover_cap=True)
    rows.append({"experiment": "turnover_cap_on", "value": 0, **metrics})
    metrics = evaluate(RiskConfig(), vol_target=False, turnover_cap=False)
    rows.append({"experiment": "both_off", "value": 0, **metrics})
    print("overlay ablations done", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "reports" / "sensitivity_v2" / "risk_overlay.csv", index=False, encoding="utf-8-sig")
    summary = {
        "params": PARAMS,
        "window": [str(start.date()), str(end.date())],
        "regime_overlay": False,
        "rows": rows,
    }
    (ROOT / "reports" / "sensitivity_v2" / "risk_overlay.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with pd.option_context("display.width", 200):
        print(
            table[
                ["experiment", "value", "total_return", "cagr", "sharpe", "max_drawdown", "exposure", "turnover"]
            ].round(4).to_string(index=False)
        )
    print("report: reports/sensitivity_v2/risk_overlay.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
