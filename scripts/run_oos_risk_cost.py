"""Out-of-sample sensitivity of volatility targeting and trading costs.

Two questions that published results must answer but full-sample sweeps cannot:

1. Is the volatility target robust, or did 18% happen to win on this sample?
2. At what cost does the edge disappear, holding the strategy fixed?

Both are evaluated with the same strictly paired 504/126/10 walk-forward
windows.  Risk configurations are chosen *inside* each training window, so the
test period never informs the choice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.config import (  # noqa: E402
    BROAD_UNIVERSE,
    BacktestConfig,
    CostConfig,
    ResearchConfig,
    RiskConfig,
)
from quant_system.data import load_panel, load_symbol, trim_panel  # noqa: E402
from quant_system.research import walk_forward  # noqa: E402
from quant_system.report import write_json  # noqa: E402

PARAMS = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)
RESEARCH = ResearchConfig(train_days=504, test_days=126, embargo_days=10)
OUT = ROOT / "reports" / "oos_risk_cost"

TARGETS = {
    "target_10": RiskConfig(target_annual_vol=0.10),
    "target_12": RiskConfig(target_annual_vol=0.12),
    "target_15": RiskConfig(target_annual_vol=0.15),
    "target_18": RiskConfig(target_annual_vol=0.18),
    "target_22": RiskConfig(target_annual_vol=0.22),
    "target_30": RiskConfig(target_annual_vol=0.30),
    "vol_off": RiskConfig(),
}


def _summary_row(label: str, result, extra: dict | None = None) -> dict:
    summary = result.summary()
    windows = result.windows
    row = {
        "variant": label,
        "total_return": summary["total_return"],
        "cagr": summary["cagr"],
        "sharpe": summary["sharpe"],
        "max_drawdown": summary["max_drawdown"],
        "exposure": summary["exposure"],
        "turnover": summary["turnover"],
        "positive_windows": int((windows["test_return"] > 0).sum()),
        "windows": int(len(windows)),
    }
    if extra:
        row.update(extra)
    return row


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "data" / "cache"
    panel = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    bench = load_symbol("sh000300", cache, adjust="")["close"]
    grid = {k: [v] for k, v in PARAMS.items()}
    print(f"symbols={len(panel)}", flush=True)

    rows: list[dict] = []
    windows_frames: dict[str, pd.DataFrame] = {}

    # 1. Fixed risk configuration, one run per target.  The strategy parameters
    #    are held fixed too; only the target changes.  This isolates the overlay.
    for name, risk in TARGETS.items():
        use_vol = name != "vol_off"
        result = walk_forward(
            panel,
            "low_vol_core",
            grid,
            benchmark=bench,
            research=RESEARCH,
            backtest=BacktestConfig(initial_cash=1_000_000),
            risk=risk,
            portfolio_kwargs={"use_vol_target": use_vol},
            verbose=False,
        )
        rows.append(_summary_row(name, result, {"kind": "fixed_target"}))
        windows_frames[name] = result.windows
        print(
            f"{name:12s} ret={rows[-1]['total_return']:+.2%} "
            f"sharpe={rows[-1]['sharpe']:.3f} mdd={rows[-1]['max_drawdown']:.2%} "
            f"pos={rows[-1]['positive_windows']}/{rows[-1]['windows']}",
            flush=True,
        )

    # 2. The target itself selected in each training window.  If the choice is
    #    unstable across windows, a fixed 18% is an artefact of the full sample.
    online = walk_forward(
        panel,
        "low_vol_core",
        grid,
        benchmark=bench,
        research=RESEARCH,
        backtest=BacktestConfig(initial_cash=1_000_000),
        risk=RiskConfig(),
        risk_variants={k: v for k, v in TARGETS.items() if k != "vol_off"},
        verbose=False,
    )
    rows.append(_summary_row("target_selected_in_training", online, {"kind": "selected"}))
    windows_frames["target_selected_in_training"] = online.windows
    print(
        f"selected     ret={rows[-1]['total_return']:+.2%} "
        f"sharpe={rows[-1]['sharpe']:.3f} "
        f"choices={online.selections}",
        flush=True,
    )

    # 3. Alternative volatility forecast: use the ex-ante covariance of the
    #    *current* basket instead of lagged realised P&L.  This was a weak
    #    full-sample improvement, so it is evaluated out of sample rather than
    #    promoted to the default on that basis.
    cov_result = walk_forward(
        panel,
        "low_vol_core",
        grid,
        benchmark=bench,
        research=RESEARCH,
        backtest=BacktestConfig(initial_cash=1_000_000),
        risk=RiskConfig(),
        portfolio_kwargs={"use_vol_target": True, "vol_forecast": "ex_ante"},
        verbose=False,
    )
    rows.append(_summary_row("vol_forecast_ex_ante", cov_result, {"kind": "forecast"}))
    windows_frames["vol_forecast_ex_ante"] = cov_result.windows
    print(
        f"ex_ante      ret={rows[-1]['total_return']:+.2%} "
        f"sharpe={rows[-1]['sharpe']:.3f} mdd={rows[-1]['max_drawdown']:.2%}",
        flush=True,
    )

    # 4. Cost curve.  Keep the default target and vary only slippage, so the
    #    reader can see the break-even round-trip cost.
    for slippage in (0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0):
        result = walk_forward(
            panel,
            "low_vol_core",
            grid,
            benchmark=bench,
            research=RESEARCH,
            backtest=BacktestConfig(
                initial_cash=1_000_000,
                costs=CostConfig(slippage_bps=slippage),
            ),
            risk=RiskConfig(),
            verbose=False,
        )
        rows.append(
            _summary_row(
                f"slippage_{slippage:g}bp",
                result,
                {"kind": "cost", "slippage_bps": slippage},
            )
        )
        windows_frames[f"slippage_{slippage:g}bp"] = result.windows
        print(
            f"slip={slippage:4.0f}bp ret={rows[-1]['total_return']:+.2%} "
            f"sharpe={rows[-1]['sharpe']:.3f}",
            flush=True,
        )

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    for name, frame in windows_frames.items():
        frame.to_csv(OUT / f"windows_{name}.csv", index=False, encoding="utf-8-sig")
    write_json(
        {
            "params": PARAMS,
            "research": RESEARCH.__dict__,
            "targets": {k: v.__dict__ for k, v in TARGETS.items()},
            "selected": online.selections,
            "rows": rows,
            "note": (
                "504/126/10 walk-forward; risk target and cost are varied one at a time; "
                "the selected variant is chosen only inside each training window."
            ),
        },
        OUT / "summary.json",
    )
    with pd.option_context("display.width", 200):
        print(
            table[
                [
                    "variant",
                    "total_return",
                    "sharpe",
                    "max_drawdown",
                    "exposure",
                    "positive_windows",
                    "windows",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )
    print(f"report: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
