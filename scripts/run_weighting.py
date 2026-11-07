"""Does the construction rule add anything after the basket is chosen?

The default is equal weight. With only six names, equal weight is a strong
implicit bet that all six have the same risk. This script holds the selection
engine fixed and changes only how the chosen names are sized:

* equal weight (baseline),
* inverse volatility over 40/60/120 bars, locked at each basket change,
* inverse volatility with a 15% single-name cap.

Weights are recomputed only when the basket changes, so the comparison does not
silently turn into a daily rebalancing strategy. Both the full sample and the
504/126/10 walk-forward are reported; the walk-forward is the decision metric.
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
    ResearchConfig,
    RiskConfig,
)
from quant_system.data import (  # noqa: E402
    load_panel,
    load_symbol,
    price_frame,
    trim_panel,
    usable_window,
)
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.report import write_json  # noqa: E402
from quant_system.research import walk_forward  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402

PARAMS = dict(top_n=6, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)
RESEARCH = ResearchConfig(train_days=504, test_days=126, embargo_days=10)
OUT = ROOT / "reports" / "weighting"


def inverse_vol_at_changes(
    selection: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    window: int,
    *,
    max_weight: float | None = None,
) -> pd.DataFrame:
    """Size selected names by 1/vol, recomputed only on basket changes.

    The selection matrix is constant between rebalances. Detecting rows where
    the set of held names changes and forward-filling the resulting weights
    keeps turnover tied to the strategy cadence rather than to daily vol noise.
    """
    held = selection.gt(0)
    changed = held.ne(held.shift(1)).any(axis=1)
    changed.iloc[0] = True
    close = price_frame(panel, "close").reindex(columns=selection.columns)
    vol = close.pct_change(fill_method=None).rolling(window).std(ddof=0)
    vol = vol.replace(0, np.nan)
    raw = (1.0 / vol).where(held)
    raw = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    if max_weight is not None:
        raw = raw.clip(upper=max_weight)
        raw = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    result = raw.where(changed).ffill().fillna(0.0)
    return result


def evaluate(panel, benchmark, weights, *, slippage: float = 5.0):
    from quant_system.config import CostConfig

    cfg = BacktestConfig(
        initial_cash=1_000_000,
        costs=CostConfig(slippage_bps=slippage),
    )
    return Backtester(cfg).run(panel, weights, benchmark=benchmark)


def variants(panel, benchmark) -> dict[str, pd.DataFrame]:
    raw = build_strategy("low_vol_core", **PARAMS).generate(panel)
    out = {
        "equal": raw,
        "invvol_40": inverse_vol_at_changes(raw, panel, 40),
        "invvol_60": inverse_vol_at_changes(raw, panel, 60),
        "invvol_120": inverse_vol_at_changes(raw, panel, 120),
        "invvol_60_cap15": inverse_vol_at_changes(raw, panel, 60, max_weight=0.15),
    }
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "data" / "cache"
    panel = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    start, end = usable_window(panel, min_active_ratio=0.95)
    panel = {symbol: frame.loc[start:end] for symbol, frame in panel.items()}
    benchmark = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
    print(f"window {start.date()}..{end.date()} symbols={len(panel)}", flush=True)

    rows = []
    for label, raw in variants(panel, benchmark).items():
        weights = build_portfolio(raw, panel, benchmark=benchmark, risk=RiskConfig(), use_regime=False)
        result = evaluate(panel, benchmark, weights)
        metrics = result.metrics.to_dict()
        rows.append(
            {
                "variant": label,
                "mode": "full_sample",
                "total_return": metrics["total_return"],
                "cagr": metrics["cagr"],
                "annual_volatility": metrics["annual_volatility"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "turnover": metrics["turnover"],
                "exposure": metrics["exposure"],
            }
        )
        result.equity.to_frame("equity").to_csv(OUT / f"equity_full_{label}.csv")
        print(
            f"full {label:18s} ret={metrics['total_return']:+.2%} "
            f"sharpe={metrics['sharpe']:.3f} mdd={metrics['max_drawdown']:.2%}",
            flush=True,
        )

    # Walk-forward: the selection is rebuilt inside every window, so the
    # weighting rule must be applied to the window-specific basket. The
    # ``weight_transform`` hook does that after selection and before overlays,
    # which keeps the comparison identical to the equal-weight path.
    transforms = {
        "invvol_40": lambda raw, hist: inverse_vol_at_changes(raw, hist, 40),
        "invvol_60": lambda raw, hist: inverse_vol_at_changes(raw, hist, 60),
        "invvol_120": lambda raw, hist: inverse_vol_at_changes(raw, hist, 120),
        "invvol_60_cap15": lambda raw, hist: inverse_vol_at_changes(raw, hist, 60, max_weight=0.15),
    }
    for label, transform in transforms.items():
        result = walk_forward(
            panel,
            "low_vol_core",
            {key: [value] for key, value in PARAMS.items()},
            benchmark=benchmark,
            research=RESEARCH,
            backtest=BacktestConfig(initial_cash=1_000_000),
            risk=RiskConfig(),
            portfolio_kwargs={"use_regime": False},
            weight_transform=transform,
            verbose=False,
        )
        summary = result.summary()
        rows.append(
            {
                "variant": label,
                "mode": "walkforward",
                "total_return": summary["total_return"],
                "cagr": summary["cagr"],
                "annual_volatility": summary["annual_volatility"],
                "sharpe": summary["sharpe"],
                "max_drawdown": summary["max_drawdown"],
                "turnover": summary["turnover"],
                "exposure": summary["exposure"],
            }
        )
        result.equity.to_frame("equity").to_csv(OUT / f"equity_oos_{label}.csv")
        print(
            f"oos  {label:18s} ret={summary['total_return']:+.2%} "
            f"sharpe={summary['sharpe']:.3f} mdd={summary['max_drawdown']:.2%}",
            flush=True,
        )

    # Equal-weight OOS from the library, so the baseline is computed by the same
    # path as every other published walk-forward number.
    equal = walk_forward(
        panel,
        "low_vol_core",
        {key: [value] for key, value in PARAMS.items()},
        benchmark=benchmark,
        research=RESEARCH,
        backtest=BacktestConfig(initial_cash=1_000_000),
        risk=RiskConfig(),
        portfolio_kwargs={"use_regime": False},
        verbose=False,
    )
    summary = equal.summary()
    rows.append(
        {
            "variant": "equal",
            "mode": "walkforward",
            "total_return": summary["total_return"],
            "cagr": summary["cagr"],
            "annual_volatility": summary["annual_volatility"],
            "sharpe": summary["sharpe"],
            "max_drawdown": summary["max_drawdown"],
            "turnover": summary["turnover"],
            "exposure": summary["exposure"],
        }
    )
    print(
        f"oos  {'equal':18s} ret={summary['total_return']:+.2%} "
        f"sharpe={summary['sharpe']:.3f} mdd={summary['max_drawdown']:.2%}",
        flush=True,
    )
    equal.equity.to_frame("equity").to_csv(OUT / "equity_oos_equal.csv")

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "comparison.csv", index=False, encoding="utf-8-sig")
    write_json(
        {
            "window": [str(start.date()), str(end.date())],
            "params": PARAMS,
            "research": {"train_days": 504, "test_days": 126, "embargo_days": 10},
            "rows": rows,
            "note": (
                "Inverse-vol weights are recomputed only when the basket changes. "
                "Walk-forward is the decision metric; full sample is descriptive."
            ),
        },
        OUT / "summary.json",
    )
    with pd.option_context("display.width", 200):
        print(table.round(4).to_string(index=False))
    print(f"report: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
