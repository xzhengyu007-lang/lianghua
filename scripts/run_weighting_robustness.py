"""Robustness of inverse-vol sizing for the 6/2 low-vol book.

The first weighting run showed inverse-vol 120 lifting the 504/126/10
out-of-sample result from +126.5% to +137.8%. A single schedule is weak
evidence, so this script repeats the comparison across:

* five walk-forward window lengths,
* six shifted start dates,
* the point-in-time pool (survivors plus real delisted names),
* the full-sample cost sweep.

Only the sizing rule changes; selection, overlays and execution stay identical.
The output keeps every negative result.
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
    CostConfig,
    ResearchConfig,
    RiskConfig,
)
from quant_system.data import (  # noqa: E402
    align_panel,
    load_panel,
    load_symbol,
    pit_universe,
    trim_panel,
    usable_window,
)
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.report import write_json  # noqa: E402
from quant_system.research import walk_forward  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from run_weighting import inverse_vol_at_changes  # noqa: E402

PARAMS = dict(top_n=6, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)
WINDOWS = [(252, 63, 5), (378, 126, 10), (504, 126, 10), (630, 126, 10), (756, 126, 10)]
SHIFTS = [0, 20, 40, 60, 80, 100]
SLIPPAGE = [0, 5, 10, 20, 30, 50]
OUT = ROOT / "reports" / "weighting_robustness"


def transform(raw, panel):
    return inverse_vol_at_changes(raw, panel, 120)


def run(panel, benchmark, research, *, weight_transform=None, slippage=5.0):
    return walk_forward(
        panel,
        "low_vol_core",
        {key: [value] for key, value in PARAMS.items()},
        benchmark=benchmark,
        research=research,
        backtest=BacktestConfig(
            initial_cash=1_000_000,
            costs=CostConfig(slippage_bps=slippage),
        ),
        risk=RiskConfig(),
        portfolio_kwargs={"use_regime": False},
        weight_transform=weight_transform,
        verbose=False,
    )


def row(label: str, result, extra: dict | None = None) -> dict:
    summary = result.summary()
    data = {
        "variant": label,
        "total_return": summary["total_return"],
        "cagr": summary["cagr"],
        "annual_volatility": summary["annual_volatility"],
        "sharpe": summary["sharpe"],
        "max_drawdown": summary["max_drawdown"],
        "information_ratio": summary["information_ratio"],
        "exposure": summary["exposure"],
        "positive_windows": int((result.windows["test_return"] > 0).sum()),
        "windows": int(len(result.windows)),
    }
    if extra:
        data.update(extra)
    return data


def shift_start(base: ResearchConfig, dates: pd.DatetimeIndex, days: int) -> ResearchConfig:
    if days == 0:
        return base
    first = dates[base.train_days + base.embargo_days]
    pos = int(dates.searchsorted(first, side="left")) + days
    if pos >= len(dates):
        return base
    return ResearchConfig(
        start=str(dates[pos].date()),
        train_days=base.train_days,
        test_days=base.test_days,
        embargo_days=base.embargo_days,
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "data" / "cache"
    survivors = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    start, end = usable_window(survivors, min_active_ratio=0.95)
    benchmark = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
    survivors = {symbol: frame.loc[start:end] for symbol, frame in survivors.items()}
    aligned_survivors, dates = align_panel(survivors)
    print(f"window {start.date()}..{end.date()} symbols={len(survivors)}", flush=True)

    payload: dict = {
        "window": [str(start.date()), str(end.date())],
        "params": PARAMS,
        "weight_variant": "invvol_120",
        "note": "Only the sizing rule changes; selection and overlays are fixed.",
    }

    # 1. Window lengths.
    length_rows = []
    for train, test, embargo in WINDOWS:
        research = ResearchConfig(train_days=train, test_days=test, embargo_days=embargo)
        for label, fn in (("equal", None), ("invvol_120", transform)):
            result = run(aligned_survivors, benchmark, research, weight_transform=fn)
            length_rows.append(
                row(label, result, {"train_days": train, "test_days": test, "embargo_days": embargo})
            )
            print(
                f"{train}/{test}/{embargo} {label}: {length_rows[-1]['total_return']:+.2%} "
                f"sharpe={length_rows[-1]['sharpe']:.3f}",
                flush=True,
            )
    pd.DataFrame(length_rows).to_csv(OUT / "window_lengths.csv", index=False, encoding="utf-8-sig")
    payload["window_lengths"] = length_rows

    # 2. Start shifts.
    shift_rows = []
    base = ResearchConfig(train_days=504, test_days=126, embargo_days=10)
    for days in SHIFTS:
        research = shift_start(base, dates, days)
        for label, fn in (("equal", None), ("invvol_120", transform)):
            result = run(aligned_survivors, benchmark, research, weight_transform=fn)
            shift_rows.append(row(label, result, {"shift_days": days, "start": research.start}))
            print(
                f"shift {days:3d} {label}: {shift_rows[-1]['total_return']:+.2%} "
                f"sharpe={shift_rows[-1]['sharpe']:.3f}",
                flush=True,
            )
    pd.DataFrame(shift_rows).to_csv(OUT / "start_shifts.csv", index=False, encoding="utf-8-sig")
    payload["start_shifts"] = shift_rows

    # 3. Point-in-time pool, same first test date as the survivor run.
    usable_start, _ = usable_window(aligned_survivors, min_active_ratio=0.9)
    usable_pos = int(dates.searchsorted(usable_start, side="left"))
    first_test = dates[usable_pos + 504 + 10]
    research = ResearchConfig(
        start=str(first_test.date()), train_days=504, test_days=126, embargo_days=10
    )
    universe = pit_universe()
    delisted = {symbol: name for symbol, name in universe.items() if symbol not in BROAD_UNIVERSE}
    pit = load_panel(list(survivors) + list(delisted), cache)
    pit = {
        symbol: frame.loc[start:end]
        for symbol, frame in pit.items()
        if len(frame.loc[start:end])
    }
    aligned_pit, _ = align_panel(pit)
    pit_rows = []
    for label, fn in (("equal", None), ("invvol_120", transform)):
        result = run(aligned_pit, benchmark, research, weight_transform=fn)
        pit_rows.append(row(label, result, {"pool": "point-in-time", "symbols": len(pit)}))
        print(
            f"pit {label}: {pit_rows[-1]['total_return']:+.2%} "
            f"sharpe={pit_rows[-1]['sharpe']:.3f}",
            flush=True,
        )
    pd.DataFrame(pit_rows).to_csv(OUT / "pit.csv", index=False, encoding="utf-8-sig")
    payload["point_in_time"] = pit_rows

    # 4. Full-sample cost sweep.
    cost_rows = []
    for slippage in SLIPPAGE:
        for label, fn in (("equal", None), ("invvol_120", transform)):
            raw = build_strategy("low_vol_core", **PARAMS).generate(aligned_survivors)
            sized = fn(raw, aligned_survivors) if fn else raw
            weights = build_portfolio(
                sized, aligned_survivors, benchmark=benchmark, risk=RiskConfig(), use_regime=False
            )
            result = Backtester(
                BacktestConfig(
                    initial_cash=1_000_000,
                    costs=CostConfig(slippage_bps=slippage),
                )
            ).run(aligned_survivors, weights, benchmark=benchmark)
            metrics = result.metrics.to_dict()
            cost_rows.append(
                {
                    "variant": label,
                    "slippage_bps": slippage,
                    "total_return": metrics["total_return"],
                    "sharpe": metrics["sharpe"],
                    "max_drawdown": metrics["max_drawdown"],
                }
            )
            print(
                f"{slippage:2d}bp {label}: {metrics['total_return']:+.2%} "
                f"sharpe={metrics['sharpe']:.3f}",
                flush=True,
            )
    pd.DataFrame(cost_rows).to_csv(OUT / "cost_sweep.csv", index=False, encoding="utf-8-sig")
    payload["cost_sweep"] = cost_rows

    write_json(payload, OUT / "summary.json")
    print(f"report: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
