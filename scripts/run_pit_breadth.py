"""Point-in-time comparison of fixed top_n=6 vs top_n=8.

The survivor-only panel says a narrower book is better. The point-in-time pool
adds real delisted names, so it is the more honest membership test. This script
holds every other knob fixed, uses one shared trading calendar and identical
504/126/10 test windows for all four arms:

    survivor-only  x  {6/2, 8/2}
    point-in-time  x  {6/2, 8/2}

It also writes a descriptive full-sample comparison, clearly separated from the
walk-forward result. No default is changed here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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

FIXED = {"vol_window": 60, "trend_window": 120, "rebalance_days": 20}
WIDTHS = {"top_n=6,cap=2": (6, 2), "top_n=8,cap=2": (8, 2)}
OUT = ROOT / "reports" / "pit_breadth_6_vs_8"


def params(top_n: int, cap: int) -> dict:
    return {**FIXED, "top_n": top_n, "max_per_sector": cap}


def same_calendar(panel: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    return {symbol: frame.loc[frame.index.isin(calendar)] for symbol, frame in panel.items()}


def full_sample(panel, benchmark, param: dict) -> dict:
    raw = build_strategy("low_vol_core", **param).generate(panel)
    weights = build_portfolio(raw, panel, benchmark=benchmark, risk=RiskConfig(), use_regime=False)
    result = Backtester(BacktestConfig(initial_cash=1_000_000)).run(panel, weights, benchmark=benchmark)
    return result.metrics.to_dict()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "data" / "cache"
    survivors = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    start, end = usable_window(survivors, min_active_ratio=0.95)
    benchmark = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
    calendar = pd.DatetimeIndex(benchmark.index)

    survivors = same_calendar(
        {symbol: frame.loc[start:end] for symbol, frame in survivors.items()}, calendar
    )
    universe = pit_universe()
    delisted = {symbol: name for symbol, name in universe.items() if symbol not in BROAD_UNIVERSE}
    pit_raw = load_panel(list(survivors) + list(delisted), cache)
    pit = same_calendar(
        {
            symbol: frame.loc[start:end]
            for symbol, frame in pit_raw.items()
            if len(frame.loc[start:end])
        },
        calendar,
    )
    print(
        f"window {start.date()}..{end.date()} survivor={len(survivors)} "
        f"pit={len(pit)} extra={len(pit) - len(survivors)}",
        flush=True,
    )

    # Both arms must start their first test window on the same bar. Without an
    # explicit start, walk_forward derives it from each arm's own date union.
    aligned_survivors, survivor_dates = align_panel(survivors)
    usable_start, _ = usable_window(aligned_survivors, min_active_ratio=0.9)
    usable_pos = int(survivor_dates.searchsorted(usable_start, side="left"))
    first_test = survivor_dates[usable_pos + 504 + 10]
    research = ResearchConfig(
        start=str(first_test.date()),
        train_days=504,
        test_days=126,
        embargo_days=10,
    )
    print(f"first test {first_test.date()}", flush=True)

    pools = {"survivor-only": survivors, "point-in-time": pit}
    rows = []
    windows_frames = []
    payload: dict = {
        "window": [str(start.date()), str(end.date())],
        "first_test": str(first_test.date()),
        "survivors": len(survivors),
        "pit_symbols": len(pit),
        "delisted_extra": len(pit) - len(survivors),
        "research": {
            "train_days": 504,
            "test_days": 126,
            "embargo_days": 10,
        },
        "fixed_params": FIXED,
    }

    for pool_name, panel in pools.items():
        aligned, _ = align_panel(panel)
        for width_name, (top_n, cap) in WIDTHS.items():
            started = time.time()
            result = walk_forward(
                aligned,
                "low_vol_core",
                {key: [value] for key, value in params(top_n, cap).items()},
                benchmark=benchmark,
                research=research,
                backtest=BacktestConfig(initial_cash=1_000_000),
                risk=RiskConfig(),
                portfolio_kwargs={"use_regime": False},
                verbose=False,
            )
            summary = result.summary()
            row = {
                "pool": pool_name,
                "width": width_name,
                "top_n": top_n,
                "max_per_sector": cap,
                "total_return": summary["total_return"],
                "cagr": summary["cagr"],
                "annual_volatility": summary["annual_volatility"],
                "sharpe": summary["sharpe"],
                "max_drawdown": summary["max_drawdown"],
                "information_ratio": summary["information_ratio"],
                "exposure": summary["exposure"],
                "positive_windows": int((result.windows["test_return"] > 0).sum()),
                "windows": int(len(result.windows)),
                "elapsed_seconds": round(time.time() - started, 1),
            }
            rows.append(row)
            frame = result.windows.copy()
            frame["pool"] = pool_name
            frame["width"] = width_name
            windows_frames.append(frame)
            result.equity.to_frame("equity").to_csv(
                OUT / f"equity_{pool_name}_{width_name.replace(',', '_').replace('=', '')}.csv"
            )
            print(
                f"{pool_name:14s} {width_name}: ret={row['total_return']:+.2%} "
                f"sharpe={row['sharpe']:.3f} mdd={row['max_drawdown']:.2%} "
                f"hit={row['positive_windows']}/{row['windows']}",
                flush=True,
            )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "comparison.csv", index=False, encoding="utf-8-sig")
    all_windows = pd.concat(windows_frames, ignore_index=True)
    all_windows.to_csv(OUT / "windows.csv", index=False, encoding="utf-8-sig")
    payload["walkforward"] = rows

    # Paired per-window differences: the same dates, only membership or width
    # changes. Report both contrasts and do not hide the sign.
    paired = all_windows.pivot_table(
        index=["test_start", "test_end"],
        columns=["pool", "width"],
        values="test_return",
        aggfunc="first",
    )
    paired_rows = []
    contrasts = [
        ("width_6_minus_8_survivor", ("survivor-only", "top_n=6,cap=2"), ("survivor-only", "top_n=8,cap=2")),
        ("width_6_minus_8_pit", ("point-in-time", "top_n=6,cap=2"), ("point-in-time", "top_n=8,cap=2")),
        ("pit_minus_survivor_6", ("point-in-time", "top_n=6,cap=2"), ("survivor-only", "top_n=6,cap=2")),
        ("pit_minus_survivor_8", ("point-in-time", "top_n=8,cap=2"), ("survivor-only", "top_n=8,cap=2")),
    ]
    for label, left, right in contrasts:
        if left not in paired.columns or right not in paired.columns:
            continue
        diff = (paired[left] - paired[right]).dropna()
        if diff.empty:
            continue
        mean = float(diff.mean())
        std = float(diff.std(ddof=1)) if len(diff) > 1 else float("nan")
        paired_rows.append(
            {
                "contrast": label,
                "windows": int(len(diff)),
                "mean_difference": mean,
                "std_difference": std,
                "t_like": mean / std * (len(diff) ** 0.5) if std and std > 0 else float("nan"),
                "positive_windows": int((diff > 0).sum()),
            }
        )
    paired_table = pd.DataFrame(paired_rows)
    paired_table.to_csv(OUT / "paired_differences.csv", index=False, encoding="utf-8-sig")
    payload["paired_differences"] = paired_rows

    # Descriptive full-sample table. This is not a substitute for the OOS test.
    full_rows = []
    for pool_name, panel in pools.items():
        for width_name, (top_n, cap) in WIDTHS.items():
            metrics = full_sample(panel, benchmark, params(top_n, cap))
            full_rows.append(
                {
                    "pool": pool_name,
                    "width": width_name,
                    "total_return": metrics["total_return"],
                    "cagr": metrics["cagr"],
                    "sharpe": metrics["sharpe"],
                    "max_drawdown": metrics["max_drawdown"],
                    "turnover": metrics["turnover"],
                }
            )
            print(
                f"full {pool_name:14s} {width_name}: {metrics['total_return']:+.2%} "
                f"sharpe={metrics['sharpe']:.3f}",
                flush=True,
            )
    pd.DataFrame(full_rows).to_csv(OUT / "full_sample.csv", index=False, encoding="utf-8-sig")
    payload["full_sample"] = full_rows

    write_json(payload, OUT / "summary.json")
    print(comparison.round(4).to_string(index=False))
    if len(paired_table):
        print(paired_table.round(4).to_string(index=False))
    print(f"report: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
