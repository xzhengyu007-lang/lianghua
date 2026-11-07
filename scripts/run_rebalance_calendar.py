"""Rebalance calendar: fixed 20-bar cadence vs calendar month-end.

The strategy rebalances every ``rebalance_days`` bars starting from the first
eligible date. That is an arbitrary calendar anchor: a live book would more
naturally rebalance at month-end. This script separates two questions:

1. Does the *phase* of the 20-bar cycle matter? The same cadence is started at
   six different offsets, which is a direct test for calendar luck.
2. Does month-end rebalancing (every ~21 bars, anchored on the last trading day
   of each month) beat the fixed cadence? This is what a discretionary desk
   would actually do.

Everything is evaluated on the same 504/126/10 walk-forward windows, plus the
fully held-out final year. No default is changed.
"""

from __future__ import annotations

import sys
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
    trim_panel,
    usable_window,
)
from quant_system.metrics import performance_metrics  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.report import write_json  # noqa: E402
from quant_system.research import walk_forward  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402

FIXED = {"vol_window": 60, "trend_window": 120, "max_per_sector": 2, "rebalance_days": 20}
CUT = "2025-11-05"
OUT = ROOT / "reports" / "rebalance_calendar"


def month_end_weights(panel, top_n: int) -> pd.DataFrame:
    """Same engine, but refresh the basket only on each month's last bar."""
    return build_strategy(
        "low_vol_core",
        **{**FIXED, "top_n": top_n},
        rebalance_on="month_end",
    ).generate(panel)


def run_walk(panel, benchmark, top_n: int, research: ResearchConfig, *, month_end: bool):
    if not month_end:
        grid = {
            "vol_window": [FIXED["vol_window"]],
            "trend_window": [FIXED["trend_window"]],
            "rebalance_days": [FIXED["rebalance_days"]],
            "max_per_sector": [FIXED["max_per_sector"]],
            "top_n": [top_n],
        }
        return walk_forward(
            panel, "low_vol_core", grid, benchmark=benchmark, research=research,
            backtest=BacktestConfig(initial_cash=1_000_000), risk=RiskConfig(),
            portfolio_kwargs={"use_regime": False}, verbose=False,
        )
    grid = {
        "vol_window": [FIXED["vol_window"]],
        "trend_window": [FIXED["trend_window"]],
        "max_per_sector": [FIXED["max_per_sector"]],
        "top_n": [top_n],
        "rebalance_on": ["month_end"],
    }
    return walk_forward(
        panel, "low_vol_core", grid, benchmark=benchmark, research=research,
        backtest=BacktestConfig(initial_cash=1_000_000), risk=RiskConfig(),
        portfolio_kwargs={"use_regime": False}, verbose=False,
    )


def summarise(label: str, result, extra: dict | None = None) -> dict:
    summary = result.summary()
    windows = result.windows
    row = {
        "variant": label,
        "total_return": summary["total_return"],
        "cagr": summary["cagr"],
        "annual_volatility": summary["annual_volatility"],
        "sharpe": summary["sharpe"],
        "max_drawdown": summary["max_drawdown"],
        "information_ratio": summary["information_ratio"],
        "exposure": summary["exposure"],
        "positive_windows": int((windows["test_return"] > 0).sum()),
        "windows": int(len(windows)),
    }
    if extra:
        row.update(extra)
    return row


def heldout(panel, benchmark, cut: pd.Timestamp, top_n: int, month_end: bool) -> dict:
    if month_end:
        raw = month_end_weights(panel, top_n)
    else:
        raw = build_strategy("low_vol_core", **{**FIXED, "top_n": top_n}).generate(panel)
    weights = build_portfolio(raw, panel, benchmark=benchmark, risk=RiskConfig(), use_regime=False)
    result = Backtester(BacktestConfig(initial_cash=1_000_000)).run(panel, weights, benchmark=benchmark)
    segment = result.equity.loc[cut:]
    metrics = performance_metrics(segment.pct_change(fill_method=None).dropna(), equity=segment)
    return {
        "total_return": metrics.total_return,
        "sharpe": metrics.sharpe,
        "max_drawdown": metrics.max_drawdown,
        "bars": len(segment),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "data" / "cache"
    panel = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    start, end = usable_window(panel, min_active_ratio=0.95)
    panel = {s: f.loc[start:end] for s, f in panel.items()}
    benchmark = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
    aligned, dates = align_panel(panel)
    usable_start, _ = usable_window(aligned, min_active_ratio=0.9)
    usable_pos = int(dates.searchsorted(usable_start, side="left"))
    first_test = dates[usable_pos + 504 + 10]
    research = ResearchConfig(start=str(first_test.date()), train_days=504, test_days=126, embargo_days=10)
    print(f"window {start.date()}..{end.date()} symbols={len(panel)} first_test={first_test.date()}", flush=True)

    payload: dict = {
        "window": [str(start.date()), str(end.date())],
        "first_test": str(first_test.date()),
        "fixed_params": FIXED,
        "cut": CUT,
        "note": (
            "Month-end arms refresh the basket only on the last bar of a month "
            "and hold it frozen in between. Walk-forward windows are strictly "
            "out of sample; the held-out year uses pre-cut information only."
        ),
    }

    print("walk-forward", flush=True)
    rows, frames = [], []
    arms = [("8/2 every 20 bars", 8, False), ("8/2 month-end", 8, True),
            ("6/2 every 20 bars", 6, False), ("6/2 month-end", 6, True)]
    for label, top_n, month_end in arms:
        result = run_walk(panel, benchmark, top_n, research, month_end=month_end)
        row = summarise(label, result, {"top_n": top_n, "month_end": month_end})
        rows.append(row)
        frame = result.windows.copy()
        frame["variant"] = label
        frames.append(frame)
        print(f"{label}: {row['total_return']:+.2%} sharpe={row['sharpe']:.3f} "
              f"mdd={row['max_drawdown']:.2%} pos={row['positive_windows']}/{row['windows']}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "walkforward.csv", index=False, encoding="utf-8-sig")
    detail = pd.concat(frames, ignore_index=True)
    detail.to_csv(OUT / "windows.csv", index=False, encoding="utf-8-sig")
    payload["walkforward"] = rows

    pivot = detail.pivot(index="test_start", columns="variant", values="test_return")
    paired = {}
    for me, base in (("8/2 month-end", "8/2 every 20 bars"), ("6/2 month-end", "6/2 every 20 bars")):
        diff = (pivot[me] - pivot[base]).dropna()
        paired[f"{me} - {base}"] = {
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std(ddof=1)),
            "t_like": float(diff.mean() / diff.std(ddof=1) * len(diff) ** 0.5) if diff.std(ddof=1) else None,
            "wins": int((diff > 0).sum()),
            "n": int(len(diff)),
        }
    payload["paired_month_end_minus_fixed"] = paired

    print("held-out final year", flush=True)
    cut = pd.Timestamp(CUT)
    hold_rows = []
    for label, top_n, month_end in arms:
        metrics = heldout(panel, benchmark, cut, top_n, month_end)
        hold_rows.append({"variant": label, **metrics})
        print(f"heldout {label}: {metrics['total_return']:+.2%} sharpe={metrics['sharpe']:.3f} "
              f"mdd={metrics['max_drawdown']:.2%}", flush=True)
    pd.DataFrame(hold_rows).to_csv(OUT / "heldout.csv", index=False, encoding="utf-8-sig")
    payload["heldout"] = hold_rows
    bench = benchmark.loc[cut:]
    payload["benchmark_heldout_return"] = float(bench.iloc[-1] / bench.iloc[0] - 1)

    write_json(payload, OUT / "summary.json")
    print(f"report: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
