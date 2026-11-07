"""Out-of-sample test of the rank buffer (the "hold longer" knob).

The full-sample sweep in section 23.3 reported buffer_rank=2.0 at +230.9% /
Sharpe 1.168 for the 6/2 candidate, the best point in a local neighbourhood.
A single full-sample optimum is exactly what curve-fitting looks like, so this
script re-tests it where it matters:

* fixed 6/2 and 8/2 under 504/126/10 walk-forward windows,
* the same arms on the point-in-time pool (survivors + real delisted names),
* a fully held-out final year using only pre-cut information.

Nothing here changes a default. Negative results are written as-is.
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
    pit_universe,
    trim_panel,
    usable_window,
)
from quant_system.metrics import performance_metrics  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.report import write_json  # noqa: E402
from quant_system.research import walk_forward  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402

FIXED = {"vol_window": 60, "trend_window": 120, "max_per_sector": 2, "rebalance_days": 20}
ARMS = [("6/2 buf1.25", 6, 1.25), ("6/2 buf1.5", 6, 1.5), ("6/2 buf2.0", 6, 2.0),
        ("8/2 buf1.5", 8, 1.5), ("8/2 buf2.0", 8, 2.0)]
PIT_ARMS = [("6/2 buf1.5", 6, 1.5), ("6/2 buf2.0", 6, 2.0)]
CUT = "2025-11-05"
OUT = ROOT / "reports" / "buffer_rank"


def params(top_n: int, buffer_rank: float) -> dict:
    return {**FIXED, "top_n": top_n, "buffer_rank": buffer_rank}


def run_walk(panel, benchmark, top_n: int, buffer_rank: float, research: ResearchConfig):
    grid = {key: [value] for key, value in params(top_n, buffer_rank).items()}
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


def heldout(panel, benchmark, cut: pd.Timestamp, top_n: int, buffer_rank: float) -> dict:
    raw = build_strategy("low_vol_core", **params(top_n, buffer_rank)).generate(panel)
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
    survivors = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    start, end = usable_window(survivors, min_active_ratio=0.95)
    survivors = {symbol: frame.loc[start:end] for symbol, frame in survivors.items()}
    benchmark = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
    calendar = pd.DatetimeIndex(benchmark.index)
    print(f"window {start.date()}..{end.date()} survivor={len(survivors)}", flush=True)

    aligned, dates = align_panel(survivors)
    usable_start, _ = usable_window(aligned, min_active_ratio=0.9)
    usable_pos = int(dates.searchsorted(usable_start, side="left"))
    first_test = dates[usable_pos + 504 + 10]
    research = ResearchConfig(start=str(first_test.date()), train_days=504, test_days=126, embargo_days=10)
    print(f"first test {first_test.date()}", flush=True)

    payload: dict = {
        "window": [str(start.date()), str(end.date())],
        "first_test": str(first_test.date()),
        "fixed_params": FIXED,
        "cut": CUT,
        "note": (
            "Walk-forward windows are strictly out of sample; the held-out year "
            "uses only pre-cut information. Full-sample optima are not evidence."
        ),
    }

    print("survivor walk-forward", flush=True)
    rows, frames = [], []
    for label, top_n, buffer_rank in ARMS:
        result = run_walk(survivors, benchmark, top_n, buffer_rank, research)
        rows.append(summarise(label, result))
        frame = result.windows.copy()
        frame["variant"] = label
        frames.append(frame)
        row = rows[-1]
        print(f"{label}: {row['total_return']:+.2%} sharpe={row['sharpe']:.3f} "
              f"mdd={row['max_drawdown']:.2%} pos={row['positive_windows']}/{row['windows']}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "survivor_walkforward.csv", index=False, encoding="utf-8-sig")
    detail = pd.concat(frames, ignore_index=True)
    detail.to_csv(OUT / "survivor_windows.csv", index=False, encoding="utf-8-sig")
    payload["survivor_walkforward"] = rows

    # Paired per-window differences against the incumbent 1.5 buffer.
    pivot = detail.pivot(index="test_start", columns="variant", values="test_return")
    paired = {}
    for label in pivot.columns:
        if label == "6/2 buf1.5":
            continue
        diff = (pivot[label] - pivot["6/2 buf1.5"]).dropna()
        paired[label] = {
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std(ddof=1)),
            "t_like": float(diff.mean() / diff.std(ddof=1) * len(diff) ** 0.5) if diff.std(ddof=1) else None,
            "wins": int((diff > 0).sum()),
            "n": int(len(diff)),
        }
    payload["paired_vs_6_2_buf1_5"] = paired

    print("point-in-time walk-forward", flush=True)
    universe = pit_universe()
    delisted = {s: n for s, n in universe.items() if s not in BROAD_UNIVERSE}
    pit_raw = load_panel(list(survivors) + list(delisted), cache)
    pit = {
        s: frame.loc[frame.index.isin(calendar)]
        for s, frame in pit_raw.items()
        if len(frame.loc[start:end])
    }
    print(f"pit symbols {len(pit)}", flush=True)
    pit_rows = []
    for label, top_n, buffer_rank in PIT_ARMS:
        result = run_walk(pit, benchmark, top_n, buffer_rank, research)
        row = summarise(label, result)
        pit_rows.append(row)
        print(f"PIT {label}: {row['total_return']:+.2%} sharpe={row['sharpe']:.3f} "
              f"mdd={row['max_drawdown']:.2%}", flush=True)
    pd.DataFrame(pit_rows).to_csv(OUT / "pit_walkforward.csv", index=False, encoding="utf-8-sig")
    payload["pit_walkforward"] = pit_rows

    print("held-out final year", flush=True)
    cut = pd.Timestamp(CUT)
    hold_rows = []
    for label, top_n, buffer_rank in ARMS:
        metrics = heldout(survivors, benchmark, cut, top_n, buffer_rank)
        hold_rows.append({"variant": label, **metrics})
        print(f"heldout {label}: {metrics['total_return']:+.2%} "
              f"sharpe={metrics['sharpe']:.3f} mdd={metrics['max_drawdown']:.2%}", flush=True)
    pd.DataFrame(hold_rows).to_csv(OUT / "heldout.csv", index=False, encoding="utf-8-sig")
    payload["heldout"] = hold_rows
    bench = benchmark.loc[cut:]
    payload["benchmark_heldout_return"] = float(bench.iloc[-1] / bench.iloc[0] - 1)

    write_json(payload, OUT / "summary.json")
    print(f"report: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
