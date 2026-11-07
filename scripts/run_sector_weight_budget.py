"""Sector *weight* budget instead of a sector *count* cap.

The default caps how many names a sector may contribute (2), which implies a
25% sector weight in an 8-name equal-weight book and 33% in a 6-name book.
Capping the sector weight itself is a different control: it de-risks a
concentrated sector without forcing the optimizer into other sectors.

A cap that merely lowers gross exposure is confounded with the volatility
target, which will lever the book back up. Every capped arm is therefore paired
with a "matched exposure" control that keeps the baseline sector mix but scales
the whole book by the same per-date factor. If a sector budget adds anything,
it must beat that control, not just the un-scaled baseline.

Nothing here changes a default; negative results are kept.
"""

from __future__ import annotations

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
from quant_system.sectors import sector_of  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402

FIXED = {"vol_window": 60, "trend_window": 120, "rebalance_days": 20}
CUT = "2025-11-05"
OUT = ROOT / "reports" / "sector_weight_budget"


def cap_sectors(raw: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Scale each sector down to ``cap`` of the book; excess stays in cash."""
    sectors = [sector_of(column) for column in raw.columns]
    result = raw.copy()
    for date in raw.index:
        row = raw.loc[date]
        held = row[row > 0]
        if held.empty:
            continue
        totals: dict[str, float] = {}
        for column in held.index:
            sector = sector_of(column)
            totals[sector] = totals.get(sector, 0.0) + float(row[column])
        factors = {
            sector: min(1.0, cap / total) if total > 1e-12 else 1.0
            for sector, total in totals.items()
        }
        for column in held.index:
            result.loc[date, column] = float(row[column]) * factors[sector_of(column)]
    return result


def matched_exposure(raw: pd.DataFrame, capped: pd.DataFrame) -> pd.DataFrame:
    """Baseline sector mix at the capped book's total exposure, same date."""
    base_total = raw.sum(axis=1).replace(0, np.nan)
    cap_total = capped.sum(axis=1)
    factor = (cap_total / base_total).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
    return raw.mul(factor, axis=0).fillna(0.0)


def transform_for(cap: float | None, matched: bool = False):
    if cap is None and not matched:
        return None

    def transform(raw: pd.DataFrame, _panel) -> pd.DataFrame:
        if cap is None:
            return raw
        capped = cap_sectors(raw, cap)
        return matched_exposure(raw, capped) if matched else capped

    return transform


def run_walk(panel, benchmark, top_n: int, cap: float | None, matched: bool, research):
    grid = {
        "vol_window": [FIXED["vol_window"]],
        "trend_window": [FIXED["trend_window"]],
        "rebalance_days": [FIXED["rebalance_days"]],
        "top_n": [top_n],
        "max_per_sector": [2],
    }
    return walk_forward(
        panel,
        "low_vol_core",
        grid,
        benchmark=benchmark,
        research=research,
        backtest=BacktestConfig(initial_cash=1_000_000),
        risk=RiskConfig(),
        portfolio_kwargs={"use_regime": False},
        weight_transform=transform_for(cap, matched),
        verbose=False,
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


def heldout(panel, benchmark, cut: pd.Timestamp, top_n: int, cap: float | None, matched: bool) -> dict:
    raw = build_strategy("low_vol_core", **FIXED, top_n=top_n, max_per_sector=2).generate(panel)
    transform = transform_for(cap, matched)
    if transform is not None:
        raw = transform(raw, panel)
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
    survivors = {s: f.loc[start:end] for s, f in survivors.items()}
    benchmark = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
    calendar = pd.DatetimeIndex(benchmark.index)
    aligned, dates = align_panel(survivors)
    usable_start, _ = usable_window(aligned, min_active_ratio=0.9)
    usable_pos = int(dates.searchsorted(usable_start, side="left"))
    first_test = dates[usable_pos + 504 + 10]
    research = ResearchConfig(start=str(first_test.date()), train_days=504, test_days=126, embargo_days=10)
    print(f"window {start.date()}..{end.date()} symbols={len(survivors)} first_test={first_test.date()}", flush=True)

    # 8/2 implies a 25% sector weight, so 20%/15% are binding; 25% is the
    # non-binding reference. 6/2 implies 33%, so 25% binds there.
    arms = [
        ("8/2 no cap", 8, None, False),
        ("8/2 sector 20%", 8, 0.20, False),
        ("8/2 sector 20% matched", 8, 0.20, True),
        ("8/2 sector 15%", 8, 0.15, False),
        ("8/2 sector 15% matched", 8, 0.15, True),
        ("6/2 no cap", 6, None, False),
        ("6/2 sector 25%", 6, 0.25, False),
        ("6/2 sector 25% matched", 6, 0.25, True),
    ]

    payload: dict = {
        "window": [str(start.date()), str(end.date())],
        "first_test": str(first_test.date()),
        "fixed_params": FIXED,
        "cut": CUT,
        "note": (
            "Sector caps keep the excess in cash; the matched arm scales the "
            "baseline book to the same total exposure on the same date, so any "
            "difference is composition rather than de-risking. Walk-forward "
            "windows are strictly out of sample."
        ),
    }

    print("walk-forward", flush=True)
    rows, frames = [], []
    for label, top_n, cap, matched in arms:
        result = run_walk(survivors, benchmark, top_n, cap, matched, research)
        row = summarise(label, result, {"top_n": top_n, "sector_cap": cap, "matched": matched})
        rows.append(row)
        frame = result.windows.copy()
        frame["variant"] = label
        frames.append(frame)
        print(f"{label}: {row['total_return']:+.2%} sharpe={row['sharpe']:.3f} "
              f"mdd={row['max_drawdown']:.2%} exposure={row['exposure']:.3f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "walkforward.csv", index=False, encoding="utf-8-sig")
    detail = pd.concat(frames, ignore_index=True)
    detail.to_csv(OUT / "windows.csv", index=False, encoding="utf-8-sig")
    payload["walkforward"] = rows

    pivot = detail.pivot(index="test_start", columns="variant", values="test_return")
    paired = {}
    for capped, matched in (
        ("8/2 sector 20%", "8/2 sector 20% matched"),
        ("8/2 sector 15%", "8/2 sector 15% matched"),
        ("6/2 sector 25%", "6/2 sector 25% matched"),
    ):
        diff = (pivot[capped] - pivot[matched]).dropna()
        paired[f"{capped} - matched"] = {
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std(ddof=1)),
            "t_like": float(diff.mean() / diff.std(ddof=1) * len(diff) ** 0.5) if diff.std(ddof=1) else None,
            "wins": int((diff > 0).sum()),
            "n": int(len(diff)),
        }
    payload["paired_capped_minus_matched"] = paired

    print("held-out final year", flush=True)
    cut = pd.Timestamp(CUT)
    hold_rows = []
    for label, top_n, cap, matched in arms:
        metrics = heldout(survivors, benchmark, cut, top_n, cap, matched)
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
