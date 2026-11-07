"""Sector-relative low-vol: rank names *inside* their own sector.

The published decomposition says the low-vol basket's excess is mostly sector
allocation: banks, utilities and energy are naturally low-vol, so a global
low-vol ranking keeps buying them. Two earlier attempts to fix this failed --
sector momentum and "pick the low-vol sector" -- but both changed the *sector
allocation* itself. This script keeps the same sector mix and only changes the
within-sector name choice:

* global low-vol (default),
* within-sector low-vol rank, equal weight across the same 8/6 slots,
* within-sector low-vol rank with a sector cap of 2.

The test is deliberately narrow: same dates, same cadence, same overlays, same
sector cap. If within-sector ranking helps, the improvement cannot come from
sector timing. Negative results are kept.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
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
    price_frame,
    trim_panel,
    usable_window,
)
from quant_system.indicators import cross_sectional_zscore, sma  # noqa: E402
from quant_system.metrics import performance_metrics  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.report import write_json  # noqa: E402
from quant_system.research import walk_forward  # noqa: E402
from quant_system.sectors import sector_of  # noqa: E402
from quant_system.strategies import LowVolCoreStrategy, _traded  # noqa: E402

FIXED = {"vol_window": 60, "trend_window": 120, "rebalance_days": 20}
CUT = "2025-11-05"
OUT = ROOT / "reports" / "sector_relative"


@dataclass
class SectorRelativeLowVol(LowVolCoreStrategy):
    """Low-vol core ranked within each sector, then merged globally."""

    sector_relative: bool = True
    name: str = "sector_relative_lowvol"

    def score(self, panel):
        close = price_frame(panel, "close")
        returns = close.pct_change(fill_method=None)
        vol = returns.rolling(self.vol_window).std(ddof=0) * np.sqrt(252)
        trend = close / sma(close, self.trend_window) - 1.0
        eligible = _traded(panel, close) & (close > sma(close, self.trend_window))
        low_vol = (-vol).where(eligible)
        if not self.sector_relative:
            scores = self.vol_weight * cross_sectional_zscore(low_vol) + self.trend_weight * cross_sectional_zscore(trend)
            return scores, eligible
        # Percentile inside each sector. A name is only comparable to peers in
        # the same industry, which removes the "banks are always lowest vol"
        # effect without changing how many names each sector contributes.
        scores = low_vol.copy()
        sectors = {column: sector_of(column) for column in low_vol.columns}
        for sector in sorted(set(sectors.values())):
            columns = [c for c in low_vol.columns if sectors[c] == sector]
            if len(columns) < 2:
                continue
            scores[columns] = low_vol[columns].rank(axis=1, pct=True)
        # Trend confirmation still applies, at a much smaller weight.
        scores = scores + 0.05 * cross_sectional_zscore(trend)
        return scores.where(eligible), eligible


def build(top_n: int, relative: bool):
    if relative:
        return SectorRelativeLowVol(top_n=top_n, **FIXED)
    return LowVolCoreStrategy(top_n=top_n, **FIXED)


def run_walk(panel, benchmark, top_n: int, relative: bool, research):
    import quant_system.research as research_module

    original = research_module.build_strategy

    def factory(_name, **params):
        return build(params.get("top_n", top_n), relative)

    research_module.build_strategy = factory
    try:
        grid = {
            "vol_window": [FIXED["vol_window"]],
            "trend_window": [FIXED["trend_window"]],
            "rebalance_days": [FIXED["rebalance_days"]],
            "max_per_sector": [2],
            "top_n": [top_n],
        }
        return walk_forward(
            panel, "low_vol_core", grid, benchmark=benchmark, research=research,
            backtest=BacktestConfig(initial_cash=1_000_000), risk=RiskConfig(),
            portfolio_kwargs={"use_regime": False}, verbose=False,
        )
    finally:
        research_module.build_strategy = original


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


def sector_share(weights: pd.DataFrame) -> pd.DataFrame:
    totals = weights.sum(axis=1).replace(0, np.nan)
    shares = {}
    for column in weights.columns:
        sector = sector_of(column)
        value = weights[column] / totals
        shares[sector] = shares.get(sector, 0.0) + value
    return pd.DataFrame(shares).fillna(0.0)


def heldout(panel, benchmark, cut: pd.Timestamp, top_n: int, relative: bool) -> dict:
    raw = build(top_n, relative).generate(panel)
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
            "Within-sector ranking changes only which names are chosen inside "
            "each industry; the sector cap and every overlay stay fixed. "
            "Walk-forward windows are strictly out of sample."
        ),
    }

    arms = [("8/2 global low-vol", 8, False), ("8/2 within-sector", 8, True),
            ("6/2 global low-vol", 6, False), ("6/2 within-sector", 6, True)]
    print("walk-forward", flush=True)
    rows, frames = [], []
    for label, top_n, relative in arms:
        result = run_walk(panel, benchmark, top_n, relative, research)
        row = summarise(label, result, {"top_n": top_n, "sector_relative": relative})
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
    for rel, base in (("8/2 within-sector", "8/2 global low-vol"), ("6/2 within-sector", "6/2 global low-vol")):
        diff = (pivot[rel] - pivot[base]).dropna()
        paired[f"{rel} - {base}"] = {
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std(ddof=1)),
            "t_like": float(diff.mean() / diff.std(ddof=1) * len(diff) ** 0.5) if diff.std(ddof=1) else None,
            "wins": int((diff > 0).sum()),
            "n": int(len(diff)),
        }
    payload["paired_within_minus_global"] = paired

    # Sector exposure is the mechanism: if within-sector ranking does not move
    # sector shares, it cannot fix the "all banks" critique either way.
    print("sector exposure", flush=True)
    exposure = {}
    for label, top_n, relative in arms:
        raw = build(top_n, relative).generate(panel)
        weights = build_portfolio(raw, panel, benchmark=benchmark, risk=RiskConfig(), use_regime=False)
        shares = sector_share(weights)
        exposure[label] = {
            sector: float(shares[sector].mean()) for sector in shares.columns
        }
        top = sorted(exposure[label].items(), key=lambda item: -item[1])[:3]
        print(f"{label}: " + ", ".join(f"{k} {v:.1%}" for k, v in top), flush=True)
    payload["mean_sector_exposure"] = exposure

    print("held-out final year", flush=True)
    cut = pd.Timestamp(CUT)
    hold_rows = []
    for label, top_n, relative in arms:
        metrics = heldout(panel, benchmark, cut, top_n, relative)
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
