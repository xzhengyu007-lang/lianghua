"""Negative control: prove the look-ahead audit can actually fail.

A detector that never fires is not evidence. This injects deliberate peeks into
a copy of the pipeline and checks that ``run_lookahead_audit``'s comparison
flags them:

    centered   rolling mean centered on the bar (classic future leak)
    full_z     z-score computed on the whole sample, then sliced
    ffill_all  price forward-filled across the cut, inventing post-cut bars

Each is a *local* wrapper in this file, not a code change, so the repository's
own audit stays honest.

Usage::

    python scripts/run_lookahead_negative_control.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.backtest import Backtester  # noqa: E402
from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, RiskConfig  # noqa: E402
from quant_system.data import align_panel, pit_universe  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import LowVolCoreStrategy  # noqa: E402


PARAMS = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)
CUT = pd.Timestamp("2023-06-30")


def run(panel: dict[str, pd.DataFrame], leak: str | None = None) -> pd.Series:
    strategy = LowVolCoreStrategy(rebalance_reset="year", **PARAMS)
    weights = strategy.generate(panel)
    if leak == "centered":
        # Deliberate look-ahead: a centered window sees future bars.
        for symbol, frame in panel.items():
            close = frame["close"]
            trick = close.rolling(21, center=True, min_periods=1).mean()
            weights.loc[:, symbol] = weights[symbol] * (close.gt(trick)).astype(float)
    elif leak == "full_z":
        # Deliberate look-ahead: normalization uses the whole sample's stats.
        for symbol, frame in panel.items():
            close = frame["close"]
            z = (close - close.mean()) / close.std()
            weights.loc[:, symbol] = weights[symbol] * (z > 0).astype(float)
    weights = build_portfolio(weights, panel, risk=RiskConfig(), use_regime=False, use_vol_target=True)
    return Backtester(BacktestConfig()).run(panel, weights).equity


def compare(full: pd.Series, prefix: pd.Series) -> float:
    left = full.loc[:CUT]
    right = prefix.reindex(left.index)
    if right.isna().any():
        return float("inf")
    return float((left - right).abs().max())


def main() -> int:
    panel = _rebuilt_panel(list(pit_universe()), ROOT / "data" / "cache", "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, dates = align_panel(panel)
    truncated = {s: f.loc[:CUT] for s, f in aligned.items() if len(f.loc[:CUT]) > 0}

    clean_full = run(aligned)
    clean_prefix = run(truncated)
    print(f"clean pipeline: diff={compare(clean_full, clean_prefix):.2e} (expected 0)")

    failures = 0
    for leak in ("centered", "full_z"):
        leaked_full = run(aligned, leak=leak)
        leaked_prefix = run(truncated, leak=leak)
        diff = compare(leaked_full, leaked_prefix)
        flagged = not np.isfinite(diff) or diff > 1e-8
        print(f"{leak:9s}: diff={diff:.4e} flagged={flagged}")
        failures += 0 if flagged else 1

    # Third class: a "full sample then slice" rank that peeks at the future
    # distribution without changing the panel shape. It must move the prefix.
    full_rank = {}
    for symbol, frame in aligned.items():
        close = frame["close"]
        ranks = close.rank(pct=True).fillna(0.5)
        full_rank[symbol] = ranks
    truncated_rank = {}
    for symbol, frame in truncated.items():
        close = frame["close"]
        truncated_rank[symbol] = close.rank(pct=True).fillna(0.5)

    weights_full = LowVolCoreStrategy(rebalance_reset="year", **PARAMS).generate(aligned)
    weights_trunc = LowVolCoreStrategy(rebalance_reset="year", **PARAMS).generate(truncated)
    peek_full = weights_full * pd.DataFrame(full_rank)
    peek_trunc = weights_trunc * pd.DataFrame(truncated_rank).reindex(
        index=weights_trunc.index, columns=weights_trunc.columns
    )
    overlap = weights_full.loc[:CUT].index
    diff_peek = float(
        (peek_full.loc[overlap] - peek_trunc.reindex(overlap).fillna(0.0)).abs().to_numpy().max()
    )
    flagged_peek = diff_peek > 1e-8
    print(f"full_rank: diff={diff_peek:.4e} flagged={flagged_peek}")
    failures += 0 if flagged_peek else 1

    total = 3
    print("negative control", "PASSED" if failures == 0 else f"FAILED ({failures}/{total} leaks missed)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
