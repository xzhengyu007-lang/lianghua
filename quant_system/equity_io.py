"""Read back a saved report's equity curve without losing the first day.

``reports/<run>/equity.csv`` starts at the *end* of day one, because the engine
marks the book to market before it writes the file. ``metrics.json`` however is
computed from the full daily return stream, which still contains that first
session. Naively taking ``equity.pct_change()`` therefore drops one day of
return and quietly shifts Sharpe (0.046 on the 2019-2021 window, enough to flip
a "nothing has changed" claim into "this variant is worse").

This module rebuilds the exact return stream the metrics were computed from:
the first day's return is recovered as ``equity[0] / initial_cash - 1``.

Usage::

    from quant_system.equity_io import daily_returns_from_equity

    returns = daily_returns_from_equity(Path("reports/wf_pit267_2019_2021"))
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_INITIAL_CASH = 1_000_000.0


def load_equity(report_dir: Path) -> pd.Series:
    """Return the saved equity level series for ``report_dir``."""
    equity_file = Path(report_dir) / "equity.csv"
    if not equity_file.exists():
        raise FileNotFoundError(equity_file)
    frame = pd.read_csv(equity_file, encoding="utf-8-sig", index_col=0, parse_dates=True)
    if frame.empty:
        raise ValueError(f"{equity_file} is empty")
    equity = frame.iloc[:, 0].astype(float)
    equity.index = pd.to_datetime(equity.index)
    equity.name = "equity"
    return equity


def daily_returns_from_equity(
    report_dir: Path,
    *,
    initial_cash: float = DEFAULT_INITIAL_CASH,
) -> pd.Series:
    """Rebuild the daily return stream that ``metrics.json`` was computed from.

    The first observation's return is the only one the level series cannot
    provide, so it is reconstructed against ``initial_cash``.
    """
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")
    equity = load_equity(report_dir)
    if len(equity) == 1:
        raise ValueError(f"{report_dir} holds a single equity point; no returns to rebuild")
    returns = equity.pct_change().dropna()
    first_day = pd.Series(
        [float(equity.iloc[0]) / float(initial_cash) - 1.0],
        index=equity.index[:1],
    )
    returns = pd.concat([first_day, returns])
    returns.name = Path(report_dir).name
    return returns
