"""Candidate cross-sectional signals, all strictly causal.

Every factor returns a DataFrame aligned to the close matrix where a *higher*
value means "more attractive", so a positive IC is always good news regardless
of the raw sign of the economic effect. The library exists to answer one
question: is there anything left after low volatility has been controlled for?

Nothing here looks forward: all rolling windows are shifted where the window
itself would otherwise include the signal bar's own return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import price_frame
from .indicators import realised_volatility


def _amount(panel: dict[str, pd.DataFrame], close: pd.DataFrame) -> pd.DataFrame:
    volume = price_frame(panel, "volume").reindex_like(close)
    return close * volume


def factor_library(
    panel: dict[str, pd.DataFrame],
    *,
    vol_window: int = 60,
) -> dict[str, pd.DataFrame]:
    close = price_frame(panel, "close")
    volume = price_frame(panel, "volume").reindex_like(close)
    returns = close.pct_change(fill_method=None)

    out: dict[str, pd.DataFrame] = {}

    # Low volatility itself: the incumbent signal, kept as the control.
    out["low_vol"] = -realised_volatility(panel, vol_window).reindex_like(close)

    # Short-term reversal: recent winners mean-revert at the one-week horizon.
    out["reversal_5"] = -(close / close.shift(5) - 1.0)

    # MAX effect: names with lottery-like daily spikes underperform.
    out["max_effect"] = -returns.rolling(20).max()

    # Abnormal volume: turnover far above its own recent norm is distribution.
    average_volume = volume.rolling(60).mean()
    out["abnormal_volume"] = -(volume / average_volume.replace(0.0, np.nan))

    # Amihud illiquidity: price impact per unit of traded value.
    amount = _amount(panel, close).replace(0.0, np.nan)
    out["amihud"] = -(returns.abs() / amount).rolling(20).mean() * 1e9

    # 12-1 momentum: skip the most recent month to avoid the reversal window.
    out["momentum_12_1"] = close.shift(21) / close.shift(252) - 1.0

    # Proximity to the 52-week high (anchoring / overhang).
    out["high_52w"] = close / close.rolling(252).max()

    for name, frame in out.items():
        out[name] = frame.replace([np.inf, -np.inf], np.nan)
    return out


def partial_ic(
    factor: pd.DataFrame,
    control: pd.DataFrame,
    future: pd.DataFrame,
) -> pd.Series:
    """Spearman IC of ``factor`` after stripping ``control`` cross-sectionally.

    Each day the factor and the forward return are rank-transformed, both are
    regressed on the rank of the control, and the correlation of the two
    residuals is recorded. That is the daily partial rank correlation, i.e.
    the information the factor carries that low volatility does not.
    """
    values: dict[pd.Timestamp, float] = {}
    common = factor.index.intersection(control.index).intersection(future.index)
    for date in common:
        s = factor.loc[date].replace([np.inf, -np.inf], np.nan).dropna()
        c = control.loc[date].replace([np.inf, -np.inf], np.nan).dropna()
        f = future.loc[date].replace([np.inf, -np.inf], np.nan).dropna()
        shared = s.index.intersection(c.index).intersection(f.index)
        if len(shared) < 10:
            continue
        s_rank = s.reindex(shared).rank(pct=True)
        c_rank = c.reindex(shared).rank(pct=True)
        f_rank = f.reindex(shared).rank(pct=True)
        design = np.column_stack([np.ones(len(shared)), c_rank.to_numpy(dtype=float)])
        beta_s, *_ = np.linalg.lstsq(design, s_rank.to_numpy(dtype=float), rcond=None)
        beta_f, *_ = np.linalg.lstsq(design, f_rank.to_numpy(dtype=float), rcond=None)
        resid_s = s_rank.to_numpy(dtype=float) - design @ beta_s
        resid_f = f_rank.to_numpy(dtype=float) - design @ beta_f
        if resid_s.std() == 0 or resid_f.std() == 0:
            continue
        values[date] = float(np.corrcoef(resid_s, resid_f)[0, 1])
    return pd.Series(values, dtype=float).sort_index()


def rank_correlation(factor: pd.DataFrame, control: pd.DataFrame) -> float:
    """Average daily cross-sectional rank correlation between two signals."""
    values = []
    for date in factor.index.intersection(control.index):
        s = factor.loc[date].replace([np.inf, -np.inf], np.nan)
        c = control.loc[date].replace([np.inf, -np.inf], np.nan)
        joined = pd.concat([s, c], axis=1).dropna()
        if len(joined) < 10:
            continue
        values.append(joined.iloc[:, 0].corr(joined.iloc[:, 1], method="spearman"))
    series = pd.Series(values, dtype=float).dropna()
    return float(series.mean()) if not series.empty else float("nan")
