"""Vectorized indicators with explicit warm-up handling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    return series.rolling(window, min_periods=min_periods or window).mean()


def ema(series: pd.Series, span: int, min_periods: int | None = None) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=min_periods or span).mean()


def true_range(frame: pd.DataFrame) -> pd.Series:
    prev_close = frame["close"].shift(1)
    ranges = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    return true_range(frame).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100.0 - 100.0 / (1.0 + rs)
    return result.where(avg_loss != 0, 100.0).where(avg_gain.notna(), np.nan)


def realized_vol(series: pd.Series, window: int = 20, periods: int = 252) -> pd.Series:
    return series.pct_change(fill_method=None).rolling(window, min_periods=window).std(ddof=0) * np.sqrt(periods)


def realised_volatility(
    panel: dict[str, pd.DataFrame], window: int
) -> pd.DataFrame:
    """Annualised close-to-close volatility on each symbol's *own* bars.

    ``close.pct_change()`` on a date-aligned panel turns every missing bar into
    a zero return.  A name suspended for a year therefore shows a volatility of
    exactly 0.0 and wins a low-volatility ranking outright -- the single
    largest distortion the point-in-time pool exposed.  Returns are computed
    per symbol on its real bars, so a suspension simply contributes no
    observation instead of a fake zero.

    The output is indexed by every date any symbol traded; a symbol with fewer
    than ``window`` real returns on a date is NaN rather than zero.
    """
    if window <= 1:
        raise ValueError("window must be > 1")
    columns: dict[str, pd.Series] = {}
    for symbol, frame in panel.items():
        close = frame["close"].dropna()
        if len(close) < 2:
            continue
        returns = close.pct_change().dropna()
        columns[symbol] = returns.rolling(window).std(ddof=0) * np.sqrt(252)
    if not columns:
        return pd.DataFrame(dtype=float)
    return pd.DataFrame(columns).sort_index()


def adx(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    up = frame["high"].diff()
    down = -frame["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = true_range(frame).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / tr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / tr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def cross_sectional_rank(frame: pd.DataFrame, *, ascending: bool = False) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True, ascending=ascending)


def rolling_beta(asset: pd.Series, benchmark: pd.Series, window: int = 60) -> pd.Series:
    asset_ret = asset.pct_change(fill_method=None)
    bench_ret = benchmark.pct_change(fill_method=None)
    covariance = asset_ret.rolling(window, min_periods=window).cov(bench_ret)
    variance = bench_ret.rolling(window, min_periods=window).var(ddof=0)
    return covariance / variance.replace(0.0, np.nan)

