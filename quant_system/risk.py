"""Portfolio-level risk overlays applied before execution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RiskConfig
from .data import price_frame


def _return_frame(panel: dict[str, pd.DataFrame], columns) -> pd.DataFrame:
    """Per-symbol returns on real bars, reindexed to the union calendar.

    A date-aligned ``pct_change`` fabricates a 0.0 return for every missing
    bar.  That is harmless for a single day, but a multi-year suspension then
    looks like a perfectly calm asset and biases both the covariance and the
    portfolio volatility proxy.  Keeping NaN here makes the rolling estimator
    skip the observation instead of averaging in a fake zero.
    """
    frames = {}
    for symbol in columns:
        frame = panel.get(symbol)
        if frame is None or frame.empty:
            continue
        close = frame["close"].dropna()
        if len(close) < 2:
            continue
        frames[symbol] = close.pct_change().dropna()
    if not frames:
        return pd.DataFrame(dtype=float)
    return pd.DataFrame(frames).reindex(columns=list(columns)).sort_index()


def _window_sums(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing ``window``-row sums, current row included, computed in place."""
    out = np.cumsum(values, axis=0, dtype=np.float64, out=values)
    if out.shape[0] > window:
        # The two slices are disjoint, so the in-place subtraction is safe and
        # avoids a second cube-sized temporary.
        out[window:] -= out[:-window]
    return out


def _rolling_covariance_variance(
    returns: pd.DataFrame,
    weights: np.ndarray,
    window: int,
    min_periods: int,
    *,
    pair_chunk: int = 512,
) -> np.ndarray:
    """Per-date variance ``w' S w`` with pairwise joint observations.

    ``S_ij`` is estimated on the bars where *both* assets traded:

        S_ij = [sum(x_i x_j) - sum(x_i 1_j) * sum(x_j 1_i) / n_ij] / n_ij

    so a multi-year suspension can no longer inject a run of fabricated zero
    returns.  The sum is built pair by pair instead of materialising a
    ``(date, asset, asset)`` cube: a book holds a handful of names, so only
    pairs that are ever co-held need work and the cost stays proportional to
    that pair count rather than to ``n_assets ** 2``.

    Returns one value per row of ``returns.index``.  NaN marks a date where a
    pair that carried weight lacked ``min_periods`` joint bars; the caller
    carries the previous causal estimate forward instead of reporting a
    risk-free portfolio.
    """
    n_rows, n_assets = returns.shape
    if n_rows == 0 or n_assets == 0:
        return np.zeros(n_rows, dtype=float)
    values = returns.to_numpy(dtype=np.float64)
    present = ~np.isnan(values)
    filled = np.where(present, values, 0.0)
    flag = present.astype(np.float64)
    w = np.nan_to_num(
        np.asarray(weights, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    if w.shape != (n_rows, n_assets):
        raise ValueError("weights must align with the return matrix")
    active = w != 0.0
    if not active.any():
        return np.zeros(n_rows, dtype=float)
    # ``co[i, j]`` counts the bars on which both i and j carry weight.
    co = active.astype(np.float64).T @ active.astype(np.float64)
    left_idx, right_idx = np.nonzero(co > 0.0)
    variance = np.zeros(n_rows, dtype=np.float64)
    covered = np.ones(n_rows, dtype=bool)
    for start in range(0, len(left_idx), pair_chunk):
        li = left_idx[start : start + pair_chunk]
        ri = right_idx[start : start + pair_chunk]
        xa = filled[:, li]
        xb = filled[:, ri]
        pa = flag[:, li]
        pb = flag[:, ri]
        cross = _window_sums(xa * xb, window)
        joint = _window_sums(pa * pb, window)
        left_sum = _window_sums(xa * pb, window)
        right_sum = _window_sums(xb * pa, window)
        enough = joint >= min_periods
        counts = np.where(enough, joint, 1.0)
        cov = cross / counts
        cov -= (left_sum / counts) * (right_sum / counts)
        if not enough.all():
            held = (w[:, li] != 0.0) & (w[:, ri] != 0.0)
            covered &= ~(held & ~enough).any(axis=1)
            cov[~enough] = 0.0
        variance += np.einsum("ti,ti->t", w[:, li] * w[:, ri], cov)
    variance[~covered] = np.nan
    return variance


def _ex_ante_volatility(
    weights: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    cfg: RiskConfig,
    *,
    through: pd.Timestamp | None = None,
) -> pd.Series:
    """Volatility of the *current* basket from trailing asset covariance.

    Unlike the lagged realised-P&L proxy, this reacts immediately when the
    selected names change. It uses only returns through the signal close.

    ``through`` truncates the calculation to dates at or before that bar.  The
    covariance is a backward-looking rolling window, so a truncated series
    gives bit-identical values on the retained dates while skipping the rest.
    The realised-forecast path needs the fallback only for its first few
    warm-up bars, and evaluating 1800 covariance blocks to use 20 of them was
    the single largest cost in a backtest.
    """
    values = pd.Series(np.nan, index=weights.index, dtype=float)
    if weights.empty or not len(weights.columns):
        return values
    returns = _return_frame(panel, weights.columns)
    if through is not None:
        returns = returns.loc[:through]
    if returns.empty:
        return values
    # The book is a step function on the trading calendar, so forward-fill it
    # onto the return rows: each row must be priced with the book actually
    # held on that bar, never with a future rebalance.
    aligned = weights.reindex(returns.index, method="ffill")
    variance = _rolling_covariance_variance(
        returns,
        aligned.to_numpy(dtype=float),
        cfg.vol_lookback,
        max(10, cfg.vol_lookback // 2),
    )
    series = pd.Series(variance, index=returns.index).ffill()
    # A weight row can fall on a date the price panel does not contain (for
    # example a test frame with a shorter calendar).  Carrying the most recent
    # covariance block forward is the correct causal choice; returning NaN
    # would make the caller fall back to a full-scale book.
    rows = returns.index.searchsorted(pd.DatetimeIndex(weights.index), side="right") - 1
    # ``through`` must not fabricate an estimate for later dates: the truncated
    # series simply stops, and carrying its last value past the cut would leak
    # a stale forecast into rows the caller did not ask about.
    valid = (rows >= 0) & (pd.DatetimeIndex(weights.index) <= returns.index[-1])
    if not valid.any():
        return values
    picked = series.to_numpy()[rows[valid]]
    values.iloc[np.flatnonzero(valid)] = np.sqrt(
        np.maximum(picked, 0.0) * 252.0
    )
    return values


def ex_ante_volatility(
    index: pd.DatetimeIndex,
    columns: list[str],
    panel: dict[str, pd.DataFrame],
    config: RiskConfig | None = None,
) -> pd.Series:
    """Annualised covariance-based volatility for a fixed universe.

    The covariance depends only on prices and the column set, never on the
    weights.  A rebalance-phase sweep runs the same universe through 20
    different weight paths, so computing this once per window instead of once
    per phase is both faster and a guarantee that every phase sees the same
    risk forecast.
    """
    cfg = config or RiskConfig()
    cfg.validate()
    shell = pd.DataFrame(0.0, index=pd.DatetimeIndex(index), columns=list(columns))
    return _ex_ante_volatility(shell, panel, cfg)


def breadth_timing_scale(
    weights_index: pd.Index,
    panel: dict[str, pd.DataFrame],
    *,
    lookback: int = 120,
    history_months: int = 24,
    lower_tercile: float = 1 / 3,
    de_risk: float = 0.25,
) -> pd.Series:
    """Market-breadth de-risking, causal in both directions.

    RESEARCH_LOG 23.58: the share of names above their 120-day average predicts
    *next* month's excess return (t=-3.9 over 78 months). The rule is only
    meaningful if its threshold comes from the past: a lower-tercile reading
    against an *expanding* history, evaluated on month-end observations and
    applied to the following month.

    The same rule also improves the equal-weight pool and the index, so this is
    market timing rather than a low-vol enhancement. It is opt-in (``default
    off``) for that reason and because 78 months is a short sample.

    Returns a daily scale series in ``{de_risk, 1.0}`` aligned to
    ``weights_index``. Signals are formed at the month-end close and apply from
    the next bar, so no same-day look-ahead is possible.
    """
    if not 0 < lower_tercile < 1:
        raise ValueError("lower_tercile must be in (0, 1)")
    if not 0 <= de_risk <= 1:
        raise ValueError("de_risk must be in [0, 1]")
    if history_months < 2:
        raise ValueError("history_months must be at least 2")

    close = price_frame(panel, "close")
    if close.empty:
        return pd.Series(1.0, index=weights_index)
    volume = price_frame(panel, "volume").reindex_like(close)
    traded = close.notna() & (volume > 0)
    sma = close.rolling(lookback, min_periods=max(20, lookback // 3)).mean()
    traded_count = traded.sum(axis=1).replace(0, np.nan)
    breadth = ((close > sma) & traded).sum(axis=1) / traded_count
    breadth = breadth.dropna()
    if breadth.empty:
        return pd.Series(1.0, index=weights_index)

    # Month-end observations, keeping a real Timestamp so the shift below is
    # unambiguous. ``groupby(...).last()`` with a two-key list returns a
    # MultiIndex whose last level is the month number, not a date.
    month_end = breadth.groupby([breadth.index.year, breadth.index.month]).tail(1)
    flags = []
    for position in range(len(month_end)):
        history = month_end.iloc[:position]
        if len(history) < history_months:
            flags.append(1.0)  # not enough history: stay fully invested
            continue
        threshold = float(history.quantile(lower_tercile))
        flags.append(float(de_risk) if float(month_end.iloc[position]) <= threshold else 1.0)
    signal = pd.Series(flags, index=month_end.index, dtype=float)

    # A flag formed at month-end M only becomes known after that close, so it
    # applies from the *next* month. Reindexing onto the daily calendar with
    # method="ffill" after shifting the signal date forward one month does that
    # without leaking the current month's close into the same month.
    signal.index = signal.index + pd.DateOffset(months=1)
    scale = signal.reindex(weights_index, method="ffill").fillna(1.0)
    return scale.astype(float)


def apply_volatility_target(
    weights: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    config: RiskConfig | None = None,
    *,
    min_scale: float = 0.25,
    forecast: str = "realised",
    ex_ante: pd.Series | None = None,
) -> pd.DataFrame:
    """Scale exposure using only returns observable at the signal close.

    ``forecast="realised"`` (default) estimates volatility from the previous
    basket's realised P&L and falls back to ex-ante covariance while that
    history is short. ``forecast="ex_ante"`` always uses the current basket's
    covariance; walk-forward testing showed it trades some return for a
    shallower drawdown, so it is opt-in rather than the default.
    """
    if forecast not in {"realised", "ex_ante"}:
        raise ValueError("forecast must be 'realised' or 'ex_ante'")
    cfg = config or RiskConfig()
    cfg.validate()
    if forecast == "ex_ante":
        fallback = (
            ex_ante.reindex(weights.index)
            if ex_ante is not None
            else _ex_ante_volatility(weights, panel, cfg)
        )
        fallback_scale = (cfg.target_annual_vol / fallback.replace(0.0, np.nan)).clip(
            min_scale, 1.0
        )
        scale = fallback_scale.reindex(weights.index).fillna(1.0)
        return weights.mul(scale, axis=0)
    close = price_frame(panel, "close").reindex(columns=weights.columns)
    # Portfolio-level: a missing bar means the holding simply did not move, so
    # a zero contribution is correct here.  The per-asset covariance above is
    # the place where a fabricated zero would bias the estimate.
    returns = close.pct_change(fill_method=None)
    held = weights.shift(1).reindex(returns.index).fillna(0.0)
    proxy = (held * returns).sum(axis=1)
    forecast_series = proxy.rolling(
        cfg.vol_lookback, min_periods=max(10, cfg.vol_lookback // 2)
    ).std(ddof=0) * np.sqrt(252)
    scale = (cfg.target_annual_vol / forecast_series.replace(0.0, np.nan)).clip(
        min_scale, 1.0
    )
    missing = scale.reindex(weights.index).isna()
    if missing.any():
        if ex_ante is not None:
            # ``ex_ante`` was computed over the whole window.  The rolling
            # covariance is backward-looking, so its values on the warm-up
            # bars are identical to a truncated computation.
            fallback = ex_ante.reindex(weights.index)
        else:
            last_needed = pd.DatetimeIndex(weights.index[missing.to_numpy()]).max()
            fallback = _ex_ante_volatility(weights, panel, cfg, through=last_needed)
        fallback_scale = (cfg.target_annual_vol / fallback.replace(0.0, np.nan)).clip(
            min_scale, 1.0
        )
    else:
        fallback_scale = pd.Series(np.nan, index=weights.index, dtype=float)
    scale = scale.reindex(weights.index).fillna(fallback_scale).fillna(1.0)
    return weights.mul(scale, axis=0)


def apply_drawdown_guard(
    weights: pd.DataFrame,
    strategy_returns: pd.Series,
    config: RiskConfig | None = None,
) -> pd.DataFrame:
    """Reduce exposure after a portfolio drawdown; re-risk on new highs."""
    cfg = config or RiskConfig()
    returns = strategy_returns.reindex(weights.index).fillna(0.0)
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    severity = (drawdown.abs() / cfg.max_drawdown_stop).clip(0.0, 1.0)
    scale = (1.0 - 0.75 * severity).clip(0.25, 1.0)
    return weights.mul(scale, axis=0)


def cap_turnover(
    weights: pd.DataFrame,
    max_daily_turnover: float = 0.35,
) -> pd.DataFrame:
    """Limit one-way daily weight movement without introducing lookahead."""
    if max_daily_turnover <= 0:
        raise ValueError("max_daily_turnover must be positive")
    result = weights.copy().fillna(0.0)
    previous = pd.Series(0.0, index=result.columns)
    for date in result.index:
        desired = result.loc[date]
        change = (desired - previous).abs()
        total = float(change.sum())
        if total > max_daily_turnover:
            alpha = max_daily_turnover / total
            desired = previous + alpha * (desired - previous)
            result.loc[date] = desired
        previous = result.loc[date]
    return result

