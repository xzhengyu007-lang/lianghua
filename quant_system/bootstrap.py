"""Block bootstrap for daily strategy returns.

The walk-forward tables in this repo are point estimates over one path. With
8-11 overlapping test windows those numbers carry real sampling error, and the
usual "window win rate" ignores it. This module resamples *blocks* of daily
returns so the serial dependence (20-day rebalance cycle, volatility clustering)
is preserved, and reports percentile intervals.

Two entry points matter for research:

- ``bootstrap_summary``: interval for one return series.
- ``paired_bootstrap_diff``: interval for the *difference* between two series
  sharing the same calendar, which is the right null for "is parameter A really
  better than B" since both see identical market days.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def default_block_length(n: int) -> int:
    """A simple n^(1/3) rule, floored at 5 and capped at a quarter of the sample."""
    if n <= 0:
        raise ValueError("sample length must be positive")
    return int(min(max(5, round(n ** (1.0 / 3.0))), max(5, n // 4)))


def stationary_bootstrap_indices(
    n: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Politis-Romano stationary bootstrap: geometric block lengths, wrapped."""
    if n <= 0:
        raise ValueError("sample length must be positive")
    if block_length <= 0:
        raise ValueError("block length must be positive")
    p = 1.0 / float(block_length)
    idx = np.empty(n, dtype=np.int64)
    idx[0] = int(rng.integers(0, n))
    for t in range(1, n):
        if rng.random() < p:
            idx[t] = int(rng.integers(0, n))
        else:
            idx[t] = (idx[t - 1] + 1) % n
    return idx


def circular_block_indices(
    n: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fixed-length circular blocks, concatenated to length n."""
    if n <= 0:
        raise ValueError("sample length must be positive")
    if block_length <= 0:
        raise ValueError("block length must be positive")
    blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=blocks)
    offsets = np.arange(block_length)
    idx = (starts[:, None] + offsets[None, :]).ravel() % n
    return idx[:n]


def _metrics_from_returns(ret: np.ndarray) -> dict[str, float]:
    if ret.size == 0:
        return {"cagr": np.nan, "sharpe": np.nan, "max_drawdown": np.nan, "annual_volatility": np.nan}
    equity = np.cumprod(1.0 + ret)
    total = float(equity[-1] - 1.0)
    years = ret.size / TRADING_DAYS
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0) if total > -1.0 else -1.0
    vol = float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS)) if ret.size > 1 else np.nan
    sharpe = float(ret.mean() * TRADING_DAYS / vol) if np.isfinite(vol) and vol > 0 else np.nan
    peak = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
    mdd = float((equity / peak - 1.0).min())
    return {"cagr": cagr, "sharpe": sharpe, "max_drawdown": mdd, "annual_volatility": vol}


def _as_array(returns: Sequence[float] | pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(returns, dtype=float).ravel()
    if not np.isfinite(arr).all():
        arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("returns are empty")
    return arr


def bootstrap_metrics(
    returns: Sequence[float] | pd.Series | np.ndarray,
    *,
    n_boot: int = 2000,
    block_length: int | None = None,
    seed: int = 0,
    method: str = "stationary",
) -> pd.DataFrame:
    """Resample ``returns`` and return one row of metrics per bootstrap draw."""
    arr = _as_array(returns)
    n = arr.size
    block = block_length or default_block_length(n)
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    sampler: Callable[..., np.ndarray] = (
        stationary_bootstrap_indices if method == "stationary" else circular_block_indices
    )
    if method not in {"stationary", "circular"}:
        raise ValueError(f"unknown bootstrap method: {method}")
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        idx = sampler(n, block, rng)
        rows.append(_metrics_from_returns(arr[idx]))
    frame = pd.DataFrame(rows)
    frame.attrs["block_length"] = block
    frame.attrs["n"] = n
    frame.attrs["method"] = method
    return frame


def _summarize(samples: pd.DataFrame, point: Mapping[str, float], low: float, high: float) -> dict:
    out: dict[str, dict[str, float]] = {}
    for column in samples.columns:
        series = samples[column].dropna()
        out[column] = {
            "point": float(point.get(column, np.nan)),
            f"p{int(low * 100)}": float(series.quantile(low)),
            f"p{int(high * 100)}": float(series.quantile(high)),
            "median": float(series.median()),
        }
    return out


def bootstrap_summary(
    returns: Sequence[float] | pd.Series | np.ndarray,
    *,
    n_boot: int = 2000,
    block_length: int | None = None,
    seed: int = 0,
    low: float = 0.025,
    high: float = 0.975,
    method: str = "stationary",
) -> dict:
    """Point estimate plus percentile interval for cagr / sharpe / mdd / vol."""
    arr = _as_array(returns)
    samples = bootstrap_metrics(
        arr, n_boot=n_boot, block_length=block_length, seed=seed, method=method
    )
    point = _metrics_from_returns(arr)
    out = _summarize(samples, point, low, high)
    out["_meta"] = {
        "n_obs": int(arr.size),
        "n_boot": int(n_boot),
        "block_length": int(samples.attrs["block_length"]),
        "method": method,
        "seed": int(seed),
    }
    return out


@dataclass
class PairedBootstrapResult:
    metric: str
    observed: float
    mean: float
    median: float
    low: float
    high: float
    prob_positive: float
    samples: np.ndarray | None = None

    def to_dict(self) -> dict[str, float]:
        return {
            "metric": self.metric,
            "observed": self.observed,
            "mean": self.mean,
            "median": self.median,
            "low": self.low,
            "high": self.high,
            "prob_positive": self.prob_positive,
        }


def paired_bootstrap_diff(
    returns_a: Sequence[float] | pd.Series | np.ndarray,
    returns_b: Sequence[float] | pd.Series | np.ndarray,
    *,
    metric: str = "sharpe",
    n_boot: int = 2000,
    block_length: int | None = None,
    seed: int = 0,
    low: float = 0.025,
    high: float = 0.975,
    method: str = "stationary",
    keep_samples: bool = False,
) -> PairedBootstrapResult:
    """Bootstrap the difference ``metric(a) - metric(b)`` on shared blocks.

    Both series must be the same length: the two strategies trade the same
    calendar, so blocks are drawn once and applied to both. That removes market
    timing luck from the comparison, which a naive independent bootstrap keeps.
    """
    a = _as_array(returns_a)
    b = _as_array(returns_b)
    if a.size != b.size:
        raise ValueError("paired bootstrap needs equal-length series")
    sampler: Callable[..., np.ndarray] = (
        stationary_bootstrap_indices if method == "stationary" else circular_block_indices
    )
    if method not in {"stationary", "circular"}:
        raise ValueError(f"unknown bootstrap method: {method}")
    n = a.size
    block = block_length or default_block_length(n)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = sampler(n, block, rng)
        ma = _metrics_from_returns(a[idx]).get(metric, np.nan)
        mb = _metrics_from_returns(b[idx]).get(metric, np.nan)
        diffs[i] = ma - mb
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        raise ValueError("no finite bootstrap differences")
    observed = _metrics_from_returns(a).get(metric, np.nan) - _metrics_from_returns(b).get(metric, np.nan)
    return PairedBootstrapResult(
        metric=metric,
        observed=float(observed),
        mean=float(diffs.mean()),
        median=float(np.median(diffs)),
        low=float(np.quantile(diffs, low)),
        high=float(np.quantile(diffs, high)),
        prob_positive=float((diffs > 0).mean()),
        samples=diffs if keep_samples else None,
    )
