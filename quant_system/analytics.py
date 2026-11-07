"""Cross-sectional factor analytics: IC, decay, and quantile spreads."""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(close: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Close-to-close forward return, aligned to the signal date."""
    return close.shift(-horizon) / close - 1.0


def spearman_ic(signal: pd.DataFrame, future: pd.DataFrame) -> pd.Series:
    """Daily cross-sectional Spearman rank correlation."""
    aligned_signal, aligned_future = signal.align(future, join="inner")
    values = {}
    for date in aligned_signal.index:
        s = aligned_signal.loc[date].replace([np.inf, -np.inf], np.nan)
        f = aligned_future.loc[date].replace([np.inf, -np.inf], np.nan)
        joined = pd.concat([s, f], axis=1).dropna()
        if len(joined) < 5:
            continue
        values[date] = joined.iloc[:, 0].corr(joined.iloc[:, 1], method="spearman")
    return pd.Series(values, dtype=float).sort_index()


def ic_summary(ic: pd.Series) -> dict[str, float]:
    clean = ic.dropna()
    if clean.empty:
        return {"mean_ic": float("nan"), "ic_ir": float("nan"), "positive_ratio": float("nan"), "t_stat": float("nan")}
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else float("nan")
    return {
        "mean_ic": mean,
        "ic_ir": float(mean / std) if std and np.isfinite(std) else float("nan"),
        "positive_ratio": float((clean > 0).mean()),
        "t_stat": float(mean / (std / np.sqrt(len(clean)))) if std and len(clean) > 1 else float("nan"),
    }


def quantile_spread(signal: pd.DataFrame, future: pd.DataFrame, quantiles: int = 5) -> pd.DataFrame:
    """Mean forward return per cross-sectional quantile, by date."""
    aligned_signal, aligned_future = signal.align(future, join="inner")
    rows = []
    for date in aligned_signal.index:
        s = aligned_signal.loc[date].replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < quantiles:
            continue
        try:
            buckets = pd.qcut(s.rank(method="first"), quantiles, labels=False)
        except ValueError:
            continue
        f = aligned_future.loc[date].reindex(buckets.index)
        for bucket in range(quantiles):
            rows.append({"date": date, "quantile": bucket + 1, "forward_return": float(f[buckets == bucket].mean())})
    if not rows:
        return pd.DataFrame(columns=["date", "quantile", "forward_return"])
    return pd.DataFrame(rows)


def factor_decay(signal: pd.DataFrame, close: pd.DataFrame, horizons: tuple[int, ...] = (1, 5, 10, 20)) -> pd.DataFrame:
    rows = []
    for horizon in horizons:
        ic = spearman_ic(signal, forward_returns(close, horizon))
        summary = ic_summary(ic)
        rows.append({"horizon": horizon, **summary, "observations": int(ic.notna().sum())})
    return pd.DataFrame(rows).set_index("horizon")


def factor_report(panel: dict[str, pd.DataFrame], factors: dict[str, pd.DataFrame], horizon: int = 5) -> dict[str, dict]:
    from .data import panel_to_frame

    close = panel_to_frame(panel, "close")
    future = forward_returns(close, horizon)
    output = {}
    for name, signal in factors.items():
        ic = spearman_ic(signal, future)
        output[name] = {**ic_summary(ic), "observations": int(ic.notna().sum())}
    return output
