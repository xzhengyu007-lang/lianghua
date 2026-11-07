"""Performance analytics for daily return and equity series."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


TRADING_DAYS = 252


@dataclass
class PerformanceMetrics:
    total_return: float
    cagr: float
    annual_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    profit_factor: float
    best_day: float
    worst_day: float
    skew: float
    kurtosis: float
    turnover: float
    exposure: float
    var_95: float
    cvar_95: float
    alpha: float
    beta: float
    information_ratio: float
    tracking_error: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def max_drawdown(equity: pd.Series) -> float:
    values = equity.dropna().astype(float)
    if values.empty:
        return float("nan")
    peak = values.cummax()
    return float((values / peak - 1.0).min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    values = equity.astype(float)
    return values / values.cummax() - 1.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator is None or not np.isfinite(denominator):
        return float("nan")
    if abs(denominator) < 1e-12:
        return 0.0
    return float(numerator / denominator)


def performance_metrics(
    returns: pd.Series,
    *,
    equity: pd.Series | None = None,
    benchmark_returns: pd.Series | None = None,
    turnover: float = float("nan"),
    exposure: float = float("nan"),
    risk_free: float = 0.0,
) -> PerformanceMetrics:
    ret = returns.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if ret.empty:
        raise ValueError("returns series is empty")
    eq = equity.reindex(ret.index).astype(float) if equity is not None else (1 + ret).cumprod()
    years = max(len(ret) / TRADING_DAYS, 1 / TRADING_DAYS)
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0) if eq.iloc[0] != 0 else float("nan")
    cagr = float((1 + total) ** (1 / years) - 1) if total > -1 else -1.0
    vol = float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(ret) > 1 else float("nan")
    excess = ret - risk_free / TRADING_DAYS
    sharpe = _safe_ratio(float(excess.mean() * TRADING_DAYS), vol)
    downside = excess[excess < 0]
    downside_dev = float(np.sqrt((downside**2).mean()) * np.sqrt(TRADING_DAYS)) if len(downside) else float("nan")
    sortino = _safe_ratio(float(excess.mean() * TRADING_DAYS), downside_dev)
    mdd = max_drawdown(eq)
    calmar = _safe_ratio(cagr, abs(mdd))
    positive = ret[ret > 0].sum()
    negative = ret[ret < 0].sum()
    profit_factor = _safe_ratio(float(positive), abs(float(negative)))
    var95 = float(ret.quantile(0.05))
    tail = ret[ret <= var95]
    cvar95 = float(tail.mean()) if len(tail) else var95
    alpha = beta = ir = te = float("nan")
    if benchmark_returns is not None:
        bench = benchmark_returns.reindex(ret.index).replace([np.inf, -np.inf], np.nan).dropna()
        joined = pd.concat([ret.rename("asset"), bench.rename("benchmark")], axis=1).dropna()
        if len(joined) > 10:
            asset_r = joined["asset"]
            bench_r = joined["benchmark"]
            cov = float(asset_r.cov(bench_r))
            bench_var = float(bench_r.var())
            beta = _safe_ratio(cov, bench_var)
            alpha_daily = float(asset_r.mean() - risk_free / TRADING_DAYS - beta * bench_r.mean()) if np.isfinite(beta) else float("nan")
            alpha = alpha_daily * TRADING_DAYS
            active = asset_r - bench_r
            te = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(active) > 1 else float("nan")
            ir = _safe_ratio(float(active.mean() * TRADING_DAYS), te)
    return PerformanceMetrics(
        total_return=total,
        cagr=cagr,
        annual_volatility=vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=mdd,
        calmar=calmar,
        win_rate=float((ret > 0).mean()),
        profit_factor=profit_factor,
        best_day=float(ret.max()),
        worst_day=float(ret.min()),
        skew=float(ret.skew()),
        kurtosis=float(ret.kurtosis()),
        turnover=float(turnover),
        exposure=float(exposure),
        var_95=var95,
        cvar_95=cvar95,
        alpha=float(alpha),
        beta=float(beta),
        information_ratio=float(ir),
        tracking_error=float(te),
    )


def monthly_returns(equity: pd.Series) -> pd.DataFrame:
    monthly = equity.resample("ME").last().pct_change(fill_method=None)
    if len(monthly):
        monthly.iloc[0] = equity.resample("ME").last().iloc[0] / equity.iloc[0] - 1
    table = monthly.to_frame("return")
    table["year"] = table.index.year
    table["month"] = table.index.month
    return table


def drawdown_table(equity: pd.Series, top: int = 5) -> pd.DataFrame:
    dd = drawdown_series(equity)
    underwater = dd < 0
    episodes: list[dict] = []
    start = None
    for date, is_under in underwater.items():
        if is_under and start is None:
            start = date
        if not is_under and start is not None:
            segment = dd.loc[start:date]
            episodes.append(
                {"start": start, "end": date, "depth": float(segment.min()), "days": len(segment)}
            )
            start = None
    if start is not None:
        segment = dd.loc[start:]
        episodes.append(
            {"start": start, "end": dd.index[-1], "depth": float(segment.min()), "days": len(segment)}
        )
    if not episodes:
        return pd.DataFrame(columns=["start", "end", "depth", "days"])
    return pd.DataFrame(episodes).sort_values("depth").head(top).reset_index(drop=True)

