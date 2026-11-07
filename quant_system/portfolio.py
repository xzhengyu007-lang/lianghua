"""Signal-to-portfolio transformations and risk overlays."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BacktestConfig, RiskConfig
from .data import price_frame
from .indicators import realised_volatility
from .risk import apply_volatility_target, breadth_timing_scale, cap_turnover


def normalize_targets(
    targets: pd.DataFrame,
    symbols: list[str],
    dates: pd.DatetimeIndex,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Clamp a raw target frame exactly the way the backtester does.

    The live signal and the backtest must agree on the book they describe, so
    this lives in one place instead of being reimplemented in ``signals.py``.
    Order matters: a single-name weight is capped first, then the whole book is
    scaled down to the gross limit. Doing it the other way round would let a
    two-name book pass through at 50% per name.
    """
    frame = targets.reindex(index=dates, columns=symbols).astype(float)
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    frame = frame.clip(upper=config.max_position_weight)
    gross = frame.sum(axis=1)
    scale = (config.max_gross_exposure / gross).where(
        gross > config.max_gross_exposure, 1.0
    )
    return frame.mul(scale, axis=0).fillna(0.0)


def equal_weight(selection: pd.DataFrame) -> pd.DataFrame:
    counts = selection.sum(axis=1).replace(0, np.nan)
    return selection.div(counts, axis=0).fillna(0.0)


def inverse_volatility_weights(
    selection: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    window: int = 40,
) -> pd.DataFrame:
    close = price_frame(panel, "close").reindex(columns=selection.columns)
    vol = realised_volatility(panel, window).reindex_like(close).replace(0, np.nan)
    inverse = (1.0 / vol).where(selection.astype(bool))
    denominator = inverse.sum(axis=1).replace(0, np.nan)
    return inverse.div(denominator, axis=0).fillna(0.0)


def regime_filter(
    weights: pd.DataFrame,
    benchmark: pd.Series,
    *,
    fast: int = 20,
    slow: int = 60,
) -> pd.DataFrame:
    """Cut equity exposure when the benchmark is below its slow trend."""
    fast_ma = benchmark.rolling(fast).mean()
    slow_ma = benchmark.rolling(slow).mean()
    risk_on = (benchmark > slow_ma) & (fast_ma > slow_ma)
    scale = risk_on.reindex(weights.index).fillna(False).astype(float)
    return weights.mul(scale, axis=0)


def build_portfolio(
    weights: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    *,
    benchmark: pd.Series | None = None,
    risk: RiskConfig | None = None,
    use_regime: bool = False,
    use_vol_target: bool = True,
    use_turnover_cap: bool = False,
    vol_forecast: str = "realised",
    ex_ante_vol: pd.Series | None = None,
) -> pd.DataFrame:
    """Apply portfolio overlays.

    ``use_regime`` defaults to **off**. Walk-forward tests showed the
    benchmark-trend switch cut out-of-sample return from +74% to +4% while
    only modestly reducing drawdown, so it is opt-in rather than default.

    ``use_turnover_cap`` spreads a rebalance over several days when the one-way
    weight change exceeds ``max_daily_turnover``. It is **off by default**: the
    execution layer already works orders against a 20-day ADV participation cap,
    which spreads a large basket change over several days with the *right*
    constraint (market liquidity rather than an arbitrary weight delta). Turning
    the portfolio-level cap on as well cost 14pp of total return and 0.05 of
    Sharpe for 1.1pp of drawdown on the full sample, because the two constraints
    delay the same trades twice. It remains available for reproducing that
    experiment.
    """
    return _build_portfolio_impl(
        weights,
        panel,
        benchmark=benchmark,
        risk=risk,
        use_regime=use_regime,
        use_vol_target=use_vol_target,
        use_turnover_cap=use_turnover_cap,
        vol_forecast=vol_forecast,
        ex_ante_vol=ex_ante_vol,
    )


def _build_portfolio_impl(
    weights: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    *,
    benchmark: pd.Series | None = None,
    risk: RiskConfig | None = None,
    use_regime: bool = False,
    use_vol_target: bool = True,
    use_turnover_cap: bool = False,
    vol_forecast: str = "realised",
    ex_ante_vol: pd.Series | None = None,
    use_breadth_timing: bool | None = None,
) -> pd.DataFrame:
    cfg = risk or RiskConfig()
    result = weights.reindex(columns=sorted(panel)).fillna(0.0)
    if use_breadth_timing is None:
        use_breadth_timing = cfg.use_breadth_timing
    if use_breadth_timing:
        scale = breadth_timing_scale(
            result.index,
            panel,
            lookback=cfg.breadth_lookback,
            history_months=cfg.breadth_history_months,
            de_risk=cfg.breadth_de_risk,
        )
        result = result.mul(scale, axis=0)
    if use_regime and benchmark is not None:
        result = regime_filter(result, benchmark)
    if use_vol_target:
        scaled = apply_volatility_target(
            result, panel, cfg, forecast=vol_forecast, ex_ante=ex_ante_vol
        )
        # Lock one sizing factor per selection episode. Recomputing a continuous
        # factor every day would create a stream of tiny rebalances.
        basket = result.gt(0)
        change = basket.ne(basket.shift(1)).any(axis=1).to_numpy()
        totals = result.sum(axis=1).to_numpy(dtype=float)
        scaled_totals = scaled.sum(axis=1).to_numpy(dtype=float)
        safe = np.where(totals > 1e-12, totals, 1.0)
        ratio = np.where(totals > 1e-12, scaled_totals / safe, 1.0)
        # Only the bars where the basket changes set a new factor; every other
        # bar carries the previous one forward.  Vectorising this replaces one
        # ``.loc[date]`` pair per bar (about 3k pandas calls per window) with a
        # single ffill, and the resulting series is identical.
        factors = pd.Series(
            np.where(change, ratio, np.nan), index=result.index, dtype=float
        ).ffill().fillna(1.0)
        result = result.mul(factors, axis=0)
    # Turnover is *primarily* controlled by the strategy's rebalance cadence and
    # ranking buffer. The explicit cap below additionally spreads a large basket
    # switch across several days, which mirrors how a real desk would work an
    # order against a participation limit.
    if use_turnover_cap:
        result = cap_turnover(result, cfg.max_daily_turnover)
    return result
