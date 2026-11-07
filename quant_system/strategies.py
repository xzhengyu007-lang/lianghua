"""Long-only cross-sectional strategies with explicit signal lags.

All strategies share one selection engine:

* indicators are computed on a price series that is forward-filled only inside
  each symbol's real bar range (suspensions do not poison rolling windows),
* a name is eligible only if it has a real bar on that date,
* the portfolio is rebalanced on a fixed cadence and incumbents are protected
  by a rank buffer against the *full* cross-section, not merely against the
  names that would have been selected anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import availability, price_frame
from .indicators import cross_sectional_rank, cross_sectional_zscore, realised_volatility, rsi, sma
from .sectors import sector_of


def rank_scores(scores: pd.DataFrame, eligible: pd.DataFrame | None = None) -> pd.DataFrame:
    """Descending cross-sectional rank (1 = best) among eligible names."""
    valid = scores.where(eligible) if eligible is not None else scores
    return valid.rank(axis=1, ascending=False, method="first")


def _select_weights(
    scores: pd.DataFrame,
    top_n: int,
    *,
    eligible: pd.DataFrame | None = None,
    available: pd.DataFrame | None = None,
    max_per_sector: int | None = None,
    rebalance_days: int = 20,
    buffer_rank: float = 1.5,
    rebalance_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Rank-buffered, equal-weight, sector-capped selection.

    An incumbent is kept as long as it stays inside the top
    ``ceil(top_n * buffer_rank)`` of the eligible cross-section. Vacant slots
    are filled by the best non-incumbents, subject to the sector cap.

    ``rebalance_dates`` replaces the fixed ``rebalance_days`` cadence with an
    explicit calendar (e.g. month-end). A forced rebalance still happens when a
    holding has passed its last real bar, so a delisted name cannot be frozen
    into the book.
    """
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if rebalance_days <= 0:
        raise ValueError("rebalance_days must be positive")
    if buffer_rank < 1.0:
        raise ValueError("buffer_rank must be >= 1")
    valid = scores.replace([np.inf, -np.inf], np.nan)
    if eligible is not None:
        valid = valid.where(eligible.reindex_like(valid).fillna(False))
    result = pd.DataFrame(0.0, index=valid.index, columns=valid.columns)
    sectors = {column: sector_of(column) for column in valid.columns}
    # A temporary suspension is not a delisting: the name can reappear. Only a
    # date after its final real bar is irreversible. Availability comes from
    # raw bars, never from the score, because a name can trade after its trend
    # filter or factor window has gone stale.
    if available is None:
        usable = valid.notna()
    else:
        usable = available.reindex_like(valid).fillna(False).astype(bool)
    last_real: dict[str, pd.Timestamp] = {}
    for column in usable.columns:
        real_bars = usable.index[usable[column]]
        if len(real_bars):
            last_real[column] = real_bars[-1]
    if rebalance_dates is not None:
        calendar = pd.DatetimeIndex(sorted(pd.DatetimeIndex(rebalance_dates).unique()))
        on_calendar = valid.index.isin(calendar)
    else:
        calendar = None
        on_calendar = None
    current: list[str] = []
    days_held = rebalance_days

    for date, row in valid.iterrows():
        ranked = row.dropna().sort_values(ascending=False)
        expired = [s for s in current if date > last_real.get(s, pd.NaT)]
        forced = bool(expired)
        if expired:
            current = [s for s in current if s not in expired]
        # With an explicit calendar the cadence is *only* the calendar; mixing
        # in ``days_held`` would fire an extra rebalance on the bar after the
        # month-end signal and silently change the strategy.
        scheduled = (
            bool(on_calendar[valid.index.get_loc(date)])
            if calendar is not None
            else days_held >= rebalance_days
        )
        # Under an explicit calendar the very first position must also wait for
        # a calendar date; otherwise the book is opened on an arbitrary bar and
        # the strategy is no longer the one under test.
        open_on_schedule = calendar is not None and not current and not forced
        if (not current and not open_on_schedule) or scheduled or forced:
            keep_limit = int(np.ceil(top_n * buffer_rank))
            rank_of = {symbol: i for i, symbol in enumerate(ranked.index)}
            survivors = [
                s for s in current if s in rank_of and rank_of[s] < keep_limit
            ]
            target_count = min(top_n, len(ranked))
            chosen: list[str] = []
            counts: dict[str, int] = {}
            for symbol in survivors:
                if len(chosen) >= target_count:
                    break
                sector = sectors[symbol]
                if max_per_sector is not None and counts.get(sector, 0) >= max_per_sector:
                    continue
                chosen.append(symbol)
                counts[sector] = counts.get(sector, 0) + 1
            for symbol in ranked.index:
                if len(chosen) >= target_count:
                    break
                if symbol in chosen:
                    continue
                sector = sectors[symbol]
                if max_per_sector is not None and counts.get(sector, 0) >= max_per_sector:
                    continue
                chosen.append(symbol)
                counts[sector] = counts.get(sector, 0) + 1
            current = chosen
            days_held = 0
        if current:
            result.loc[date, current] = 1.0 / len(current)
            days_held += 1
    return result


def interval_dates(
    index: pd.DatetimeIndex,
    every: int,
    *,
    offset: int = 0,
    reset: str = "start",
) -> pd.DatetimeIndex:
    """Every ``every``-th trading bar, shifted by ``offset`` bars.

    ``reset="start"`` is the historical cadence: rebalance on bars 20, 40,
    60, ... counted from the first bar of the sample.  That anchor moves when
    the research window moves, which is why shifting the start by one session
    can change a full-sample result by more than 50pp (RESEARCH_LOG 23.20).

    ``reset="year"`` restarts the count on the first trading bar of each
    calendar year.  The anchor is then a property of the market calendar
    rather than of the backtest window, so the same date rebalances regardless
    of where the sample begins.  A window that starts mid-year still has an
    arbitrary phase for that first partial year, but the error cannot
    accumulate: it is re-anchored every January.
    """
    if every <= 0:
        raise ValueError("every must be positive")
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if reset not in {"start", "year"}:
        raise ValueError("reset must be 'start' or 'year'")
    if len(index) <= offset:
        return pd.DatetimeIndex([])
    if reset == "start":
        positions = range(offset, len(index), every)
        return pd.DatetimeIndex([index[position] for position in positions])
    years = pd.Series(index).dt.year.to_numpy()
    starts = np.flatnonzero(np.r_[True, years[1:] != years[:-1]])
    positions: list[int] = []
    for block, start in enumerate(starts):
        stop = starts[block + 1] if block + 1 < len(starts) else len(index)
        positions.extend(range(start + offset, stop, every))
    return pd.DatetimeIndex([index[position] for position in positions])


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading bar of each calendar month in ``index``."""
    if len(index) == 0:
        return pd.DatetimeIndex([])
    periods = pd.Series(index).dt.to_period("M")
    # ``shift(-1)`` introduces a trailing NA, which must not be treated as a
    # mask value: the final bar is always the last bar of its month.
    is_last = (periods != periods.shift(-1)).fillna(True).to_numpy(dtype=bool)
    return pd.DatetimeIndex(index[is_last])


@dataclass
class _BaseStrategy:
    top_n: int = 6
    max_per_sector: int | None = 2
    rebalance_days: int = 20
    buffer_rank: float = 1.5
    rebalance_on: str = "interval"
    # Calendar phase offset for the interval cadence.  ``0`` reproduces the
    # historical "every N bars from the first bar" behaviour.  A non-zero value
    # delays the first rebalance, which is how the anchor-sensitivity study
    # shifts the whole calendar by one day at a time.
    rebalance_offset: int = 0
    # ``start`` reproduces the historical "count from the first bar" cadence.
    # ``year`` re-anchors the cadence every January, which makes the rebalance
    # calendar independent of where the research window happens to begin.
    rebalance_reset: str = "start"
    name: str = "base"

    def score(self, panel: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError

    def select(
        self,
        scores: pd.DataFrame,
        eligible: pd.DataFrame,
        available: pd.DataFrame,
        *,
        offset: int | None = None,
        rebalance_on: str | None = None,
    ) -> pd.DataFrame:
        """Turn cached scores into target weights for one rebalance calendar.

        Splitting this out of :meth:`generate` lets a phase sweep reuse the
        factor computation.  Scoring is by far the expensive half, and the
        anchor study needs the *identical* cross-section under every phase;
        recomputing it 20 times would be both slower and a chance for drift.
        """
        cadence = self.rebalance_on if rebalance_on is None else rebalance_on
        if cadence not in {"interval", "month_end"}:
            raise ValueError("rebalance_on must be 'interval' or 'month_end'")
        phase = self.rebalance_offset if offset is None else offset
        if phase < 0:
            raise ValueError("rebalance offset must be >= 0")
        calendar = None
        if cadence == "month_end":
            calendar = month_end_dates(scores.index)
        elif phase or self.rebalance_reset != "start":
            calendar = interval_dates(
                scores.index,
                self.rebalance_days,
                offset=phase,
                reset=self.rebalance_reset,
            )
        return _select_weights(
            scores,
            self.top_n,
            eligible=eligible,
            available=available,
            max_per_sector=self.max_per_sector,
            rebalance_days=self.rebalance_days,
            buffer_rank=self.buffer_rank,
            rebalance_dates=calendar,
        )

    def generate(self, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        scores, eligible = self.score(panel)
        close = price_frame(panel, "close")
        return self.select(scores, eligible, _traded(panel, close))

    def prepare(self, panel: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Score a panel once, returning everything a phase sweep needs."""
        scores, eligible = self.score(panel)
        close = price_frame(panel, "close")
        return scores, eligible, _traded(panel, close)


def _traded(panel: dict[str, pd.DataFrame], close: pd.DataFrame) -> pd.DataFrame:
    return availability(panel).reindex(columns=close.columns).fillna(False)


@dataclass
class MomentumStrategy(_BaseStrategy):
    lookback: int = 60
    skip: int = 5
    vol_window: int = 40
    trend_window: int = 120
    name: str = "momentum"

    def score(self, panel: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        close = price_frame(panel, "close")
        momentum = close.shift(self.skip) / close.shift(self.lookback + self.skip) - 1.0
        vol = realised_volatility(panel, self.vol_window).reindex_like(close)
        scores = cross_sectional_zscore(momentum) - 0.5 * cross_sectional_zscore(vol)
        eligible = _traded(panel, close) & (close > sma(close, self.trend_window))
        return scores, eligible


@dataclass
class MeanReversionStrategy(_BaseStrategy):
    rsi_window: int = 14
    entry_rsi: float = 35.0
    exit_rsi: float = 55.0
    trend_window: int = 120
    name: str = "mean_reversion"

    def score(self, panel: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        close = price_frame(panel, "close")
        rsi_values = pd.DataFrame(
            {column: rsi(close[column], self.rsi_window) for column in close.columns}
        ).reindex(columns=close.columns)
        trend = _traded(panel, close) & (close > sma(close, self.trend_window))
        scores = (self.entry_rsi - rsi_values).where(trend)
        scores = scores.where(scores > 0)
        return scores, trend


@dataclass
class LowVolatilityStrategy(_BaseStrategy):
    vol_window: int = 60
    trend_window: int = 120
    name: str = "low_volatility"

    def score(self, panel: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        close = price_frame(panel, "close")
        vol = realised_volatility(panel, self.vol_window).reindex_like(close)
        eligible = _traded(panel, close) & (close > sma(close, self.trend_window))
        return -vol, eligible


@dataclass
class BreakoutStrategy(_BaseStrategy):
    entry_window: int = 55
    exit_window: int = 20
    name: str = "breakout"

    def score(self, panel: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        close = price_frame(panel, "close")
        highs = price_frame(panel, "high")
        entry_high = highs.shift(1).rolling(self.entry_window).max()
        exit_low = close.shift(1).rolling(self.exit_window).min()
        breakout = close > entry_high
        recent_break = breakout.rolling(5, min_periods=1).max().astype(bool)
        broken = (close < exit_low).rolling(3, min_periods=1).max().astype(bool)
        eligible = _traded(panel, close) & recent_break & ~broken
        scores = (close / entry_high - 1.0).where(eligible)
        return scores, eligible


@dataclass
class MultiFactorStrategy(_BaseStrategy):
    momentum_window: int = 60
    trend_window: int = 120
    vol_window: int = 40
    weights: dict[str, float] = field(
        default_factory=lambda: {"momentum": 0.5, "trend": 0.25, "low_vol": 0.25}
    )
    name: str = "multi_factor"

    def score(self, panel: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        close = price_frame(panel, "close")
        momentum = cross_sectional_zscore(close.shift(5) / close.shift(self.momentum_window + 5) - 1.0)
        trend = cross_sectional_zscore(close / sma(close, self.trend_window) - 1.0)
        vol = cross_sectional_zscore(
            -realised_volatility(panel, self.vol_window).reindex_like(close)
        )
        scores = (
            self.weights.get("momentum", 0.0) * momentum
            + self.weights.get("trend", 0.0) * trend
            + self.weights.get("low_vol", 0.0) * vol
        )
        eligible = _traded(panel, close) & (close > sma(close, self.trend_window))
        return scores, eligible


@dataclass
class LowVolCoreStrategy(_BaseStrategy):
    """Low-volatility core with a weak trend confirmation.

    The factor study showed a positive IC for low volatility at every horizon
    and a negative 20-day IC for 60-day momentum in this universe, so low
    volatility is the primary signal and trend only confirms it.
    """

    top_n: int = 8
    vol_window: int = 60
    trend_window: int = 120
    vol_weight: float = 0.8
    trend_weight: float = 0.2
    # The published baseline keeps only names above their trend line, which
    # leaves cash when the whole cross-section is below it. These switches are
    # research-only and default to the exact historical behaviour.
    trend_filter: bool = True
    trend_fallback: bool = False
    fallback_penalty: float = 5.0
    name: str = "low_vol_core"

    def score(self, panel: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        close = price_frame(panel, "close")
        traded = _traded(panel, close)
        vol = realised_volatility(panel, self.vol_window).reindex_like(close)
        trend_ok = traded & (close > sma(close, self.trend_window))
        trend = close / sma(close, self.trend_window) - 1.0
        scores = self.vol_weight * cross_sectional_zscore(-vol) + self.trend_weight * cross_sectional_zscore(trend)
        if not self.trend_filter:
            return scores, traded
        if self.trend_fallback:
            if self.fallback_penalty < 0:
                raise ValueError("fallback_penalty must be non-negative")
            scores = scores.where(trend_ok, scores - self.fallback_penalty)
            return scores, traded
        return scores, trend_ok


@dataclass
class SectorNeutralLowVolStrategy(_BaseStrategy):
    """Low volatility ranked within each sector, then balanced across sectors.

    The full-sample study showed the raw low-vol basket's excess return is
    mostly sector allocation (banks/utilities). Ranking inside sectors keeps
    the cross-sectional information but removes the industry tilt. ``per_sector``
    names are held from every sector that has enough eligible names, equal
    weight inside a sector and equal weight across sectors.
    """

    per_sector: int = 3
    vol_window: int = 60
    trend_window: int = 120
    name: str = "sector_neutral_lowvol"

    def score(self, panel: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        close = price_frame(panel, "close")
        vol = realised_volatility(panel, self.vol_window).reindex_like(close)
        trend = close / sma(close, self.trend_window) - 1.0
        eligible = _traded(panel, close) & (close > sma(close, self.trend_window))
        low_vol = (-vol).where(eligible)
        sectors = {column: sector_of(column) for column in low_vol.columns}
        scores = low_vol.copy()
        for sector in set(sectors.values()):
            columns = [c for c in low_vol.columns if sectors[c] == sector]
            if len(columns) < 2:
                continue
            # Percentile inside the sector so every sector is comparable.
            scores[columns] = low_vol[columns].rank(axis=1, pct=True)
        return scores, eligible

    def generate(self, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        scores, eligible = self.score(panel)
        if self.per_sector <= 0:
            raise ValueError("per_sector must be positive")
        close = price_frame(panel, "close")
        sectors = pd.Series({c: sector_of(c) for c in close.columns})
        universe_size = max(1, len(sectors))
        # Hold ``per_sector`` names from every sector, so the global basket is
        # much wider than per_sector and the sector cap must equal per_sector.
        raw = _select_weights(
            scores,
            self.per_sector * universe_size,
            eligible=eligible,
            available=_traded(panel, close),
            max_per_sector=self.per_sector,
            rebalance_days=self.rebalance_days,
            buffer_rank=self.buffer_rank,
        )
        # Re-weight: equal inside each sector, equal across active sectors.
        active_sectors = raw.T.groupby(sectors).sum().T.gt(1e-12)
        counts = active_sectors.sum(axis=1).replace(0, np.nan)
        inside = raw.T.groupby(sectors).transform(lambda x: x.sum()).T.replace(0, np.nan)
        result = raw / inside
        return result.div(counts, axis=0).fillna(0.0)


@dataclass
class LowVolPlusStrategy(_BaseStrategy):
    """Low volatility plus optional extra signals (illiquidity, 12-1 momentum).

    Built as the testbed for RESEARCH_LOG 23.41: every extra signal is a weight
    that defaults to zero, so the base behaviour is identical to ``low_vol_core``.

    RESEARCH_LOG 23.41: after controlling for 60-day volatility, illiquidity
    still carries an IC of +0.09 inside the calmest third of the universe, so it
    is the one candidate that is genuinely orthogonal to the incumbent signal.
    The risk is capacity, not information: illiquid names are small, so this
    variant must be judged with impact costs switched on.
    """

    top_n: int = 8
    vol_window: int = 60
    trend_window: int = 120
    illiquidity_window: int = 20
    vol_weight: float = 0.5
    trend_weight: float = 0.2
    illiquidity_weight: float = 0.3
    momentum_weight: float = 0.0
    momentum_window: int = 252
    momentum_skip: int = 21
    blend: str = "zscore"
    calm_fraction: float = 0.0
    name: str = "low_vol_plus"

    def _standardise(self, frame: pd.DataFrame) -> pd.DataFrame:
        # Amihud has a fat right tail: its cross-sectional z-score reaches +20
        # while a volatility z-score lives inside +/-2, so a "30% illiquidity
        # weight" is really a 95% weight. Percentile ranks keep every signal on
        # the same bounded scale (RESEARCH_LOG 23.41).
        if self.blend == "rank":
            return cross_sectional_rank(frame, ascending=True)
        return cross_sectional_zscore(frame)

    def score(self, panel: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        close = price_frame(panel, "close")
        volume = price_frame(panel, "volume").reindex_like(close)
        vol = realised_volatility(panel, self.vol_window).reindex_like(close)
        trend = close / sma(close, self.trend_window) - 1.0
        amount = (close * volume).replace(0.0, np.nan)
        illiquidity = (close.pct_change(fill_method=None).abs() / amount).rolling(
            self.illiquidity_window
        ).mean()
        momentum = close.shift(self.momentum_skip) / close.shift(self.momentum_window) - 1.0
        scores = (
            self.vol_weight * self._standardise(-vol)
            + self.trend_weight * self._standardise(trend)
            + self.illiquidity_weight * self._standardise(illiquidity)
        )
        if self.momentum_weight:
            scores = scores + self.momentum_weight * self._standardise(momentum)
        eligible = _traded(panel, close) & (close > sma(close, self.trend_window))
        if self.calm_fraction and self.calm_fraction < 1.0:
            # The illiquidity premium is only monotone *inside* the calm third;
            # unconditionally it saturates by the second quintile, so ranking
            # the whole universe by illiquidity buys risk without return.
            calm = cross_sectional_rank(-vol, ascending=False) <= self.calm_fraction
            eligible = eligible & calm
            scores = scores.where(calm)
        return scores, eligible


STRATEGY_REGISTRY: dict[str, type] = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "low_volatility": LowVolatilityStrategy,
    "breakout": BreakoutStrategy,
    "multi_factor": MultiFactorStrategy,
    "low_vol_core": LowVolCoreStrategy,
    "sector_neutral_lowvol": SectorNeutralLowVolStrategy,
    "low_vol_plus": LowVolPlusStrategy,
    "low_vol_illiquid": LowVolPlusStrategy,
}


def build_strategy(name: str, **params) -> _BaseStrategy:
    try:
        cls = STRATEGY_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(STRATEGY_REGISTRY)}") from exc
    return cls(**params)
