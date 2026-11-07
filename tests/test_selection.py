from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_system.strategies import _select_weights, build_strategy, interval_dates


def _scores(values: dict[str, list[float]]) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=len(next(iter(values.values()))))
    return pd.DataFrame(values, index=index)


def test_rank_buffer_keeps_incumbent_against_outsider() -> None:
    # A held name slipping from 1st to 4th of 4 must survive a 2-name basket
    # with a 1.5x buffer (keep window = 3), while a 5th-place name cannot.
    scores = _scores(
        {
            "sh600000": [10, 10, 5.0],
            "sh600004": [9, 9, 9.0],
            "sh600008": [8, 8, 8.0],
            "sh600010": [7, 7, 7.0],
        }
    )
    weights = _select_weights(scores, 2, rebalance_days=1, buffer_rank=1.5)
    first = set(weights.iloc[0][weights.iloc[0] > 0].index)
    second = set(weights.iloc[1][weights.iloc[1] > 0].index)
    third = set(weights.iloc[2][weights.iloc[2] > 0].index)
    assert first == {"sh600000", "sh600004"}
    assert second == {"sh600000", "sh600004"}
    assert third == {"sh600004", "sh600008"}


def test_sector_neutral_strategy_balances_sectors(sample_panel) -> None:
    strategy = build_strategy(
        "sector_neutral_lowvol",
        per_sector=1,
        vol_window=20,
        trend_window=40,
        rebalance_days=5,
    )
    weights = strategy.generate(sample_panel)
    totals = weights.sum(axis=1)
    active = totals[totals > 1e-9]
    assert len(active) > 0
    assert np.allclose(active.to_numpy(), 1.0)
    # More than one name must be held once several sectors are available.
    assert weights.gt(0).sum(axis=1).max() >= 1


def test_rebalance_cadence_is_respected() -> None:
    # The incumbent is held for exactly ``rebalance_days`` bars and is then
    # replaced by the current top name.
    scores = _scores({"sh600000": [5, 4, 3, 2, 1], "sh601318": [1, 2, 3, 4, 5]})
    weights = _select_weights(scores, 1, rebalance_days=3, buffer_rank=1.0)
    held = weights.gt(0).idxmax(axis=1)
    assert held.iloc[0] == "sh600000"
    assert held.iloc[2] == "sh600000"
    assert held.iloc[3] == "sh601318"
    assert held.iloc[4] == "sh601318"


def test_sector_cap_limits_concentration() -> None:
    scores = _scores({"600519": [5], "000858": [4], "601318": [3], "600036": [2]})
    weights = _select_weights(scores, 3, max_per_sector=1, rebalance_days=1)
    chosen = set(weights.iloc[0][weights.iloc[0] > 0].index)
    assert len(chosen) <= 3
    sectors = {s: __import__("quant_system.sectors", fromlist=["sector_of"]).sector_of(s) for s in chosen}
    assert len(set(sectors.values())) == len(sectors)


def test_all_strategies_hold_cash_until_signals_exist(sample_panel) -> None:
    for name in ("momentum", "low_volatility", "low_vol_core", "multi_factor", "breakout", "mean_reversion"):
        weights = build_strategy(name).generate(sample_panel)
        assert not weights.empty
        assert (weights >= 0).all().all()
        row_sums = weights.sum(axis=1)
        assert row_sums.max() <= 1.0 + 1e-9
        assert weights.iloc[:30].sum().sum() == pytest.approx(0.0)


def test_weights_never_exceed_position_cap(sample_panel) -> None:
    # Raw strategy weights are equal-weight within the basket; the hard
    # per-name cap is applied later by the backtester.
    weights = build_strategy("low_vol_core", top_n=3).generate(sample_panel)
    assert float(weights.sum(axis=1).max()) <= 1.0 + 1e-9



def test_incumbent_is_removed_after_its_last_bar() -> None:
    # A delisted name must not be carried as a stale holding once it has passed
    # its final real bar. A short suspension, by contrast, must keep the
    # incumbent because the name can reappear.
    index = pd.bdate_range("2024-01-02", periods=4)
    scores = pd.DataFrame(
        {
            "sh600000": [10.0, 10.0, np.nan, np.nan],
            "sh600004": [np.nan, np.nan, 7.0, 7.0],
            "sh600008": [5.0, 5.0, 5.0, 5.0],
            "sh600010": [4.0, 4.0, 6.0, 4.0],
        },
        index=index,
    )
    weights = _select_weights(scores, 2, rebalance_days=10, buffer_rank=2.0)
    held = [set(row[row > 0].index) for _, row in weights.iterrows()]
    assert held[0] == {"sh600000", "sh600008"}
    assert held[1] == {"sh600000", "sh600008"}
    assert "sh600000" not in held[2]
    assert "sh600000" not in held[3]
    assert held[2] == {"sh600004", "sh600008"}


def test_short_suspension_does_not_trigger_forced_rebalance() -> None:
    index = pd.bdate_range("2024-01-02", periods=4)
    scores = pd.DataFrame(
        {
            "sh600000": [10.0, np.nan, np.nan, 9.0],
            "sh600004": [5.0, 8.0, 8.0, 5.0],
            "sh600008": [4.0, 4.0, 4.0, 4.0],
        },
        index=index,
    )
    weights = _select_weights(scores, 2, rebalance_days=10, buffer_rank=2.0)
    held = [set(row[row > 0].index) for _, row in weights.iterrows()]
    assert held[1] == {"sh600000", "sh600004"}
    assert held[3] == {"sh600000", "sh600004"}


def test_month_end_calendar_replaces_the_fixed_cadence() -> None:
    """An explicit calendar must not be mixed with ``rebalance_days``.

    A first version applied both triggers and fired an extra rebalance on the
    bar after month-end, which silently changed the strategy under test. This
    pins the intended behaviour: membership changes only on the supplied dates
    (plus forced exits), never on a 20-bar timer.
    """
    from quant_system.strategies import month_end_dates

    index = pd.bdate_range("2024-01-02", periods=45)
    values = [float(i) for i in range(len(index))]
    scores = pd.DataFrame(
        {
            "sh600000": [100.0 - v for v in values],
            "sh600004": values,
            "sh600008": [50.0] * len(index),
        },
        index=index,
    )
    ends = month_end_dates(index)
    assert len(ends) == 3
    weights = _select_weights(
        scores, 1, rebalance_days=1, buffer_rank=1.0, rebalance_dates=ends
    )
    held = weights.gt(0)
    # Nothing may be held before the first calendar date: the fixed cadence is
    # fully replaced, not merely supplemented.
    assert not held.loc[: ends[0]].iloc[:-1].any().any()
    # Membership may only change on a calendar date (a date where the ranking
    # does not change produces no membership change at all).
    changed = held.ne(held.shift(1)).any(axis=1)
    # The first bar is a false "change" because the previous row is all False;
    # ignore it and check the remaining transitions.
    transitions = index[changed].difference([index[0]])
    assert set(transitions) <= set(ends)
    assert transitions.min() >= ends[0]


def test_month_end_dates_marks_the_last_bar_of_each_month() -> None:
    from quant_system.strategies import month_end_dates

    index = pd.DatetimeIndex(
        ["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-29", "2024-03-01"]
    )
    assert [str(d.date()) for d in month_end_dates(index)] == [
        "2024-01-31",
        "2024-02-29",
        "2024-03-01",
    ]

def test_rebalance_offset_shifts_the_entire_calendar(sample_panel) -> None:
    """The anchor must be an explicit, testable parameter.

    A 20-day cadence with an arbitrary phase is the single largest source of
    path variance in this project (see RESEARCH_LOG 23.21), so the offset has
    to change *every* rebalance date, not just the first one.
    """
    from quant_system.strategies import interval_dates

    index = pd.bdate_range("2024-01-01", periods=100)
    base = interval_dates(index, 20, offset=0)
    shifted = interval_dates(index, 20, offset=1)
    assert list(base) == [index[i] for i in (0, 20, 40, 60, 80)]
    assert list(shifted) == [index[i] for i in (1, 21, 41, 61, 81)]


def test_rebalance_offset_changes_the_resulting_book(sample_panel) -> None:
    base = build_strategy("low_vol_core", top_n=3, rebalance_days=20).generate(sample_panel)
    shifted = build_strategy(
        "low_vol_core", top_n=3, rebalance_days=20, rebalance_offset=1
    ).generate(sample_panel)
    assert not base.equals(shifted)
    assert base.index.equals(shifted.index)


def test_year_reset_makes_the_rebalance_calendar_window_independent() -> None:
    """A year-anchored cadence must not move when the sample start moves.

    ``reset="start"`` counts bars from the first bar of the sample, so shifting
    the start by one session shifts every later rebalance and can change a
    full-sample result by >50pp (RESEARCH_LOG 23.20).  ``reset="year"``
    re-anchors each January; from the first full year onward the dates are a
    property of the market calendar alone.
    """
    index = pd.bdate_range("2020-01-02", "2024-12-31")
    full = interval_dates(index, 20, reset="year")
    for shift in (1, 3, 17):
        part = interval_dates(index[shift:], 20, reset="year")
        first_year = index[shift].year
        assert set(full[full.year > first_year]) == set(part[part.year > first_year])

    legacy = interval_dates(index, 20, reset="start")
    shifted = interval_dates(index[1:], 20, reset="start")
    # The old cadence really does drift: this is the bug, not a preference.
    assert set(legacy[legacy.year > 2020]) != set(shifted[shifted.year > 2020])


def test_year_reset_keeps_the_requested_phase_within_each_year() -> None:
    index = pd.bdate_range("2021-01-04", "2021-12-31")
    for offset in (0, 5, 13):
        dates = interval_dates(index, 20, offset=offset, reset="year")
        assert len(dates) > 0
        assert dates[0] == index[offset]
        assert all(index.get_loc(date) % 20 == offset % 20 for date in dates)
