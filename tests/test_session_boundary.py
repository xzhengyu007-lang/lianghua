"""Which session's close *must* be in the cache by now?

The gate used to ask ``previous_trading_day(today)``.  On Monday night that answers
"Friday" -- and the calendar it asks is derived from the same cache, so a Friday
cache certified itself as current and the desk offered Friday's book for Tuesday's
open.  The right question is "which session has closed by now", which the cache
cannot answer about itself; the exchange's own print date can.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_system.realtime import (
    Board,
    PriceLimit,
    Quote,
    TradingCalendar,
    last_completed_session,
    previous_trading_day,
    session_gap,
)

HOLIDAY = (pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-08"))  # 国庆


def _days(*, through: str = "2026-11-06") -> list[dt.date]:
    """Business days from 2026-09-21 through ``through``, minus 国庆."""
    return [
        day.date()
        for day in pd.bdate_range("2026-09-21", through)
        if not (HOLIDAY[0] <= day <= HOLIDAY[1])
    ]


@pytest.fixture()
def cal() -> TradingCalendar:
    """A calendar that ends on Friday 2026-11-06 -- exactly the failing situation."""
    return TradingCalendar(_days())


@pytest.fixture()
def cal_with_monday() -> TradingCalendar:
    """The same calendar once Monday's bar arrives."""
    return TradingCalendar(_days(through="2026-11-09"))


# --------------------------------------------------------------------------- #
# the regression itself
# --------------------------------------------------------------------------- #
def test_the_date_question_and_the_session_question_disagree_on_monday_night(cal) -> None:
    """This is the bug in one line: both were "correct", only one was useful."""
    monday_night = pd.Timestamp("2026-11-09 23:00")
    assert previous_trading_day(monday_night.date(), calendar=cal) == dt.date(2026, 11, 6)
    assert last_completed_session(monday_night, calendar=cal) == dt.date(2026, 11, 9)


# --------------------------------------------------------------------------- #
# "has the session closed yet?" -- the 15:00 boundary
# --------------------------------------------------------------------------- #
def test_after_the_close_on_a_trading_day_todays_close_is_required(cal_with_monday) -> None:
    assert last_completed_session(
        "2026-11-09 16:30", calendar=cal_with_monday
    ) == dt.date(2026, 11, 9)


def test_before_the_open_nothing_is_required_beyond_the_previous_session(cal_with_monday) -> None:
    assert last_completed_session(
        "2026-11-09 08:00", calendar=cal_with_monday
    ) == dt.date(2026, 11, 6)


def test_inside_the_session_todays_close_is_not_expected_yet(cal_with_monday) -> None:
    assert last_completed_session(
        "2026-11-09 10:00", calendar=cal_with_monday
    ) == dt.date(2026, 11, 6)


def test_weekend_expects_fridays_close(cal) -> None:
    assert last_completed_session("2026-11-07 12:00", calendar=cal) == dt.date(2026, 11, 6)


def test_a_holiday_inside_coverage_expects_the_last_real_session(cal) -> None:
    """The cache is authoritative inside coverage: no bar on 10-01 means no session."""
    assert last_completed_session("2026-10-01 12:00", calendar=cal) == dt.date(2026, 9, 30)


def test_the_witness_settles_a_holiday_beyond_coverage(cal) -> None:
    """Feed still prints 09-30 on 10-02 -> the market really did not open."""
    assert last_completed_session(
        "2026-10-02 12:00", calendar=cal, witness="2026-09-30"
    ) == dt.date(2026, 9, 30)


# --------------------------------------------------------------------------- #
# the witness: the one fact that does not come from the cache
# --------------------------------------------------------------------------- #
def test_a_witness_from_today_after_the_close_proves_the_session_ran(cal) -> None:
    assert last_completed_session(
        "2026-11-09 23:00", calendar=cal, witness="2026-11-09"
    ) == dt.date(2026, 11, 9)


def test_a_witness_from_today_during_the_session_proves_nothing_about_the_close(cal) -> None:
    assert last_completed_session(
        "2026-11-09 10:00", calendar=cal, witness="2026-11-09"
    ) == dt.date(2026, 11, 6)


def test_a_month_old_witness_means_a_dead_feed_not_a_closed_market(cal) -> None:
    """A dead feed must not freeze the gate: fall back to the calendar."""
    assert last_completed_session(
        "2026-11-09 23:00", calendar=cal, witness="2026-08-01"
    ) == dt.date(2026, 11, 9)


def test_a_witness_from_the_future_is_ignored(cal) -> None:
    assert last_completed_session(
        "2026-11-09 23:00", calendar=cal, witness="2026-11-10"
    ) == dt.date(2026, 11, 9)


def test_without_calendar_coverage_the_weekday_rule_is_the_honest_default() -> None:
    empty = TradingCalendar([])
    assert last_completed_session("2026-11-09 23:00", calendar=empty) == dt.date(2026, 11, 9)
    assert last_completed_session("2026-11-08 23:00", calendar=empty) == dt.date(2026, 11, 6)


def test_a_tz_aware_moment_is_honoured(cal) -> None:
    moment = pd.Timestamp("2026-11-09 16:00", tz="Asia/Shanghai")
    assert last_completed_session(moment, calendar=cal) == dt.date(2026, 11, 9)


# --------------------------------------------------------------------------- #
# session_gap: sessions behind, not calendar days behind
# --------------------------------------------------------------------------- #
def test_one_session_behind_is_what_friday_to_monday_means(cal) -> None:
    assert session_gap("2026-11-06", "2026-11-09", calendar=cal) == 1


def test_a_week_behind_counts_five_sessions(cal) -> None:
    assert session_gap("2026-11-06", "2026-11-13", calendar=cal) == 5


def test_a_holiday_week_is_one_session_not_seven_days(cal) -> None:
    assert session_gap("2026-09-30", "2026-10-09", calendar=cal) == 1


def test_current_data_has_no_gap(cal) -> None:
    assert session_gap("2026-11-06", "2026-11-06", calendar=cal) == 0
    assert session_gap(None, "2026-11-06", calendar=cal) == 0


def test_data_ahead_of_expectation_is_not_a_negative_gap(cal) -> None:
    assert session_gap("2026-11-09", "2026-11-06", calendar=cal) == 0


# --------------------------------------------------------------------------- #
# Board.print_date -- the witness, extracted from the round itself
# --------------------------------------------------------------------------- #
def test_print_date_takes_the_newest_exchange_stamp() -> None:
    board = Board(
        quotes={
            "sh600036": SimpleNamespace(quote_time=pd.Timestamp("2026-11-09 14:59:58")),
            "sz000001": SimpleNamespace(quote_time=pd.Timestamp("2026-11-09 15:00:03")),
        }
    )
    assert board.print_date == dt.date(2026, 11, 9)


def test_print_date_is_none_when_nothing_printed() -> None:
    empty = Board()
    assert empty.print_date is None
    assert Board(quotes={"x": SimpleNamespace(quote_time=pd.NaT)}).print_date is None


def _quote(symbol: str, stamp: str) -> Quote:
    return Quote(
        symbol=symbol,
        name="测试",
        price=10.0,
        prev_close=10.0,
        open=10.0,
        high=10.0,
        low=10.0,
        volume_lots=0.0,
        amount_wan=0.0,
        bid1=9.99,
        ask1=10.01,
        quote_time=pd.Timestamp(stamp),
        fetched_at=pd.Timestamp(stamp),
        limit=PriceLimit(pct=0.10, up=11.0, down=9.0, source="computed"),
        status="normal",
    )


def test_the_board_payload_carries_the_print_date() -> None:
    board = Board(quotes={"sh600036": _quote("sh600036", "2026-11-09 15:00:03")})
    assert board.to_dict()["print_date"] == "2026-11-09"
    assert Board().to_dict()["print_date"] is None
