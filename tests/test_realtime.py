"""Contract tests for the L1 snapshot layer (stage 1).

Nothing here touches the network: every transport is a fake, and the payloads are
built from the field map that was verified against a live ``qt.gtimg.cn``
response on 2026-11-09 (see ``quant_system/realtime.py``).  The tests that matter
most are the ugly ones -- suspension, provider sentinels, throttle, backoff and
the degraded path -- because those are the states in which a trading desk
silently lies to its user.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from quant_system.realtime import (
    CN_TZ,
    Board,
    PriceLimit,
    Quote,
    QuoteError,
    QuoteUnavailable,
    RateLimiter,
    SnapshotStore,
    TencentQuoteSource,
    annotate,
    chunks,
    fresh_floor,
    is_st,
    limit_pct,
    load_board,
    parse_payload,
    price_limit,
    refresh,
    sellable_shares,
    session_state,
    with_backoff,
)

TRADING_NOON = pd.Timestamp("2026-11-06 14:00:00", tz=CN_TZ)


# --------------------------------------------------------------------------- #
# payload builders
# --------------------------------------------------------------------------- #
def _payload(
    symbol: str = "sh600519",
    *,
    name: str = "贵州茅台",
    price: float = 1453.0,
    prev_close: float = 1452.0,
    open_: float = 1450.0,
    volume: float = 42848.0,
    when: str = "20261106161458",
    bid1: float = 1452.0,
    ask1: float = 1453.0,
    limit_up: float = 1597.2,
    limit_down: float = 1306.8,
    high: float = 1457.72,
    low: float = 1436.0,
    amount: float = 619953.0,
    fields: int = 88,
) -> str:
    """One ``v_xxx="..."`` record with the real field positions."""
    values = [""] * fields
    values[0] = "1"
    values[1] = name
    values[2] = symbol[2:]
    values[3] = str(price)
    values[4] = str(prev_close)
    values[5] = str(open_)
    values[6] = str(volume)
    values[9] = str(bid1)
    values[19] = str(ask1)
    values[30] = when
    values[33] = str(high)
    values[34] = str(low)
    values[37] = str(amount)
    values[47] = str(limit_up)
    values[48] = str(limit_down)
    return 'v_%s="%s";' % (symbol, "~".join(values))


def _price(quote: Quote) -> float:
    return quote.price


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def test_parse_reads_last_price_and_top_of_book() -> None:
    quotes = parse_payload(_payload(), fetched_at=TRADING_NOON)
    assert len(quotes) == 1
    quote = quotes[0]
    assert quote.symbol == "sh600519"
    assert quote.name == "贵州茅台"
    assert quote.price == pytest.approx(1453.0)
    assert quote.prev_close == pytest.approx(1452.0)
    assert quote.bid1 == pytest.approx(1452.0)
    assert quote.ask1 == pytest.approx(1453.0)
    assert quote.change_pct == pytest.approx(0.0689, abs=1e-3)
    # 2026-11-06 16:14:58 in Asia/Shanghai, not UTC
    assert quote.quote_time == pd.Timestamp("2026-11-06 16:14:58", tz=CN_TZ)


def test_parse_marks_a_suspended_name_suspended_not_unchanged() -> None:
    """002129 really printed price=prev_close, open=0, volume=0 on 2026-11-06."""
    payload = _payload(
        "sz002129", name="TCL中环", price=9.38, prev_close=9.38, open_=0.0,
        volume=0.0, bid1=0.0, ask1=0.0, limit_up=10.32, limit_down=8.44,
    )
    quote = parse_payload(payload, fetched_at=TRADING_NOON)[0]
    assert quote.status == "suspended"
    assert quote.can_buy is False and quote.can_sell is False


def test_parse_flags_limit_up_and_limit_down_at_the_boundary() -> None:
    up = _payload(price=1597.2, prev_close=1452.0)
    down = _payload(price=1306.8, prev_close=1452.0)
    assert parse_payload(up, fetched_at=TRADING_NOON)[0].status == "limit_up"
    quote = parse_payload(down, fetched_at=TRADING_NOON)[0]
    assert quote.status == "limit_down"
    # You can sell into a limit-up but you cannot buy one.
    up_quote = parse_payload(up, fetched_at=TRADING_NOON)[0]
    assert up_quote.can_sell is True and up_quote.can_buy is False
    assert quote.can_buy is False and quote.can_sell is False


def test_provider_sentinel_minus_one_falls_back_to_a_computed_limit() -> None:
    """The provider sends -1 for indices and sometimes for 北交所."""
    payload = _payload(price=34.28, prev_close=34.28, limit_up=-1.0, limit_down=-1.0, fields=87)
    quote = parse_payload(payload, fetched_at=TRADING_NOON)[0]
    assert quote.limit.source == "computed"
    assert quote.limit.pct == pytest.approx(0.10)
    assert quote.limit.up == pytest.approx(37.71)   # 34.28 * 1.10 = 37.708 -> 37.71
    assert quote.limit.down == pytest.approx(30.85)


def test_a_garbage_timestamp_is_not_reinterpreted_as_now() -> None:
    payload = _payload(when="0")
    quote = parse_payload(payload, fetched_at=TRADING_NOON)[0]
    assert quote.quote_time == TRADING_NOON
    annotated = annotate(quote, now=TRADING_NOON + pd.Timedelta(seconds=1))
    assert annotated.stale is False  # same instant, but never silently "fresh" for long


def test_malformed_records_are_skipped_not_zero_filled() -> None:
    payload = 'v_sh600519="1~broken";' + _payload()
    quotes = parse_payload(payload, fetched_at=TRADING_NOON)
    assert [q.symbol for q in quotes] == ["sh600519"]
    assert all(q.price > 0 for q in quotes)


# --------------------------------------------------------------------------- #
# limits
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "symbol,name,expected",
    [
        ("sh600519", "贵州茅台", 0.10),
        ("sz000001", "平安银行", 0.10),
        ("sz300750", "宁德时代", 0.20),
        ("sz301029", "怡合达", 0.20),
        ("sh688981", "中芯国际", 0.20),
        ("bj830799", "艾融软件", 0.30),
        ("sz002129", "ST中环", 0.05),      # main board ST keeps 5%
        ("sz300999", "*ST金龙", 0.20),     # ChiNext ST keeps 20%
    ],
)
def test_limit_pct_follows_the_board_and_st_rule(symbol: str, name: str, expected: float) -> None:
    assert limit_pct(symbol, name) == pytest.approx(expected)


def test_st_detection_tolerates_star_and_spacing() -> None:
    assert is_st("*ST德豪") is True
    assert is_st("ST 中环") is True
    assert is_st("宁德时代") is False


def test_price_limit_prefers_the_provider_rounding() -> None:
    limit = price_limit("sh600519", "贵州茅台", 1452.0, provider_up=1597.2, provider_down=1306.8)
    assert limit.source == "provider"
    assert (limit.up, limit.down) == (1597.2, 1306.8)


# --------------------------------------------------------------------------- #
# throttle / batching / backoff
# --------------------------------------------------------------------------- #
class FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.t = start
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


def test_rate_limiter_waits_the_remainder_of_the_interval() -> None:
    fake = FakeClock()
    limiter = RateLimiter(3.0, clock=fake.clock, sleep=fake.sleep)
    assert limiter.acquire() == pytest.approx(0.0)
    assert limiter.acquire() == pytest.approx(3.0)
    fake.t += 1.0
    assert limiter.acquire() == pytest.approx(2.0)
    assert fake.sleeps == pytest.approx([3.0, 2.0])


def test_chunks_splits_at_fifty_without_duplicates() -> None:
    symbols = ["sh%06d" % i for i in range(1, 121)] + ["sh999999", "sh000001"]
    batches = chunks(symbols, 50)
    assert [len(b) for b in batches] == [50, 50, 21]
    assert sum(len(b) for b in batches) == 121  # the repeated 000001 was dropped


def test_fetch_batches_at_fifty_and_throttles_every_request() -> None:
    fake = FakeClock()
    urls: list[str] = []

    def transport(url: str, *, timeout=None) -> bytes:
        urls.append(url)
        return _payload("sh600000", name="浦发银行", price=10.0, prev_close=9.9).encode("gbk")

    source = TencentQuoteSource(
        transport=transport,
        limiter=RateLimiter(3.0, clock=fake.clock, sleep=fake.sleep),
        batch_size=50,
    )
    symbols = ["sh%06d" % i for i in range(1, 121)]  # 120 names -> 3 requests
    source.fetch(symbols)
    assert len(urls) == 3
    assert all(url.count(",") + 1 <= 50 for url in urls)
    assert fake.sleeps == pytest.approx([3.0, 3.0])


def test_backoff_retries_with_exponential_waits_then_raises() -> None:
    fake = FakeClock()
    attempts: list[int] = []

    def transport(url: str, *, timeout=None) -> bytes:
        attempts.append(1)
        raise QuoteUnavailable("网络不可达")

    source = TencentQuoteSource(
        transport=transport,
        limiter=RateLimiter(0.0, clock=fake.clock, sleep=fake.sleep),
        attempts=3,
        base_wait=1.0,
        max_wait=8.0,  # the production cap is 1 s; this test pins the exponential rule itself
        sleep=fake.sleep,
    )
    with pytest.raises(QuoteError):
        source._fetch_batch(["sh600519"])
    assert len(attempts) == 3
    assert fake.sleeps == pytest.approx([1.0, 2.0])
    assert [row["wait"] for row in source.retry_log] == pytest.approx([1.0, 2.0])


def test_backoff_recovers_without_reporting_a_failure() -> None:
    fake = FakeClock()
    calls = {"n": 0}

    def transport(url: str, *, timeout=None) -> bytes:
        calls["n"] += 1
        if calls["n"] < 3:
            raise QuoteUnavailable("限流")
        return _payload().encode("gbk")

    source = TencentQuoteSource(
        transport=transport,
        limiter=RateLimiter(0.0, clock=fake.clock, sleep=fake.sleep),
        attempts=4,
        base_wait=1.0,
        sleep=fake.sleep,
    )
    quotes, missing, errors = source.fetch(["sh600519"])
    assert [q.symbol for q in quotes] == ["sh600519"]
    assert missing == [] and errors == []


def test_one_failing_batch_does_not_hide_the_healthy_ones() -> None:
    fake = FakeClock()

    def transport(url: str, *, timeout=None) -> bytes:
        requested = url.split("=", 1)[1].split(",")
        if "sz000002" in requested:
            raise QuoteUnavailable("HTTP 502")
        return "".join(
            _payload(symbol, name="x", price=10.0, prev_close=9.9, limit_up=10.89, limit_down=8.91)
            for symbol in requested
        ).encode("gbk")

    source = TencentQuoteSource(
        transport=transport,
        limiter=RateLimiter(0.0, clock=fake.clock, sleep=fake.sleep),
        batch_size=2,
        attempts=1,
        sleep=fake.sleep,
    )
    quotes, missing, errors = source.fetch(["sh600519", "sz000001", "sz000002"])
    assert [q.symbol for q in quotes] == ["sh600519", "sz000001"]
    assert missing == ["sz000002"]
    assert errors and "取价失败" in errors[0]


# --------------------------------------------------------------------------- #
# staleness
# --------------------------------------------------------------------------- #
def test_a_fresh_in_session_print_is_not_stale() -> None:
    quote = parse_payload(
        _payload(when="20261106135930"), fetched_at=TRADING_NOON
    )[0]
    annotated = annotate(quote, now=TRADING_NOON)
    assert annotated.stale is False
    assert annotated.age_seconds == pytest.approx(30.0)
    assert annotated.message.startswith("现价")


def test_yesterdays_close_is_stale_once_the_market_opens() -> None:
    friday_close = parse_payload(_payload(when="20261106150001"), fetched_at=TRADING_NOON)[0]
    monday_1000 = pd.Timestamp("2026-11-09 10:00:00", tz=CN_TZ)
    annotated = annotate(friday_close, now=monday_1000)
    assert annotated.stale is True
    assert annotated.can_buy is False  # an old price may never size an order
    assert "延迟" in annotated.message


def test_before_the_open_the_previous_close_is_current_but_labelled() -> None:
    """At 02:20 on Monday, Friday's close IS the latest possible print.

    Calling it "stale" would cry wolf every morning; calling it "现价" would be a
    lie.  The label carries the truth: 参考上一交易日收盘, and the timestamp is
    right there.
    """
    friday_close = parse_payload(_payload(when="20261106150001"), fetched_at=TRADING_NOON)[0]
    monday_pre_open = pd.Timestamp("2026-11-09 02:20:00", tz=CN_TZ)
    annotated = annotate(friday_close, now=monday_pre_open)
    assert annotated.stale is False
    assert "上一交易日收盘" in annotated.message
    assert annotated.age_seconds > 0


def test_a_print_from_two_sessions_ago_is_stale_before_the_open() -> None:
    thursday = parse_payload(_payload(when="20261105150001"), fetched_at=TRADING_NOON)[0]
    monday_pre_open = pd.Timestamp("2026-11-09 02:20:00", tz=CN_TZ)
    annotated = annotate(thursday, now=monday_pre_open)
    assert annotated.stale is True
    assert annotated.can_buy is False
    assert "延迟" in annotated.message


def test_a_lagging_print_during_the_session_is_stale_even_if_it_is_today() -> None:
    quote = parse_payload(_payload(when="20261106093005"), fetched_at=TRADING_NOON)[0]
    annotated = annotate(quote, now=TRADING_NOON, stale_after=60)
    assert annotated.stale is True


@pytest.mark.parametrize(
    "stamp,expected",
    [
        ("2026-11-06 02:20:00", "pre_open"),
        ("2026-11-06 09:20:00", "auction"),
        ("2026-11-06 10:00:00", "open"),
        ("2026-11-06 12:00:00", "lunch"),
        ("2026-11-06 14:30:00", "open"),
        ("2026-11-06 16:00:00", "after_close"),
        ("2026-11-07 10:00:00", "closed"),
    ],
)
def test_session_state_covers_every_window(stamp: str, expected: str) -> None:
    assert session_state(pd.Timestamp(stamp, tz=CN_TZ)) == expected


def test_fresh_floor_changes_with_the_session() -> None:
    floor_open, label_open = fresh_floor(pd.Timestamp("2026-11-09 10:00:00", tz=CN_TZ))
    assert floor_open == pd.Timestamp("2026-11-09 09:30:00", tz=CN_TZ)
    assert label_open == "今日盘中"
    floor_pre, _ = fresh_floor(pd.Timestamp("2026-11-09 02:20:00", tz=CN_TZ))
    assert floor_pre == pd.Timestamp("2026-11-06 09:30:00", tz=CN_TZ)


# --------------------------------------------------------------------------- #
# T+1
# --------------------------------------------------------------------------- #
def test_sellable_shares_respects_t_plus_one_and_freezes() -> None:
    assert sellable_shares(1000) == 1000
    assert sellable_shares(1000, bought_today=400) == 600
    assert sellable_shares(1000, bought_today=400, frozen=200) == 400
    assert sellable_shares(500, bought_today=500) == 0
    assert sellable_shares(0) == 0


# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #
def test_store_round_trip_keeps_every_field_and_the_collection_time(tmp_path) -> None:
    store = SnapshotStore(tmp_path / "quotes.sqlite")
    quote = parse_payload(_payload(), fetched_at=TRADING_NOON)[0]
    store.save([quote], run_at=TRADING_NOON, requested=1)
    stored = store.latest(["sh600519"])["sh600519"]
    assert stored.price == pytest.approx(quote.price)
    assert stored.bid1 == pytest.approx(quote.bid1)
    assert stored.limit.up == pytest.approx(quote.limit.up)
    assert stored.quote_time == quote.quote_time
    assert stored.fetched_at == TRADING_NOON
    assert stored.status == "normal"


def test_store_history_is_newest_first(tmp_path) -> None:
    store = SnapshotStore(tmp_path / "quotes.sqlite")
    for minutes in (0, 3, 6):
        moment = TRADING_NOON + pd.Timedelta(minutes=minutes)
        quote = parse_payload(_payload(), fetched_at=moment)[0]
        store.save([quote], run_at=moment)
    history = store.history("sh600519", limit=10)
    assert len(history) == 3
    assert list(history["fetched_at"]) == sorted(history["fetched_at"], reverse=True)


def test_store_records_a_failed_round_with_zero_quotes(tmp_path) -> None:
    store = SnapshotStore(tmp_path / "quotes.sqlite")
    store.save([], run_at=TRADING_NOON, requested=3, errors=["数据源不可用"])
    runs = store.runs()
    assert len(runs) == 1
    assert runs.iloc[0]["returned"] == 0
    assert "数据源不可用" in runs.iloc[0]["errors"]


def test_throttle_remainder_survives_a_process_restart(tmp_path) -> None:
    """Two quick CLI runs must not double the request rate."""
    from quant_system.realtime import throttle_remainder

    store = SnapshotStore(tmp_path / "quotes.sqlite")
    assert throttle_remainder(store, now=TRADING_NOON) == pytest.approx(0.0)
    store.save([], run_at=TRADING_NOON)
    assert throttle_remainder(store, now=TRADING_NOON + pd.Timedelta(seconds=1)) == pytest.approx(2.0)
    assert throttle_remainder(store, now=TRADING_NOON + pd.Timedelta(seconds=4)) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# the degraded path: the one that decides whether this desk can be trusted
# --------------------------------------------------------------------------- #
def test_refresh_degrades_to_the_stored_snapshot_and_never_zeroes_a_price(tmp_path) -> None:
    store = SnapshotStore(tmp_path / "quotes.sqlite")
    good = parse_payload(_payload(), fetched_at=TRADING_NOON)[0]
    store.save([good], run_at=TRADING_NOON)

    def dead_transport(url: str, *, timeout=None) -> bytes:
        raise QuoteUnavailable("断网")

    fake = FakeClock()
    source = TencentQuoteSource(
        transport=dead_transport,
        limiter=RateLimiter(0.0, clock=fake.clock, sleep=fake.sleep),
        attempts=2,
        base_wait=0.5,
        sleep=fake.sleep,
    )
    monday = pd.Timestamp("2026-11-09 10:00:00", tz=CN_TZ)
    board = refresh(["sh600519"], source=source, store=store, now=monday)

    assert board.degraded is True
    assert board.errors, "a failed round must carry a human-readable error"
    assert board.missing == ["sh600519"]
    assert set(board.quotes) == {"sh600519"}
    quote = board.quotes["sh600519"]
    assert quote.price == pytest.approx(1453.0), "must keep the real last price"
    assert quote.stale is True
    assert quote.can_buy is False
    assert "延迟" in quote.message
    assert any("本地最后快照" in error for error in board.errors)
    assert "已降级" in board.markdown()


def test_refresh_saves_the_round_even_when_it_failed(tmp_path) -> None:
    store = SnapshotStore(tmp_path / "quotes.sqlite")

    def dead_transport(url: str, *, timeout=None) -> bytes:
        raise QuoteUnavailable("DNS 失败")

    fake = FakeClock()
    source = TencentQuoteSource(
        transport=dead_transport,
        limiter=RateLimiter(0.0, clock=fake.clock, sleep=fake.sleep),
        attempts=1,
        sleep=fake.sleep,
    )
    refresh(["sh600519"], source=source, store=store, now=TRADING_NOON)
    runs = store.runs()
    assert len(runs) == 1 and runs.iloc[0]["returned"] == 0


def test_refresh_on_a_healthy_source_reports_no_degradation(tmp_path) -> None:
    def transport(url: str, *, timeout=None) -> bytes:
        return _payload(when="20261106135940").encode("gbk")

    fake = FakeClock()
    source = TencentQuoteSource(
        transport=transport,
        limiter=RateLimiter(0.0, clock=fake.clock, sleep=fake.sleep),
    )
    board = refresh(["sh600519"], source=source, store=None, now=TRADING_NOON)
    assert board.degraded is False
    assert board.quotes["sh600519"].can_buy is True
    assert board.tradable == 1
    assert board.fresh_ratio == 1.0


def test_one_stale_name_does_not_degrade_the_whole_board() -> None:
    """A broad universe contains delisted names; that is not a broken feed.

    If one old print flipped the board badge, the badge would be on every day and
    would stop meaning anything -- so staleness stays per-row (and as a ratio).
    """

    def transport(url: str, *, timeout=None) -> bytes:
        live = _payload("sh600519", when="20261106135940")
        dead = _payload(
            "sh600999", name="退市样本", price=3.2, prev_close=3.2, open_=0.0,
            volume=0.0, when="20240102100000", limit_up=3.52, limit_down=2.88,
        )
        return (live + dead).encode("gbk")

    fake = FakeClock()
    board = refresh(
        ["sh600519", "sh600999"],
        source=TencentQuoteSource(
            transport=transport, limiter=RateLimiter(0.0, clock=fake.clock, sleep=fake.sleep)
        ),
        now=TRADING_NOON,
    )
    assert board.degraded is False
    assert board.quotes["sh600999"].stale is True
    assert board.quotes["sh600519"].stale is False
    assert board.fresh_ratio == pytest.approx(0.5)
    assert board.tradable == 1


def test_offline_board_reads_only_the_store(tmp_path) -> None:
    store = SnapshotStore(tmp_path / "quotes.sqlite")
    assert load_board(["sh600519"], store=store, now=TRADING_NOON).errors
    stale_quote = parse_payload(_payload(when="20261106110000"), fetched_at=TRADING_NOON)[0]
    store.save([stale_quote], run_at=TRADING_NOON)
    board = load_board(["sh600519"], store=store, now=TRADING_NOON)
    assert board.source == "store"
    assert board.quotes["sh600519"].price == pytest.approx(1453.0)
    assert board.degraded is True  # three hours old at 14:00 -> stale, and says so
    assert "延迟" in board.quotes["sh600519"].message


def test_board_json_payload_is_complete_enough_for_the_desk(tmp_path) -> None:
    store = SnapshotStore(tmp_path / "quotes.sqlite")
    board = refresh(
        ["sh600519"],
        source=TencentQuoteSource(
            transport=lambda url, **_: _payload(when="20261106135940").encode("gbk"),
            limiter=RateLimiter(0.0, clock=FakeClock().clock, sleep=lambda s: None),
        ),
        store=store,
        now=TRADING_NOON,
    )
    payload = board.to_dict()
    assert payload["session_label"] == "盘中"
    quote = payload["quotes"]["sh600519"]
    for key in ("price", "bid1", "ask1", "status_label", "stale", "age_seconds", "can_buy", "message"):
        assert key in quote
    assert isinstance(payload["degraded"], bool)


def test_empty_board_is_allowed_and_reports_the_reason() -> None:
    board = Board()
    assert board.to_dict()["quotes"] == {}
    assert board.stale_symbols == [] and board.tradable == 0
    assert PriceLimit(0.1, 0.0, 0.0, "computed").to_dict()["source"] == "computed"
    assert _price(parse_payload(_payload(), fetched_at=TRADING_NOON)[0]) > 0


# --------------------------------------------------------------------------- #
# trading calendar from the cache (stage-1 gap ①)
# --------------------------------------------------------------------------- #
def _calendar(days: list[str], *, source: str = "test") -> "TradingCalendar":
    from quant_system.realtime import TradingCalendar

    return TradingCalendar([pd.Timestamp(d).date() for d in days], source=source)


def test_calendar_drops_a_real_public_holiday() -> None:
    """2026-10-01..10-07 is National Day; the cache has no bars there.

    The weekday rule calls Thursday 2026-10-01 a session.  The cache is the
    authority this repo already trusts, so the calendar must disagree with the
    weekday rule exactly here.
    """
    from quant_system.realtime import load_calendar

    calendar = load_calendar()
    if len(calendar) < 100:  # pragma: no cover - a cache-less checkout
        pytest.skip("no parquet cache to derive a calendar from")
    assert calendar.covers(pd.Timestamp("2026-10-01").date())
    assert calendar.is_trading_day(pd.Timestamp("2026-10-01").date()) is False
    assert calendar.is_trading_day(pd.Timestamp("2026-10-08").date()) is True
    assert calendar.previous_trading_day(pd.Timestamp("2026-10-08").date()) == pd.Timestamp(
        "2026-09-30"
    ).date()


def test_a_holiday_is_closed_and_its_floor_is_the_previous_session() -> None:
    day = "2026-10-01 10:00:00"  # National Day, a Thursday
    assert session_state(pd.Timestamp(day, tz=CN_TZ)) == "closed"
    floor, label = fresh_floor(pd.Timestamp(day, tz=CN_TZ))
    assert floor == pd.Timestamp("2026-09-30 09:30:00", tz=CN_TZ)
    assert label == "上一交易日收盘"


def test_calendar_skips_a_multi_day_gap() -> None:
    calendar = _calendar(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-12"])
    assert calendar.previous_trading_day(pd.Timestamp("2026-01-12").date()) == pd.Timestamp(
        "2026-01-09"
    ).date()
    assert calendar.is_trading_day(pd.Timestamp("2026-01-10").date()) is False  # Saturday
    assert calendar.is_trading_day(pd.Timestamp("2026-01-12").date()) is True


def test_outside_its_coverage_the_calendar_falls_back_to_weekdays() -> None:
    calendar = _calendar(["2026-01-05", "2026-01-06"])
    assert calendar.covers(pd.Timestamp("2030-01-02").date()) is False
    assert calendar.is_trading_day(pd.Timestamp("2030-01-02").date()) is True   # Wednesday
    assert calendar.is_trading_day(pd.Timestamp("2030-01-05").date()) is False  # Saturday


def test_calendar_is_rebuilt_after_the_ttl(tmp_path) -> None:
    from quant_system.realtime import CALENDAR_TTL_SECONDS, load_calendar

    cache = tmp_path / "cache"
    cache.mkdir()
    frame = pd.DataFrame({"close": [1.0, 1.0, 1.0]}, index=pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]))
    frame.to_parquet(cache / "sh600000_raw.parquet")

    clock = FakeClock(0.0)
    first = load_calendar(cache, clock=clock.clock)
    second = load_calendar(cache, clock=clock.clock)
    assert first is second
    assert first.last == pd.Timestamp("2026-01-07").date()
    clock.t += CALENDAR_TTL_SECONDS + 1
    frame.loc[pd.Timestamp("2026-01-08")] = 1.0
    frame.to_parquet(cache / "sh600000_raw.parquet")
    third = load_calendar(cache, clock=clock.clock)
    assert third is not first
    assert third.last == pd.Timestamp("2026-01-08").date()


# --------------------------------------------------------------------------- #
# snapshot store: WAL + retention (stage-1 gap ②)
# --------------------------------------------------------------------------- #
def test_store_uses_wal_so_readers_do_not_block_the_writer(tmp_path) -> None:
    store = SnapshotStore(tmp_path / "quotes.sqlite")
    with store._session() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_prune_keeps_the_first_and_last_row_of_old_days(tmp_path) -> None:
    store = SnapshotStore(tmp_path / "quotes.sqlite", keep_days=1)
    old_day = pd.Timestamp("2026-01-05 14:00:00", tz=CN_TZ)
    for minutes in (0, 3, 6, 9, 12):
        moment = old_day + pd.Timedelta(minutes=minutes)
        store.save([parse_payload(_payload(), fetched_at=moment)[0]], run_at=moment, auto_prune=False)
    today = TRADING_NOON
    for minutes in (0, 3, 6):
        moment = today + pd.Timedelta(minutes=minutes)
        store.save([parse_payload(_payload(when="20261106135930"), fetched_at=moment)[0]], run_at=moment, auto_prune=False)

    result = store.prune(now=TRADING_NOON + pd.Timedelta(days=1), keep_days=1)
    assert result["deleted"] == 3  # 5 old rows -> first + last survive
    history = store.history("sh600519", limit=50)
    old_rows = history[history["fetched_at"].str.startswith("2026-01-05")]
    assert len(old_rows) == 2
    assert set(old_rows["fetched_at"]) == {
        "2026-01-05T14:00:00+08:00",
        "2026-01-05T14:12:00+08:00",
    }
    assert len(store.runs()) == 3  # the three runs of the retained day survive


def test_save_prunes_at_most_once_per_day(tmp_path) -> None:
    store = SnapshotStore(tmp_path / "quotes.sqlite", keep_days=0)
    day_one = pd.Timestamp("2026-01-05 10:00:00", tz=CN_TZ)
    for minutes in (0, 3, 6):
        moment = day_one + pd.Timedelta(minutes=minutes)
        store.save([parse_payload(_payload(), fetched_at=moment)[0]], run_at=moment)
    assert len(store.history("sh600519", limit=50)) == 3
    # The next day's first save prunes the old day down to first + last, then adds.
    next_day = day_one + pd.Timedelta(days=1)
    store.save([parse_payload(_payload(), fetched_at=next_day)[0]], run_at=next_day)
    assert len(store.history("sh600519", limit=50)) == 3  # 2 for the old day + 1 new


# --------------------------------------------------------------------------- #
# second source + cross-check (stage-1 gap ③)
# --------------------------------------------------------------------------- #
def _sina_payload(
    symbol: str = "sh600519",
    *,
    name: str = "贵州茅台",
    price: float = 1453.0,
    prev_close: float = 1452.0,
    open_: float = 1450.0,
    volume_shares: float = 4_284_772.0,
    when: tuple[str, str] = ("2026-11-06", "13:59:30"),
) -> str:
    values = [""] * 34
    values[0] = name
    values[1] = "%.3f" % open_
    values[2] = "%.3f" % prev_close
    values[3] = "%.3f" % price
    values[4] = "1457.720"
    values[5] = "1436.000"
    values[6] = "%.3f" % max(price - 1, 0.0)
    values[7] = "%.3f" % max(price + 1, 0.0)
    values[8] = "%.0f" % volume_shares
    values[9] = "6199528289.000"
    values[30] = when[0]
    values[31] = when[1]
    return 'var hq_str_%s="%s";' % (symbol, ",".join(values))


def test_parse_sina_reads_price_book_and_time() -> None:
    from quant_system.realtime import parse_sina_payload

    quote = parse_sina_payload(_sina_payload(), fetched_at=TRADING_NOON)[0]
    assert quote.price == pytest.approx(1453.0)
    assert quote.prev_close == pytest.approx(1452.0)
    assert quote.bid1 == pytest.approx(1452.0)
    assert quote.ask1 == pytest.approx(1454.0)
    assert quote.volume_lots == pytest.approx(42847.72)  # shares -> lots
    assert quote.amount_wan == pytest.approx(619_952.83, rel=1e-4)  # 元 -> 万元
    assert quote.quote_time == pd.Timestamp("2026-11-06 13:59:30", tz=CN_TZ)
    assert quote.sources == ("sina",)


def test_sina_reports_zero_for_a_suspended_name_and_we_never_print_0_00() -> None:
    from quant_system.realtime import parse_sina_payload

    payload = _sina_payload("sz002129", name="TCL中环", price=0.0, prev_close=9.38, open_=0.0, volume_shares=0.0)
    quote = parse_sina_payload(payload, fetched_at=TRADING_NOON)[0]
    assert quote.status in {"no_quote", "suspended"}
    assert quote.can_buy is False and quote.can_sell is False
    board = Board(quotes={quote.symbol: annotate(quote, now=TRADING_NOON)})
    assert "| — |" in board.markdown(), board.markdown()


class _FakeSource:
    name = "fake"

    def __init__(self, quotes, *, error: str | None = None) -> None:
        self._quotes = list(quotes)
        self.warnings: list[str] = []
        self.retry_log: list[dict] = []
        self.calls = 0
        self._error = error

    def fetch(self, symbols):
        self.calls += 1
        if self._error:
            raise QuoteUnavailable(self._error)
        return list(self._quotes), [], []


def test_cross_check_flags_a_price_disagreement_as_suspect() -> None:
    from quant_system.realtime import CrossCheckSource, parse_sina_payload

    primary = _FakeSource([parse_payload(_payload(), fetched_at=TRADING_NOON)[0]])
    secondary = _FakeSource([parse_sina_payload(_sina_payload(price=1400.0), fetched_at=TRADING_NOON)[0]])
    source = CrossCheckSource(primary=primary, secondary=secondary)
    quotes, missing, errors = source.fetch(["sh600519"])
    assert errors == [] and missing == []
    quote = quotes[0]
    assert quote.suspect is True
    assert quote.can_buy is False
    assert "双源价差" in quote.message
    assert quote.sources == ("tencent", "sina")


def test_cross_check_accepts_a_small_gap() -> None:
    from quant_system.realtime import CrossCheckSource, parse_sina_payload

    primary = _FakeSource([parse_payload(_payload(), fetched_at=TRADING_NOON)[0]])
    secondary = _FakeSource([parse_sina_payload(_sina_payload(price=1453.5), fetched_at=TRADING_NOON)[0]])
    source = CrossCheckSource(primary=primary, secondary=secondary)
    quote = source.fetch(["sh600519"])[0][0]
    assert quote.suspect is False
    assert quote.sources == ("tencent", "sina")
    assert source.warnings == []


def test_cross_check_warns_when_the_second_source_is_down() -> None:
    from quant_system.realtime import CrossCheckSource

    primary = _FakeSource([parse_payload(_payload(), fetched_at=TRADING_NOON)[0]])
    secondary = _FakeSource([], error="新浪 HTTP 502")
    source = CrossCheckSource(primary=primary, secondary=secondary)
    quotes, missing, errors = source.fetch(["sh600519"])
    assert len(quotes) == 1 and errors == []      # the desk still has a price
    assert quotes[0].suspect is False
    assert any("第二来源不可用" in w for w in source.warnings)


def test_cross_check_falls_back_to_the_second_source() -> None:
    from quant_system.realtime import CrossCheckSource, parse_sina_payload

    primary = _FakeSource([], error="腾讯连接失败")
    secondary = _FakeSource([parse_sina_payload(_sina_payload(), fetched_at=TRADING_NOON)[0]])
    source = CrossCheckSource(primary=primary, secondary=secondary)
    quotes, missing, errors = source.fetch(["sh600519"])
    assert [q.symbol for q in quotes] == ["sh600519"]
    assert quotes[0].sources == ("sina",)
    assert any("回落到新浪" in w for w in source.warnings)
    assert errors  # the primary failure is still reported


def test_a_lost_second_source_is_a_warning_not_a_degradation(tmp_path) -> None:
    from quant_system.realtime import CrossCheckSource

    store = SnapshotStore(tmp_path / "quotes.sqlite")
    source = CrossCheckSource(
        primary=_FakeSource([parse_payload(_payload(when="20261106135930"), fetched_at=TRADING_NOON)[0]]),
        secondary=_FakeSource([], error="新浪超时"),
    )
    board = refresh(["sh600519"], source=source, store=store, now=TRADING_NOON)
    assert board.degraded is False
    assert board.warnings and "第二来源不可用" in board.warnings[0]
    assert board.quotes["sh600519"].can_buy is True


def test_board_renders_a_suspect_name_with_a_marker() -> None:
    from quant_system.realtime import parse_sina_payload

    quote = parse_payload(_payload(), fetched_at=TRADING_NOON)[0]
    marked = annotate(replace(quote, suspect=True), now=TRADING_NOON)
    board = Board(quotes={marked.symbol: marked})
    assert "⚠" in board.markdown()
    assert board.suspect_symbols == ["sh600519"]
    assert board.to_dict()["quotes"]["sh600519"]["suspect"] is True


# --------------------------------------------------------------------------- #
# poller: yellow on the first failure (stage-1 gap ④)
# --------------------------------------------------------------------------- #
def test_poller_starts_yellow_then_goes_green(tmp_path) -> None:
    from quant_system.realtime import GREEN, QuotePoller

    store = SnapshotStore(tmp_path / "quotes.sqlite")
    source = _FakeSource([parse_payload(_payload(when="20261106135940"), fetched_at=TRADING_NOON)[0]])
    poller = QuotePoller(
        ["sh600519"], source=source, store=store, now=lambda: TRADING_NOON, interval=0.0
    )
    assert poller.status().state == "yellow"  # nothing fetched yet, and it says so
    status = poller.tick()
    assert status.state == GREEN
    assert status.failures == 0
    assert "可交易" in status.message
    assert status.to_dict()["board"]["quotes"]["sh600519"]["price"] == pytest.approx(1453.0)


def test_poller_turns_yellow_on_the_first_failure_and_red_after_the_threshold(tmp_path) -> None:
    from quant_system.realtime import GREEN, RED, YELLOW, QuotePoller

    store = SnapshotStore(tmp_path / "quotes.sqlite")
    good = _FakeSource([parse_payload(_payload(when="20261106135940"), fetched_at=TRADING_NOON)[0]])
    poller = QuotePoller(
        ["sh600519"], source=good, store=store, now=lambda: TRADING_NOON, red_after=3
    )
    assert poller.tick().state == GREEN

    poller._source = _FakeSource([], error="腾讯连接失败")
    first = poller.tick()
    assert first.state == YELLOW            # reacts on the first failure, no timeout wait
    assert first.failures == 1
    assert "重连中" in first.message
    assert first.board is not None and first.board.quotes, "must still show the stored snapshot"
    poller.tick()
    third = poller.tick()
    assert third.state == RED
    assert "数据源不可用" in third.message


def test_poller_thread_runs_rounds_and_stops(tmp_path) -> None:
    import time as _time

    from quant_system.realtime import QuotePoller

    store = SnapshotStore(tmp_path / "quotes.sqlite")
    source = _FakeSource([parse_payload(_payload(when="20261106135940"), fetched_at=TRADING_NOON)[0]])
    poller = QuotePoller(
        ["sh600519"], source=source, store=store, now=lambda: TRADING_NOON, interval=0.02
    )
    with poller:
        _time.sleep(0.3)
    assert source.calls >= 2
    assert poller.status().rounds >= 2
    assert poller._thread is None
