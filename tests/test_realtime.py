"""Contract tests for the L1 snapshot layer (stage 1).

Nothing here touches the network: every transport is a fake, and the payloads are
built from the field map that was verified against a live ``qt.gtimg.cn``
response on 2026-11-09 (see ``quant_system/realtime.py``).  The tests that matter
most are the ugly ones -- suspension, provider sentinels, throttle, backoff and
the degraded path -- because those are the states in which a trading desk
silently lies to its user.
"""

from __future__ import annotations

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
