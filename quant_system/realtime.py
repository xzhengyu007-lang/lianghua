"""L1 snapshot quotes for execution and monitoring -- never for research.

Iron rule
---------
Nothing here may be reached from the research path (``backtest``, ``factors``,
``signals``, ``strategies``, ``portfolio``, ``risk``, ``research``, ``metrics``,
``costs``, ``bootstrap*``, ``attribution``).  Two independent reasons:

* **Look-ahead**: a backtest decides at yesterday's close and executes at
  today's open.  A live quote arriving at 10:00 is information the backtest
  could not have had.
* **Price convention**: the daily panel uses the *rebuilt multiplicative total
  return* cache (``*_tr.parquet``), while the provider's "qfq" is additive and
  known-distorted (RESEARCH_LOG 23.19).  Mixing a raw quote into that series
  silently changes every return around an ex-dividend date.

``tests/test_realtime_isolation.py`` proves both properties: it walks the import
graph and it runs a real backtest with a transport that raises on any call.

Why L1 and not L2
-----------------
The strategy rebalances every 20 trading days into names chosen from the daily
close.  Minute bars and order-by-order flow would add plumbing without touching
the only decision the system makes.  What execution *does* need from the live
feed is narrower: best bid/ask, the last price with a timestamp, and the
tradability state (suspended / limit-up / limit-down), plus a stale flag that
never lets an old price impersonate the current one.

Field map of the snapshot payload
---------------------------------
Verified against a live ``qt.gtimg.cn`` payload on 2026-11-09 (88 fields for a
stock, 87 for 北交所; 0-based index after splitting on ``~``)::

    0  market flag (1=沪 51=深 62=北)      30 quote time  yyyyMMddHHmmss
    1  name (GBK)                          31 change amount      32 change %
    2  code                                33 high              34 low
    3  last price                          36 volume (手)       37 amount (万元)
    4  previous close                      38 turnover %        39 P/E
    5  open                                43 amplitude %       44 float mcap (亿)
    6  volume (手)                         45 total mcap (亿)   46 P/B
    7  outer volume  8 inner volume        47 limit-up price    48 limit-down price
    9  bid1 price  10 bid1 size            49 volume ratio
    ...  19  ask1 price  20 ask1 size

``47/48`` come back as ``-1`` for indices and for names the provider does not
annotate (北交所 in the sample), so the limit is recomputed from the previous
close and the board's own rule set whenever the provider value is unusable.

Known limitations (stated, not hidden)
--------------------------------------
* Holidays are not modelled: a trading day here is "Mon-Fri", the same
  simplification ``numpy.busday_count`` already implies elsewhere in this repo.
  On a holiday the board reports the previous session's close and marks it
  stale, which is the safe direction to be wrong in.
* The quote endpoint is an unofficial, free, rate-limited source: self-use only,
  never redistributed.  One round per 3 s, at most 50 symbols per request.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd

from .data import infer_price_limit, normalize_symbol

# Mainland China has had no DST since 1991, so a fixed offset is exact and needs
# no tz database -- ``zoneinfo`` finds none on a stock Windows install.
CN_TZ = timezone(timedelta(hours=8), "Asia/Shanghai")

SNAPSHOT_URL = "https://qt.gtimg.cn/q="
BATCH_LIMIT = 50
REFRESH_MIN_INTERVAL = 3.0
STALE_AFTER_SECONDS = 60.0
# Reconnect budget.  The desk polls every 3 s, so a broken round must fail fast
# and fall back to the local snapshot: two attempts with a 0.5 s wait cost at most
# 2 x connect-timeout + 0.5 s, which keeps "the board is visibly degraded" inside
# the 5 s SLA.  A single transient hiccup is still absorbed, and the next poll
# (3 s later) is the real retry.
DEFAULT_ATTEMPTS = 2
RETRY_BASE_WAIT = 0.5
RETRY_MAX_WAIT = 1.0
CONNECT_TIMEOUT = 1.5
READ_TIMEOUT = 5.0
DEFAULT_STORE = Path("reports") / "realtime" / "quotes.sqlite"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) aq-realtime/1.0",
    "Referer": "https://finance.qq.com/",
}

SESSION_MORNING = (dtime(9, 30), dtime(11, 30))
SESSION_AFTERNOON = (dtime(13, 0), dtime(15, 0))
AUCTION_START = dtime(9, 15)


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
class QuoteError(RuntimeError):
    """Base class for anything that makes a quote unusable."""


class QuoteUnavailable(QuoteError):
    """The upstream source could not be reached, or answered with garbage."""


# --------------------------------------------------------------------------- #
# calendar, sessions, limits
# --------------------------------------------------------------------------- #
def now_cn() -> pd.Timestamp:
    """Current time as a tz-aware Asia/Shanghai timestamp."""
    return pd.Timestamp(datetime.now(tz=CN_TZ))


def _as_ts(value: pd.Timestamp | datetime | str | None) -> pd.Timestamp:
    if value is None:
        return now_cn()
    ts = pd.Timestamp(value)
    return ts.tz_localize(CN_TZ) if ts.tzinfo is None else ts.tz_convert(CN_TZ)


def is_trading_day(day: date) -> bool:
    """Mon-Fri. Holidays are deliberately not modelled -- see module docstring."""
    return day.weekday() < 5


def previous_trading_day(day: date) -> date:
    cursor = day - timedelta(days=1)
    while not is_trading_day(cursor):
        cursor -= timedelta(days=1)
    return cursor


def session_state(ts: pd.Timestamp | datetime | str | None = None) -> str:
    """One of ``pre_open`` / ``auction`` / ``open`` / ``lunch`` / ``after_close`` / ``closed``.

    ``closed`` means "no session today at all" (weekend).  The distinction
    matters for staleness: outside a session the last price is *expected* to be
    the previous close, so a naive "age > 60 s means broken" rule would cry wolf
    every evening.
    """
    ts = _as_ts(ts)
    if not is_trading_day(ts.date()):
        return "closed"
    clock = ts.time()
    if clock < AUCTION_START:
        return "pre_open"
    if clock < SESSION_MORNING[0]:
        return "auction"
    if SESSION_MORNING[0] <= clock <= SESSION_MORNING[1]:
        return "open"
    if SESSION_MORNING[1] < clock < SESSION_AFTERNOON[0]:
        return "lunch"
    if SESSION_AFTERNOON[0] <= clock <= SESSION_AFTERNOON[1]:
        return "open"
    return "after_close"


SESSION_LABELS = {
    "pre_open": "开盘前",
    "auction": "集合竞价",
    "open": "盘中",
    "lunch": "午间休市",
    "after_close": "已收盘",
    "closed": "非交易日",
}


def fresh_floor(ts: pd.Timestamp | datetime | str | None = None) -> tuple[pd.Timestamp, str]:
    """Earliest quote timestamp that may still count as current at ``ts``.

    Returns ``(floor, label)``.  A quote older than ``floor`` is stale by
    definition, no matter how recently it was fetched -- that is the whole point
    of separating "when we asked" from "when the exchange printed it".
    """
    ts = _as_ts(ts)
    state = session_state(ts)
    today = ts.date()
    if state in {"open", "lunch"}:
        floor = pd.Timestamp(datetime.combine(today, SESSION_MORNING[0]), tz=CN_TZ)
        return floor, "今日盘中"
    if state == "auction":
        floor = pd.Timestamp(datetime.combine(today, AUCTION_START), tz=CN_TZ)
        return floor, "今日集合竞价"
    if state == "after_close":
        floor = pd.Timestamp(datetime.combine(today, SESSION_MORNING[0]), tz=CN_TZ)
        return floor, "今日收盘"
    previous = previous_trading_day(today)
    floor = pd.Timestamp(datetime.combine(previous, SESSION_MORNING[0]), tz=CN_TZ)
    return floor, "上一交易日收盘"


def is_st(name: str) -> bool:
    """ST / *ST names carry a 5% limit on the main boards."""
    return "ST" in str(name or "").upper().replace(" ", "")


def limit_pct(symbol: str, name: str = "") -> float:
    """Daily price-limit as a fraction, from the listing board and ST status.

    ``data.infer_price_limit`` owns the board rule (主板 10% / 创业板·科创板 20% /
    北交所 30%) and is reused rather than duplicated, so the backtest and the
    live feed cannot disagree about what a limit-up is.  The one addition here
    is ST: 5% on the main boards, while 创业板/科创板 ST names keep their 20%.
    """
    base = float(infer_price_limit(symbol))
    if is_st(name) and base == 0.10:
        return 0.05
    return base


@dataclass(frozen=True)
class PriceLimit:
    pct: float
    up: float
    down: float
    source: str  # "provider" | "computed"

    def to_dict(self) -> dict:
        return {"pct": self.pct, "up": self.up, "down": self.down, "source": self.source}


def price_limit(
    symbol: str,
    name: str = "",
    prev_close: float = 0.0,
    *,
    provider_up: float | None = None,
    provider_down: float | None = None,
) -> PriceLimit:
    """Limit-up/limit-down prices, preferring the exchange's own rounding.

    The provider publishes the exact tick-rounded limits, which is better than
    ``prev_close * (1 + pct)`` because A-share limits round to 0.01 and a naive
    float comparison then disagrees at the boundary (a stock sitting exactly on
    its limit would be classified as "normal").  When the provider sends ``-1``
    or nothing, the value is computed and flagged ``computed``.
    """
    pct = limit_pct(symbol, name)
    usable_up = provider_up is not None and provider_up > 0
    usable_down = provider_down is not None and provider_down > 0
    if usable_up and usable_down:
        return PriceLimit(pct, round(float(provider_up), 2), round(float(provider_down), 2), "provider")
    if prev_close and prev_close > 0:
        up = round(prev_close * (1 + pct), 2)
        down = round(prev_close * (1 - pct), 2)
        return PriceLimit(
            pct,
            round(float(provider_up), 2) if usable_up else up,
            round(float(provider_down), 2) if usable_down else down,
            "computed",
        )
    return PriceLimit(pct, float(provider_up or 0.0), float(provider_down or 0.0), "computed")


def sellable_shares(held: int, bought_today: int = 0, frozen: int = 0) -> int:
    """T+1: shares you may sell today.

    A-share cash equity is T+1, so shares bought today are not sellable until the
    next session; a frozen slice (pending order, pledge) is also excluded.  The
    function is deliberately arithmetic-only: the ledger owns the facts.
    """
    return max(0, int(held) - int(bought_today) - int(frozen))


# --------------------------------------------------------------------------- #
# the quote
# --------------------------------------------------------------------------- #
STATUS_LABELS = {
    "normal": "正常",
    "limit_up": "涨停",
    "limit_down": "跌停",
    "suspended": "停牌",
    "pre_open": "未开盘",
    "no_quote": "无报价",
}


@dataclass(frozen=True)
class Quote:
    symbol: str
    name: str
    price: float
    prev_close: float
    open: float
    high: float
    low: float
    volume_lots: float
    amount_wan: float
    bid1: float
    ask1: float
    quote_time: pd.Timestamp
    fetched_at: pd.Timestamp
    limit: PriceLimit
    status: str
    stale: bool = False
    age_seconds: float = 0.0
    message: str = ""

    @property
    def change_pct(self) -> float:
        if not self.prev_close:
            return 0.0
        return (self.price / self.prev_close - 1.0) * 100.0

    @property
    def can_buy(self) -> bool:
        """A stale or limit-up or suspended name cannot be bought right now."""
        return self.status == "normal" and not self.stale

    @property
    def can_sell(self) -> bool:
        """Selling into a limit-up is possible; selling a suspended name is not."""
        return self.status in {"normal", "limit_up"} and not self.stale

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "price": self.price,
            "prev_close": self.prev_close,
            "change_pct": round(self.change_pct, 4),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "volume_lots": self.volume_lots,
            "amount_wan": self.amount_wan,
            "bid1": self.bid1,
            "ask1": self.ask1,
            "quote_time": self.quote_time.isoformat(),
            "fetched_at": self.fetched_at.isoformat(),
            "limit": self.limit.to_dict(),
            "status": self.status,
            "status_label": STATUS_LABELS.get(self.status, self.status),
            "stale": bool(self.stale),
            "age_seconds": round(float(self.age_seconds), 1),
            "can_buy": self.can_buy,
            "can_sell": self.can_sell,
            "message": self.message,
        }


def _float(fields: Sequence[str], index: int, default: float = 0.0) -> float:
    try:
        value = fields[index]
    except IndexError:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _quote_time(raw: str, *, fetched_at: pd.Timestamp) -> pd.Timestamp:
    """Parse the provider's ``yyyyMMddHHmmss`` stamp; fall back to fetch time.

    A provider that sends ``0`` or garbage must not be silently reinterpreted as
    "now": that would be exactly the "old price impersonating the current one"
    failure this module exists to prevent.  On a bad stamp we keep the fetch time
    and let ``annotate`` decide, which marks the quote stale immediately.
    """
    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    if len(digits) < 14:
        return fetched_at
    try:
        parsed = datetime.strptime(digits[:14], "%Y%m%d%H%M%S")
    except ValueError:
        return fetched_at
    return pd.Timestamp(parsed, tz=CN_TZ)


def _classify(price: float, open_: float, volume_lots: float, limit: PriceLimit, state: str) -> str:
    """Tradability class from a snapshot.

    Order matters: a suspended name prints ``volume == 0`` and ``open == 0`` while
    still carrying yesterday's close in the price field -- 002129 did exactly
    that on 2026-11-06.  Treating it as "normal, unchanged" would let the desk
    plan a buy into a halted stock.
    """
    if price <= 0:
        return "no_quote"
    if state in {"pre_open", "closed"} and volume_lots <= 0:
        return "pre_open"
    if volume_lots <= 0 and open_ <= 0:
        return "suspended"
    if limit.up > 0 and price >= limit.up - 0.005:
        return "limit_up"
    if limit.down > 0 and price <= limit.down + 0.005:
        return "limit_down"
    return "normal"


def parse_payload(text: str, *, fetched_at: pd.Timestamp | None = None) -> list[Quote]:
    """Parse a snapshot payload into quotes. Pure: no network, no clock guesses.

    Malformed symbols are skipped rather than zero-filled: a name we cannot read
    is reported as missing by the caller, never as a 0.00 price.
    """
    fetched = _as_ts(fetched_at)
    state = session_state(fetched)
    quotes: list[Quote] = []
    for chunk in str(text or "").split(";"):
        chunk = chunk.strip()
        if "=" not in chunk:
            continue
        head, _, body = chunk.partition("=")
        symbol = head.strip().removeprefix("v_")
        if not symbol:
            continue
        fields = body.strip().strip('"').split("~")
        if len(fields) < 35:
            continue
        price = _float(fields, 3)
        prev_close = _float(fields, 4)
        open_ = _float(fields, 5)
        volume_lots = _float(fields, 6)
        provider_up = _float(fields, 47, default=0.0)
        provider_down = _float(fields, 48, default=0.0)
        name = fields[1] if len(fields) > 1 else ""
        limit = price_limit(
            symbol,
            name,
            prev_close,
            provider_up=provider_up if provider_up > 0 else None,
            provider_down=provider_down if provider_down > 0 else None,
        )
        quote_time = _quote_time(fields[30] if len(fields) > 30 else "", fetched_at=fetched)
        status = _classify(price, open_, volume_lots, limit, state)
        quotes.append(
            Quote(
                symbol=normalize_symbol(symbol),
                name=name,
                price=price,
                prev_close=prev_close,
                open=open_,
                high=_float(fields, 33),
                low=_float(fields, 34),
                volume_lots=volume_lots,
                amount_wan=_float(fields, 37),
                bid1=_float(fields, 9),
                ask1=_float(fields, 19),
                quote_time=quote_time,
                fetched_at=fetched,
                limit=limit,
                status=status,
                message=STATUS_LABELS.get(status, status),
            )
        )
    return quotes


def annotate(quote: Quote, *, now: pd.Timestamp | datetime | str | None = None,
             stale_after: float = STALE_AFTER_SECONDS) -> Quote:
    """Stamp freshness onto a quote.  This is the only place ``stale`` is decided.

    Two independent tests, because either one alone is wrong:

    * the print is older than ``fresh_floor`` (wrong session entirely), or
    * we are in a live session and the print is older than ``stale_after``
      seconds (feed is lagging even though it is today's print).
    """
    moment = _as_ts(now)
    age = float((moment - quote.quote_time).total_seconds())
    floor, floor_label = fresh_floor(moment)
    state = session_state(moment)
    stale = quote.quote_time < floor or (state == "open" and age > float(stale_after))
    if stale:
        minutes = max(1.0, age / 60.0)
        message = f"延迟 {minutes:.0f} 分钟（最后更新 {quote.quote_time:%m-%d %H:%M:%S}，{floor_label}）"
    elif state in {"after_close", "closed", "lunch"}:
        message = f"收盘价 {quote.quote_time:%m-%d %H:%M:%S}"
    elif state == "pre_open":
        message = f"未开盘，参考上一交易日收盘 {quote.quote_time:%m-%d %H:%M:%S}"
    else:
        message = f"现价 {quote.quote_time:%H:%M:%S}"
    return replace(quote, stale=stale, age_seconds=age, message=message)


# --------------------------------------------------------------------------- #
# rate limiting and retries
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Enforce a minimum gap between upstream requests.

    The free quote endpoint is not an official product; the project's own rule
    is one round per 3 s.  The limiter is shared by every caller, so adding a
    second consumer (the desk poller, the order validator) cannot double the
    request rate.
    """

    def __init__(
        self,
        min_interval: float = REFRESH_MIN_INTERVAL,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = float(min_interval)
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def acquire(self) -> float:
        """Block until the next request is allowed; returns the seconds waited."""
        if self._last is None:
            self._last = self._clock()
            return 0.0
        remainder = self.min_interval - (self._clock() - self._last)
        waited = 0.0
        if remainder > 0:
            self._sleep(remainder)
            waited = remainder
        self._last = self._clock()
        return waited


def with_backoff(
    fn: Callable[[], object],
    *,
    attempts: int = 4,
    base: float = 1.0,
    factor: float = 2.0,
    max_wait: float = 8.0,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, float, Exception], None] | None = None,
):
    """Retry ``fn`` with exponential backoff on :class:`QuoteError` only.

    Only transport/parse failures are retried: a ``TypeError`` in our own code
    must surface immediately instead of being retried four times and reported as
    "network trouble".
    """
    last: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return fn()
        except QuoteError as exc:  # pragma: no cover - exercised via fakes
            last = exc
            if attempt >= attempts:
                break
            wait = min(float(max_wait), float(base) * float(factor) ** (attempt - 1))
            if on_retry is not None:
                on_retry(attempt, wait, exc)
            sleep(wait)
    assert last is not None
    raise last


# --------------------------------------------------------------------------- #
# source
# --------------------------------------------------------------------------- #
def default_transport(
    url: str, *, timeout: tuple[float, float] = (CONNECT_TIMEOUT, READ_TIMEOUT)
) -> bytes:
    """One HTTP GET, returning raw bytes (the payload is GBK, not UTF-8)."""
    import requests  # local import: keeps the module importable without a network stack

    try:
        response = requests.get(url, timeout=timeout, headers=_HEADERS)
    except Exception as exc:  # requests raises a family of exceptions
        raise QuoteUnavailable(f"请求失败：{exc}") from exc
    if response.status_code >= 400:
        raise QuoteUnavailable(f"HTTP {response.status_code}")
    return response.content


def chunks(symbols: Sequence[str], size: int = BATCH_LIMIT) -> list[list[str]]:
    """Split a symbol list into batches of at most ``size``."""
    ordered = list(dict.fromkeys(symbols))
    return [ordered[i : i + size] for i in range(0, len(ordered), max(1, size))]


class TencentQuoteSource:
    """Batch L1 snapshot reader for ``qt.gtimg.cn``.

    ``transport`` and ``limiter`` are injectable so tests can drive the batching,
    the throttle and the degradation path without touching the network.
    """

    name = "tencent"

    def __init__(
        self,
        *,
        transport: Callable[..., bytes] | None = None,
        limiter: RateLimiter | None = None,
        batch_size: int = BATCH_LIMIT,
        timeout: tuple[float, float] = (CONNECT_TIMEOUT, READ_TIMEOUT),
        attempts: int = DEFAULT_ATTEMPTS,
        base_wait: float = RETRY_BASE_WAIT,
        max_wait: float = RETRY_MAX_WAIT,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._transport = transport or default_transport
        self._limiter = limiter or RateLimiter(clock=clock, sleep=sleep)
        self.batch_size = int(batch_size)
        self.timeout = timeout
        self.attempts = int(attempts)
        self.base_wait = float(base_wait)
        self.max_wait = float(max_wait)
        self._sleep = sleep
        self.retry_log: list[dict] = []

    def _fetch_batch(self, batch: Sequence[str]) -> list[Quote]:
        symbols = [normalize_symbol(s) for s in batch]
        url = SNAPSHOT_URL + ",".join(symbols)

        def once() -> list[Quote]:
            self._limiter.acquire()
            payload = self._transport(url, timeout=self.timeout)
            text = payload.decode("gbk", errors="replace") if isinstance(payload, bytes) else str(payload)
            if not text.strip():
                raise QuoteUnavailable("返回内容为空")
            quotes = parse_payload(text)
            if not quotes:
                raise QuoteUnavailable("返回内容无法解析（可能被限流或接口已变更）")
            return quotes

        return with_backoff(
            once,
            attempts=self.attempts,
            base=self.base_wait,
            max_wait=self.max_wait,
            sleep=self._sleep,
            on_retry=lambda attempt, wait, exc: self.retry_log.append(
                {"attempt": attempt, "wait": wait, "error": str(exc), "symbols": len(symbols)}
            ),
        )

    def fetch(self, symbols: Iterable[str]) -> tuple[list[Quote], list[str], list[str]]:
        """Return ``(quotes, missing, errors)``.

        A failing batch does not abort the round: the healthy batches are still
        worth showing, and the failure is reported so the board can degrade
        visibly instead of silently dropping names.
        """
        requested = [normalize_symbol(s) for s in symbols]
        quotes: list[Quote] = []
        errors: list[str] = []
        for batch in chunks(requested, self.batch_size):
            try:
                quotes.extend(self._fetch_batch(batch))
            except QuoteError as exc:
                errors.append(f"{len(batch)} 只标的取价失败：{exc}")
        got = {q.symbol for q in quotes}
        missing = [s for s in requested if s not in got]
        return quotes, missing, errors


# --------------------------------------------------------------------------- #
# snapshot store
# --------------------------------------------------------------------------- #
_SCHEMA = """
CREATE TABLE IF NOT EXISTS quotes (
    symbol       TEXT NOT NULL,
    name         TEXT,
    price        REAL,
    prev_close   REAL,
    open_price   REAL,
    high         REAL,
    low          REAL,
    volume_lots  REAL,
    amount_wan   REAL,
    bid1         REAL,
    ask1         REAL,
    limit_pct    REAL,
    limit_up     REAL,
    limit_down   REAL,
    limit_source TEXT,
    status       TEXT,
    quote_time   TEXT NOT NULL,
    fetched_at   TEXT NOT NULL,
    source       TEXT,
    PRIMARY KEY (symbol, quote_time, fetched_at)
);
CREATE INDEX IF NOT EXISTS idx_quotes_symbol_fetched ON quotes (symbol, fetched_at DESC);
CREATE TABLE IF NOT EXISTS fetch_runs (
    run_at    TEXT PRIMARY KEY,
    requested INTEGER,
    returned  INTEGER,
    errors    TEXT
);
"""


class SnapshotStore:
    """SQLite history of every snapshot we ever saw, with its collection time.

    Why persist them at all: the desk must be able to show "the last price we
    actually saw, and when", after a restart or a network loss.  That is the
    difference between degrading honestly and inventing a number.
    """

    def __init__(self, path: str | Path = DEFAULT_STORE) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        return conn

    def save(
        self,
        quotes: Iterable[Quote],
        *,
        run_at: pd.Timestamp | None = None,
        requested: int | None = None,
        errors: Sequence[str] = (),
    ) -> int:
        moment = _as_ts(run_at)
        rows = [
            (
                q.symbol, q.name, q.price, q.prev_close, q.open, q.high, q.low,
                q.volume_lots, q.amount_wan, q.bid1, q.ask1,
                q.limit.pct, q.limit.up, q.limit.down, q.limit.source,
                q.status, q.quote_time.isoformat(), q.fetched_at.isoformat(), "tencent",
            )
            for q in quotes
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO quotes VALUES (%s)" % ",".join("?" * 19), rows
            )
            conn.execute(
                "INSERT OR REPLACE INTO fetch_runs VALUES (?,?,?,?)",
                (moment.isoformat(), int(requested or len(rows)), len(rows), json.dumps(list(errors), ensure_ascii=False)),
            )
        return len(rows)

    def latest(self, symbols: Iterable[str] | None = None) -> dict[str, Quote]:
        """Newest stored snapshot per symbol (no staleness decision here)."""
        query = (
            "SELECT * FROM quotes WHERE fetched_at = "
            "(SELECT MAX(fetched_at) FROM quotes AS inner_q WHERE inner_q.symbol = quotes.symbol)"
        )
        params: list[str] = []
        if symbols is not None:
            wanted = [normalize_symbol(s) for s in symbols]
            if not wanted:
                return {}
            query += " AND symbol IN (%s)" % ",".join("?" * len(wanted))
            params.extend(wanted)
        with self._connect() as conn:
            rows = conn.execute(query + " ORDER BY symbol", params).fetchall()
        return {row["symbol"]: self._to_quote(row) for row in rows}

    def history(self, symbol: str, limit: int = 50) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM quotes WHERE symbol = ? ORDER BY fetched_at DESC LIMIT ?",
                (normalize_symbol(symbol), int(limit)),
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(row) for row in rows])

    def runs(self, limit: int = 20) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM fetch_runs ORDER BY run_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def last_run_at(self) -> pd.Timestamp | None:
        with self._connect() as conn:
            row = conn.execute("SELECT MAX(run_at) AS last FROM fetch_runs").fetchone()
        if row is None or row["last"] is None:
            return None
        return pd.Timestamp(row["last"])

    @staticmethod
    def _to_quote(row: sqlite3.Row) -> Quote:
        return Quote(
            symbol=row["symbol"],
            name=row["name"] or "",
            price=float(row["price"] or 0.0),
            prev_close=float(row["prev_close"] or 0.0),
            open=float(row["open_price"] or 0.0),
            high=float(row["high"] or 0.0),
            low=float(row["low"] or 0.0),
            volume_lots=float(row["volume_lots"] or 0.0),
            amount_wan=float(row["amount_wan"] or 0.0),
            bid1=float(row["bid1"] or 0.0),
            ask1=float(row["ask1"] or 0.0),
            quote_time=pd.Timestamp(row["quote_time"]),
            fetched_at=pd.Timestamp(row["fetched_at"]),
            limit=PriceLimit(
                pct=float(row["limit_pct"] or 0.0),
                up=float(row["limit_up"] or 0.0),
                down=float(row["limit_down"] or 0.0),
                source=row["limit_source"] or "stored",
            ),
            status=row["status"] or "no_quote",
        )


# --------------------------------------------------------------------------- #
# the round
# --------------------------------------------------------------------------- #
@dataclass
class Board:
    """One refresh round, including everything that went wrong."""

    quotes: dict[str, Quote] = field(default_factory=dict)
    fetched_at: pd.Timestamp | None = None
    errors: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    degraded: bool = False
    source: str = "tencent"
    session: str = "unknown"

    @property
    def stale_symbols(self) -> list[str]:
        return sorted(s for s, q in self.quotes.items() if q.stale)

    @property
    def tradable(self) -> int:
        return sum(1 for q in self.quotes.values() if q.status == "normal" and not q.stale)

    @property
    def fresh_ratio(self) -> float:
        """Share of quotes that are actually current -- what the top badge shows."""
        if not self.quotes:
            return 0.0
        fresh = sum(1 for q in self.quotes.values() if not q.stale)
        return round(fresh / len(self.quotes), 4)

    def to_dict(self) -> dict:
        return {
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at is not None else None,
            "session": self.session,
            "session_label": SESSION_LABELS.get(self.session, self.session),
            "source": self.source,
            "degraded": bool(self.degraded),
            "errors": list(self.errors),
            "missing": list(self.missing),
            "stale_symbols": self.stale_symbols,
            "tradable": self.tradable,
            "fresh_ratio": self.fresh_ratio,
            "quotes": {symbol: quote.to_dict() for symbol, quote in sorted(self.quotes.items())},
        }

    def markdown(self) -> str:
        header = "| 标的 | 名称 | 现价 | 涨跌 | 状态 | 买一/卖一 | 新鲜度 |"
        rule = "|---|---|---:|---:|---|---|---|"
        lines = [header, rule]
        for symbol, q in sorted(self.quotes.items()):
            lines.append(
                "| %s | %s | %.2f | %+.2f%% | %s | %.2f/%.2f | %s |"
                % (
                    symbol, q.name, q.price, q.change_pct,
                    STATUS_LABELS.get(q.status, q.status), q.bid1, q.ask1, q.message,
                )
            )
        summary = "行情 %d 只｜可交易 %d 只｜滞后 %d 只｜%s" % (
            len(self.quotes), self.tradable, len(self.stale_symbols),
            SESSION_LABELS.get(self.session, self.session),
        )
        if self.degraded:
            summary += "｜**已降级**：" + "；".join(self.errors or ["数据源不可用"])
        if self.missing:
            summary += "｜缺报价：" + ",".join(sorted(self.missing)[:8])
        return "\n".join(lines) + "\n\n" + summary + "\n"


def refresh(
    symbols: Iterable[str],
    *,
    source: TencentQuoteSource | None = None,
    store: SnapshotStore | None = None,
    now: pd.Timestamp | datetime | str | None = None,
    stale_after: float = STALE_AFTER_SECONDS,
) -> Board:
    """One refresh round: fetch, annotate, persist, and degrade loudly.

    The failure path is the important one.  When the source is unreachable the
    board falls back to the newest stored snapshot and marks every price stale
    with its real age -- never to "0.00", never to yesterday's close dressed up
    as today's.  ``degraded`` is what the UI badge and the order validator read.
    """
    moment = _as_ts(now)
    wanted = [normalize_symbol(s) for s in symbols]
    src = source or TencentQuoteSource()
    board = Board(fetched_at=moment, source=getattr(src, "name", "tencent"), session=session_state(moment))
    try:
        quotes, missing, errors = src.fetch(wanted)
    except QuoteError as exc:  # defensive: fetch() already degrades per batch
        quotes, missing, errors = [], wanted, [str(exc)]

    annotated = {q.symbol: annotate(q, now=moment, stale_after=stale_after) for q in quotes}
    board.missing = list(missing)
    board.errors = list(errors)

    if annotated:
        board.quotes = annotated
    elif store is not None:
        stored = store.latest(wanted)
        if stored:
            board.quotes = {s: annotate(q, now=moment, stale_after=stale_after) for s, q in stored.items()}
            board.errors.append("数据源不可用：以下为本地最后快照，价格已按真实时间标注为滞后")

    # "Degraded" means the round itself is untrustworthy -- a source failure or a
    # name with no quote.  A single stale print is *not* board-level degradation:
    # a broad universe legitimately contains delisted names whose last print is
    # years old, and flagging the whole board for that would train the user to
    # ignore the badge.  Staleness is reported per row and as a ratio.
    board.degraded = bool(board.errors) or bool(board.missing)
    if store is not None:
        store.save(quotes, run_at=moment, requested=len(wanted), errors=board.errors)
    return board


def load_board(
    symbols: Iterable[str] | None = None,
    *,
    store: SnapshotStore | None = None,
    now: pd.Timestamp | datetime | str | None = None,
    stale_after: float = STALE_AFTER_SECONDS,
) -> Board:
    """Offline board from the local store -- what the desk shows with no network."""
    moment = _as_ts(now)
    destination = store or SnapshotStore()
    stored = destination.latest(symbols)
    board = Board(
        quotes={s: annotate(q, now=moment, stale_after=stale_after) for s, q in stored.items()},
        fetched_at=moment,
        source="store",
        session=session_state(moment),
    )
    if not stored:
        board.errors.append("本地没有历史快照：先在线刷新一次")
    # Offline is degraded by construction: there is no live feed, so anything not
    # verifiably fresh has to be presented as such rather than as a current price.
    board.degraded = bool(board.errors) or any(q.stale for q in board.quotes.values())
    return board


def throttle_remainder(
    store: SnapshotStore | None = None,
    *,
    min_interval: float = REFRESH_MIN_INTERVAL,
    now: pd.Timestamp | datetime | str | None = None,
) -> float:
    """Seconds still owed before the next upstream request, across processes.

    The in-process :class:`RateLimiter` cannot see a previous CLI invocation, so
    two quick ``aq quote`` runs would double the request rate against a free and
    unofficial endpoint.  Every round's wall-clock time is recorded in the store,
    so the gap is measured rather than assumed.
    """
    if store is None:
        return 0.0
    last = store.last_run_at()
    if last is None:
        return 0.0
    gap = float((_as_ts(now) - last).total_seconds())
    return max(0.0, float(min_interval) - gap)
