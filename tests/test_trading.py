"""Tests for the tiered trading layer (stage 3): the ten red lines, one by one.

Written as the red-line list rather than as module coverage, because that is the
acceptance criterion: each test names the line it guards, and a line without a test
is a line that only exists in a document.

No network.  Every quote is a fake with just the fields the validator reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_system.config import CostConfig
from quant_system.paper import Ledger, Position, apply_fills
from quant_system.realtime import PriceLimit, Quote
from quant_system.trading import (
    AuditLog,
    IdempotencyBook,
    Order,
    OrderState,
    RiskLimits,
    TradingGate,
    TradingLevel,
    can_transition,
    check_limits,
    daily_loss_halt,
    daily_statement,
    reconcile,
    validate_order,
    write_alerts,
    write_broker_files,
)
from quant_system.trading.idempotency import DuplicateOrderError
from quant_system.trading.pipeline import plan_session
from quant_system.trading.recon import append_equity_point, equity_curve

TRADE_DATE = "2026-11-09"
NOON = pd.Timestamp("2026-11-06 14:00:00", tz="Asia/Shanghai")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _quote(symbol="sh600519", price=100.0, *, status="normal", stale=False, suspect=False,
           prev_close=99.0, pct=0.10) -> Quote:
    return Quote(
        symbol=symbol, name="样本", price=price, prev_close=prev_close, open=price,
        high=price, low=price, volume_lots=1000.0, amount_wan=100.0,
        bid1=price - 0.01, ask1=price + 0.01, quote_time=NOON, fetched_at=NOON,
        limit=PriceLimit(pct, round(prev_close * (1 + pct), 2), round(prev_close * (1 - pct), 2), "computed"),
        status=status, stale=stale, suspect=suspect,
    )


def _ledger(cash=1_000_000.0, positions=None) -> Ledger:
    return Ledger(
        cash=cash,
        positions={symbol: Position(shares=shares, avg_cost=90.0) for symbol, shares in (positions or {}).items()},
        as_of="2026-11-06",
    )


class FakeBoard:
    def __init__(self, quotes, degraded=False, errors=None, fetched_at=NOON):
        self.quotes = {quote.symbol: quote for quote in quotes}
        self.degraded = degraded
        self.errors = errors or []
        self.fetched_at = fetched_at


def _order(**kwargs) -> Order:
    base = dict(order_id="x", trade_date=TRADE_DATE, symbol="sh600519", side="buy",
                shares=100, limit_price=100.0)
    base.update(kwargs)
    return Order(**base)


# --------------------------------------------------------------------------- #
# red line 1: no real order path below L3
# --------------------------------------------------------------------------- #
def test_default_level_is_paper_and_cannot_submit(tmp_path) -> None:
    gate = TradingGate(tmp_path / "gate.json")
    assert gate.level is TradingLevel.L1_PAPER
    assert gate.can_submit(TRADE_DATE) is False
    assert gate.state.kill_switch is False


def test_l3_needs_an_explicit_daily_authorisation(tmp_path) -> None:
    gate = TradingGate(tmp_path / "gate.json")
    gate.set_level(TradingLevel.L3_FULL_AUTO, trade_date="2026-11-09")
    assert gate.l3_authorized_for("2026-11-09") is True
    assert gate.l3_authorized_for("2026-11-10") is False   # expires at midnight
    assert gate.can_submit("2026-11-10") is False


def test_l3_without_a_date_is_refused(tmp_path) -> None:
    gate = TradingGate(tmp_path / "gate.json")
    with pytest.raises(ValueError):
        gate.set_level(TradingLevel.L3_FULL_AUTO)


def test_the_source_tree_contains_no_order_submission_path() -> None:
    """Red line 1, strongest form: the capability does not exist.

    If a broker gateway is ever added, this test fails and the reviewer is forced to
    look at L3 handling before any order can leave the machine.
    """
    root = Path(__file__).resolve().parent.parent / "quant_system" / "trading"
    forbidden = ("requests.post", "urlopen", "xtquant", "easytrader", "send_order",
                 "insert_order", "place_order", "socket.socket")
    offenders: list[str] = []
    for path in sorted(root.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append("%s: %s" % (path.name, token))
    assert offenders == [], offenders


def test_plan_session_writes_no_broker_file_at_l1(tmp_path) -> None:
    gate = TradingGate(tmp_path / "gate.json")
    outcome = plan_session(
        ledger=_ledger(), target_weights=pd.Series({"sh600519": 0.2}),
        prices=pd.Series({"sh600519": 100.0}), gate=gate, trade_date=TRADE_DATE,
        equity=1_000_000.0, book=IdempotencyBook(tmp_path / "book.jsonl"),
        audit=AuditLog(tmp_path / "audit.jsonl"), output_dir=tmp_path / "orders",
    )
    assert outcome.written is None
    assert not (tmp_path / "orders").exists()
    assert "不生成券商导入文件" in " ".join(outcome.warnings)
    assert outcome.orders, "L1 still records the intent and the audit trail"


def test_plan_session_writes_the_broker_file_at_l2(tmp_path) -> None:
    gate = TradingGate(tmp_path / "gate.json")
    gate.set_level(TradingLevel.L2_SEMI_AUTO)
    outcome = plan_session(
        ledger=_ledger(), target_weights=pd.Series({"sh600519": 0.2}),
        prices=pd.Series({"sh600519": 100.0}), gate=gate, trade_date=TRADE_DATE,
        equity=1_000_000.0, book=IdempotencyBook(tmp_path / "book.jsonl"),
        audit=AuditLog(tmp_path / "audit.jsonl"), output_dir=tmp_path / "orders",
    )
    assert outcome.written is not None
    assert (tmp_path / "orders" / "orders.csv").exists()
    assert (tmp_path / "orders" / "checklist.md").exists()


# --------------------------------------------------------------------------- #
# red line 2: ticket / position / daily limits
# --------------------------------------------------------------------------- #
def test_a_ticket_above_the_notional_cap_is_refused() -> None:
    limits = RiskLimits(max_order_notional=50_000)
    report = check_limits([_order(shares=1000, limit_price=100.0)], limits=limits, equity=1_000_000.0)
    assert report.ok is False
    assert {violation.code for violation in report.violations} == {"max_order_notional"}


def test_daily_new_exposure_counts_what_was_already_bought_today() -> None:
    limits = RiskLimits(max_daily_new_exposure=100_000)
    report = check_limits(
        [_order(shares=500, limit_price=100.0)], limits=limits, equity=1_000_000.0,
        existing_daily_buys=60_000.0,
    )
    assert report.ok is False
    assert any(v.code == "max_daily_new_exposure" and v.scope == "portfolio" for v in report.violations)


def test_the_position_cap_counts_the_existing_holding() -> None:
    limits = RiskLimits(max_position_weight=0.25, max_order_notional=10_000_000)
    report = check_limits(
        [_order(shares=2000, limit_price=100.0)], limits=limits, equity=1_000_000.0,
        holdings={"sh600519": 1000},
    )
    assert report.ok is False
    assert any(v.code == "max_position_weight" for v in report.violations)


# --------------------------------------------------------------------------- #
# red line 3: daily loss halt + kill switch
# --------------------------------------------------------------------------- #
def test_the_daily_loss_breaker_fires_at_the_threshold() -> None:
    halt, message = daily_loss_halt(965_000.0, 1_000_000.0)
    assert halt is True and "熔断" in message
    halt, _ = daily_loss_halt(980_000.0, 1_000_000.0)
    assert halt is False


def test_a_missing_previous_equity_does_not_halt() -> None:
    assert daily_loss_halt(900_000.0, 0.0)[0] is False


def test_the_breaker_sets_the_kill_switch_and_stops_the_session(tmp_path) -> None:
    gate = TradingGate(tmp_path / "gate.json")
    outcome = plan_session(
        ledger=_ledger(), target_weights=pd.Series({"sh600519": 0.2}),
        prices=pd.Series({"sh600519": 100.0}), gate=gate, trade_date=TRADE_DATE,
        equity=950_000.0, equity_prev_close=1_000_000.0,
        book=IdempotencyBook(tmp_path / "book.jsonl"), audit=AuditLog(tmp_path / "audit.jsonl"),
    )
    assert outcome.halted is True
    assert "熔断" in outcome.halt_reason
    assert gate.state.kill_switch is True
    assert outcome.orders == []
    assert AuditLog(tmp_path / "audit.jsonl").count("halt") == 1


def test_the_kill_switch_blocks_everything_and_needs_a_reset(tmp_path) -> None:
    gate = TradingGate(tmp_path / "gate.json")
    gate.kill("测试停机")
    outcome = plan_session(
        ledger=_ledger(), target_weights=pd.Series({"sh600519": 0.2}),
        prices=pd.Series({"sh600519": 100.0}), gate=gate, trade_date=TRADE_DATE,
        equity=1_000_000.0, book=IdempotencyBook(tmp_path / "book.jsonl"),
        audit=AuditLog(tmp_path / "audit.jsonl"),
    )
    assert outcome.halted is True and "kill switch" in outcome.halt_reason
    gate.reset_kill()
    assert gate.state.kill_switch is False


def test_kill_also_drops_the_level_back_to_paper(tmp_path) -> None:
    gate = TradingGate(tmp_path / "gate.json")
    gate.set_level(TradingLevel.L3_FULL_AUTO, trade_date="2026-11-09")
    gate.kill("事故")
    assert gate.level is TradingLevel.L1_PAPER
    assert gate.state.enabled_until is None


def test_a_corrupt_state_file_fails_closed(tmp_path) -> None:
    path = tmp_path / "gate.json"
    path.write_text("{not json", encoding="utf-8")
    gate = TradingGate(path)
    assert gate.state.kill_switch is True
    assert gate.can_submit(TRADE_DATE) is False


# --------------------------------------------------------------------------- #
# red line 4: idempotency
# --------------------------------------------------------------------------- #
def test_the_same_key_cannot_be_reserved_twice(tmp_path) -> None:
    book = IdempotencyBook(tmp_path / "book.jsonl")
    book.reserve(trade_date=TRADE_DATE, symbol="sh600519", side="buy", order_id="a")
    with pytest.raises(DuplicateOrderError):
        book.reserve(trade_date=TRADE_DATE, symbol="sh600519", side="buy", order_id="b")


def test_a_released_key_can_be_reused(tmp_path) -> None:
    book = IdempotencyBook(tmp_path / "book.jsonl")
    book.reserve(trade_date=TRADE_DATE, symbol="sh600519", side="buy", order_id="a")
    book.release((TRADE_DATE, "sh600519", "buy"))
    book.reserve(trade_date=TRADE_DATE, symbol="sh600519", side="buy", order_id="b")
    assert len(book) == 1


def test_the_opposite_side_and_other_days_are_independent(tmp_path) -> None:
    book = IdempotencyBook(tmp_path / "book.jsonl")
    book.reserve(trade_date=TRADE_DATE, symbol="sh600519", side="buy", order_id="a")
    book.reserve(trade_date=TRADE_DATE, symbol="sh600519", side="sell", order_id="b")
    book.reserve(trade_date="2026-11-10", symbol="sh600519", side="buy", order_id="c")
    assert len(book) == 3


def test_idempotency_survives_a_restart(tmp_path) -> None:
    path = tmp_path / "book.jsonl"
    IdempotencyBook(path).reserve(trade_date=TRADE_DATE, symbol="sh600519", side="buy", order_id="a")
    reopened = IdempotencyBook(path)
    assert reopened.is_reserved(trade_date=TRADE_DATE, symbol="sh600519", side="buy")
    with pytest.raises(DuplicateOrderError):
        reopened.reserve(trade_date=TRADE_DATE, symbol="sh600519", side="buy", order_id="b")


def test_replanning_the_same_day_is_deduplicated_end_to_end(tmp_path) -> None:
    """The dangerous path: the operator (or a cron) runs the plan twice."""
    common = dict(
        ledger=_ledger(), target_weights=pd.Series({"sh600519": 0.2}),
        prices=pd.Series({"sh600519": 100.0}), trade_date=TRADE_DATE,
        equity=1_000_000.0, output_dir=tmp_path / "orders",
    )
    gate = TradingGate(tmp_path / "gate.json")
    book = IdempotencyBook(tmp_path / "book.jsonl")
    audit = AuditLog(tmp_path / "audit.jsonl")
    first = plan_session(gate=gate, book=book, audit=audit, **common)
    assert len(first.orders) == 1
    second = plan_session(gate=gate, book=book, audit=audit, **common)
    assert second.orders == []
    assert second.rejected and "已有一张有效订单" in " ".join(second.rejected[0]["reasons"])
    assert audit.count("order.duplicate") == 1


# --------------------------------------------------------------------------- #
# red line 5: pre-trade validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "overrides,expect",
    [
        ({"shares": 150}, "整手"),
        ({"limit_price": 0.0}, "限价必须为正"),
        ({"shares": -100}, "股数必须为正"),
        ({"quote": _quote(status="suspended")}, "停牌"),
        ({"quote": _quote(status="limit_up")}, "涨停"),
        ({"quote": _quote(stale=True)}, "行情滞后"),
        ({"quote": _quote(suspect=True)}, "双源"),
        ({"quote": _quote(price=0.0)}, "没有有效报价"),
    ],
)
def test_validation_explains_every_refusal_in_chinese(overrides, expect) -> None:
    kwargs = {"shares": 100, "limit_price": 100.0, "quote": None}
    kwargs.update(overrides)
    quote = kwargs.pop("quote")
    result = validate_order(
        symbol="sh600519", side="buy", trade_date=TRADE_DATE, ledger=_ledger(),
        quote=quote, **kwargs,
    )
    assert result.ok is False
    assert any(expect in reason for reason in result.reasons), result.reasons


def test_selling_more_than_t_plus_one_allows_is_refused() -> None:
    result = validate_order(
        symbol="sh600519", side="sell", shares=1000, limit_price=100.0, trade_date=TRADE_DATE,
        ledger=_ledger(positions={"sh600519": 1000}), bought_today=600,
    )
    assert result.ok is False
    assert any("T+1" in reason and "可卖只有 400" in reason for reason in result.reasons)


def test_buying_beyond_cash_is_refused_with_the_shortfall() -> None:
    result = validate_order(
        symbol="sh600519", side="buy", shares=10_000, limit_price=100.0, trade_date=TRADE_DATE,
        ledger=_ledger(cash=50_000.0),
    )
    assert result.ok is False
    assert any("现金不足" in reason for reason in result.reasons)


def test_a_limit_price_outside_the_daily_band_is_refused() -> None:
    result = validate_order(
        symbol="sh600519", side="buy", shares=100, limit_price=200.0, trade_date=TRADE_DATE,
        quote=_quote(price=100.0, prev_close=99.0), ledger=_ledger(),
    )
    assert result.ok is False
    assert any("涨停价" in reason for reason in result.reasons)


def test_a_clean_order_passes() -> None:
    result = validate_order(
        symbol="sh600519", side="buy", shares=900, limit_price=100.0, trade_date=TRADE_DATE,
        ledger=_ledger(), quote=_quote(), equity=1_000_000.0,
    )
    assert result.ok is True, result.reasons
    assert result.cash_after is not None and result.cash_after > 0


# --------------------------------------------------------------------------- #
# order state machine
# --------------------------------------------------------------------------- #
def test_the_state_machine_rejects_illegal_transitions() -> None:
    assert can_transition(OrderState.SUBMITTING, OrderState.NOTTRADED) is True
    assert can_transition(OrderState.ALLTRADED, OrderState.NOTTRADED) is False
    order = _order()
    filled = order.transition(OrderState.NOTTRADED)
    with pytest.raises(ValueError):
        filled.transition(OrderState.ALLTRADED)   # needs filled == shares
    assert filled.transition(OrderState.ALLTRADED, filled_shares=100).state is OrderState.ALLTRADED


def test_alltraded_requires_a_full_fill() -> None:
    order = _order(shares=200)
    with pytest.raises(ValueError):
        order.transition(OrderState.ALLTRADED, filled_shares=100)


# --------------------------------------------------------------------------- #
# red lines 6/7: audit log
# --------------------------------------------------------------------------- #
def test_the_audit_records_the_six_required_facts(tmp_path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    audit.record(
        "order.accepted", trade_date=TRADE_DATE, symbol="sh600519", side="buy", shares=100,
        price=100.0, order_id="a", signal_fingerprint="deadbeefdeadbeef",
        snapshot_time="2026-11-09T10:00:00+08:00", level=2, source="signal",
    )
    row = audit.events()[0]
    for field in ("at", "symbol", "shares", "price", "signal_fingerprint", "snapshot_time", "source"):
        assert row[field], field


def test_the_audit_never_contains_a_credential(tmp_path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    audit.record("order.accepted", trade_date=TRADE_DATE, symbol="sh600519", side="buy",
                 shares=100, price=100.0, source="signal")
    assert audit.scan_for_secrets() == []


def test_the_signal_fingerprint_is_stable_and_order_independent() -> None:
    from quant_system.trading import signal_fingerprint

    a = signal_fingerprint("2026-11-06", {"sh600519": 0.1, "sz000001": 0.2}, "low_vol_core")
    b = signal_fingerprint("2026-11-06", {"sz000001": 0.2, "sh600519": 0.1}, "low_vol_core")
    c = signal_fingerprint("2026-11-06", {"sh600519": 0.1, "sz000001": 0.3}, "low_vol_core")
    assert a == b and a != c


# --------------------------------------------------------------------------- #
# red line 8/9: reconciliation and the daily statement
# --------------------------------------------------------------------------- #
def test_a_position_difference_raises_an_alert_and_never_touches_the_ledger(tmp_path) -> None:
    ledger = _ledger(positions={"sh600519": 1000})
    before = ledger.to_dict()
    report = reconcile(ledger, {"sh600519": 1200}, broker_cash=ledger.cash, trade_date=TRADE_DATE)
    assert report.ok is False
    assert any("账本少" in alert for alert in report.alerts)
    assert ledger.to_dict() == before, "对账不允许改账本"
    path = write_alerts(report, tmp_path / "recon")
    assert path.exists() and "账本少" in path.read_text(encoding="utf-8")


def test_a_cash_difference_is_reported_with_both_numbers() -> None:
    ledger = _ledger(cash=100_000.0)
    report = reconcile(ledger, {}, broker_cash=99_900.0, trade_date=TRADE_DATE)
    assert report.ok is False
    assert any("现金差" in alert for alert in report.alerts)


def test_matching_books_reconcile_clean(tmp_path) -> None:
    ledger = _ledger(cash=50_000.0, positions={"sh600519": 1000})
    report = reconcile(ledger, {"sh600519": 1000}, broker_cash=50_000.0, trade_date=TRADE_DATE)
    assert report.ok is True and report.alerts == []


def test_the_daily_statement_reports_fills_fees_and_equity(tmp_path) -> None:
    ledger = _ledger(cash=1_000_000.0)
    fills = pd.DataFrame([{"symbol": "sh600519", "side": "buy", "shares": 100, "price": 100.0}])
    booked = apply_fills(ledger, fills, date=TRADE_DATE, costs=CostConfig())
    statement = daily_statement(booked, pd.Series({"sh600519": 101.0}), trade_date=TRADE_DATE)
    assert statement["fills"] == 1
    assert statement["fees_today_total"] > 0
    assert statement["positions"] == 1
    assert statement["equity"] > 0


def test_the_equity_curve_is_append_only_and_idempotent_per_day(tmp_path) -> None:
    path = tmp_path / "equity.jsonl"
    base = {"cash": 1000.0, "market_value": 0.0, "positions": 0, "fees_today_total": 0.0}
    append_equity_point(path, dict(base, trade_date="2026-11-09", equity=1000.0))
    append_equity_point(path, dict(base, trade_date="2026-11-10", equity=1010.0))
    append_equity_point(path, dict(base, trade_date="2026-11-10", equity=1015.0))  # same day again
    curve = equity_curve(path)
    assert len(curve) == 2
    assert curve.iloc[-1]["equity"] == pytest.approx(1015.0)
    assert curve.iloc[-1]["return"] == pytest.approx(0.015)


# --------------------------------------------------------------------------- #
# L2 broker files
# --------------------------------------------------------------------------- #
def test_broker_file_is_importable_and_refuses_bad_rows(tmp_path) -> None:
    orders = [_order(shares=300, limit_price=42.5)]
    result = write_broker_files(orders, tmp_path / "out", trade_date=TRADE_DATE,
                               names={"sh600519": "贵州茅台"}, data_as_of="2026-11-06")
    text = (tmp_path / "out" / "orders.csv").read_text(encoding="utf-8-sig")
    assert "600519" in text and "买入" in text and "42.50" in text
    assert result.rows == 1 and result.total_buy == pytest.approx(12_750.0)
    checklist = (tmp_path / "out" / "checklist.md").read_text(encoding="utf-8")
    assert "六个问题" in checklist and "不构成投资建议" in checklist

    with pytest.raises(ValueError):
        write_broker_files([_order(shares=150)], tmp_path / "bad", trade_date=TRADE_DATE)


# --------------------------------------------------------------------------- #
# portfolio-level rejects vs ticket-level rejects
# --------------------------------------------------------------------------- #
def test_a_portfolio_limit_halts_the_whole_batch(tmp_path) -> None:
    """Each ticket fits; the batch does not.  Trimming would change the strategy."""
    gate = TradingGate(tmp_path / "gate.json")
    outcome = plan_session(
        ledger=_ledger(),
        target_weights=pd.Series({"sh600519": 0.2, "sz000001": 0.2}),
        prices=pd.Series({"sh600519": 100.0, "sz000001": 50.0}), gate=gate,
        trade_date=TRADE_DATE, equity=1_000_000.0,
        book=IdempotencyBook(tmp_path / "book.jsonl"), audit=AuditLog(tmp_path / "audit.jsonl"),
        limits=RiskLimits(max_order_notional=500_000.0, max_daily_new_exposure=300_000.0),
    )
    assert outcome.halted is True
    assert "组合限额" in outcome.halt_reason
    assert outcome.orders == []


def test_one_bad_ticket_does_not_block_the_others(tmp_path) -> None:
    gate = TradingGate(tmp_path / "gate.json")
    board = FakeBoard([
        _quote("sh600519", 100.0, status="suspended"),
        _quote("sz000001", 50.0, prev_close=49.5),
    ])
    outcome = plan_session(
        ledger=_ledger(),
        target_weights=pd.Series({"sh600519": 0.1, "sz000001": 0.1}),
        prices=pd.Series({"sh600519": 100.0, "sz000001": 50.0}), gate=gate,
        trade_date=TRADE_DATE, equity=1_000_000.0, quotes=board,
        book=IdempotencyBook(tmp_path / "book.jsonl"), audit=AuditLog(tmp_path / "audit.jsonl"),
    )
    assert [order.symbol for order in outcome.orders] == ["sz000001"]
    assert outcome.rejected and outcome.rejected[0]["symbol"] == "sh600519"
    assert outcome.halted is False


# --------------------------------------------------------------------------- #
# the whole loop: plan -> fills -> statement -> reconcile
# --------------------------------------------------------------------------- #
def test_the_loop_closes(tmp_path) -> None:
    gate = TradingGate(tmp_path / "gate.json")
    gate.set_level(TradingLevel.L2_SEMI_AUTO)
    ledger = _ledger(cash=1_000_000.0)
    outcome = plan_session(
        ledger=ledger, target_weights=pd.Series({"sh600519": 0.2}),
        prices=pd.Series({"sh600519": 100.0}), gate=gate, trade_date=TRADE_DATE,
        equity=1_000_000.0, book=IdempotencyBook(tmp_path / "book.jsonl"),
        audit=AuditLog(tmp_path / "audit.jsonl"), output_dir=tmp_path / "orders",
        snapshot_time="2026-11-09T10:00:00+08:00",
    )
    assert outcome.written is not None
    assert (tmp_path / "orders" / "orders.csv").exists()

    booked = apply_fills(
        ledger,
        pd.DataFrame([{"symbol": "sh600519", "side": "buy", "shares": 2000, "price": 100.0}]),
        date=TRADE_DATE,
    )
    statement = daily_statement(booked, pd.Series({"sh600519": 101.0}), trade_date=TRADE_DATE)
    append_equity_point(tmp_path / "equity.jsonl", statement)
    report = reconcile(booked, {"sh600519": 2000}, broker_cash=booked.cash, trade_date=TRADE_DATE)
    assert report.ok is True
    assert equity_curve(tmp_path / "equity.jsonl").iloc[-1]["positions"] == 1


# --------------------------------------------------------------------------- #
# the paper replay script: the 20-session equity curve must be reproducible
# --------------------------------------------------------------------------- #
def test_paper_replay_builds_one_equity_point_per_session(tmp_path) -> None:
    """Regression for "missing price for held symbol".

    ``plan_orders`` values the whole book to size orders, so a name that drops out
    of the target basket while still held must still have a price in the vector --
    the replay, the CLI plan and the desk all had to be fixed for that.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from run_paper_replay import replay  # noqa: E402

    rng = np.random.default_rng(5)
    index = pd.bdate_range("2025-01-02", periods=140)
    panel = {}
    for i in range(6):
        close = pd.Series(10.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.008 + i * 0.001, len(index)))), index=index)
        panel["sh6000%02d" % i] = pd.DataFrame(
            {"open": close * 1.001, "high": close * 1.01, "low": close * 0.99,
             "close": close, "volume": 900_000.0},
            index=index,
        )
    summary = replay(
        panel, sessions=3, equity=200_000.0, output=tmp_path / "replay",
        cache_dir=tmp_path / "no-cache",  # prices come from the panel, not the cache
    )
    assert summary["sessions"] == 3
    assert summary["rows"] and all(row["exec_day"] for row in summary["rows"])
    assert (tmp_path / "replay" / "equity.jsonl").exists()
    assert (tmp_path / "replay" / "summary.json").exists()
