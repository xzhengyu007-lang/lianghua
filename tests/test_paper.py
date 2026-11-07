"""Paper ledger invariants: no oversell, no overdraft, cash reconciles."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quant_system.config import CostConfig
from quant_system.paper import (
    Ledger,
    Position,
    apply_fills,
    plan_orders,
    positions_frame,
    reconcile,
    summarize,
)


def _prices(**kwargs) -> pd.Series:
    return pd.Series(kwargs, dtype=float)


def test_plan_sells_before_buys_and_respects_cash() -> None:
    ledger = Ledger(cash=10_000.0, positions={"sh600000": Position(shares=1000, avg_cost=5.0)})
    prices = _prices(sh600000=5.0, sh600036=40.0)
    orders = plan_orders(ledger, pd.Series({"sh600036": 0.5}), prices, cash_buffer=0.02)
    assert not orders.empty
    assert orders["side"].tolist() == ["sell", "buy"]
    buys = orders[orders["side"] == "buy"]
    assert buys["shares"].sum() * 40.0 <= ledger.cash + 1000 * 5.0


def test_stale_suspended_position_is_not_valued_or_traded() -> None:
    """A stale close cannot size today's order; leave the position untouched."""
    ledger = Ledger(
        cash=10_000.0,
        positions={"sh600000": Position(shares=1000, avg_cost=5.0)},
    )
    prices = _prices(sh600000=5.0, sh600036=40.0)
    orders = plan_orders(
        ledger,
        pd.Series({"sh600036": 0.5}),
        prices,
        stale_symbols={"sh600000"},
    )
    assert not orders.empty
    assert "sh600000" not in set(orders["symbol"])
    assert (orders["side"] == "buy").all()


def test_ledger_round_trip(tmp_path: Path) -> None:
    ledger = Ledger(cash=1000.0, positions={"sh600000": Position(200, 9.5)})
    path = ledger.save(tmp_path / "ledger.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    restored = Ledger.load(path)
    assert restored.cash == pytest.approx(1000.0)
    assert restored.positions["sh600000"].shares == 200
    assert restored.positions["sh600000"].avg_cost == pytest.approx(9.5)


def test_oversell_is_rejected() -> None:
    ledger = Ledger(cash=0.0, positions={"sh600000": Position(100, 10.0)})
    fills = pd.DataFrame(
        [{"symbol": "sh600000", "side": "sell", "shares": 200, "price": 10.0}]
    )
    with pytest.raises(ValueError, match="cannot sell"):
        apply_fills(ledger, fills, date="2026-01-05")


def test_overdraft_is_rejected() -> None:
    ledger = Ledger(cash=500.0)
    fills = pd.DataFrame(
        [{"symbol": "sh600000", "side": "buy", "shares": 100, "price": 10.0}]
    )
    with pytest.raises(ValueError, match="available"):
        apply_fills(ledger, fills, date="2026-01-05")


def test_apply_fills_updates_cash_and_cost_basis() -> None:
    costs = CostConfig(commission_rate=0.00025, min_commission=5.0, transfer_fee_rate=0.00001)
    ledger = Ledger(cash=100_000.0)
    fills = pd.DataFrame([{"symbol": "sh600000", "side": "buy", "shares": 1000, "price": 10.0}])
    after = apply_fills(ledger, fills, date="2026-01-05", costs=costs)
    expected_fee = 5.0 + 1000 * 10.0 * 0.00001
    assert after.cash == pytest.approx(100_000.0 - 10_000.0 - expected_fee)
    assert after.positions["sh600000"].shares == 1000
    assert after.positions["sh600000"].avg_cost == pytest.approx((10_000 + expected_fee) / 1000)
    assert ledger.cash == pytest.approx(100_000.0), "input ledger must not be mutated"

    sell = pd.DataFrame([{"symbol": "sh600000", "side": "sell", "shares": 1000, "price": 11.0}])
    closed = apply_fills(after, sell, date="2026-02-02", costs=costs)
    assert "sh600000" not in closed.positions
    assert closed.cash > after.cash
    assert closed.trades[-1]["realized_pnl"] > 0


def test_summary_and_positions_frame() -> None:
    ledger = Ledger(cash=1000.0, positions={"sh600000": Position(100, 10.0)})
    prices = _prices(sh600000=12.0)
    summary = summarize(ledger, prices)
    assert summary["equity"] == pytest.approx(1000 + 1200)
    frame = positions_frame(ledger, prices)
    assert frame.loc[0, "market_value"] == pytest.approx(1200)
    assert frame.loc[0, "weight"] == pytest.approx(1200 / 2200)


def test_reconcile_reports_only_differences() -> None:
    ledger = Ledger(cash=1000.0, positions={"sh600000": Position(100, 10.0)})
    diff = reconcile(ledger, {"sh600000": 100}, broker_cash=1000.0)
    assert diff.empty
    diff = reconcile(ledger, {"sh600000": 200, "sh600036": 50}, broker_cash=900.0)
    assert set(diff["item"]) == {"sh600000", "sh600036", "cash"}
    assert diff.set_index("item").loc["sh600036", "difference"] == -50


def test_fills_without_side_column_are_inferred() -> None:
    """Broker exports often omit `side`; infer it from the current position."""
    ledger = Ledger(cash=100_000.0, positions={})
    fills = pd.DataFrame(
        [
            {"symbol": "sh600000", "price": 10.0, "shares": 1000},
            {"symbol": "sh600000", "price": 11.0, "shares": 400},
        ]
    )
    first = apply_fills(ledger, fills.iloc[[0]], date="2026-11-06")
    assert first.positions["sh600000"].shares == 1000
    assert first.cash < 100_000.0

    second = apply_fills(first, fills.iloc[[1]], date="2026-11-07")
    assert second.positions["sh600000"].shares == 600
    assert second.trades[-1]["side"] == "sell"
    assert second.cash > first.cash


def test_fills_missing_required_columns_raise_a_clear_error() -> None:
    ledger = Ledger(cash=100_000.0, positions={})
    fills = pd.DataFrame([{"symbol": "sh600000", "price": 10.0}])
    with pytest.raises(ValueError, match="missing"):
        apply_fills(ledger, fills, date="2026-11-06")
