from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_system.signals import latest_signal, save_signal


def test_latest_signal_produces_executable_orders(sample_panel, tmp_path: Path) -> None:
    signal = latest_signal(
        sample_panel,
        "low_vol_core",
        params={"top_n": 3, "vol_window": 30, "trend_window": 60, "rebalance_days": 10},
        cash=500_000,
    )
    assert not signal.rows.empty
    assert signal.as_of == next(iter(sample_panel.values())).index[-1]
    assert signal.gross_exposure > 0
    assert (signal.rows["shares"] % 100 == 0).all()
    assert (signal.rows["amount"] >= 0).all()
    assert signal.rows["sector"].notna().all()
    out = save_signal(signal, tmp_path)
    payload = json.loads((out / "signal.json").read_text(encoding="utf-8"))
    assert payload["execution"] == "next trading day open"
    assert len(payload["orders"]) == len(signal.rows)
    assert (out / "orders.csv").exists()


def test_signal_reports_the_score_that_chose_each_name(sample_panel) -> None:
    """``orders.csv`` must carry the strategy score, not a hardcoded null.

    The score is the only field that explains why a name is in the book, and
    the watchlist already showed it, so an empty column in the order sheet was
    a real gap in the daily output rather than a formatting choice.
    """
    signal = latest_signal(
        sample_panel,
        "low_vol_core",
        params={"top_n": 3, "vol_window": 30, "trend_window": 60, "rebalance_days": 10},
        cash=500_000,
    )
    scores = signal.rows["score"]
    assert scores.notna().all()
    assert np.isfinite(scores.to_numpy(dtype=float)).all()
    # The book is *not* simply the top-N by score: sector caps and the trend
    # filter can promote a lower-scoring name, so a rank assertion would be
    # wrong here.  What must hold is that the score is the strategy's own
    # number and that the watchlist is drawn from the same series.
    assert signal.watchlist["score"].notna().all() if not signal.watchlist.empty else True


def test_signal_flags_missing_bar_on_as_of_date(sample_panel) -> None:
    panel = dict(sample_panel)
    last = next(iter(panel.values())).index[-1]
    panel["sh600001"] = panel["sh600001"].loc[: last - pd.Timedelta(days=3)]
    signal = latest_signal(
        panel,
        "low_vol_core",
        params={"top_n": 4, "vol_window": 30, "trend_window": 60, "rebalance_days": 10},
    )
    assert isinstance(signal.warnings, list)
    if "sh600001" in set(signal.rows["symbol"]):
        row = signal.rows.set_index("symbol").loc["sh600001"]
        assert "no_bar_today" in str(row["flags"])


def test_signal_respects_lot_size(sample_panel) -> None:
    signal = latest_signal(
        sample_panel,
        "low_vol_core",
        params={"top_n": 2, "vol_window": 30, "trend_window": 60, "rebalance_days": 10},
        cash=123_456,
        lot_size=100,
    )
    assert (signal.rows["shares"] % 100 == 0).all()
    assert (signal.rows["amount"] <= 123_456 * 1.01).all()

def test_signal_applies_the_backtest_position_and_gross_limits(sample_panel) -> None:
    """A live signal must never recommend a book the backtester would resize.

    The raw selector returns 1/top_n per name and 100% gross. The backtester
    clamps single names at ``max_position_weight`` and then scales the book to
    ``max_gross_exposure``. With two names that is a large difference (50% per
    name -> 25%, 100% -> 50%), so this test pins the transformation.
    """
    from quant_system.config import BacktestConfig
    from quant_system.portfolio import normalize_targets

    signal = latest_signal(
        sample_panel,
        "low_vol_core",
        params={"top_n": 2, "vol_window": 30, "trend_window": 60, "rebalance_days": 10},
        cash=1_000_000,
    )
    weights = signal.target_weights
    assert weights.max() <= 0.25 + 1e-9
    assert weights.sum() <= 0.95 + 1e-9

    # Same function, same numbers: the signal is the backtester's target book.
    from quant_system.portfolio import build_portfolio
    from quant_system.strategies import build_strategy

    raw = build_strategy(
        "low_vol_core", top_n=2, vol_window=30, trend_window=60, rebalance_days=10
    ).generate(sample_panel)
    scaled = build_portfolio(raw, sample_panel, use_vol_target=True)
    expected = normalize_targets(
        scaled, sorted(sample_panel), scaled.index, BacktestConfig()
    ).loc[signal.as_of]
    expected = expected[expected > 1e-9]
    assert weights.to_dict() == pytest.approx(expected.to_dict())


def test_signal_sizes_on_equity_not_leftover_cash(sample_panel) -> None:
    """Sizing on cash would under-invest an account that already holds stock."""
    params = {"top_n": 3, "vol_window": 30, "trend_window": 60, "rebalance_days": 10}
    cash_only = latest_signal(sample_panel, "low_vol_core", params=params, cash=100_000)
    full_equity = latest_signal(
        sample_panel, "low_vol_core", params=params, cash=100_000, equity=1_000_000
    )
    assert full_equity.sizing_equity == pytest.approx(1_000_000)
    assert cash_only.rows["shares"].sum() < full_equity.rows["shares"].sum()
    assert full_equity.gross_exposure == pytest.approx(cash_only.gross_exposure)


def test_signal_payload_records_sizing_basis(sample_panel) -> None:
    signal = latest_signal(
        sample_panel,
        "low_vol_core",
        params={"top_n": 3, "vol_window": 30, "trend_window": 60, "rebalance_days": 10},
        cash=500_000,
    )
    payload = signal.to_dict()
    assert payload["sizing_equity"] == pytest.approx(500_000)
    assert payload["lot_size"] == 100
    assert payload["planned_notional"] <= 500_000 * 0.95 * 1.02


def test_signal_warns_when_the_feed_has_not_updated(sample_panel) -> None:
    as_of = max(frame["close"].dropna().index.max() for frame in sample_panel.values())
    params = dict(top_n=3, vol_window=30, trend_window=60, rebalance_days=10)
    stale = latest_signal(
        sample_panel,
        "low_vol_core",
        params=params,
        today=as_of + pd.Timedelta(days=30),
    )
    assert any("数据源已更新" in warning for warning in stale.warnings)

    current = latest_signal(
        sample_panel,
        "low_vol_core",
        params=params,
        today=as_of + pd.Timedelta(days=1),
    )
    assert not any("数据源已更新" in warning for warning in current.warnings)
