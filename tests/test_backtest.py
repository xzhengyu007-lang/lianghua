from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_system.backtest import Backtester, simulate_buy_and_hold
from quant_system.config import BacktestConfig, CostConfig
from quant_system.costs import execution_price, transaction_costs


def _panel() -> dict[str, pd.DataFrame]:
    index = pd.bdate_range("2024-01-02", periods=100)
    result = {}
    for i, symbol in enumerate(["sh600000", "sh600036", "sz000001"]):
        close = pd.Series(10 + np.arange(100) * 0.05 + i, index=index)
        frame = pd.DataFrame(
            {
                "open": close * 1.001,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000.0,
            },
            index=index,
        )
        result[symbol] = frame
    return result


def test_execution_price_is_adverse() -> None:
    costs = CostConfig(slippage_bps=10)
    assert execution_price("buy", 100, costs) == pytest.approx(100.1)
    assert execution_price("sell", 100, costs) == pytest.approx(99.9)


def test_transaction_costs_min_commission_and_stamp() -> None:
    costs = CostConfig(min_commission=5, commission_rate=0.00025, stamp_duty_rate=0.0005)
    buy = transaction_costs("buy", 100, 10.0, open_price=10.0, config=costs)
    assert buy.commission == pytest.approx(5.0)
    assert buy.stamp_duty == 0.0
    sell = transaction_costs("sell", 1000, 10.0, open_price=10.0, config=costs)
    assert sell.stamp_duty == pytest.approx(5.0)


def test_backtest_cash_and_equity_identity() -> None:
    panel = _panel()
    dates = next(iter(panel.values())).index
    targets = pd.DataFrame(1 / 3, index=dates, columns=sorted(panel))
    targets.iloc[:2] = 0.0
    result = Backtester(BacktestConfig(initial_cash=100_000, max_position_weight=0.5)).run(panel, targets)
    assert len(result.equity) == len(dates)
    assert result.equity.iloc[-1] > 0
    assert result.metrics.max_drawdown <= 0
    if len(result.trades):
        assert (result.trades["shares"] > 0).all()


def test_lot_size_respected() -> None:
    panel = _panel()
    dates = next(iter(panel.values())).index
    targets = pd.DataFrame(0.0, index=dates, columns=sorted(panel))
    targets.iloc[2:, 0] = 0.5
    result = Backtester(BacktestConfig(initial_cash=200_000, lot_size=100)).run(panel, targets)
    assert len(result.trades) > 0
    assert (result.trades["shares"] % 100 == 0).all()


def test_buy_and_hold_has_trades() -> None:
    panel = _panel()
    result = simulate_buy_and_hold(panel, config=BacktestConfig(initial_cash=100_000, max_position_weight=0.5))
    assert result.metadata["trades"] > 0
    assert result.metrics.total_return != 0


def test_delisted_pool_window_uses_common_sample() -> None:
    """Extending a panel with long-dead names must not move the test window.

    Regression for the point-in-time comparison: delisted names carried history
    back to 2004, and merging them into the survivor panel silently shifted the
    backtest start (and the benchmark join). The comparison is only valid when
    every pool is clipped to the same sample.
    """
    from quant_system.data import align_panel

    long_index = pd.bdate_range("2015-01-02", periods=1600)
    sample = long_index[-400:]

    def frame_on(index: pd.DatetimeIndex, base: float) -> pd.DataFrame:
        close = pd.Series(base + np.arange(len(index)) * 0.01, index=index)
        return pd.DataFrame(
            {
                "open": close * 1.001,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000.0,
            },
            index=index,
        )

    panel = {
        "sh600000": frame_on(sample, 10.0),
        "sh600036": frame_on(sample, 11.0),
        # Dead name: stops trading well before the survivor sample ends.
        "sz000999": frame_on(long_index[:900], 8.0),
    }

    _, raw_dates = align_panel(panel)
    _, clipped_dates = align_panel(
        {symbol: frame.loc[sample[0] : sample[-1]] for symbol, frame in panel.items()}
    )

    assert raw_dates.min() < sample[0]
    assert clipped_dates.min() == sample[0]
    assert clipped_dates.max() == sample[-1]
    assert list(clipped_dates) == list(sample)

def test_delisted_exits_counts_actual_liquidations() -> None:
    """A retiring name that was never held must not count as an exit."""
    index = pd.bdate_range("2024-01-02", periods=40)
    price = pd.Series(10.0, index=index)
    panel = {
        "sh600000": pd.DataFrame(
            {
                "open": price, "high": price, "low": price, "close": price,
                "volume": 2_000_000.0,
            },
            index=index,
        )
    }
    # A second name stops trading halfway through but is never targeted.
    late = index[:20]
    panel["sh600004"] = pd.DataFrame(
        {
            "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
            "volume": 2_000_000.0,
        },
        index=late,
    )
    targets = pd.DataFrame(0.0, index=index, columns=["sh600000", "sh600004"])
    targets.loc[index[5]:, "sh600000"] = 0.5
    result = Backtester(BacktestConfig(initial_cash=100_000)).run(panel, targets)
    assert result.metadata["delisted_candidates"] == 1
    assert result.metadata["delisted_exits"] == 0


def test_turnover_is_interval_not_annualised(sample_panel) -> None:
    from quant_system.backtest import Backtester
    from quant_system.config import BacktestConfig
    from quant_system.portfolio import build_portfolio
    from quant_system.strategies import build_strategy

    raw = build_strategy("low_vol_core", top_n=2, vol_window=30, trend_window=60, rebalance_days=10).generate(sample_panel)
    weights = build_portfolio(raw, sample_panel, use_vol_target=False)
    result = Backtester(BacktestConfig(initial_cash=1_000_000)).run(sample_panel, weights)
    years = len(result.equity) / 252.0
    assert result.metadata["annual_turnover"] == pytest.approx(
        result.metrics.turnover / years
    )
    assert result.metadata["turnover_definition"].startswith("turnover = one-way")
