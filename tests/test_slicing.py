from __future__ import annotations

import numpy as np
import pandas as pd

from quant_system.backtest import Backtester, slice_result
from quant_system.config import BacktestConfig


def _panel(n: int = 120):
    index = pd.bdate_range("2024-01-02", periods=n)
    out = {}
    for i, symbol in enumerate(["sh600000", "sh600036", "sz000001"]):
        close = pd.Series(10 + i + np.arange(n) * 0.02, index=index)
        out[symbol] = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1e6,
            },
            index=index,
        )
    return out


def test_slice_keeps_positions_from_before_the_window() -> None:
    panel = _panel()
    index = next(iter(panel.values())).index
    targets = pd.DataFrame(1 / 3, index=index, columns=sorted(panel))
    targets.iloc[:2] = 0.0
    full = Backtester(BacktestConfig(initial_cash=300_000)).run(panel, targets)
    sliced = slice_result(full, start=index[60])
    assert sliced.equity.index[0] == index[60]
    # Exposure must reflect the inherited basket, not a fresh flat start.
    assert sliced.metrics.exposure > 0.5
    assert sliced.equity.iloc[0] == full.equity.loc[index[60]]


def test_slice_metrics_match_full_period_returns() -> None:
    panel = _panel()
    index = next(iter(panel.values())).index
    targets = pd.DataFrame(1 / 3, index=index, columns=sorted(panel))
    targets.iloc[:2] = 0.0
    full = Backtester(BacktestConfig(initial_cash=300_000)).run(panel, targets)
    sliced = slice_result(full, start=index[30], end=index[90])
    expected = full.equity.loc[index[30] : index[90]]
    pd.testing.assert_series_equal(sliced.equity, expected, check_freq=False)
    assert sliced.metrics.total_return == (
        expected.iloc[-1] / expected.iloc[0] - 1
    )