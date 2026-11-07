from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_system.metrics import max_drawdown, monthly_returns, performance_metrics


def test_max_drawdown_simple() -> None:
    equity = pd.Series([100, 120, 90, 95, 130], index=pd.bdate_range("2024-01-01", periods=5))
    assert max_drawdown(equity) == pytest.approx(-0.25)


def test_performance_metrics_constant_growth() -> None:
    index = pd.bdate_range("2024-01-01", periods=252)
    returns = pd.Series(0.001, index=index)
    equity = (1 + returns).cumprod() * 100
    metrics = performance_metrics(returns, equity=equity)
    assert metrics.total_return > 0.2
    assert metrics.max_drawdown == pytest.approx(0.0)
    assert metrics.win_rate == 1.0
    assert np.isfinite(metrics.sharpe)


def test_performance_metrics_benchmark() -> None:
    index = pd.bdate_range("2024-01-01", periods=120)
    rng = np.random.default_rng(1)
    bench = pd.Series(rng.normal(0.0003, 0.01, 120), index=index)
    asset = 0.5 * bench + pd.Series(rng.normal(0.0001, 0.004, 120), index=index)
    metrics = performance_metrics(asset, benchmark_returns=bench)
    assert np.isfinite(metrics.beta)
    assert metrics.beta < 1.0
    assert np.isfinite(metrics.information_ratio)


def test_monthly_returns_shape() -> None:
    index = pd.bdate_range("2024-01-01", periods=90)
    equity = pd.Series(np.linspace(100, 120, 90), index=index)
    table = monthly_returns(equity)
    assert {"year", "month", "return"}.issubset(table.columns)
    assert len(table) >= 3


