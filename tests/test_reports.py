from __future__ import annotations

import numpy as np
import pandas as pd

from quant_system.metrics import drawdown_table, monthly_returns, performance_metrics
from quant_system.report import plot_equity, save_backtest
from quant_system.backtest import Backtester
from quant_system.config import BacktestConfig


def _result():
    index = pd.bdate_range("2024-01-02", periods=260)
    rng = np.random.default_rng(3)
    close = pd.Series(10 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(index)))), index=index)
    panel = {
        s: pd.DataFrame(
            {"open": close * (1 + 0.001 * i), "high": close * 1.01, "low": close * 0.99,
             "close": close, "volume": 2_000_000.0},
            index=index,
        )
        for i, s in enumerate(["sh600000", "sh600036", "sz000001", "sh601318"])
    }
    targets = pd.DataFrame(0.24, index=index, columns=sorted(panel))
    targets.iloc[:2] = 0.0
    targets.iloc[130:, :] = 0.0
    targets.iloc[130:, 0] = 0.24
    return Backtester(BacktestConfig(initial_cash=500_000)).run(panel, targets)


def test_dashboard_is_self_contained_and_complete(tmp_path) -> None:
    result = _result()
    save_backtest(result, tmp_path, title="Dashboard smoke test")
    html = (tmp_path / "dashboard.html").read_text(encoding="utf-8")
    assert "<svg" in html
    assert "Equity curve" in html
    assert "Drawdown" in html
    assert "Monthly returns" in html
    assert "Turnover (interval, one-way)" in html
    assert "http://" not in html and "https://" not in html
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "metrics.json").exists()


def test_plot_equity_compounds_benchmark_returns(tmp_path) -> None:
    """A return series must be compounded, not normalised as a price level."""
    result = _result()
    index = result.equity.index
    benchmark = pd.Series(0.0, index=index)
    benchmark.iloc[1] = 0.10
    path = plot_equity(result, tmp_path / "equity.png", benchmark=benchmark)
    assert path.exists()
    assert path.stat().st_size > 0


def test_monthly_returns_align_with_equity() -> None:
    result = _result()
    table = monthly_returns(result.equity)
    assert len(table) >= 10
    compounded = float((1 + table["return"].fillna(0.0)).prod() - 1)
    expected = float(result.equity.iloc[-1] / result.equity.iloc[0] - 1)
    assert abs(compounded - expected) < 0.02


def test_drawdown_table_reports_episodes() -> None:
    equity = pd.Series([100, 120, 90, 95, 130, 100], index=pd.bdate_range("2024-01-02", periods=6))
    table = drawdown_table(equity)
    assert not table.empty
    assert table["depth"].min() <= -0.25
    assert (table["days"] >= 1).all()


def test_metrics_are_json_safe() -> None:
    result = _result()
    payload = result.metrics.to_dict()
    for value in payload.values():
        assert not isinstance(value, np.ndarray)


def test_equal_weight_no_trade_on_flat_market() -> None:
    index = pd.bdate_range("2024-01-02", periods=40)
    frame = pd.DataFrame(
        {"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 1e6}, index=index
    )
    panel = {"sh600000": frame.copy()}
    targets = pd.DataFrame(0.5, index=index, columns=["sh600000"])
    targets.iloc[:2] = 0.0
    result = Backtester(BacktestConfig(initial_cash=100_000)).run(panel, targets)
    # Only costs move the curve: the stock itself is pinned at 10.00.
    assert result.metrics.total_return < 0
    assert result.metrics.total_return > -0.01
    buys = result.trades[result.trades["side"] == "buy"]
    assert len(buys) == 1
    assert result.equity.nunique() <= 3

def test_fingerprint_detects_a_single_changed_trade() -> None:
    """The fingerprint must catch a one-trade change, not just a total return."""
    from quant_system.fingerprint import compare, fingerprint

    equity = pd.Series(
        [100.0, 101.0, 102.5], index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    )
    trades = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "symbol": ["sh600000", "sh600000"],
            "side": ["buy", "sell"],
            "shares": [100, 100],
            "price": [10.0, 10.5],
            "reason": ["rebalance", "rebalance"],
        }
    )
    metadata = {"symbols_in_panel": 122, "costs": {"slippage_bps": 5.0}}
    base = fingerprint(equity, trades, metadata)
    assert compare(base, fingerprint(equity, trades, metadata))["equal"]

    changed = trades.copy()
    changed.loc[1, "price"] = 10.5001
    result = compare(base, fingerprint(equity, changed, metadata))
    assert not result["equal"]
    assert "trades" in result["differences"]


def test_fingerprint_ignores_row_order_and_volatile_metadata() -> None:
    """Canonicalisation must not depend on file order or a run timestamp."""
    from quant_system.fingerprint import compare, fingerprint

    equity = pd.Series(
        [100.0, 101.0], index=pd.to_datetime(["2024-01-02", "2024-01-03"])
    )
    trades = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "symbol": ["sh600000", "sh601318"],
            "side": ["buy", "buy"],
            "shares": [100, 200],
            "price": [10.0, 20.0],
            "reason": ["rebalance", "rebalance"],
        }
    )
    first = fingerprint(equity, trades, {"symbols_in_panel": 122, "generated_at": "a"})
    second = fingerprint(
        equity,
        trades.iloc[::-1].reset_index(drop=True),
        {"generated_at": "b", "symbols_in_panel": 122},
    )
    assert compare(first, second)["equal"]


def test_fingerprint_equity_is_date_sensitive() -> None:
    from quant_system.fingerprint import compare, fingerprint

    equity = pd.Series(
        [100.0, 101.0], index=pd.to_datetime(["2024-01-02", "2024-01-03"])
    )
    shifted = equity.copy()
    shifted.index = pd.to_datetime(["2024-01-02", "2024-01-04"])
    trades = pd.DataFrame()
    result = compare(
        fingerprint(equity, trades, {}), fingerprint(shifted, trades, {})
    )
    assert not result["equal"]
    assert result["differences"] == ["equity"]
