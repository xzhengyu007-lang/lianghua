from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_system.bootstrap_report import load_returns
from quant_system.equity_io import daily_returns_from_equity, load_equity


def _write_report(root: Path, equity: list[float], start: str = "2024-01-02") -> Path:
    index = pd.bdate_range(start, periods=len(equity))
    frame = pd.DataFrame({"equity": equity}, index=index)
    frame.index.name = "date"
    frame.to_csv(root / "equity.csv")
    return root


def test_first_day_return_is_recovered(tmp_path: Path) -> None:
    # equity.csv stores the level *after* day one, so the +2% open is hidden.
    _write_report(tmp_path, [1_020_000.0, 1_010_000.0, 1_030_000.0])
    returns = daily_returns_from_equity(tmp_path)
    assert len(returns) == 3
    assert returns.iloc[0] == pytest.approx(0.02)
    assert returns.iloc[1] == pytest.approx(1_010_000.0 / 1_020_000.0 - 1.0)
    # Compounding from the initial cash must land on the final level.
    assert float(np.prod(1.0 + returns.to_numpy())) == pytest.approx(1.03)


def test_plain_pct_change_would_drop_a_day(tmp_path: Path) -> None:
    _write_report(tmp_path, [1_020_000.0, 1_010_000.0, 1_030_000.0])
    equity = load_equity(tmp_path)
    naive = equity.pct_change().dropna()
    assert len(naive) == 2
    assert len(daily_returns_from_equity(tmp_path)) == 3
    assert naive.sum() != pytest.approx(0.03)


def test_bootstrap_report_loader_uses_the_same_reconstruction(tmp_path: Path) -> None:
    _write_report(tmp_path, [1_005_000.0, 1_007_000.0])
    pd.testing.assert_series_equal(
        load_returns(tmp_path).astype(float),
        daily_returns_from_equity(tmp_path).astype(float),
    )


def test_single_point_report_is_rejected(tmp_path: Path) -> None:
    _write_report(tmp_path, [1_000_000.0])
    with pytest.raises(ValueError):
        daily_returns_from_equity(tmp_path)


def test_missing_equity_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        daily_returns_from_equity(tmp_path / "missing_report")


def test_initial_cash_must_be_positive(tmp_path: Path) -> None:
    _write_report(tmp_path, [1_000_000.0, 1_001_000.0])
    with pytest.raises(ValueError):
        daily_returns_from_equity(tmp_path, initial_cash=0.0)
