"""Tests for the tradability helper (23.75).

The point of this study is that "the bar exists and volume is positive" is an
assumption every execution result rests on. These tests pin the four ways it is
measured: cross-sectional breadth, held name-days without a bar, the longest
such run, and how much book weight sits in a name that cannot be traded.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_tradability", ROOT / "scripts" / "run_tradability.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tradability = _load_module()


def _tradable(rows):
    index = pd.bdate_range("2024-01-02", periods=len(rows))
    return pd.DataFrame(rows, index=index).fillna(False).astype(bool)


def _positions(path: Path, rows) -> None:
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    frame.to_csv(path / "positions.csv", index=False)


def test_breadth_counts_tradable_names_per_year():
    frame = _tradable([
        {"AAA": True, "BBB": True, "CCC": True},
        {"AAA": True, "BBB": True, "CCC": True},
        {"AAA": True, "BBB": False, "CCC": False},
    ])
    table = tradability.breadth_table(frame).iloc[0]
    assert table["days"] == 3
    assert table["median_tradable"] == 3.0
    assert table["min_tradable"] == 1
    # 90% of a 3-name pool is 2.7, so the 2-name day also counts as thin.
    assert abs(table["share_days_below_90pct"] - 1 / 3) < 1e-12


def test_holding_exposure_counts_name_days_without_a_bar(tmp_path):
    index = pd.bdate_range("2024-01-02", periods=3)
    tradable = pd.DataFrame(
        {"AAA": [True, True, True], "BBB": [False, False, False]}, index=index
    )
    rows = [
        {"date": day, "symbol": sym, "shares": 100, "close": 10.0, "market_value": 1000.0, "weight": 0.5}
        for day in index
        for sym in ("AAA", "BBB")
    ]
    _positions(tmp_path, rows)

    row = tradability.holdings_exposure(tmp_path, tradable)
    assert row["held_name_days"] == 6
    assert row["suspended_name_days"] == 3
    assert abs(row["suspended_share"] - 0.5) < 1e-12


def test_closed_positions_are_not_counted_as_exposure(tmp_path):
    index = pd.bdate_range("2024-01-02", periods=2)
    tradable = pd.DataFrame({"AAA": [False, False]}, index=index)
    rows = [
        {"date": day, "symbol": "AAA", "shares": 0, "close": 10.0, "market_value": 0.0, "weight": 0.0}
        for day in index
    ]
    _positions(tmp_path, rows)

    row = tradability.holdings_exposure(tmp_path, tradable)
    assert row["held_days"] == 0


def test_longest_suspension_measures_the_consecutive_run(tmp_path):
    index = pd.bdate_range("2024-01-02", periods=5)
    tradable = pd.DataFrame(
        {"AAA": [False, False, False, True, False], "BBB": [True] * 5}, index=index
    )
    rows = [
        {"date": day, "symbol": sym, "shares": 100, "close": 10.0, "market_value": 1000.0, "weight": 0.5}
        for day in index
        for sym in ("AAA", "BBB")
    ]
    _positions(tmp_path, rows)

    top = tradability.longest_holding_suspensions(tmp_path, tradable)
    assert top.iloc[0]["symbol"] == "AAA"
    assert top.iloc[0]["longest_suspension_days_held"] == 3


def test_frozen_exposure_measures_weight_that_cannot_be_traded(tmp_path):
    index = pd.bdate_range("2024-01-02", periods=2)
    tradable = pd.DataFrame(
        {"AAA": [True, False], "BBB": [False, True]}, index=index
    )
    rows = [
        {"date": index[0], "symbol": "AAA", "shares": 100, "close": 10.0, "market_value": 1000.0, "weight": 0.5},
        {"date": index[0], "symbol": "BBB", "shares": 100, "close": 10.0, "market_value": 1000.0, "weight": 0.5},
        {"date": index[1], "symbol": "AAA", "shares": 100, "close": 10.0, "market_value": 1000.0, "weight": 0.4},
        {"date": index[1], "symbol": "BBB", "shares": 100, "close": 10.0, "market_value": 1000.0, "weight": 0.6},
    ]
    _positions(tmp_path, rows)

    row = tradability.frozen_exposure(tmp_path, tradable)
    assert row["days"] == 2
    assert row["days_with_frozen"] == 2
    assert abs(row["share_days_frozen"] - 1.0) < 1e-12
    assert abs(row["mean_frozen_weight"] - 0.45) < 1e-12
    assert abs(row["max_frozen_weight"] - 0.5) < 1e-12
    assert row["frozen_name_days"] == 2


def test_clean_window_reports_no_frozen_exposure(tmp_path):
    index = pd.bdate_range("2024-01-02", periods=2)
    tradable = pd.DataFrame({"AAA": [True, True], "BBB": [True, True]}, index=index)
    rows = [
        {"date": day, "symbol": sym, "shares": 100, "close": 10.0, "market_value": 1000.0, "weight": 0.5}
        for day in index
        for sym in ("AAA", "BBB")
    ]
    _positions(tmp_path, rows)

    row = tradability.frozen_exposure(tmp_path, tradable)
    assert row["days_with_frozen"] == 0
    assert row["mean_frozen_weight"] == 0.0
    assert row["frozen_name_days"] == 0
