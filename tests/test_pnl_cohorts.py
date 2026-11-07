"""Tests for the realised-P&L cohort decomposition (23.79)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_pnl_cohorts", ROOT / "scripts" / "run_pnl_cohorts.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pnl = _load_module()


def test_cohort_of_splits_on_the_code_suffix():
    codes = {"600519", "000858"}
    assert pnl.cohort_of("sh600519", codes) == "listed"
    assert pnl.cohort_of("sz000858", codes) == "listed"
    assert pnl.cohort_of("sh600145", codes) == "delisted"


def test_summarise_splits_exposure_and_pnl(tmp_path):
    index = pd.bdate_range("2024-01-02", periods=2)
    positions = pd.DataFrame(
        [
            {"date": d, "symbol": s, "shares": 100, "close": 10.0, "market_value": 1000.0, "weight": 0.5}
            for d in index
            for s in ("sh600519", "sh600145")
        ]
    )
    positions.to_csv(tmp_path / "positions.csv", index=False)
    trades = pd.DataFrame(
        [
            {"date": index[0], "symbol": "sh600519", "side": "sell", "shares": 100, "price": 10.0,
             "notional": 1000.0, "reason": "rebalance", "commission": 0.0, "stamp_duty": 0.0,
             "transfer_fee": 0.0, "slippage": 0.0, "fees": 0.0, "realized_pnl": 100.0},
            {"date": index[1], "symbol": "sh600145", "side": "sell", "shares": 100, "price": 10.0,
             "notional": 1000.0, "reason": "rebalance", "commission": 0.0, "stamp_duty": 0.0,
             "transfer_fee": 0.0, "slippage": 0.0, "fees": 0.0, "realized_pnl": 300.0},
        ]
    )
    trades.to_csv(tmp_path / "trades.csv", index=False)

    rows = pnl.summarise(tmp_path, "toy", {"600519"})
    by = {r["cohort"]: r for r in rows}
    assert abs(by["listed"]["exposure_share"] - 0.5) < 1e-12
    assert abs(by["delisted"]["exposure_share"] - 0.5) < 1e-12
    assert by["listed"]["realised_pnl"] == 100.0
    assert by["delisted"]["realised_pnl"] == 300.0
    assert abs(by["delisted"]["pnl_share"] - 0.75) < 1e-12


def test_single_name_cohort_is_fully_concentrated(tmp_path):
    index = pd.bdate_range("2024-01-02", periods=1)
    pd.DataFrame(
        [{"date": index[0], "symbol": "sh600145", "shares": 100, "close": 10.0,
          "market_value": 1000.0, "weight": 1.0}]
    ).to_csv(tmp_path / "positions.csv", index=False)
    pd.DataFrame(
        [{"date": index[0], "symbol": "sh600145", "side": "sell", "shares": 100, "price": 10.0,
          "notional": 1000.0, "reason": "rebalance", "commission": 0.0, "stamp_duty": 0.0,
          "transfer_fee": 0.0, "slippage": 0.0, "fees": 0.0, "realized_pnl": -50.0}]
    ).to_csv(tmp_path / "trades.csv", index=False)

    rows = pnl.summarise(tmp_path, "toy", {"600519"})
    by = {r["cohort"]: r for r in rows}
    assert by["delisted"]["names_held"] == 1
    assert abs(by["delisted"]["win_rate"]) < 1e-12
    assert by["listed"]["names_held"] == 0
