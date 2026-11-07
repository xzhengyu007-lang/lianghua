"""Tests for the execution-timing study (23.101)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_execution_timing", ROOT / "scripts" / "run_execution_timing.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


timing = _load_module()


def test_delays_start_at_zero_and_are_small():
    assert timing.DELAYS[0] == 0
    assert max(timing.DELAYS) <= 5


def test_shifting_weights_moves_the_execution_date_only():
    index = pd.bdate_range("2024-01-02", periods=4)
    weights = pd.DataFrame({"a": [0.5, 0.5, 0.0, 0.0]}, index=index)
    shifted = weights.shift(1).fillna(0.0)
    assert shifted["a"].tolist() == [0.0, 0.5, 0.5, 0.0]


def _panel():
    path = ROOT / "reports" / "execution_timing" / "panel.csv"
    if not path.exists():
        pytest.skip("run scripts/run_execution_timing.py first")
    return pd.read_csv(path)


def test_delay_hurts_in_two_windows_and_helps_in_the_reversal_window():
    """The finding is a direction, and the reversal window must stay a reversal."""
    data = _panel()
    def sharpe(window, delay):
        return float(data[(data["window"] == window) & (data["delay_days"] == delay)]["sharpe"].iloc[0])

    for window in ("2019-2021", "2024-2026"):
        assert sharpe(window, 3) < sharpe(window, 0), window
    assert sharpe("2022-2023", 3) > sharpe("2022-2023", 0)


def test_the_penalty_is_small_at_one_day():
    data = _panel()
    for window in data["window"].unique():
        base = float(data[(data["window"] == window) & (data["delay_days"] == 0)]["sharpe"].iloc[0])
        one = float(data[(data["window"] == window) & (data["delay_days"] == 1)]["sharpe"].iloc[0])
        assert abs(one - base) < 0.10, (window, one - base)
