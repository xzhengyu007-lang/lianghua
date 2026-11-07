"""Tests for the cadence study on the blend (23.91)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_cadence_blend", ROOT / "scripts" / "run_cadence_blend.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cadence = _load_module()


def test_cadence_grid_brackets_the_incumbent_default():
    assert 20 in cadence.CADENCES and 60 in cadence.CADENCES
    assert cadence.REFERENCE == 20
    assert list(cadence.CADENCES) == sorted(cadence.CADENCES)


def test_median_holding_counts_consecutive_spells():
    index = pd.bdate_range("2024-01-02", periods=6)
    frame = pd.DataFrame(0.0, index=index, columns=["a"])
    frame.iloc[0:3, 0] = 0.5
    frame.iloc[4:6, 0] = 0.5
    assert cadence.median_holding(frame) == 2.5


def _panel():
    path = ROOT / "reports" / "cadence_blend" / "panel.csv"
    if not path.exists():
        pytest.skip("run scripts/run_cadence_blend.py first")
    return pd.read_csv(path)


def test_turnover_falls_monotonically_as_the_cadence_lengthens():
    data = _panel()
    for window, block in data.groupby("window"):
        series = block.sort_values("cadence")["turnover"].to_list()
        assert all(series[i] >= series[i + 1] for i in range(len(series) - 1)), window


def test_holding_grows_with_the_cadence():
    data = _panel()
    for window, block in data.groupby("window"):
        series = block.sort_values("cadence")["median_holding_days"].to_list()
        assert all(series[i] < series[i + 1] for i in range(len(series) - 1)), window
