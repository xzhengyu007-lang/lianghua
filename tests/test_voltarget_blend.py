"""Tests for the volatility-target re-pricing on the blend (23.85)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_voltarget_blend", ROOT / "scripts" / "run_voltarget_blend.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


voltarget = _load_module()


def test_reference_target_is_measured():
    assert voltarget.REFERENCE in voltarget.TARGETS
    assert list(voltarget.TARGETS) == sorted(voltarget.TARGETS)


def test_phases_cover_one_full_cycle():
    assert voltarget.PHASES == tuple(range(20))


def _panel():
    path = ROOT / "reports" / "voltarget_blend" / "panel.csv"
    if not path.exists():
        pytest.skip("run scripts/run_voltarget_blend.py first")
    return pd.read_csv(path)


def test_twelve_percent_draws_down_less_than_eighteen_in_every_window():
    """The one claim that has to hold wherever the baseline is set."""
    data = _panel()
    for window, sub in data.groupby("window"):
        twelve = sub[sub["target"] == 0.12].iloc[0]
        eighteen = sub[sub["target"] == 0.18].iloc[0]
        assert twelve["max_drawdown"] > eighteen["max_drawdown"], window


def test_ten_percent_is_dominated_by_twelve():
    """Same drawdown within half a point, strictly less return."""
    data = _panel()
    for window, sub in data.groupby("window"):
        ten = sub[sub["target"] == 0.10].iloc[0]
        twelve = sub[sub["target"] == 0.12].iloc[0]
        assert ten["total_return"] < twelve["total_return"], window
        assert ten["max_drawdown"] >= twelve["max_drawdown"] - 0.005, window
