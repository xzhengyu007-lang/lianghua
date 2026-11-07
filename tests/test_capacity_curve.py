"""Tests for the capacity curve helper (23.72)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_capacity_curve", ROOT / "scripts" / "run_capacity_curve.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capacity = _load_module()


def test_identical_weights_have_zero_drift():
    idx = pd.bdate_range("2024-01-02", periods=3)
    weights = pd.DataFrame({"AAA": [0.5, 0.4, 0.3], "BBB": [0.5, 0.6, 0.7]}, index=idx)
    drift = capacity.weight_drift(weights, weights.copy())
    assert (drift.abs() < 1e-12).all()


def test_missing_half_of_a_position_is_counted():
    idx = pd.bdate_range("2024-01-02", periods=1)
    realized = pd.DataFrame({"AAA": [0.0]}, index=idx)
    wanted = pd.DataFrame({"AAA": [0.5]}, index=idx)
    # L1 = 0.5, so the half-L1 (fraction of the book in the wrong place) is 0.25
    assert abs(capacity.weight_drift(realized, wanted).iloc[0] - 0.25) < 1e-12


def test_unfilled_order_shows_up_as_drift():
    idx = pd.bdate_range("2024-01-02", periods=1)
    realized = pd.DataFrame({"AAA": [0.5], "BBB": [0.0]}, index=idx)
    wanted = pd.DataFrame({"AAA": [0.5], "BBB": [0.5]}, index=idx)
    # BBB wanted 0.5 but filled 0.0 -> half-L1 = 0.25
    assert abs(capacity.weight_drift(realized, wanted).iloc[0] - 0.25) < 1e-12
