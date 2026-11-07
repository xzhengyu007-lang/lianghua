"""Tests for the capacity-curve helpers (23.72 / 23.84).

The script itself rotted once: a report field referenced a local variable from
another function, so the published table could not be reproduced until it was
fixed. These pin the helper's arithmetic so the executable part stays honest.
"""

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


def test_weight_drift_is_half_the_l1_distance():
    realized = pd.DataFrame({"a": [0.5], "b": [0.5]})
    wanted = pd.DataFrame({"a": [0.6], "b": [0.4]})
    drift = capacity.weight_drift(realized, wanted)
    assert abs(drift.iloc[0] - 0.1) < 1e-12


def test_identical_weights_have_no_drift():
    frame = pd.DataFrame({"a": [0.5], "b": [0.5]})
    drift = capacity.weight_drift(frame, frame.copy())
    assert abs(drift.iloc[0]) < 1e-12


def test_only_common_names_are_compared():
    realized = pd.DataFrame({"a": [0.5]})
    wanted = pd.DataFrame({"a": [0.5], "b": [0.5]})
    drift = capacity.weight_drift(realized, wanted)
    assert abs(drift.iloc[0]) < 1e-12


def test_cash_levels_are_ascending_and_span_the_curve():
    levels = capacity.CASH_LEVELS
    assert list(levels) == sorted(levels)
    assert levels[0] <= 1e6 and levels[-1] >= 1e9
