"""Tests for the volatility-target calibration study (23.62).

The published claim is narrow: the scaler never levers up, a lower target means
less gross exposure, and the forecast window changes *how much* of the same
signal is carried rather than which names are held. These tests pin the two
mechanical facts so a future refactor of ``build_portfolio`` cannot silently
invert them while every report still looks plausible.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_vol_calibration", ROOT / "scripts" / "run_vol_calibration.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


calib = _load_module()


@pytest.fixture
def variant_outputs(sample_panel):
    low = calib.run_variant(sample_panel, 0.06, 40)
    high = calib.run_variant(sample_panel, 0.18, 40)
    return low, high


def test_scaler_never_levers_up(variant_outputs):
    for _, _, weights in variant_outputs:
        gross = weights.sum(axis=1)
        assert float(gross.max()) <= 1.0 + 1e-9


def test_lower_target_holds_less(variant_outputs):
    low_gross = variant_outputs[0][2].sum(axis=1)
    high_gross = variant_outputs[1][2].sum(axis=1)
    common = low_gross.index.intersection(high_gross.index)
    assert len(common) > 0
    assert float(low_gross.loc[common].mean()) < float(high_gross.loc[common].mean())


def test_forecast_window_does_not_change_the_basket(sample_panel):
    short = calib.run_variant(sample_panel, 0.12, 20)[2]
    long = calib.run_variant(sample_panel, 0.12, 120)[2]
    common = short.index.intersection(long.index)
    columns = [c for c in short.columns if c in long.columns]
    held_short = short.loc[common, columns].to_numpy() > 0
    held_long = long.loc[common, columns].to_numpy() > 0
    assert np.array_equal(held_short, held_long)
