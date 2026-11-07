"""Tests for the breadth/IC decomposition helper (23.76)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_breadth_ic", ROOT / "scripts" / "run_breadth_ic.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


breadth = _load_module()


def _pair(n_names: int = 10, n_days: int = 40):
    index = pd.bdate_range("2024-01-02", periods=n_days)
    columns = [f"s{i:02d}" for i in range(n_names)]
    rng = np.random.default_rng(7)
    signal = pd.DataFrame(rng.normal(size=(n_days, n_names)), index=index, columns=columns)
    future = signal * 0.5 + pd.DataFrame(
        rng.normal(scale=0.1, size=(n_days, n_names)), index=index, columns=columns
    )
    return signal, future


def test_matched_ic_reproduces_full_ic_when_target_equals_the_pool():
    """Subsampling the whole pool is a permutation, so the rank IC cannot move."""
    signal, future = _pair()
    full = breadth.spearman_ic(signal, future)
    matched = breadth.matched_ic(signal, future, target_n=10, draws=3)
    assert len(matched) == len(full)
    assert np.allclose(matched.to_numpy(), full.reindex(matched.index).to_numpy(), atol=1e-12)


def test_matched_ic_skips_cross_sections_smaller_than_the_target():
    signal, future = _pair(n_names=6)
    assert len(breadth.matched_ic(signal, future, target_n=10, draws=2)) == 0


def test_yearly_averages_ic_within_each_calendar_year():
    index = pd.to_datetime(["2023-12-27", "2023-12-28", "2024-01-02", "2024-01-03"])
    frame = pd.DataFrame(
        {
            "n_names": [10, 10, 8, 8],
            "n_listed": [10, 10, 8, 8],
            "n_delisted": [0, 0, 0, 0],
            "ic_full": [0.10, 0.20, 0.30, -0.10],
            "ic_listed": [0.10, 0.20, 0.30, -0.10],
            "ic_delisted": [0.0, 0.0, 0.0, 0.0],
            "ic_matched": [0.10, 0.20, 0.30, -0.10],
        },
        index=index,
    )
    table = breadth.yearly(frame).set_index("year")
    assert table.loc[2023, "days"] == 2
    assert abs(table.loc[2023, "ic_full"] - 0.15) < 1e-12
    assert abs(table.loc[2024, "ic_full"] - 0.10) < 1e-12
    assert table.loc[2024, "median_names"] == 8.0


def test_ols_recovers_a_known_linear_relation():
    rng = np.random.default_rng(3)
    x = rng.normal(size=200)
    z = rng.normal(size=200)
    frame = pd.DataFrame({"x": x, "z": z, "y": 1.5 + 2.0 * x - 0.5 * z})
    result = breadth.ols(frame, "y", ["x", "z"])
    assert abs(result["coefficients"]["intercept"]["estimate"] - 1.5) < 1e-8
    assert abs(result["coefficients"]["x"]["estimate"] - 2.0) < 1e-8
    assert abs(result["coefficients"]["z"]["estimate"] + 0.5) < 1e-8
    assert result["coefficients"]["x"]["t"] > 100
