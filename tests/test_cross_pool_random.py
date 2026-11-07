"""Tests for the random sub-pool breadth study (23.93)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_cross_pool_random", ROOT / "scripts" / "run_cross_pool_random.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


random_pools = _load_module()


def test_target_vol_matches_the_published_configuration():
    assert random_pools.TARGET_VOL == 0.12


def _summary():
    path = ROOT / "reports" / "cross_pool_random" / "summary.csv"
    if not path.exists():
        pytest.skip("run scripts/run_cross_pool_random.py first")
    return pd.read_csv(path).sort_values("size").reset_index(drop=True)


def test_median_sharpe_rises_with_breadth():
    series = _summary()["sharpe_median"].to_list()
    assert all(series[i] < series[i + 1] for i in range(len(series) - 1))


def test_narrow_pools_stay_positive_most_of_the_time():
    """The point that separates 23.93 from 23.92: narrow is not the same as broken."""
    data = _summary()
    narrowest = data.iloc[0]
    assert narrowest["share_positive"] > 0.5
    assert narrowest["sharpe_p10"] > 0


def test_turnover_falls_as_the_pool_narrows():
    series = _summary()["turnover_median"].to_list()
    assert all(series[i] < series[i + 1] for i in range(len(series) - 1))
