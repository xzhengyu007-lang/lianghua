"""Tests for the selection-character study (23.81)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_selection_character", ROOT / "scripts" / "run_selection_character.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


character = _load_module()


def test_percentile_of_returns_the_share_of_the_pool_at_or_below():
    pool = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert abs(character.percentile_of(3.0, pool) - 0.75) < 1e-12
    assert abs(character.percentile_of(0.5, pool) - 0.0) < 1e-12
    assert abs(character.percentile_of(9.0, pool) - 1.0) < 1e-12


def test_percentile_of_is_nan_when_the_comparison_is_impossible():
    assert math.isnan(character.percentile_of(1.0, pd.Series([], dtype=float)))
    assert math.isnan(character.percentile_of(float("nan"), pd.Series([1.0, 2.0])))


def test_percentile_of_ignores_missing_pool_values():
    pool = pd.Series([1.0, float("nan"), 3.0])
    # two usable observations: 1.0 and 3.0; a 2.0 sits above half of them
    assert abs(character.percentile_of(2.0, pool) - 0.5) < 1e-12


def test_periods_partition_the_panel_in_order():
    spans = [character.PERIODS[k] for k in ("2019-2021", "2022-2023", "2024-2026")]
    for start, end in spans:
        assert start < end
    assert spans[0][1] < spans[1][0]
    assert spans[1][1] < spans[2][0]
