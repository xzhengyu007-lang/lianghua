"""Tests for the calendar-year breakdown (23.71)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_year_table", ROOT / "scripts" / "run_year_table.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


year_table = _load_module()


def test_return_is_compounded_within_the_year():
    returns = pd.Series([0.10, -0.10], index=pd.bdate_range("2024-01-02", periods=2))
    stats = year_table.year_stats(returns)
    assert abs(stats["return"] - (1.10 * 0.90 - 1.0)) < 1e-12


def test_drawdown_is_reported_as_a_negative_number():
    returns = pd.Series([0.05, -0.20, 0.02], index=pd.bdate_range("2024-01-02", periods=3))
    stats = year_table.year_stats(returns)
    assert -0.21 < stats["max_drawdown"] < 0.0


def test_hit_rate_counts_positive_days_only():
    returns = pd.Series([0.01, 0.0, -0.01, 0.02], index=pd.bdate_range("2024-01-02", periods=4))
    stats = year_table.year_stats(returns)
    assert stats["hit_rate"] == 0.5
    assert stats["days"] == 4
