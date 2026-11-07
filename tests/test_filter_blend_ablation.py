"""Tests for the filter/blend ablation (23.110)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_filter_blend_ablation", ROOT / "scripts" / "run_filter_blend_ablation.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass needs the module registered
    spec.loader.exec_module(module)
    return module


ablation = _load_module()

BASE = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2,
            rebalance_days=20, rebalance_reset="year")


def _panel(days=220):
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2024-01-02", periods=days)
    panel = {}
    for k in range(6):
        drift = 0.0008 if k % 2 == 0 else -0.0008  # half trend up, half down
        close = pd.Series(10.0 * np.exp(np.cumsum(rng.normal(drift, 0.014, days))), index=index)
        volume = pd.Series(rng.integers(1000, 5000, days).astype(float), index=index)
        panel["sh%06d" % (600000 + k)] = pd.DataFrame({
            "open": close.shift(1).fillna(close),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume,
        })
    return panel


def test_blend_off_scores_pure_volatility():
    panel = _panel()
    on, _ = ablation.Ablation(use_filter=True, use_blend=True, **BASE).score(panel)
    off, _ = ablation.Ablation(use_filter=True, use_blend=False, **BASE).score(panel)
    assert not on.equals(off)


def test_filter_off_widens_eligibility():
    panel = _panel()
    _, filtered = ablation.Ablation(use_filter=True, use_blend=True, **BASE).score(panel)
    _, unfiltered = ablation.Ablation(use_filter=False, use_blend=True, **BASE).score(panel)
    assert not filtered.equals(unfiltered)
    assert unfiltered.to_numpy().sum() >= filtered.to_numpy().sum()


def _panel_csv():
    path = ROOT / "reports" / "filter_blend_ablation" / "panel.csv"
    if not path.exists():
        pytest.skip("run scripts/run_filter_blend_ablation.py first")
    return pd.read_csv(path)


def test_all_four_variants_are_measured_in_every_window():
    data = _panel_csv()
    assert data["variant"].nunique() == 4
    assert len(data) == 12


def test_no_cell_is_statistically_significant():
    path = ROOT / "reports" / "filter_blend_ablation" / "paired.csv"
    if not path.exists():
        pytest.skip("run scripts/run_filter_blend_ablation.py first")
    pairs = pd.read_csv(path)
    assert int(((pairs["low"] > 0) | (pairs["high"] < 0)).sum()) == 0
