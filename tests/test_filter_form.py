"""Tests for the trend-filter form sweep (23.116)."""

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
        "run_filter_form", ROOT / "scripts" / "run_filter_form.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass needs the module registered
    spec.loader.exec_module(module)
    return module


sweep = _load_module()

BASE = dict(top_n=8, vol_window=20, trend_window=60, max_per_sector=2,
            rebalance_days=20, rebalance_reset="year")


def _panel(days=200):
    rng = np.random.default_rng(3)
    index = pd.bdate_range("2024-01-02", periods=days)
    panel = {}
    for k in range(4):
        drift = 0.0010 if k < 2 else -0.0010
        close = pd.Series(10.0 * np.exp(np.cumsum(rng.normal(drift, 0.01, days))), index=index)
        panel["sh%06d" % (600000 + k)] = pd.DataFrame({
            "open": close.shift(1).fillna(close),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": pd.Series(2000.0, index=index),
        })
    return panel


def test_no_filter_admits_at_least_as_many_names():
    panel = _panel()
    _, none = sweep.Filtered(filter_kind="none", **BASE).score(panel)
    _, sma = sweep.Filtered(filter_kind="sma", filter_window=120, **BASE).score(panel)
    assert none.to_numpy().sum() >= sma.to_numpy().sum()
    assert not none.equals(sma)


def test_sma_filter_is_the_documented_rule():
    panel = _panel()
    from quant_system.data import price_frame
    from quant_system.indicators import sma

    close = price_frame(panel, "close")
    _, eligible = sweep.Filtered(filter_kind="sma", filter_window=120, **BASE).score(panel)
    expected = close > sma(close, 120)
    # every eligible cell must also be above its own SMA
    assert not (eligible & ~expected).to_numpy().any()


def test_an_unknown_filter_kind_is_rejected():
    with pytest.raises(ValueError):
        sweep.Filtered(filter_kind="magic", **BASE).score(_panel())


def _blend():
    path = ROOT / "reports" / "filter_form" / "blend.csv"
    if not path.exists():
        pytest.skip("run scripts/run_filter_form.py --blend first")
    return pd.read_csv(path)


def test_the_longer_window_wins_in_every_window_after_phase_blending():
    """The finding: SMA250 beats SMA120 across the board, post-blend."""
    blend = _blend()
    pivot = blend.pivot_table(index="window", columns="variant", values="sharpe")
    long_col = next(c for c in pivot.columns if "250" in c)
    short_col = next(c for c in pivot.columns if "120" in c)
    assert (pivot[long_col] > pivot[short_col]).all()


def test_the_single_phase_gap_is_larger_than_the_blended_gap():
    """Phase averaging shrinks the gap but does not remove it."""
    blend = _blend()
    pivot = blend.pivot_table(index="window", columns="variant", values="sharpe")
    long_col = next(c for c in pivot.columns if "250" in c)
    short_col = next(c for c in pivot.columns if "120" in c)
    blended_gap = float((pivot[long_col] - pivot[short_col]).min())
    single = pd.read_csv(ROOT / "reports" / "filter_form" / "panel.csv")
    sp = single.pivot_table(index="window", columns="variant", values="sharpe")
    s_long = next(c for c in sp.columns if c.startswith("SMA250"))
    s_short = next(c for c in sp.columns if c.startswith("SMA120"))
    assert float((sp[s_long] - sp[s_short]).min()) > blended_gap
