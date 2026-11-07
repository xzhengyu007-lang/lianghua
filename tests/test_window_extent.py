"""Tests for the window-extent sweep and the 23.118 correction (23.118)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _panel():
    path = ROOT / "reports" / "window_extent" / "panel.csv"
    if not path.exists():
        pytest.skip("run scripts/run_window_extent.py first")
    return pd.read_csv(path)


def test_the_curve_has_an_interior_peak_not_a_boundary_maximum():
    """23.116 claimed monotone; the extended sweep must show it turning."""
    data = _panel().set_index("window")["full_sharpe"]
    peak = int(data.idxmax())
    assert peak not in (int(data.index.min()), int(data.index.max()))
    assert data.loc[peak] > data.loc[300] > data.loc[500]


def test_both_ends_are_worse_than_the_plateau():
    data = _panel().set_index("window")["full_sharpe"]
    assert data.loc[60] < data.loc[120]
    assert data.loc[500] < data.loc[250]


def test_the_plateau_members_are_not_distinguishable_from_each_other():
    """250 vs 300 differ by less than one config's own phase span."""
    data = _panel().set_index("window")
    gap = float(data.loc[250, "full_sharpe"] - data.loc[300, "full_sharpe"])
    span = float(data.loc[250, "phase_max"] - data.loc[250, "phase_min"])
    assert 0 < gap < span


def test_turnover_falls_monotonically_with_the_window():
    data = _panel().sort_values("window")
    assert data["turnover"].is_monotonic_decreasing


def test_the_render_only_mode_rebuilt_the_report():
    report = ROOT / "reports" / "window_extent" / "report.md"
    if not report.exists():
        pytest.skip("run scripts/run_window_extent.py --render-only first")
    text = report.read_text(encoding="utf-8")
    assert "| 窗口 |" in text
    for window in ("60", "120", "250", "300", "400", "500"):
        assert "| %s |" % window in text
