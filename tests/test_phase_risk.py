"""Guard for the phase-risk comparison (23.65).

The claim is "a lower risk budget compresses the calendar spread only
slightly and leaves the spread-per-unit-PnL untouched". Both halves are
checkable from the saved report, so they stay checkable after a refactor.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "reports" / "phase_risk_comparison" / "summary.csv"


def _summary() -> pd.DataFrame:
    if not SUMMARY.exists():
        pytest.skip("run scripts/run_phase_risk_comparison.py first")
    table = pd.read_csv(SUMMARY)
    if "target" not in table.columns:
        pytest.skip("summary.csv predates the grouped schema")
    return table


def _pair(table: pd.DataFrame, group: str) -> tuple[pd.Series, pd.Series]:
    low = table[(table["group"] == group) & (table["target"] == "12%")]
    high = table[(table["group"] == group) & (table["target"] == "18%")]
    if low.empty or high.empty:
        pytest.skip(f"no 12%/18% pair for {group}")
    return low.iloc[0], high.iloc[0]


def test_lower_budget_narrows_the_absolute_spread_but_only_a_little() -> None:
    low, high = _pair(_summary(), "2024-2026")
    assert low.mean_spread < high.mean_spread
    shrink = 1.0 - low.mean_spread / high.mean_spread
    assert 0.0 < shrink < 0.20


def test_relative_spread_is_basically_unchanged() -> None:
    low, high = _pair(_summary(), "2024-2026")
    assert 1.5 < low.ratio < 3.5
    assert abs(low.ratio / high.ratio - 1.0) < 0.15


def test_both_runs_cover_the_same_windows() -> None:
    low, high = _pair(_summary(), "2024-2026")
    assert low.windows == high.windows


def test_every_window_group_has_both_budgets() -> None:
    table = _summary()
    for group in table["group"].dropna().unique():
        targets = set(table[table["group"] == group]["target"])
        assert {"12%", "18%"} <= targets, f"{group} is missing a risk budget"
