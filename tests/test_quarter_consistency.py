"""Tests for the quarterly sign test (23.104)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_quarter_consistency", ROOT / "scripts" / "run_quarter_consistency.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


quarters = _load_module()


def test_quarterly_compounds_within_each_quarter():
    index = pd.bdate_range("2024-01-02", periods=2)
    returns = pd.Series([0.10, -0.05], index=index)
    result = quarters.quarterly(returns)
    expected = (1.10 * 0.95) - 1.0
    assert abs(result.iloc[0] - expected) < 1e-12


def test_sign_test_counts_wins_and_reports_a_p_value():
    series = pd.Series([0.01] * 8 + [-0.01] * 2)
    row = quarters.sign_test(series, "toy")
    assert row["n"] == 10 and row["wins"] == 8
    assert abs(row["win_rate"] - 0.8) < 1e-12
    assert 0 < row["p_value"] < 0.10


def _summary():
    path = ROOT / "reports" / "quarter_consistency" / "summary.json"
    if not path.exists():
        pytest.skip("run scripts/run_quarter_consistency.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_pool_itself_is_a_coin_flip():
    """The reference line must not look like a winner, or the comparison is broken."""
    rows = {r["series"]: r for r in _summary()["tests"]}
    pool = next(v for k, v in rows.items() if "参照" in k)
    assert 0.3 < pool["win_rate"] < 0.7


def test_the_sign_test_is_not_claimed_to_be_powerful():
    """The finding is that it is not a shortcut; the summary must say so."""
    payload = _summary()
    assert payload["sign_test_power"] < 0.8
    assert payload["quarters_for_80pct_power"] > 40
