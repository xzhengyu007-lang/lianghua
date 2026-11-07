"""Tests for the holding-persistence helper (23.74)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_holding_persistence", ROOT / "scripts" / "run_holding_persistence.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


persistence = _load_module()


def _flags(rows):
    index = pd.bdate_range("2024-01-02", periods=len(rows))
    return pd.DataFrame(rows, index=index).fillna(0.0).astype(float) > 0


def test_retention_counts_names_kept_from_the_previous_basket():
    flags = _flags([
        {"AAA": 1, "BBB": 1, "CCC": 0},
        {"AAA": 1, "BBB": 1, "CCC": 0},
        {"AAA": 1, "BBB": 0, "CCC": 1},
    ])
    sets = persistence.holding_sets(flags)
    table = persistence.rebalance_table(sets)
    assert table["initial"].sum() == 1
    changes = table[~table["initial"]].reset_index(drop=True)
    assert len(changes) == 1
    assert abs(changes["retention"].iloc[0] - 0.5) < 1e-12
    assert changes["removed"].iloc[0] == 1
    assert changes["added"].iloc[0] == 1


def test_spells_measure_consecutive_trading_days():
    flags = _flags([
        {"AAA": 1},
        {"AAA": 1},
        {"AAA": 0},
        {"AAA": 1},
    ])
    spells = persistence.holding_spells(flags)
    assert spells["days"].tolist() == [2, 1]


def test_unchanged_basket_is_not_counted_as_a_change():
    flags = _flags([{"AAA": 1, "BBB": 1}, {"AAA": 1, "BBB": 1}])
    summary = persistence.summarise("toy", flags)
    assert summary["changes"] == 0


def test_summary_averages_names_held_per_day():
    flags = _flags([
        {"AAA": 1, "BBB": 1},
        {"AAA": 1, "BBB": 0},
    ])
    summary = persistence.summarise("toy", flags)
    assert summary["avg_names"] == 1.5
    assert summary["min_names"] == 1
    assert summary["max_names"] == 2
    assert summary["distinct_names"] == 2
