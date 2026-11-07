"""Tests for the selection-slack measurement (23.80)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_selection_slack", ROOT / "scripts" / "run_selection_slack.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


slack = _load_module()


def _weights(rows) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=len(rows))
    return pd.DataFrame(rows, index=index).fillna(0.0)


def test_rebalance_dates_track_changes_in_the_held_set():
    weights = _weights([
        {"a": 0.5, "b": 0.0, "c": 0.0},
        {"a": 0.5, "b": 0.0, "c": 0.0},
        {"a": 0.0, "b": 0.5, "c": 0.0},
        {"a": 0.0, "b": 0.5, "c": 0.0},
        {"a": 0.0, "b": 0.0, "c": 0.5},
    ])
    dates = slack.rebalance_dates(weights)
    assert len(dates) == 3
    assert dates[0] == weights.index[0]
    assert dates[1] == weights.index[2]
    assert dates[2] == weights.index[4]


def test_rebalance_dates_ignore_the_warmup_prefix():
    weights = _weights([
        {"a": 0.0, "b": 0.0},
        {"a": 0.0, "b": 0.0},
        {"a": 0.5, "b": 0.0},
        {"a": 0.5, "b": 0.0},
    ])
    dates = slack.rebalance_dates(weights)
    assert dates == [weights.index[2]]


def test_unchanged_weights_produce_a_single_rebalance():
    weights = _weights([{"a": 0.0}, {"a": 0.5}, {"a": 0.5}, {"a": 0.5}])
    assert slack.rebalance_dates(weights) == [weights.index[1]]
