"""Tests for the rank-overlap study (23.111)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_rank_overlap", ROOT / "scripts" / "run_rank_overlap.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass needs the module registered
    spec.loader.exec_module(module)
    return module


overlap = _load_module()


def test_held_sets_record_only_dates_where_the_basket_changes():
    index = pd.bdate_range("2024-01-02", periods=3)
    weights = pd.DataFrame(
        [
            {"a": 0.5, "b": 0.5, "c": 0.0},
            {"a": 0.5, "b": 0.5, "c": 0.0},
            {"a": 0.0, "b": 0.5, "c": 0.5},
        ],
        index=index,
    )
    sets = overlap.held_sets(weights)
    assert len(sets) == 2
    assert sets[index[0]] == {"a", "b"}
    assert sets[index[2]] == {"b", "c"}


def _summary():
    path = ROOT / "reports" / "rank_overlap" / "summary.json"
    if not path.exists():
        pytest.skip("run scripts/run_rank_overlap.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_blend_is_not_decorative():
    """It must actually change baskets, or the whole ablation would be a null."""
    payload = _summary()
    assert payload["share_identical"] < 0.5
    assert payload["mean_overlap"] < payload["top_n"]


def test_but_the_change_carries_no_measurable_edge():
    """The finding: high activity, no information."""
    payload = _summary()
    assert abs(payload["edge_mean"]) < 0.10
    assert 0.3 < payload["edge_positive_share"] < 0.7
