"""Tests for the cross-pool independence study (23.92)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_cross_pool", ROOT / "scripts" / "run_cross_pool.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cross = _load_module()


def test_pool_settings_are_the_ones_the_doc_claims():
    assert cross.TARGET_VOL == 0.12
    assert cross.MIN_NAMES >= 20


def _summary():
    path = ROOT / "reports" / "cross_pool" / "summary.json"
    if not path.exists():
        pytest.skip("run scripts/run_cross_pool.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def test_pools_are_positively_correlated_so_independence_is_partial():
    payload = _summary()
    assert 0.0 < payload["mean_pairwise_correlation"] < 1.0
    assert payload["effective_independent_pools"] < payload["n_pools"]


def test_splitting_destroys_the_edge_rather_than_diversifying_it():
    """The headline: the combination is far worse than one broad pool."""
    payload = _summary()
    assert payload["combined_sharpe"] < payload["whole_pool_sharpe"]
    assert payload["combined_sharpe"] < payload["best_single_pool_sharpe"]


def test_every_pool_has_enough_names_for_the_slot_count():
    payload = _summary()
    for row in payload["pools"]:
        if row["pool"] == "ALL":
            continue
        assert row["names"] >= cross.MIN_NAMES, row["pool"]
