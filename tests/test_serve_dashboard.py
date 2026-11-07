"""Tests for the live dashboard server (scripts/serve_dashboard.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "serve_dashboard", ROOT / "scripts" / "serve_dashboard.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


server = _load_module()


def test_defaults_match_the_published_cli_defaults():
    """Regression: feeding slippage 0 silently beat the archived baseline."""
    assert server.DEFAULT_RUN["slippage_bps"] == 5.0
    assert server.DEFAULT_RUN["target_vol"] == 0.18
    assert server.DEFAULT_RUN["rebalance_reset"] == "year"
    assert server.DEFAULT_RUN["top_n"] == 8


def test_clean_clamps_out_of_range_knobs():
    _, applied = server._clean({"top_n": 999, "vol_window": 1, "target_vol": 9.0})
    assert applied["top_n"] == 40
    assert applied["vol_window"] == 5
    assert applied["target_vol"] == 0.60


def test_clean_reads_numbers_from_strings():
    _, applied = server._clean({"top_n": "6", "vol_weight": "0.5"})
    assert applied["top_n"] == 6
    assert applied["vol_weight"] == 0.5


def test_clean_falls_back_on_garbage_values():
    _, applied = server._clean({"top_n": "abc", "rebalance_reset": "monthly"})
    assert applied["top_n"] == 8
    assert applied["rebalance_reset"] == "year"


def test_zero_sector_cap_means_unlimited_not_zero():
    _, applied = server._clean({"max_per_sector": 0})
    assert applied["max_per_sector"] == 0
