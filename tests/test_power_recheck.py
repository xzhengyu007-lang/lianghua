"""Tests for the detection-power re-check (23.89)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_power_recheck", ROOT / "scripts" / "run_power_recheck.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


power = _load_module()


def test_grid_covers_the_deployable_sharpe():
    assert 0.90 in power.SHARPES
    assert 1.16 in power.SHARPES
    assert 7.3 in power.YEARS


def test_analytic_years_scale_inversely_with_sharpe():
    from run_power_study import Z95, analytic_years

    assert analytic_years(0.90, power_z=Z95) > analytic_years(1.16, power_z=Z95)
    assert 6.5 < analytic_years(0.90, power_z=Z95) < 6.8


def _summary():
    path = ROOT / "reports" / "power_recheck" / "summary.json"
    if not path.exists():
        pytest.skip("run scripts/run_power_recheck.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def test_bootstrap_interval_brackets_the_point_estimate():
    recorded = _summary()["recorded"]
    assert recorded["sharpe_low"] < recorded["sharpe_point"] < recorded["sharpe_high"]


def test_power_is_monotone_in_sharpe_at_the_recorded_horizon():
    payload = _summary()
    by_key = {(row["sharpe"], row["years"]): row["power"] for row in payload["power"]}
    horizon = 7.3
    assert by_key[(0.90, horizon)] < by_key[(1.16, horizon)]
    assert by_key[(0.80, horizon)] < by_key[(0.90, horizon)]
