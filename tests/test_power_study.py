"""Tests for the Sharpe power curve (23.64).

The claim is a structural one - short samples cannot resolve a Sharpe of 1.0 -
so the guard is that the simulator stays honest: a better true Sharpe must not
need more data, three years must stay in the noise, and the analytic formula
must agree with the textbook value.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_power_study", ROOT / "scripts" / "run_power_study.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


power = _load_module()


def test_analytic_years_match_the_textbook_value():
    years = power.analytic_years(1.0, power_z=power.Z95)
    assert 5.6 < years < 5.9


def test_three_years_is_still_noise():
    result = power.simulate(1.0, 3, draws=2000, seed=11)
    assert result["power"] < 0.15


def test_long_sample_resolves_a_strong_sharpe():
    result = power.simulate(1.5, 12, draws=2000, seed=12)
    assert result["power"] > 0.9


def test_power_increases_with_sample_length():
    short = power.simulate(1.0, 4, draws=2000, seed=13)["power"]
    long = power.simulate(1.0, 10, draws=2000, seed=14)["power"]
    assert short < long


def test_simulation_is_reproducible():
    first = power.simulate(1.0, 5, draws=500, seed=99)
    second = power.simulate(1.0, 5, draws=500, seed=99)
    assert first == second
