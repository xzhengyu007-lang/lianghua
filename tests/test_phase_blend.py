"""Tests for the phase-blend re-check (23.83)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_phase_blend", ROOT / "scripts" / "run_phase_blend.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


blend = _load_module()


def test_blend_uses_all_twenty_phases():
    assert blend.PHASES == tuple(range(20))


def test_windows_are_ordered_and_disjoint():
    spans = [blend.WINDOWS[k] for k in ("2019-2021", "2022-2023", "2024-2026")]
    for start, end in spans:
        assert start < end
    assert spans[0][1] < spans[1][0]
    assert spans[1][1] < spans[2][0]


def _comparison():
    path = ROOT / "reports" / "phase_blend" / "summary.json"
    if not path.exists():
        pytest.skip("run scripts/run_phase_blend.py first")
    return json.loads(path.read_text(encoding="utf-8"))["comparison"]


def test_blend_is_less_volatile_than_every_single_phase():
    """The one claim from 23.22 that has to survive the rebuild."""
    for row in _comparison():
        assert row["blend_vol"] <= row["phase_vol_min"] + 1e-12, row["window"]


def test_blend_lands_within_five_points_of_the_phase_median():
    for row in _comparison():
        gap = abs(row["blend_return"] - row["phase_median_return"]) * 100
        assert gap <= 5.0, (row["window"], gap)
