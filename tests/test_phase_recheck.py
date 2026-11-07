"""Tests for the calendar-phase re-check (23.82)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_phase_recheck", ROOT / "scripts" / "run_phase_recheck.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase = _load_module()


def test_phases_cover_one_full_rebalance_cycle():
    assert phase.PHASES == tuple(range(20))


def test_both_anchor_modes_are_measured():
    assert set(phase.MODES) == {"start", "year"}


def test_windows_are_ordered_and_disjoint():
    spans = [phase.WINDOWS[k] for k in ("2019-2021", "2022-2023", "2024-2026")]
    for start, end in spans:
        assert start < end
    assert spans[0][1] < spans[1][0]
    assert spans[1][1] < spans[2][0]


def test_min_median_max_are_ordered_in_every_cell():
    summary = ROOT / "reports" / "phase_recheck" / "summary.json"
    if not summary.exists():
        import pytest

        pytest.skip("run scripts/run_phase_recheck.py first")
    payload = json.loads(summary.read_text(encoding="utf-8"))
    for row in payload["spread"]:
        assert row["ret_min"] <= row["ret_median"] <= row["ret_max"]
        assert row["sharpe_min"] <= row["sharpe_median"] <= row["sharpe_max"]
        assert row["ret_spread_pp"] >= 0
