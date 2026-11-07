"""Tests for the trend_window=250 re-release gate (23.117)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _summary():
    path = ROOT / "reports" / "window_gate" / "summary.json"
    if not path.exists():
        pytest.skip("run scripts/run_window_gate.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_candidate_beats_the_published_config_on_the_headline():
    payload = _summary()
    assert payload["SMA250"]["full_sharpe"] > payload["SMA120"]["full_sharpe"]


def test_the_whole_phase_distribution_moves_up_not_just_the_mean():
    """The one gate that matters: not phase luck, a shifted distribution."""
    payload = _summary()
    assert payload["SMA250"]["phase_span"]["min"] > payload["SMA120"]["phase_span"]["min"]
    assert payload["SMA250"]["phase_span"]["max"] > payload["SMA120"]["phase_span"]["max"]


def test_the_candidate_does_not_buy_return_with_drawdown():
    payload = _summary()
    assert payload["SMA250"]["full_max_drawdown"] > payload["SMA120"]["full_max_drawdown"]


def test_the_candidate_trades_less():
    payload = _summary()
    assert payload["SMA250"]["full_turnover"] < payload["SMA120"]["full_turnover"]


def test_the_2019_handicap_is_real_and_documented():
    """250-day warmup on a truncated panel means SMA250 sits out 2019 entirely."""
    payload = _summary()
    assert payload["SMA250"]["by_year"]["2019"]["return"] == 0.0
    assert payload["SMA120"]["by_year"]["2019"]["return"] > 0.0
