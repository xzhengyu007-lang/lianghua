"""Tests for the style/size regime measurement (23.135)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


regime = _load("run_style_regime", "run_style_regime.py")


def _summary():
    path = ROOT / "reports" / "style_regime" / "summary.json"
    if not path.exists():
        pytest.skip("run scripts/run_style_regime.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_size_regression_uses_the_whole_sample():
    """Two legs with different coverage silently shorten the regression."""
    payload = _summary()
    size = payload["monthly"].get("size_vs_strategy")
    if not size:
        pytest.skip("size regression not computed")
    assert size["n"] > 180, "size regression looks truncated to the CSI300 span"
    assert payload["size_large_leg"] in ("sh000016", "sh000300")


def test_the_size_relation_is_negative_and_significant():
    """The finding: large-cap leadership helps, and it is not a rounding error."""
    size = _summary()["monthly"]["size_vs_strategy"]
    assert size["r"] < -0.1
    assert size["t"] < -2.0


def test_the_within_universe_vol_factor_is_not_the_explanation():
    style = _summary()["monthly"]["style_vs_strategy"]
    assert abs(style["r"]) < 0.10


def test_annual_buckets_agree_in_direction_with_the_monthly_sign():
    payload = _summary()
    assert payload["large_cap_leads"]["mean_strategy_return"] > \
        payload["small_cap_leads"]["mean_strategy_return"]
