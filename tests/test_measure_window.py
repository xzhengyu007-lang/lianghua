"""Tests for the window measurement on extended history (23.129)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


measure = _load("measure_window", "measure_window.py")


def _write(cache: Path, symbol: str, suffix: str, start: str, rows: int) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    index = pd.bdate_range(start, periods=rows)
    pd.DataFrame({"close": np.linspace(10, 12, rows)}, index=index).to_parquet(
        cache / (symbol + suffix))


def test_load_panel_prefers_the_longest_series(tmp_path):
    _write(tmp_path, "sh600000", "_qfq.parquet", "2020-01-01", 200)
    _write(tmp_path, "sh600000", "_raw.parquet", "2010-01-04", 3000)
    panel = measure.load_panel(tmp_path, ["sh600000"])
    assert len(panel["sh600000"]) == 3000


def test_load_panel_skips_symbols_without_data(tmp_path):
    _write(tmp_path, "sh600000", "_raw.parquet", "2010-01-04", 100)
    panel = measure.load_panel(tmp_path, ["sh600000", "sh600001"])
    assert list(panel) == ["sh600000"]


def _summary():
    path = ROOT / "reports" / "window_measure" / "summary.json"
    if not path.exists():
        pytest.skip("run scripts/measure_window.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_current_rule_barely_moves_the_window():
    """The finding: extending history alone buys ~1.8 years, not 8."""
    payload = _summary()
    assert payload["years"] < 10
    assert payload["years"] > 7.14


def test_relaxing_coverage_buys_history_monotonically():
    sweep = _summary()["sweep"]
    assert len(sweep) >= 6
    years = [row["years"] for row in sweep]
    assert years == sorted(years), "looser coverage must not shorten the window"
    assert years[-1] > 15.0


def test_the_cheap_ratio_still_keeps_a_wide_cross_section():
    """0.75 buys 16 years while keeping ~98 names -- above 23.93's knee."""
    sweep = {row["ratio"]: row for row in _summary()["sweep"]}
    row = sweep.get(0.75)
    assert row is not None
    assert row["names_at_start"] >= 90
    assert row["years"] > 15.0
