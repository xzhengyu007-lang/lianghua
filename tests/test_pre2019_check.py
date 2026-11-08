"""Tests for the pre-2019 out-of-sample check (23.131)."""

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


check = _load("run_pre2019_check", "run_pre2019_check.py")


def _frames(tmp: Path, code: str, hfq_scale: float, raw_last: float):
    index = pd.bdate_range("2015-01-05", periods=400)
    base = np.linspace(10, 12, 400)
    hfq = pd.DataFrame({
        "open": base * hfq_scale, "high": base * hfq_scale * 1.01,
        "low": base * hfq_scale * 0.99, "close": base * hfq_scale,
        "volume": np.full(400, 1000.0),
    }, index=index)
    hfq.to_parquet(tmp / (code + "_hfq.parquet"))
    raw = pd.DataFrame({"close": (base * raw_last / 12.0)}, index=index)
    raw.to_parquet(tmp / (code + "_raw.parquet"))


def test_hfq_is_rescaled_to_raw_price_levels(tmp_path):
    """Unscaled hfq would make 100-share lots unaffordable and fake the fills."""
    _frames(tmp_path, "sh600000", hfq_scale=1000.0, raw_last=12.0)
    panel = check.load_scaled(tmp_path)
    close = panel["sh600000"]["close"]
    assert close.iloc[-1] == pytest.approx(12.0, rel=1e-6)
    assert close.max() < 100


def test_rescaling_preserves_returns(tmp_path):
    _frames(tmp_path, "sh600000", hfq_scale=1000.0, raw_last=12.0)
    panel = check.load_scaled(tmp_path)
    close = panel["sh600000"]["close"]
    expected = 12.0 / 10.0
    assert float(close.iloc[-1] / close.iloc[0]) == pytest.approx(expected, rel=1e-6)


def _summary():
    path = ROOT / "reports" / "pre2019_check" / "summary.json"
    if not path.exists():
        pytest.skip("run scripts/run_pre2019_check.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_earlier_window_is_not_worse():
    """The finding: the edge existed before 2019, so this must not invert."""
    rows = {row["half"]: row for row in _summary()["rows"]}
    assert rows["pre-2019"]["sharpe"] > 0.5
    assert rows["pre-2019"]["sharpe"] > rows["post-2019"]["sharpe"]


def test_the_analysis_records_the_survivorship_caveat():
    payload = _summary()
    assert "幸存者" in payload["note"] or payload["symbols"] < 128
    assert payload["phases"] >= 2
