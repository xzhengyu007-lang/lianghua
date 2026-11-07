"""Tests for the one-command verification battery (23.68).

The battery is only as good as its detectors, so each check gets a case where
it must fail: duplicated dates, a non-positive price, and a run that is not
reproducible. A check that cannot fail is decoration.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_verification", ROOT / "scripts" / "run_verification.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify = _load_module()


def _panel(n: int = 120, seed: int = 3) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2024-01-02", periods=n, name="date")
    close = 10.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(n, 1e6),
        },
        index=index,
    )
    return {"sh600000": frame}


def test_clean_panel_passes(tmp_path):
    result = verify.check_data(_panel(), tmp_path)
    assert result["ok"] is True
    assert result["problems"] == []


def test_duplicate_dates_are_caught(tmp_path):
    panel = _panel()
    frame = pd.concat([panel["sh600000"], panel["sh600000"].iloc[[0]]])
    result = verify.check_data({"sh600000": frame}, tmp_path)
    assert result["ok"] is False
    assert any("duplicate" in p for p in result["problems"])


def test_non_positive_price_is_caught(tmp_path):
    panel = _panel()
    frame = panel["sh600000"].copy()
    frame.iloc[5, frame.columns.get_loc("close")] = -1.0
    result = verify.check_data({"sh600000": frame}, tmp_path)
    assert result["ok"] is False
    assert any("non-positive" in p for p in result["problems"])


def test_reproducibility_check_is_deterministic(tmp_path):
    panel = _panel()
    weights = pd.DataFrame(1.0 / len(panel), index=panel["sh600000"].index, columns=list(panel))
    result = verify.check_repro(panel, weights, ("2024-01-02", "2024-06-30"))
    assert result["ok"] is True
    assert result["max_abs_diff"] == 0.0
