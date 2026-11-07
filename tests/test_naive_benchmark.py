"""Tests for the head-to-head against the obvious alternative (23.109)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_naive_benchmark", ROOT / "scripts" / "run_naive_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass needs the module registered
    spec.loader.exec_module(module)
    return module


naive = _load_module()


def test_the_naive_variant_drops_the_trend_filter():
    """Naive means -vol and nothing else; any filter would defeat the comparison."""
    import inspect

    source = inspect.getsource(naive.NaiveLowVol)
    assert "realised_volatility" in source
    assert "sma(" not in source
    assert "trend_ok" not in source


def _panel():
    path = ROOT / "reports" / "naive_benchmark" / "panel.csv"
    if not path.exists():
        pytest.skip("run scripts/run_naive_benchmark.py first")
    return pd.read_csv(path)


def test_the_naive_book_is_far_cheaper_to_run():
    data = _panel()
    pivot = data.pivot_table(index="window", columns="variant", values="turnover")
    plain = next(c for c in pivot.columns if c.startswith("A"))
    published = next(c for c in pivot.columns if c.startswith("C"))
    assert (pivot[published] > pivot[plain] * 1.5).all()


def test_the_one_clear_gap_is_the_first_window():
    """The finding: the trend filter is what saves 2019-2021, not the ranking."""
    data = _panel()
    pivot = data.pivot_table(index="window", columns="variant", values="sharpe")
    plain = next(c for c in pivot.columns if c.startswith("A"))
    published = next(c for c in pivot.columns if c.startswith("C"))
    gaps = (pivot[published] - pivot[plain]).abs()
    assert gaps.idxmax() == "2019-2021"


def test_no_cell_is_statistically_significant():
    path = ROOT / "reports" / "naive_benchmark" / "paired.csv"
    if not path.exists():
        pytest.skip("run scripts/run_naive_benchmark.py first")
    pairs = pd.read_csv(path)
    assert int(((pairs["low"] > 0) | (pairs["high"] < 0)).sum()) == 0
