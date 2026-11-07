"""Tests for the microstructure cost floor (23.100)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_cost_floor", ROOT / "scripts" / "run_cost_floor.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


floor = _load_module()


def _positions(tmp_path: Path, prices: list[float]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    index = pd.bdate_range("2024-01-02", periods=2)
    rows = []
    for day in index:
        for i, price in enumerate(prices):
            rows.append(
                {
                    "date": day,
                    "symbol": "sh%06d" % (600000 + i),
                    "shares": 100,
                    "close": price,
                    "market_value": 100.0 * price,
                    "weight": 1.0 / len(prices),
                }
            )
    pd.DataFrame(rows).to_csv(tmp_path / "positions.csv", index=False)
    return tmp_path


def test_tick_cost_scales_with_one_over_price(tmp_path):
    cheap = floor.window_cost(_positions(tmp_path / "a", [5.0]))
    rich = floor.window_cost(_positions(tmp_path / "b", [50.0]))
    assert abs(cheap["tick_bp_weighted"] - 20.0) < 1e-6
    assert abs(rich["tick_bp_weighted"] - 2.0) < 1e-6


def test_half_spread_is_half_a_tick(tmp_path):
    info = floor.window_cost(_positions(tmp_path, [10.0]))
    assert abs(info["half_spread_bp"] - 0.5 * info["tick_bp_weighted"]) < 1e-9


def test_the_floor_is_never_below_the_published_assumption(tmp_path):
    """max(floor, published) means the excess can only be zero or positive."""
    for price in (1.0, 5.0, 20.0, 200.0):
        info = floor.window_cost(_positions(tmp_path / str(price), [price]))
        assert info["excess_per_side_bp"] >= -1e-9


def test_cheap_basket_costs_more_than_the_published_assumption(tmp_path):
    info = floor.window_cost(_positions(tmp_path, [5.0]))
    assert info["excess_per_side_bp"] > 0


def _summary():
    path = ROOT / "reports" / "cost_floor" / "summary.json"
    if not path.exists():
        pytest.skip("run scripts/run_cost_floor.py first")
    import json

    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def test_the_excess_shrinks_as_the_book_moves_up_market():
    """23.81 says the held price rose; the floor should therefore bind less."""
    rows = {r["window"]: r for r in _summary()}
    assert rows["2019-2021"]["excess_per_side_bp"] > rows["2024-2026"]["excess_per_side_bp"]
    assert rows["2024-2026"]["excess_per_side_bp"] == 0.0
