"""Tests for the look-ahead audit harness (scripts/run_lookahead_audit.py).

The audit script itself runs the full pipeline twice on real data. These tests
pin down the machinery it relies on, so a broken comparison or a broken
scrambler cannot silently turn a leaking strategy into a green check.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "run_lookahead_audit", ROOT / "scripts" / "run_lookahead_audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_audit_module()
CUT = pd.Timestamp("2023-06-30")


def _panel(n_symbols: int = 3, n: int = 250) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2023-01-02", periods=n, name="date")
    panel = {}
    for i in range(n_symbols):
        returns = rng.normal(0.0004, 0.01, n)
        close = 10.0 * np.exp(np.cumsum(returns))
        panel[f"sh60000{i}"] = pd.DataFrame(
            {
                "open": close * 1.001,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.full(n, 1_000_000.0),
            },
            index=index,
        )
    return panel


def _equity(index: pd.DatetimeIndex, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = 1_000_000.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.008, len(index)))
    return pd.Series(values, index=index)


def test_compare_identical_prefix_passes():
    equity = _equity(pd.bdate_range("2023-01-02", periods=200))
    later = equity.copy()
    later.iloc[-10:] *= 1.5
    result = audit.compare(equity, later, CUT)
    assert result["ok"] is True
    assert result["max_abs_diff"] == 0.0
    assert result["max_rel_diff"] <= 1e-10


def test_compare_one_basis_point_difference_fails():
    equity = _equity(pd.bdate_range("2023-01-02", periods=200))
    shifted = equity.copy()
    shifted.iloc[5] *= 1.0001
    result = audit.compare(equity, shifted, CUT)
    assert result["ok"] is False
    assert result["max_rel_diff"] > 1e-6
    assert result["first_mismatch"] == str(equity.index[5].date())


def test_compare_missing_date_fails():
    equity = _equity(pd.bdate_range("2023-01-02", periods=200))
    gapped = equity.drop(equity.index[7])
    result = audit.compare(equity, gapped, CUT)
    assert result["ok"] is False
    assert result["reason"] == "missing overlapping dates"


def test_truncate_keeps_only_pre_cut_rows():
    panel = _panel()
    clipped = audit.truncate(panel, CUT)
    for symbol, frame in clipped.items():
        assert frame.index.max() <= CUT
        assert len(frame) == len(panel[symbol].loc[:CUT])
        pd.testing.assert_frame_equal(frame, panel[symbol].loc[:CUT])


def test_scramble_future_leaves_the_past_untouched():
    panel = _panel()
    scrambled = audit.scramble_future(panel, CUT, seed=5)
    for symbol, frame in scrambled.items():
        pd.testing.assert_frame_equal(frame.loc[:CUT], panel[symbol].loc[:CUT])
        assert len(frame) == len(panel[symbol])


def _path_returns(frame: pd.DataFrame, cut: pd.Timestamp) -> np.ndarray:
    """Returns of the post-cut path measured from the pre-cut closing price."""
    before = frame.loc[frame.index <= cut, "close"]
    after = frame.loc[frame.index > cut, "close"]
    joined = pd.concat([before.iloc[[-1]], after]).to_numpy(dtype=float)
    return joined[1:] / joined[:-1] - 1.0


def test_scramble_future_preserves_post_cut_return_multiset():
    panel = _panel()
    scrambled = audit.scramble_future(panel, CUT, seed=5)
    for symbol, frame in panel.items():
        original = _path_returns(frame, CUT)
        rebuilt = _path_returns(scrambled[symbol], CUT)
        assert original.size == rebuilt.size
        np.testing.assert_allclose(np.sort(original), np.sort(rebuilt), rtol=1e-9, atol=1e-12)
        # A permutation that leaves the path intact would be a no-op; make sure
        # the scrambled closes are actually different after the cut.
        assert not np.allclose(
            frame.loc[frame.index > CUT, "close"],
            scrambled[symbol].loc[scrambled[symbol].index > CUT, "close"],
        )


def test_scramble_future_is_deterministic_for_a_fixed_seed():
    panel = _panel()
    first = audit.scramble_future(panel, CUT, seed=99)
    second = audit.scramble_future(panel, CUT, seed=99)
    for symbol in panel:
        pd.testing.assert_frame_equal(first[symbol], second[symbol])
