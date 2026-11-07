"""Tests for the IC horizon machinery (23.66).

``ic_series`` is the only non-trivial piece: it filters by eligibility,
drops thin cross-sections and returns a Spearman IC per date. A silent bug
there would invert the whole decay curve, so the guard is behavioural.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_signal_horizon", ROOT / "scripts" / "run_signal_horizon.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


horizon = _load_module()


def _frames(n_dates: int = 30, n_names: int = 60, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2024-01-02", periods=n_dates, name="date")
    columns = [f"s{i:03d}" for i in range(n_names)]
    return pd.DataFrame(rng.normal(0, 0.05, (n_dates, n_names)), index=index, columns=columns)


def _all_true(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(True, index=frame.index, columns=frame.columns)


def test_perfect_predictor_gets_ic_one():
    forward = _frames()
    ic = horizon.ic_series(forward.copy(), _all_true(forward), forward)
    assert len(ic) == len(forward)
    np.testing.assert_allclose(ic.to_numpy(), 1.0, atol=1e-9)


def test_noise_predictor_stays_near_zero():
    forward = _frames(seed=7)
    noise = pd.DataFrame(
        np.random.default_rng(99).normal(size=forward.shape),
        index=forward.index,
        columns=forward.columns,
    )
    ic = horizon.ic_series(noise, _all_true(forward), forward)
    assert abs(float(ic.mean())) < 0.1


def test_thin_cross_sections_are_dropped():
    forward = _frames(n_names=10)
    ic = horizon.ic_series(forward.copy(), _all_true(forward), forward)
    assert ic.empty


def test_eligibility_mask_is_applied():
    forward = _frames(n_names=40)
    mask = np.zeros(forward.shape, dtype=bool)
    mask[:, :30] = True
    eligible = pd.DataFrame(mask, index=forward.index, columns=forward.columns)

    ic = horizon.ic_series(forward.copy(), eligible, forward)
    assert len(ic) == len(forward)

    date = forward.index[0]
    expected, _ = spearmanr(forward.loc[date, forward.columns[:30]], forward.loc[date, forward.columns[:30]])
    assert ic.loc[date] == expected

    only_fifteen = eligible.copy()
    only_fifteen.iloc[:, 15:] = False
    assert horizon.ic_series(forward.copy(), only_fifteen, forward).empty
