from __future__ import annotations

import numpy as np
import pytest

from quant_system.bootstrap import (
    bootstrap_metrics,
    bootstrap_summary,
    circular_block_indices,
    default_block_length,
    paired_bootstrap_diff,
    stationary_bootstrap_indices,
)


def test_default_block_length_is_reasonable() -> None:
    assert default_block_length(700) == 9
    assert default_block_length(10) >= 5
    with pytest.raises(ValueError):
        default_block_length(0)


def test_stationary_indices_have_right_length_and_range() -> None:
    rng = np.random.default_rng(3)
    idx = stationary_bootstrap_indices(500, 10, rng)
    assert idx.shape == (500,)
    assert idx.min() >= 0 and idx.max() < 500
    # With geometric blocks of mean 10 the draw should not be a pure permutation
    assert len(np.unique(idx)) < 500


def test_circular_indices_wrap_within_block() -> None:
    rng = np.random.default_rng(5)
    idx = circular_block_indices(7, 3, rng)
    assert idx.shape == (7,)
    assert idx.min() >= 0 and idx.max() < 7


def test_bootstrap_is_seed_reproducible() -> None:
    rng = np.random.default_rng(11)
    ret = rng.normal(0.0005, 0.01, 300)
    first = bootstrap_metrics(ret, n_boot=50, seed=7)
    second = bootstrap_metrics(ret, n_boot=50, seed=7)
    assert np.allclose(first["sharpe"].to_numpy(), second["sharpe"].to_numpy())


def test_bootstrap_summary_brackets_point_estimate() -> None:
    rng = np.random.default_rng(13)
    ret = rng.normal(0.0006, 0.009, 600)
    summary = bootstrap_summary(ret, n_boot=400, seed=1)
    sharpe = summary["sharpe"]
    assert sharpe["p2"] < sharpe["point"] < sharpe["p97"]
    assert summary["_meta"]["n_obs"] == 600


def test_paired_diff_is_zero_for_identical_series() -> None:
    rng = np.random.default_rng(17)
    ret = rng.normal(0.0004, 0.01, 250)
    result = paired_bootstrap_diff(ret, ret, n_boot=100, seed=2)
    assert result.observed == pytest.approx(0.0)
    assert result.low == pytest.approx(0.0)
    assert result.high == pytest.approx(0.0)


def test_paired_diff_centres_on_noise() -> None:
    rng = np.random.default_rng(19)
    a = rng.normal(0.0, 0.01, 400)
    b = rng.normal(0.0, 0.01, 400)
    a -= a.mean()
    b -= b.mean()
    result = paired_bootstrap_diff(a, b, n_boot=400, seed=3)
    assert result.low < 0 < result.high
    assert 0.2 < result.prob_positive < 0.8


def test_paired_diff_rejects_unequal_lengths() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_diff(np.zeros(10), np.zeros(11), n_boot=10)
