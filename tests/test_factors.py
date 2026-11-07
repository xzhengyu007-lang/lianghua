from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_system.analytics import forward_returns
from quant_system.data import price_frame
from quant_system.factors import factor_library, partial_ic, rank_correlation
from quant_system.strategies import LowVolCoreStrategy, LowVolPlusStrategy, STRATEGY_REGISTRY


@pytest.fixture
def panel() -> dict[str, pd.DataFrame]:
    from tests.conftest import make_frame

    return {
        f"sh60000{i}": make_frame(seed=i + 3, drift=0.0002 + i * 0.00004,
                                  volatility=0.008 + i * 0.004)
        for i in range(12)
    }


def test_factor_library_returns_finite_causal_frames(panel) -> None:
    factors = factor_library(panel)
    expected = {
        "low_vol", "reversal_5", "max_effect", "abnormal_volume",
        "amihud", "momentum_12_1", "high_52w",
    }
    assert set(factors) == expected

    cut = list(panel)[0]
    cutoff = panel[cut].index[-40]
    truncated = {k: v.loc[:cutoff] for k, v in panel.items()}
    short = factor_library(truncated)

    for name, frame in factors.items():
        finite = frame.replace([np.inf, -np.inf], np.nan)
        assert np.isfinite(finite.dropna(how="all").to_numpy(dtype=float)).all(), name
        common = short[name].index.intersection(frame.index)
        pd.testing.assert_frame_equal(
            short[name].loc[common].dropna(how="all"),
            frame.loc[common].dropna(how="all"),
            check_exact=False,
            rtol=1e-9,
        )


def test_partial_ic_of_the_control_against_itself_is_zero(panel) -> None:
    factors = factor_library(panel)
    future = forward_returns(price_frame(panel, "close"), 5)
    control = factors["low_vol"]
    partial = partial_ic(control, control, future).dropna()
    assert len(partial) > 50
    assert abs(float(partial.mean())) < 0.02


def test_rank_correlation_is_one_for_a_signal_with_itself(panel) -> None:
    factors = factor_library(panel)
    assert rank_correlation(factors["amihud"], factors["amihud"]) == pytest.approx(1.0)
    assert abs(rank_correlation(factors["amihud"], -factors["amihud"]) + 1.0) < 1e-9


def test_low_vol_plus_with_no_extra_signal_equals_the_core(panel) -> None:
    """The testbed must not silently change the baseline it is compared against."""
    core_scores, core_eligible = LowVolCoreStrategy().score(panel)
    plus = LowVolPlusStrategy(vol_weight=0.8, trend_weight=0.2, illiquidity_weight=0.0,
                              momentum_weight=0.0)
    plus_scores, plus_eligible = plus.score(panel)
    pd.testing.assert_frame_equal(core_scores, plus_scores)
    pd.testing.assert_frame_equal(core_eligible, plus_eligible)


def test_both_registry_names_resolve_to_the_testbed() -> None:
    assert STRATEGY_REGISTRY["low_vol_plus"] is STRATEGY_REGISTRY["low_vol_illiquid"]


def test_low_vol_trend_switches_default_to_the_published_baseline(panel) -> None:
    published = LowVolCoreStrategy().score(panel)
    explicit = LowVolCoreStrategy(trend_filter=True, trend_fallback=False).score(panel)
    pd.testing.assert_frame_equal(published[0], explicit[0])
    pd.testing.assert_frame_equal(published[1], explicit[1])


def test_low_vol_trend_fallback_fills_a_fully_below_trend_book() -> None:
    from tests.conftest import make_frame

    falling = {
        f"sh60000{i}": make_frame(seed=20 + i, drift=-0.004, volatility=0.012)
        for i in range(12)
    }
    strict = LowVolCoreStrategy(top_n=8, max_per_sector=None)
    fallback = LowVolCoreStrategy(top_n=8, max_per_sector=None, trend_fallback=True, fallback_penalty=5.0)
    strict_weights = strict.generate(falling)
    fallback_weights = fallback.generate(falling)
    assert int(strict_weights.iloc[-1].gt(0).sum()) == 0
    assert int(fallback_weights.iloc[-1].gt(0).sum()) == 8
    assert fallback_weights.iloc[-1].sum() == pytest.approx(1.0)
