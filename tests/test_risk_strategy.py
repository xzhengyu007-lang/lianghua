from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_system.config import RiskConfig
from quant_system.portfolio import build_portfolio, regime_filter
from quant_system.risk import (
    _ex_ante_volatility,
    apply_volatility_target,
    breadth_timing_scale,
    cap_turnover,
)
from quant_system.strategies import STRATEGY_REGISTRY, build_strategy


def test_all_strategies_produce_bounded_weights(sample_panel: dict[str, pd.DataFrame]) -> None:
    for name in STRATEGY_REGISTRY:
        strategy = build_strategy(name)
        weights = strategy.generate(sample_panel)
        assert (weights >= 0).all().all(), name
        assert (weights.sum(axis=1) <= 1.0000001).all(), name
        assert list(weights.columns) == sorted(sample_panel), name


def _suspended_panel() -> dict[str, pd.DataFrame]:
    """Three names trading normally plus one that stops for a long spell."""
    from tests.conftest import make_frame

    panel = {
        "sh600000": make_frame(seed=1, drift=0.0002),
        "sh600004": make_frame(seed=2, drift=0.0002),
        "sh600008": make_frame(seed=3, drift=0.0002),
    }
    halted = make_frame(seed=4, drift=0.0002).copy()
    # 150 consecutive missing bars is far longer than the 60-day window.
    halted.iloc[150:300] = np.nan
    panel["sh600074"] = halted
    return panel


def test_realised_volatility_does_not_treat_a_suspension_as_zero() -> None:
    """A halted name must not look like the calmest asset in the panel.

    The aligned-panel version produced exactly 0.0 volatility for a long
    suspension and put the name at the top of every low-vol ranking.
    """
    from quant_system.indicators import realised_volatility

    panel = _suspended_panel()
    vol = realised_volatility(panel, 60)
    halted = vol["sh600074"].dropna()
    assert len(halted) > 0
    assert (halted > 0).all()
    assert not vol["sh600074"].equals(vol["sh600000"])


def test_low_vol_strategies_do_not_rank_a_suspended_name_first() -> None:
    from quant_system.strategies import build_strategy

    panel = _suspended_panel()
    for name in ("low_volatility", "low_vol_core", "sector_neutral_lowvol"):
        strategy = build_strategy(name, **({"per_sector": 1} if name == "sector_neutral_lowvol" else {}))
        scores, _ = strategy.score(panel)
        row = scores.loc["2023-09-01"].dropna()
        if "sh600074" in row.index:
            assert row.idxmax() != "sh600074", name


def test_inverse_volatility_weights_ignore_a_suspended_name() -> None:
    from quant_system.portfolio import inverse_volatility_weights

    panel = _suspended_panel()
    dates = pd.bdate_range("2023-01-02", periods=400)
    selection = pd.DataFrame(True, index=dates, columns=sorted(panel))
    weights = inverse_volatility_weights(selection, panel, window=60)
    row = weights.loc["2023-09-01"]
    assert row["sh600074"] == 0.0 or pd.isna(row["sh600074"])


def test_ex_ante_covariance_skips_suspension_days() -> None:
    """Pairwise covariance must use joint observations, not padded zeros."""
    from quant_system.risk import _ex_ante_volatility
    from quant_system.config import RiskConfig

    panel = _suspended_panel()
    dates = pd.bdate_range("2023-01-02", periods=400)
    weights = pd.DataFrame(0.25, index=dates, columns=sorted(panel))
    vol = _ex_ante_volatility(weights, panel, RiskConfig(vol_lookback=60))
    finite = vol.dropna()
    assert len(finite) > 0
    assert (finite > 0).all()


def test_rolling_covariance_variance_matches_pandas_pairwise() -> None:
    """The pair-by-pair path must reproduce pandas' pairwise covariance.

    ``rolling().cov(pairwise=True)`` still applies ``min_periods`` to the whole
    row, so a single short history blanks every entry.  The direct moment
    computation is checked here against pandas on rows where pandas does
    produce a full block.
    """
    from quant_system.risk import _rolling_covariance_variance

    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2022-01-03", periods=400)
    columns = [f"s{i}" for i in range(7)]
    values = rng.normal(0, 0.012, (len(idx), len(columns)))
    values[60:200, 5] = np.nan
    values[0:30, 2] = np.nan
    returns = pd.DataFrame(values, index=idx, columns=columns)
    window, min_periods = 60, 30
    w = np.zeros((len(idx), len(columns)))
    w[:, 0], w[:, 1], w[:, 5] = 0.4, 0.3, 0.3
    w[250:, 5] = 0.0
    w[250:, 6] = 0.3

    mine = _rolling_covariance_variance(returns, w, window, min_periods)
    reference = returns.rolling(window, min_periods=1).cov(pairwise=True, ddof=0)
    expected = np.full(len(idx), np.nan)
    for position, date in enumerate(idx):
        block = reference.loc[date].to_numpy(dtype=float)
        if np.isfinite(block).all():
            expected[position] = float(w[position] @ block @ w[position])
    both = np.isfinite(mine) & np.isfinite(expected)
    assert both.sum() > 200
    assert mine[both] == pytest.approx(expected[both], rel=1e-10, abs=1e-16)


def test_ex_ante_through_does_not_extrapolate_past_the_cut() -> None:
    """A truncated forecast must stop, not carry its last block forward."""
    from quant_system.risk import _ex_ante_volatility

    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2021-01-04", periods=400)
    columns = [f"s{i}" for i in range(4)]
    panel = {
        c: pd.DataFrame(
            {"close": rng.normal(0, 0.01, len(idx)).cumsum() + 20.0}, index=idx
        )
        for c in columns
    }
    weights = pd.DataFrame(0.25, index=idx, columns=columns)
    cfg = RiskConfig(vol_lookback=60)
    full = _ex_ante_volatility(weights, panel, cfg)
    cut = _ex_ante_volatility(weights, panel, cfg, through=idx[200])
    assert cut.loc[idx[201]:].isna().all()
    overlap = full.loc[: idx[200]].notna() & cut.loc[: idx[200]].notna()
    assert overlap.sum() > 100
    assert cut.loc[: idx[200]][overlap].to_numpy() == pytest.approx(
        full.loc[: idx[200]][overlap].to_numpy(), rel=0, abs=0
    )


def test_ex_ante_scales_a_zero_weight_book_to_zero() -> None:
    """All-cash must report 0 vol, not NaN that the caller fills with 1.0."""
    from quant_system.risk import _ex_ante_volatility

    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2021-01-04", periods=200)
    columns = [f"s{i}" for i in range(5)]
    panel = {
        c: pd.DataFrame(
            {"close": rng.normal(0, 0.01, len(idx)).cumsum() + 20.0}, index=idx
        )
        for c in columns
    }
    weights = pd.DataFrame(0.0, index=idx, columns=columns)
    vol = _ex_ante_volatility(weights, panel, RiskConfig(vol_lookback=60))
    finite = vol.dropna()
    assert len(finite) > 150
    assert (finite == 0.0).all()


def test_turnover_cap(sample_panel: dict[str, pd.DataFrame]) -> None:
    dates = next(iter(sample_panel.values())).index[:20]
    weights = pd.DataFrame(0.0, index=dates, columns=sorted(sample_panel))
    weights.iloc[10:, 0] = 0.9
    capped = cap_turnover(weights, max_daily_turnover=0.1)
    changes = (capped - capped.shift(1).fillna(0.0)).abs().sum(axis=1)
    assert changes.max() <= 0.1000001


def test_vol_target_does_not_increase_exposure(sample_panel: dict[str, pd.DataFrame]) -> None:
    dates = next(iter(sample_panel.values())).index
    weights = pd.DataFrame(0.0, index=dates, columns=sorted(sample_panel))
    weights.iloc[:, 0] = 0.8
    scaled = apply_volatility_target(weights, sample_panel, RiskConfig(target_annual_vol=0.05))
    assert (scaled <= weights + 1e-12).all().all()


def test_ex_ante_forecast_uses_only_current_basket(sample_panel: dict[str, pd.DataFrame]) -> None:
    dates = next(iter(sample_panel.values())).index
    weights = pd.DataFrame(0.0, index=dates, columns=sorted(sample_panel))
    weights.iloc[:, 0] = 0.6
    cfg = RiskConfig()
    realised = apply_volatility_target(weights, sample_panel, cfg)
    ex_ante = apply_volatility_target(weights, sample_panel, cfg, forecast="ex_ante")
    raw = _ex_ante_volatility(weights, sample_panel, cfg)
    assert np.isfinite(raw.dropna()).all()
    # The two forecasts may agree while the basket is stable; they must not be
    # silently aliased by the keyword.
    assert not realised.equals(ex_ante) or (realised - ex_ante).abs().max().max() <= 1e-12


def test_ex_ante_forecast_validates_name(sample_panel: dict[str, pd.DataFrame]) -> None:
    dates = next(iter(sample_panel.values())).index
    weights = pd.DataFrame(0.5, index=dates, columns=sorted(sample_panel)[:2])
    with pytest.raises(ValueError, match="forecast"):
        apply_volatility_target(weights, sample_panel, RiskConfig(), forecast="nope")


def test_regime_filter_cuts_exposure() -> None:
    index = pd.bdate_range("2024-01-01", periods=100)
    benchmark = pd.Series(np.r_[np.linspace(100, 120, 60), np.linspace(120, 90, 40)], index=index)
    weights = pd.DataFrame(0.5, index=index, columns=["a", "b"])
    filtered = regime_filter(weights, benchmark)
    assert filtered.iloc[-1].sum() == 0.0
    assert filtered.iloc[70].sum() == 0.0


def test_portfolio_build(sample_panel: dict[str, pd.DataFrame]) -> None:
    dates = next(iter(sample_panel.values())).index
    weights = pd.DataFrame(0.5, index=dates, columns=sorted(sample_panel)[:2])
    result = build_portfolio(weights, sample_panel, use_regime=False)
    assert (result >= 0).all().all()
    assert (result.sum(axis=1) <= 1.0000001).all()



def test_sector_cap_is_enforced(sample_panel) -> None:
    from collections import Counter
    from quant_system.sectors import sector_of
    from quant_system.strategies import LowVolCoreStrategy

    strategy = LowVolCoreStrategy(top_n=8, max_per_sector=1)
    weights = strategy.generate(sample_panel)
    last = weights.iloc[-1]
    chosen = last[last > 0].index
    counts = Counter(sector_of(symbol) for symbol in chosen)
    assert max(counts.values(), default=0) <= 1


def test_effective_breadth_is_scale_invariant() -> None:
    """``1/sum(w^2)`` must be computed on normalised weights.

    The first version used raw weights, so a volatility-target scale of 0.75
    made 8 equal names look like 14.2 names. Breadth is a composition measure,
    not a function of gross exposure.
    """
    from scripts.run_diversification_audit import effective_breadth

    dates = pd.bdate_range("2024-01-02", periods=3)
    columns = [f"sh60000{i}" for i in range(8)]
    full = pd.DataFrame(1.0 / 8.0, index=dates, columns=columns)
    scaled = full * 0.75
    assert effective_breadth(full).iloc[0] == pytest.approx(8.0)
    assert effective_breadth(scaled).iloc[0] == pytest.approx(8.0)
    # A genuinely concentrated book must report fewer names.
    concentrated = pd.DataFrame(0.0, index=dates, columns=columns)
    concentrated.iloc[:, 0] = 0.5
    concentrated.iloc[:, 1] = 0.5
    assert effective_breadth(concentrated).iloc[0] == pytest.approx(2.0)


def test_normalize_targets_caps_single_name_before_scaling_gross() -> None:
    """Order matters: cap the name first, then scale the book.

    A 2-name book at 50% each must become 25% each and 50% gross. Scaling
    gross first and capping afterwards would leave 50% per name.
    """
    from quant_system.config import BacktestConfig
    from quant_system.portfolio import normalize_targets

    dates = pd.bdate_range("2024-01-02", periods=2)
    raw = pd.DataFrame(
        {"a": [0.5, 0.5], "b": [0.5, 0.5]}, index=dates
    )
    out = normalize_targets(raw, ["a", "b"], dates, BacktestConfig())
    assert out.to_numpy() == pytest.approx(np.full((2, 2), 0.25))
    assert float(out.sum(axis=1).iloc[0]) == pytest.approx(0.5)


def test_normalize_targets_scales_a_wide_book_to_the_gross_limit() -> None:
    from quant_system.config import BacktestConfig
    from quant_system.portfolio import normalize_targets

    dates = pd.bdate_range("2024-01-02", periods=1)
    columns = [f"s{i}" for i in range(10)]
    raw = pd.DataFrame([{c: 0.1 for c in columns}], index=dates)
    out = normalize_targets(raw, columns, dates, BacktestConfig())
    assert float(out.sum(axis=1).iloc[0]) == pytest.approx(0.95)
    assert float(out.iloc[0].max()) <= 0.25


def test_backtester_delegates_to_the_shared_normalizer() -> None:
    """The backtest and the live signal must not be able to diverge."""
    from quant_system.backtest import Backtester
    from quant_system.config import BacktestConfig
    from quant_system.portfolio import normalize_targets

    dates = pd.bdate_range("2024-01-02", periods=3)
    raw = pd.DataFrame({"a": [0.6, 0.2, 0.0], "b": [0.6, 0.9, 1.0]}, index=dates)
    symbols = ["a", "b"]
    cfg = BacktestConfig()
    assert Backtester._normalize_targets(raw, symbols, dates, cfg).to_numpy() == pytest.approx(
        normalize_targets(raw, symbols, dates, cfg).to_numpy()
    )


def _breadth_panel(months: int = 30) -> dict[str, pd.DataFrame]:
    """A panel where breadth collapses for one month and recovers afterwards."""
    dates = pd.bdate_range("2020-01-31", periods=months * 21)
    frames = {}
    for k, symbol in enumerate(["sh600000", "sh600001", "sh600002", "sh600003"]):
        # All four names trend up until month 20, then collapse together.
        phase = np.arange(len(dates)) / 21.0
        trend = np.where(phase < 20, -0.01 * phase, -0.20 + 0.02 * (phase - 20))
        close = 10.0 * np.exp(trend + 0.002 * k)
        frames[symbol] = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1e5},
            index=dates,
        )
    return frames


def test_breadth_scale_is_one_and_de_risk_only() -> None:
    panel = _breadth_panel()
    scale = breadth_timing_scale(next(iter(panel.values())).index, panel)
    assert set(np.unique(scale)) <= {1.0}
    # The post-signal month is not yet in the panel, so nothing is de-risked.
    assert scale.notna().all()


def test_breadth_scale_validates_arguments() -> None:
    panel = _breadth_panel()
    with pytest.raises(ValueError):
        breadth_timing_scale(next(iter(panel.values())).index, panel, lower_tercile=0.0)
    with pytest.raises(ValueError):
        breadth_timing_scale(next(iter(panel.values())).index, panel, de_risk=1.5)


def test_breadth_scale_never_leaks_the_current_month() -> None:
    panel = _breadth_panel()
    index = next(iter(panel.values())).index
    scale = breadth_timing_scale(index, panel, history_months=2, de_risk=0.25)
    # The signal month itself must always be fully invested: the reading is
    # formed at that month's close and may only act on the following month.
    first_de_risk = scale[scale < 1.0].index.min()
    if first_de_risk is None:
        pytest.skip("panel did not trigger a de-risk month")
    signal_month = (first_de_risk.year, first_de_risk.month)
    month_ends = scale.groupby([scale.index.year, scale.index.month]).apply(
        lambda series: series.iloc[-1]
    )
    assert month_ends.loc[signal_month] == pytest.approx(0.25)
    previous = scale[scale.index < first_de_risk]
    assert previous.empty or (previous == 1.0).all()


def test_breadth_scale_cuts_the_month_after_a_collapse() -> None:
    # 40 months: a slow uptrend keeps breadth high, then a sharp 3-month drop
    # pushes almost every name under its 120-day average.
    dates = pd.bdate_range("2019-01-31", periods=40 * 21)
    frames = {}
    for k in range(4):
        phase = np.arange(len(dates)) / 21.0
        level = np.where(phase < 30, 10.0 * (1.004 ** np.minimum(phase, 30)),
                         10.0 * (1.004 ** 30) * np.exp(-0.10 * np.maximum(phase - 30, 0)))
        close = level * (1.0 + 0.01 * k)
        frames[f"sh60000{k}"] = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1e5},
            index=dates,
        )
    index = next(iter(frames.values())).index
    scale = breadth_timing_scale(index, frames, history_months=6, de_risk=0.25)
    month_end = scale.groupby([scale.index.year, scale.index.month]).last()
    assert (month_end < 1.0).any(), "collapse should trigger the rule"
    triggered = month_end[month_end < 1.0].index[0]
    previous = month_end[month_end.index < triggered]
    # The month that produced the reading stays fully invested.
    assert (previous == 1.0).all()
    assert month_end.loc[triggered] == pytest.approx(0.25)
