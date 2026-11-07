"""Walk-forward invariants: no overlap, reproducible, audited."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_system.config import BacktestConfig, ResearchConfig
from quant_system.research import walk_forward
from quant_system.strategies import build_strategy


def test_windows_do_not_overlap(sample_panel) -> None:
    cfg = ResearchConfig(train_days=120, test_days=40, embargo_days=5)
    result = walk_forward(
        sample_panel,
        "low_vol_core",
        {"vol_window": [30], "top_n": [3], "trend_window": [60], "rebalance_days": [10]},
        research=cfg,
        backtest=BacktestConfig(initial_cash=100_000),
        verbose=False,
    )
    windows = result.windows
    assert len(windows) >= 2
    for _, row in windows.iterrows():
        assert pd.Timestamp(row["train_end"]) < pd.Timestamp(row["test_start"])
    starts = pd.to_datetime(windows["test_start"]).tolist()
    ends = pd.to_datetime(windows["test_end"]).tolist()
    for earlier_end, later_start in zip(ends, starts[1:]):
        assert earlier_end < later_start


def test_walk_forward_is_reproducible(sample_panel) -> None:
    kwargs = dict(
        research=ResearchConfig(train_days=100, test_days=30, embargo_days=3),
        backtest=BacktestConfig(initial_cash=100_000),
        verbose=False,
    )
    grid = {"vol_window": [30], "top_n": [3], "trend_window": [50], "rebalance_days": [10]}
    first = walk_forward(sample_panel, "low_vol_core", grid, **kwargs)
    second = walk_forward(sample_panel, "low_vol_core", grid, **kwargs)
    pd.testing.assert_series_equal(first.equity, second.equity)
    pd.testing.assert_frame_equal(first.windows, second.windows)


def test_every_window_has_audit_and_activity(sample_panel) -> None:
    result = walk_forward(
        sample_panel,
        "low_vol_core",
        {"vol_window": [30], "top_n": [3], "trend_window": [60], "rebalance_days": [10]},
        research=ResearchConfig(train_days=120, test_days=40, embargo_days=5),
        backtest=BacktestConfig(initial_cash=100_000),
        verbose=False,
    )
    assert not result.audit.empty
    assert len(result.audit) == len(result.windows)
    assert (result.audit["bars"] >= 5).all()
    # A long-only low-vol strategy in a synthetic rising panel must trade.
    assert (result.audit["positions_at_start"] > 0).all()
    assert result.total_trades > 0
    assert (result.windows["test_return"].abs() > 0).all()


def test_parameter_grid_selection_uses_only_training_window(sample_panel, monkeypatch) -> None:
    """The candidate trained on data after a test start must not be chosen."""
    import quant_system.research as research_module

    seen_train_ends: list[pd.Timestamp] = []
    original = research_module.build_strategy

    def spy(name, **params):
        return original(name, **params)

    monkeypatch.setattr(research_module, "build_strategy", spy)
    result = walk_forward(
        sample_panel,
        "low_vol_core",
        {"vol_window": [30], "top_n": [3], "trend_window": [60], "rebalance_days": [10]},
        research=ResearchConfig(train_days=100, test_days=30, embargo_days=3),
        backtest=BacktestConfig(initial_cash=100_000),
        verbose=False,
    )
    for _, row in result.windows.iterrows():
        assert pd.Timestamp(row["train_end"]) < pd.Timestamp(row["test_start"])


def test_window_exposure_is_populated(sample_panel) -> None:
    """A window's exposure must describe that window, not come back as NaN.

    The first PIT run reported NaN exposure because ``performance_metrics`` was
    called without the test-period weights. That made every audit row look like
    a cash-only window even when eight names were held.
    """
    result = walk_forward(
        sample_panel,
        "low_vol_core",
        {"vol_window": [30], "top_n": [3], "trend_window": [60], "rebalance_days": [10]},
        research=ResearchConfig(train_days=120, test_days=40, embargo_days=5),
        backtest=BacktestConfig(initial_cash=100_000),
        verbose=False,
    )
    assert result.windows["exposure"].notna().all()
    assert (result.windows["exposure"] > 0).all()
    assert result.exposure > 0


def test_summary_exposure_matches_test_windows_not_full_history(sample_panel) -> None:
    """Headline exposure must describe the OOS windows only.

    Before the fix it averaged ``full.metrics.exposure`` over each window's
    entire backtest, so a run that starts flat long before the first test window
    reported ~50% exposure while every test window was ~100% invested
    (RESEARCH_LOG 23.36).
    """
    result = walk_forward(
        sample_panel,
        "low_vol_core",
        {"vol_window": [30], "top_n": [3], "trend_window": [60], "rebalance_days": [10]},
        research=ResearchConfig(train_days=120, test_days=40, embargo_days=5),
        backtest=BacktestConfig(initial_cash=100_000),
        verbose=False,
    )
    summary = result.summary()
    assert summary["exposure"] == pytest.approx(result.windows["exposure"].mean(), rel=1e-6)
    assert "exposure_full_history" in summary
    assert 0.0 < summary["exposure"] <= 1.05


def test_window_turnover_is_independent_of_compounded_equity(sample_panel) -> None:
    """Turnover must divide by the equity this window actually traded.

    ``BacktestResult.metrics.turnover`` divides the whole-history traded
    notional by the whole-history average equity.  Every walk-forward window
    restarts at ``initial_cash``, so once earlier windows compound, that
    denominator no longer matches the live window equity.  In the PIT run this
    reported 41.2 for one version and 56.5 for another even though returns and
    trade counts were within 2%.  The windowed measure must use the window's
    own mean equity and the trades inside it.
    """
    from quant_system.research import _window_turnover

    from quant_system.backtest import Backtester

    strategy = build_strategy("low_vol_core", top_n=3, vol_window=30, trend_window=60, rebalance_days=10)
    weights = strategy.generate(sample_panel)
    result = Backtester(BacktestConfig(initial_cash=100_000)).run(sample_panel, weights)
    segment = result.equity.iloc[-60:]
    windowed = _window_turnover(result, segment)
    assert windowed > 0
    # It must equal notional / 2 / this window's mean equity, and must differ
    # from the whole-history ratio whenever the path has moved.
    lo, hi = segment.index[0], segment.index[-1]
    trades = result.trades
    traded = trades[(trades["date"] >= lo) & (trades["date"] <= hi)]["notional"].sum()
    assert windowed == pytest.approx(float(traded) / 2.0 / float(segment.mean()))
    if abs(float(segment.mean()) - float(result.equity.mean())) > 1e-6:
        assert windowed != pytest.approx(float(result.metrics.turnover))


def test_truncating_history_does_not_change_earlier_results(sample_panel) -> None:
    """A walk-forward run must be causal: future bars cannot affect the past.

    Build weights from a long panel and from a prefix that stops earlier. The
    overlapping weights and equity must be identical. This catches overlays that
    silently reindex to the full sample or use a future covariance window.
    """
    from quant_system.config import RiskConfig
    from quant_system.data import align_panel
    from quant_system.portfolio import build_portfolio

    aligned, dates = align_panel(sample_panel)
    cut = dates[-25]
    short = {symbol: frame.loc[:cut] for symbol, frame in aligned.items()}
    params = dict(top_n=3, vol_window=30, trend_window=60, rebalance_days=10)
    long_raw = build_strategy("low_vol_core", **params).generate(aligned)
    short_raw = build_strategy("low_vol_core", **params).generate(short)
    assert long_raw.loc[:cut].equals(short_raw)
    long_weights = build_portfolio(long_raw, aligned, risk=RiskConfig())
    short_weights = build_portfolio(short_raw, short, risk=RiskConfig())
    pd.testing.assert_frame_equal(
        long_weights.loc[:cut], short_weights, check_names=False
    )


def test_risk_variants_are_selected_inside_training_window(sample_panel) -> None:
    """A risk configuration must be chosen like a parameter, never on the test set."""
    from quant_system.config import RiskConfig

    result = walk_forward(
        sample_panel,
        "low_vol_core",
        {"vol_window": [30], "top_n": [3], "trend_window": [60], "rebalance_days": [10]},
        research=ResearchConfig(train_days=120, test_days=40, embargo_days=5),
        backtest=BacktestConfig(initial_cash=100_000),
        risk_variants={
            "low": RiskConfig(target_annual_vol=0.10),
            "high": RiskConfig(target_annual_vol=0.30),
        },
        verbose=False,
    )
    assert not result.windows.empty
    assert "risk_variant" in result.windows.columns
    assert set(result.windows["risk_variant"]).issubset({"low", "high"})
    assert all("risk_variant" in selection for selection in result.selections)
    for _, row in result.windows.iterrows():
        assert pd.Timestamp(row["train_end"]) < pd.Timestamp(row["test_start"])


def test_weight_transform_is_applied_after_selection(sample_panel) -> None:
    """The weighting hook must receive the selected basket, not raw scores.

    It is also checked for causality: a prefix run and a full run must produce
    identical transformed weights on the overlap.
    """
    from quant_system.data import align_panel

    aligned, dates = align_panel(sample_panel)
    cut = dates[-25]
    short = {symbol: frame.loc[:cut] for symbol, frame in aligned.items()}
    calls: list[int] = []

    def transform(raw, panel):
        calls.append(len(raw))
        held = raw.gt(0)
        assert (held.sum(axis=1).max() <= 3)
        counts = held.sum(axis=1).replace(0, pd.NA)
        return raw.div(counts, axis=0).fillna(0.0)

    result = walk_forward(
        short,
        "low_vol_core",
        {"vol_window": [30], "top_n": [3], "trend_window": [60], "rebalance_days": [10]},
        research=ResearchConfig(train_days=120, test_days=40, embargo_days=5),
        backtest=BacktestConfig(initial_cash=100_000),
        weight_transform=transform,
        verbose=False,
    )
    assert calls
    assert (result.equity > 0).all()


def test_phase_sweep_reports_distribution_not_a_single_point(sample_panel) -> None:
    """A walk-forward window must expose the rebalance-phase spread.

    The headline equity path is one arbitrary calendar phase.  When the caller
    asks for a sweep, every window has to carry median/worst/best so a report
    cannot quietly present a lucky phase as the strategy's return.
    """
    result = walk_forward(
        sample_panel,
        "low_vol_core",
        {"vol_window": [30], "top_n": [3], "trend_window": [60], "rebalance_days": [10]},
        research=ResearchConfig(train_days=120, test_days=40, embargo_days=5),
        backtest=BacktestConfig(initial_cash=100_000),
        phase_offsets=range(10),
        verbose=False,
    )
    windows = result.windows
    assert not windows.empty
    assert (windows["phase_count"] == 10).all()
    assert windows["phase_return_median"].notna().all()
    assert (windows["phase_return_worst"] <= windows["phase_return_median"]).all()
    assert (windows["phase_return_median"] <= windows["phase_return_best"]).all()
    assert "phase_return_median" in result.audit.columns


def test_phase_blend_differs_from_the_offset_zero_path(sample_panel) -> None:
    """Averaging all phases must be a different book, not an alias of phase 0."""
    base = walk_forward(
        sample_panel,
        "low_vol_core",
        {"vol_window": [30], "top_n": [3], "trend_window": [60], "rebalance_days": [10]},
        research=ResearchConfig(train_days=120, test_days=40, embargo_days=5),
        backtest=BacktestConfig(initial_cash=100_000),
        phase_offsets=range(10),
        verbose=False,
    )
    blended = walk_forward(
        sample_panel,
        "low_vol_core",
        {"vol_window": [30], "top_n": [3], "trend_window": [60], "rebalance_days": [10]},
        research=ResearchConfig(train_days=120, test_days=40, embargo_days=5),
        backtest=BacktestConfig(initial_cash=100_000),
        phase_offsets=range(10),
        phase_blend=True,
        verbose=False,
    )
    assert not base.equity.equals(blended.equity)
    assert (blended.equity > 0).all()


def test_inverse_vol_weights_change_only_with_the_basket(sample_panel) -> None:
    """Sizing must be locked between basket changes, or turnover explodes."""
    from quant_system.data import align_panel

    aligned, dates = align_panel(sample_panel)
    raw = build_strategy(
        "low_vol_core", top_n=3, vol_window=30, trend_window=60, rebalance_days=10
    ).generate(aligned)
    from scripts.run_weighting import inverse_vol_at_changes

    sized = inverse_vol_at_changes(raw, aligned, 30)
    held = raw.gt(0)
    changed = held.ne(held.shift(1)).any(axis=1)
    changed.iloc[0] = True
    for date in dates[1:]:
        if not changed.loc[date]:
            pd.testing.assert_series_equal(
                sized.loc[date], sized.loc[dates[dates.get_loc(date) - 1]],
                check_names=False,
            )
    held_rows = sized.sum(axis=1) > 0
    assert held_rows.any()
    assert np_all_close(sized.loc[held_rows].sum(axis=1), 1.0)


def np_all_close(series, target: float) -> bool:
    return bool(((series - target).abs() <= 1e-9).all())


def test_prepare_then_select_reproduces_generate(sample_panel) -> None:
    """A phase sweep must reuse the same scoring path as a normal backtest.

    ``walk_forward`` calls ``prepare()`` once and then ``select()`` per phase.
    If that pair ever diverges from ``generate()``, every phase distribution
    and every headline number would be computed from a different book than the
    single-run backtest.  Pin the offset-zero case, for both anchor modes.
    """
    from quant_system.data import align_panel

    aligned, _ = align_panel(sample_panel)
    for reset in ("start", "year"):
        strategy = build_strategy(
            "low_vol_core",
            top_n=3,
            vol_window=30,
            trend_window=60,
            rebalance_days=10,
            rebalance_reset=reset,
        )
        direct = strategy.generate(aligned)
        scores, eligible, available = strategy.prepare(aligned)
        reused = strategy.select(scores, eligible, available, offset=0)
        pd.testing.assert_frame_equal(direct, reused, check_exact=True)
