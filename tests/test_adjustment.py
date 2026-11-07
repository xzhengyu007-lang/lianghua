from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_system.adjustment import (
    fit_piecewise_affine,
    rebuild_frame,
    rebuild_panel,
)


def _frame(closes, index=None):
    index = index or pd.bdate_range("2020-01-01", periods=len(closes))
    close = pd.Series(closes, index=index, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1000.0,
        },
        index=index,
    )


def _additive_series(raw, offset, event_at, new_offset):
    """Build the provider's additive qfq: qfq = raw - offset (piecewise)."""
    qfq = raw.copy()
    qfq.iloc[event_at:] = raw.iloc[event_at:] - new_offset
    qfq.iloc[:event_at] = raw.iloc[:event_at] - offset
    return qfq


def _with_ohlc(close: pd.Series, volume: float = 1000.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": volume,
        },
        index=close.index,
    )


def test_additive_adjustment_is_rebuilt_as_a_total_return_series():
    raw = _frame([10.0, 10.5, 10.2, 10.0, 10.4, 10.8])
    qfq = _additive_series(raw["close"], 2.0, 3, 1.5)
    rebuilt, fit = rebuild_frame(raw, _with_ohlc(qfq), symbol="sh600000")

    assert len(fit.events) == 1
    event = fit.events[0]
    assert event.date == raw.index[3]
    assert event.cash == pytest.approx(0.5)
    assert event.split == pytest.approx(1.0)

    # On the ex-date the holder receives 0.5 cash on a 10.2 pre-close, so the
    # total return is 10.5 / 10.2 - 1 (raw +5.88%, dividend +4.90%).
    assert rebuilt["close"].iloc[3] / rebuilt["close"].iloc[2] - 1 == pytest.approx(
        10.5 / 10.2 - 1, rel=1e-6
    )
    # Older bars are scaled by raw_event / (raw_event + cash) = 10.0 / 10.5,
    # so the first adjusted close is 10.0 * 10.0 / 10.5.  The final close is
    # 10.8, and the total return follows directly.
    expected = 10.8 / (10.0 * 10.0 / 10.5) - 1
    assert rebuilt["close"].iloc[-1] / rebuilt["close"].iloc[0] - 1 == pytest.approx(
        expected, rel=1e-6
    )


def test_dividend_day_return_matches_raw_plus_cash():
    raw = _frame([20.0, 20.4, 19.0, 19.2, 19.1])
    qfq = _additive_series(raw["close"], 3.0, 2, 2.0)
    rebuilt, fit = rebuild_frame(raw, _with_ohlc(qfq), symbol="sh600000")
    total_return = rebuilt["close"].pct_change()
    # raw falls 20.4 -> 19.0; the 1.0 cash dividend restores the holder to
    # 20.0, i.e. a -1.96% total return rather than -6.86%.
    assert float(fit.events[0].cash) == pytest.approx(1.0)
    assert total_return.iloc[2] == pytest.approx(20.0 / 20.4 - 1, rel=1e-6)


def test_bonus_issue_scales_prices_and_volume_but_keeps_wealth():
    # 1-for-1 bonus: the traded price halves from 10 to 5 on the event bar.
    # The provider's qfq series is anchored at the latest bar, so the older
    # segment is the one that gets rescaled (slope 2.0).
    raw = _frame([10.0, 10.6, 9.8, 5.1, 5.4, 5.0])
    # The provider shows the pre-event bars at half the traded price, i.e.
    # qfq = raw / 2 before the event and qfq = raw after it.
    qfq = raw["close"].copy()
    qfq.iloc[:3] = raw["close"].iloc[:3] / 2.0
    rebuilt, fit = rebuild_frame(raw, _with_ohlc(qfq), symbol="sh600000")
    assert len(fit.events) == 1
    event = fit.events[0]
    assert event.split == pytest.approx(2.0)
    assert event.cash == pytest.approx(0.0, abs=1e-9)
    # Each pre-event share becomes two shares, so volume doubles.
    assert rebuilt["volume"].iloc[0] == pytest.approx(2000.0)
    # Adjusted prices are continuous across the bonus: the holder's wealth
    # does not jump just because the exchange halved the quoted price.
    # raw 9.8 -> 5.1 is -48%, but one old share became two: the wealth ratio
    # is (2 * 5.1) / 9.8 = 1.0408.
    assert rebuilt["close"].iloc[3] / rebuilt["close"].iloc[2] == pytest.approx(
        (2.0 * 5.1) / 9.8, rel=1e-6
    )


def test_rebuild_rejects_an_unfittable_series():
    raw = _frame(np.linspace(10, 20, 80))
    rng = np.random.default_rng(7)
    noise = pd.Series(10 + np.cumsum(rng.normal(0, 2, 80)), index=raw.index)
    rebuilt, fit = rebuild_frame(raw, _with_ohlc(noise), symbol="sh600000")
    # A random walk cannot be described by a handful of affine segments, so
    # the fit must say so rather than returning a clean-looking series.
    assert fit.problems
    assert "poor" in " ".join(fit.problems) or fit.diagnostics["segments"] > 20


def test_rebuild_flags_a_series_that_is_not_affine():
    raw = _frame(np.linspace(10.0, 20.0, 120))
    # Curved relation: a single affine segment cannot describe it.
    curved = pd.Series(20.0 + (raw["close"].to_numpy() - 15.0) ** 2, index=raw.index)
    _, fit = rebuild_frame(raw, _with_ohlc(curved), symbol="sh600000")
    assert fit.problems


def test_fit_reports_implausible_corporate_action_instead_of_hiding_it():
    raw = _frame(np.linspace(10.0, 20.0, 60))
    qfq = raw["close"].copy()
    # A 50x "split" is not a real A-share action and must be flagged.
    qfq.iloc[30:] = qfq.iloc[30:] / 50.0
    fit = fit_piecewise_affine(raw["close"], qfq)
    assert fit.problems


def test_panel_rebuild_skips_symbols_without_raw_files(tmp_path):
    cache = tmp_path / "cache"
    raw_dir = tmp_path / "raw"
    cache.mkdir()
    raw_dir.mkdir()
    closes = np.linspace(10.0, 12.0, 30)
    raw = _frame(closes)
    qfq = _frame(closes - 1.0)
    raw.to_parquet(raw_dir / "sh600000_raw.parquet")
    qfq.to_parquet(cache / "sh600000_qfq.parquet")
    panel, fits = rebuild_panel(["sh600000", "sz000001"], cache, raw_dir)
    assert list(panel) == ["sh600000"]
    assert list(fits) == ["sh600000"]

def test_leading_raw_bar_without_a_qfq_factor_is_dropped_not_fatal():
    # Some delisted names have one raw bar before the provider's qfq history
    # starts.  The factor there is unknowable, so the bar is dropped and the
    # loss is recorded; aborting the whole universe for one bar is not.
    raw_index = pd.bdate_range("2020-01-01", periods=7)
    qfq_index = raw_index[1:]
    raw_close = pd.Series(
        [10.0, 10.2, 10.1, 10.4, 10.3, 10.6, 10.8], index=raw_index, dtype=float
    )
    raw = _with_ohlc(raw_close)
    qfq_close = pd.Series([10.2, 10.1, 10.4, 10.3, 10.6, 10.8], index=qfq_index)
    rebuilt, fit = rebuild_frame(raw, _with_ohlc(qfq_close), symbol="sh600401")

    assert len(rebuilt) == len(qfq_index)
    assert rebuilt.index[0] == qfq_index[0]
    assert fit.diagnostics["leading_bars_dropped"] == 1
    assert any("leading raw bars dropped" in problem for problem in fit.problems)
