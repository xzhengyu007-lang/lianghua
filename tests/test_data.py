from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from quant_system.data import (
    TencentProvider,
    align_panel,
    infer_price_limit,
    load_symbol,
    normalize_symbol,
    panel_to_frame,
    tradeability,
    validate_panel,
)


@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("600000", "sh600000"),
        ("sh600000", "sh600000"),
        ("000001", "sz000001"),
        ("300750", "sz300750"),
        ("688981", "sh688981"),
        ("830799", "bj830799"),
    ],
)
def test_normalize_symbol(symbol: str, expected: str) -> None:
    assert normalize_symbol(symbol) == expected


def test_infer_price_limit() -> None:
    assert infer_price_limit("600000") == 0.10
    assert infer_price_limit("300750") == 0.20
    assert infer_price_limit("830799") == 0.30


def test_parse_rows_deduplicates_and_sorts() -> None:
    rows = [
        ["2024-01-03", "10", "11", "12", "9", "100"],
        ["2024-01-02", "9", "10", "11", "8", "200"],
        ["2024-01-03", "10", "10.5", "11.5", "9.5", "300"],
    ]
    frame = TencentProvider.parse_rows(rows)
    assert len(frame) == 2
    assert frame.index.is_monotonic_increasing
    assert frame.loc["2024-01-03", "close"] == 10.5
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]


def test_load_symbol_drops_non_positive_adjusted_rows(tmp_path) -> None:
    """Long qfq pages can come back negative; those rows are not prices."""
    index = pd.bdate_range("2024-01-01", periods=5)
    frame = pd.DataFrame(
        {
            "open": [10.0, -1.0, 11.0, 12.0, 13.0],
            "high": [10.5, -0.5, 11.5, 12.5, 13.5],
            "low": [9.5, -1.5, 10.5, 11.5, 12.5],
            "close": [10.0, -1.0, 11.0, 12.0, 13.0],
            "volume": [100.0] * 5,
        },
        index=index,
    )
    frame.to_parquet(tmp_path / "sh600519_qfq.parquet")
    loaded = load_symbol("600519", tmp_path)
    assert len(loaded) == 4
    assert (loaded[["open", "high", "low", "close"]] > 0).all().all()


def test_download_accepts_raw_as_an_alias(tmp_path, monkeypatch) -> None:
    """``raw`` must reach the provider as an unadjusted request, not raise.

    The CLI exposes ``--adjust raw`` while the provider encodes unadjusted
    history as an empty string.  Before this was normalised, the documented
    command failed with "adjust must be qfq, hfq, or empty", which made the
    rebuilt convention impossible to bootstrap from a clean checkout.
    """
    seen: list[str] = []

    def fake_request(self, symbol, count, adjust, end=None):
        seen.append(adjust)
        return {"code": 0, "data": {}}

    monkeypatch.setattr(TencentProvider, "_request", fake_request)
    provider = TencentProvider(tmp_path, pause=0.0)
    with pytest.raises(ValueError, match="no daily data"):
        provider.download("600000", count=60, adjust="raw")
    assert seen == [""]


def test_cache_audit_accepts_a_total_return_cache_without_provider_meta(tmp_path) -> None:
    """A rebuilt ``_tr`` file has no provider ``adjust`` value.

    It is the recommended convention now, so the audit must check it (row
    counts, dates, non-positive prices) instead of ignoring it or flagging it
    for a missing provider field it can never have.
    """
    from quant_system.cache_audit import audit_cache

    index = pd.bdate_range("2024-01-02", periods=80)
    close = pd.Series(10.0 + np.arange(80) * 0.01, index=index)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1000.0,
        }
    )
    frame.to_parquet(tmp_path / "sh600000_tr.parquet")
    (tmp_path / "sh600000_tr.meta.json").write_text(
        json.dumps(
            {
                "symbol": "sh600000",
                "source": "raw+qfq piecewise-affine rebuild",
                "rows": len(frame),
                "start": str(index[0].date()),
                "end": str(index[-1].date()),
                "segments": 2,
                "events": [],
                "problems": [],
            }
        ),
        encoding="utf-8",
    )

    report = audit_cache(tmp_path)
    item = next(f for f in report.files if f.adjust == "tr")
    assert item.ok, item.problems
    assert item.rows_file == 80


def test_cache_audit_rejects_a_total_return_cache_that_disagrees_with_its_meta(tmp_path) -> None:
    from quant_system.cache_audit import audit_cache

    index = pd.bdate_range("2024-01-02", periods=80)
    close = pd.Series(10.0 + np.arange(80) * 0.01, index=index)
    frame = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1000.0}
    )
    frame.to_parquet(tmp_path / "sh600000_tr.parquet")
    (tmp_path / "sh600000_tr.meta.json").write_text(
        json.dumps(
            {
                "symbol": "sh600000",
                "source": "raw+qfq piecewise-affine rebuild",
                "rows": 79,
                "start": str(index[0].date()),
                "end": str(index[-1].date()),
            }
        ),
        encoding="utf-8",
    )
    report = audit_cache(tmp_path)
    item = next(f for f in report.files if f.adjust == "tr")
    assert not item.ok
    assert any("row count mismatch" in problem for problem in item.problems)


def test_validate_panel(sample_panel: dict[str, pd.DataFrame]) -> None:
    report = validate_panel(sample_panel)
    assert report.symbols == len(sample_panel)
    assert report.duplicate_dates == 0
    assert report.ohlc_violations == 0
    assert report.rows > 0


def test_align_panel_uses_union_by_default(sample_panel: dict[str, pd.DataFrame]) -> None:
    panel = dict(sample_panel)
    panel["sh600000"] = panel["sh600000"].iloc[:-10]
    aligned, dates = align_panel(panel)
    assert all(frame.index.equals(dates) for frame in aligned.values())
    assert len(dates) == max(len(f) for f in panel.values())
    assert aligned["sh600000"]["close"].isna().iloc[-1]
    assert aligned["sh600001"]["close"].notna().all()


def test_align_panel_require_all_restores_intersection(
    sample_panel: dict[str, pd.DataFrame],
) -> None:
    panel = dict(sample_panel)
    panel["sh600000"] = panel["sh600000"].iloc[:-10]
    aligned, dates = align_panel(panel, require_all=True)
    assert len(dates) == min(len(f) for f in panel.values())
    assert all(frame["close"].notna().all() for frame in aligned.values())


def test_availability_and_last_bar_dates(sample_panel: dict[str, pd.DataFrame]) -> None:
    from quant_system.data import availability, last_bar_dates

    panel = dict(sample_panel)
    panel["sh600000"] = panel["sh600000"].iloc[:-10]
    mask = availability(panel)
    assert not bool(mask["sh600000"].iloc[-1])
    assert bool(mask["sh600001"].iloc[-1])
    last = last_bar_dates(panel)
    assert last["sh600000"] < last["sh600001"]


def test_tradeability_blocks_limit_up_and_suspension() -> None:
    index = pd.bdate_range("2024-01-02", periods=4)
    frame = pd.DataFrame(
        {
            "open": [10.0, 11.0, 10.0, 10.0],
            "high": [10.5, 11.0, 10.5, 10.5],
            "low": [9.8, 10.9, 9.5, 9.5],
            "close": [10.0, 11.0, 10.0, 10.0],
            "volume": [1000, 1000, 0, 1000],
        },
        index=index,
    )
    access = tradeability(frame, "600000")
    assert not access.loc[index[1], "can_buy"]  # limit-up open
    assert not access.loc[index[2], "can_buy"]  # suspension
    assert not access.loc[index[2], "can_sell"]


def test_panel_to_frame(sample_panel: dict[str, pd.DataFrame]) -> None:
    close = panel_to_frame(sample_panel, "close")
    assert close.shape[1] == len(sample_panel)
    assert close.index.equals(sample_panel[next(iter(sample_panel))].index)


def test_research_window_ignores_a_single_long_history() -> None:
    """One name with an unusually long history must not drag the window back.

    `sh600837` carries 443 real bars before 2019 while every other broad-universe
    name starts in 2019. Using the raw union made the CLI run a 2017-2026
    backtest on 1-25 stocks and report +121% instead of the published +145.9%.
    """
    from quant_system.cli import _research_panel

    long_index = pd.bdate_range("2017-08-08", periods=2200)
    recent = long_index[-600:]

    def frame(index: pd.DatetimeIndex, base: float) -> pd.DataFrame:
        close = pd.Series(base + np.arange(len(index)) * 0.01, index=index)
        return pd.DataFrame(
            {
                "open": close * 1.001,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000.0,
            },
            index=index,
        )

    panel = {
        "sh600000": frame(recent, 10.0),
        "sh600036": frame(recent, 11.0),
        "sh600837": frame(long_index, 8.0),
    }
    clipped = _research_panel(panel)
    start = min(frame.index.min() for frame in clipped.values())
    assert start >= recent[0]
    assert all(frame.index.max() == recent[-1] for frame in clipped.values())



def test_pit_research_window_uses_survivor_anchor_and_normalizes_keys() -> None:
    from quant_system.cli import _research_panel

    anchor_index = pd.bdate_range("2019-06-03", periods=300)
    long_index = pd.bdate_range("2014-01-02", periods=1500)
    recent_index = anchor_index[100:250]

    def frame(index: pd.DatetimeIndex, base: float) -> pd.DataFrame:
        close = pd.Series(base + np.arange(len(index)) * 0.01, index=index)
        return pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000.0,
            },
            index=index,
        )

    panel = {
        "sh600000": frame(anchor_index, 10.0),
        "sh600036": frame(anchor_index, 11.0),
        "sh600074": frame(long_index, 5.0),
        "sh603133": frame(recent_index, 7.0),
    }
    clipped = _research_panel(panel, anchor_symbols=["600000", "600036"])
    assert set(clipped) == {"sh600000", "sh600036", "sh600074", "sh603133"}
    assert clipped["sh600000"].index.min() == anchor_index[0]
    assert clipped["sh600074"].index.min() >= anchor_index[0]
    # Late listings are clipped to the survivor window; availability, not the
    # window, makes them ineligible before their first bar.
    assert clipped["sh603133"].index.min() == recent_index[0]



def test_default_research_panel_still_applies_coverage_filter() -> None:
    from quant_system.cli import _research_panel

    index = pd.bdate_range("2019-06-03", periods=400)
    # A trailing hole, like a late listing or an unfinished download, rather
    # than a name that simply started later.
    short = index[:100].union(index[200:])
    long = index[:360]

    def frame(dates: pd.DatetimeIndex, base: float) -> pd.DataFrame:
        close = pd.Series(base + np.arange(len(dates)) * 0.01, index=dates)
        return pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000.0,
            },
            index=dates,
        )

    panel = {
        "sh600000": frame(index, 10.0),
        "sh600036": frame(index, 11.0),
        "sh600837": frame(long, 8.0),
        "sh688111": frame(short, 9.0),
    }
    clipped = _research_panel(panel)
    assert "sh688111" not in clipped
    assert "sh600837" in clipped

def test_pit_universe_contains_real_delisted_names() -> None:
    from quant_system.config import BROAD_UNIVERSE
    from quant_system.data import pit_universe

    universe = pit_universe()
    assert len(universe) == len(BROAD_UNIVERSE) + 140
    # A real delisted symbol recovered by the probe must be present and named.
    assert "sh600837" in universe
    assert universe["sh600837"]


def test_explicit_availability_prevents_stale_incumbent() -> None:
    from quant_system.strategies import _select_weights

    index = pd.bdate_range("2024-01-02", periods=4)
    scores = pd.DataFrame(
        {"sh600000": [10.0, 10.0, np.nan, np.nan], "sh600004": [5.0, 5.0, 7.0, 7.0]},
        index=index,
    )
    available = pd.DataFrame(
        {
            "sh600000": [True, True, False, False],
            "sh600004": [True, True, True, True],
        },
        index=index,
    )
    weights = _select_weights(scores, 1, available=available, rebalance_days=10)
    held = [set(row[row > 0].index) for _, row in weights.iterrows()]
    assert held[0] == {"sh600000"}
    assert held[1] == {"sh600000"}
    assert held[2] == {"sh600004"}

def test_every_pit_member_has_a_sector() -> None:
    from quant_system.data import pit_universe
    from quant_system.sectors import sector_of

    missing = [s for s in pit_universe() if sector_of(s) == "其他"]
    assert not missing, missing


def test_delisted_names_receive_sector_classifications() -> None:
    from quant_system.sectors import DELISTED_SECTOR_MAP, sector_of

    assert len(DELISTED_SECTOR_MAP) == 140
    assert sector_of("sh600837") == "金融"
    assert all(sector_of(symbol) != "其他" for symbol in DELISTED_SECTOR_MAP)

def test_cache_audit_flags_an_additive_adjustment(tmp_path) -> None:
    """Tencent's qfq subtracts a constant; the audit must not call it healthy."""
    from quant_system.cache_audit import audit_cache

    index = pd.bdate_range("2024-01-02", periods=120)
    raw_close = pd.Series(10.0 + np.arange(120) * 0.02, index=index)
    raw = pd.DataFrame(
        {
            "open": raw_close,
            "high": raw_close,
            "low": raw_close,
            "close": raw_close,
            "volume": 1000.0,
        }
    )
    # Pure additive adjustment: qfq = raw - 1.5 for the first 60 bars, then
    # qfq = raw - 1.0 (an ex-dividend step).
    qfq = raw.copy()
    qfq.loc[index[:60], ["open", "high", "low", "close"]] = (
        raw.loc[index[:60], ["open", "high", "low", "close"]] - 1.5
    )
    qfq.loc[index[60:], ["open", "high", "low", "close"]] = (
        raw.loc[index[60:], ["open", "high", "low", "close"]] - 1.0
    )
    raw.to_parquet(tmp_path / "sh600000_raw.parquet")
    qfq.to_parquet(tmp_path / "sh600000_qfq.parquet")
    for stem in ("sh600000_raw", "sh600000_qfq"):
        (tmp_path / f"{stem}.meta.json").write_text(
            json.dumps(
                {
                    "symbol": "sh600000",
                    "adjust": stem.rsplit("_", 1)[1],
                    "pages": 1,
                    "rows": len(raw),
                    "start": str(index[0].date()),
                    "end": str(index[-1].date()),
                    "warnings": [],
                }
            ),
            encoding="utf-8",
        )

    report = audit_cache(tmp_path)
    qfq_item = next(item for item in report.files if item.adjust == "qfq")
    assert not qfq_item.ok
    assert any("additive" in problem for problem in qfq_item.problems)
    assert qfq_item.adjustment_constant_share is not None
    assert qfq_item.adjustment_constant_share > 0.9


def test_cache_audit_accepts_a_multiplicative_adjustment(tmp_path) -> None:
    from quant_system.cache_audit import audit_cache

    index = pd.bdate_range("2024-01-02", periods=120)
    raw_close = pd.Series(10.0 + np.arange(120) * 0.02, index=index)
    raw = pd.DataFrame(
        {
            "open": raw_close,
            "high": raw_close,
            "low": raw_close,
            "close": raw_close,
            "volume": 1000.0,
        }
    )
    qfq = raw.copy()
    qfq.loc[index[:60], ["open", "high", "low", "close"]] = (
        raw.loc[index[:60], ["open", "high", "low", "close"]] * 0.9
    )
    raw.to_parquet(tmp_path / "sh600000_raw.parquet")
    qfq.to_parquet(tmp_path / "sh600000_qfq.parquet")
    for stem in ("sh600000_raw", "sh600000_qfq"):
        (tmp_path / f"{stem}.meta.json").write_text(
            json.dumps(
                {
                    "symbol": "sh600000",
                    "adjust": stem.rsplit("_", 1)[1],
                    "pages": 1,
                    "rows": len(raw),
                    "start": str(index[0].date()),
                    "end": str(index[-1].date()),
                    "warnings": [],
                }
            ),
            encoding="utf-8",
        )

    report = audit_cache(tmp_path)
    qfq_item = next(item for item in report.files if item.adjust == "qfq")
    assert not any("additive" in problem for problem in qfq_item.problems)

