from __future__ import annotations

import pandas as pd
import pytest

from quant_system.brief import (
    build_brief,
    data_health,
    render_html,
    render_markdown,
    risk_snapshot,
    save_brief,
)
from quant_system.paper import Ledger, Position

PARAMS = dict(top_n=3, vol_window=30, trend_window=60, max_per_sector=2, rebalance_days=10)


def test_data_health_detects_stale_symbol(sample_panel) -> None:
    panel = dict(sample_panel)
    panel["sh600000"] = panel["sh600000"].iloc[:-5]
    health = data_health(panel)
    assert health["symbols"] == 6
    assert health["on_latest_bar"] == 5
    assert health["coverage"] == pytest.approx(5 / 6, rel=1e-3)
    assert health["stale_examples"][0]["symbol"] == "sh600000"


def test_brief_contains_signal_risk_and_notes(sample_panel) -> None:
    brief = build_brief(sample_panel, "low_vol_core", params=PARAMS)
    assert len(brief.signal.rows) > 0
    assert brief.risk["positions"] > 0
    assert brief.risk["ex_ante_vol"] > 0
    markdown = render_markdown(brief)
    for heading in ("## 数据", "## 信号", "## 风险"):
        assert heading in markdown
    payload = brief.to_dict()
    assert payload["as_of"] == str(brief.as_of.date())
    assert payload["signal"]["orders"]


def test_beta_is_reported_only_with_a_benchmark(sample_panel) -> None:
    weights = pd.Series({"sh600000": 0.3, "sh600001": 0.3, "sh600002": 0.3})
    without = risk_snapshot(weights, sample_panel)
    assert without["beta"] is None
    benchmark = sample_panel["sh600001"]["close"]
    with_bench = risk_snapshot(weights, sample_panel, benchmark)
    assert with_bench["beta"] is not None
    assert with_bench["sectors"]


def test_save_brief_writes_three_artifacts(sample_panel, tmp_path) -> None:
    brief = build_brief(sample_panel, "low_vol_core", params=PARAMS)
    out = save_brief(brief, tmp_path / "brief")
    assert (out / "brief.md").exists()
    assert (out / "brief.json").exists()
    html = (out / "brief.html").read_text(encoding="utf-8")
    assert "<pre>" in html
    assert render_html(brief).startswith("<!doctype html>")


def test_ledger_block_reports_drift(sample_panel, tmp_path) -> None:
    ledger = Ledger(cash=100_000.0, positions={"sh600000": Position(shares=1000, avg_cost=10.0)})
    path = ledger.save(tmp_path / "ledger.json")
    brief = build_brief(sample_panel, "low_vol_core", params=PARAMS, ledger_path=path)
    assert brief.ledger is not None
    assert brief.ledger["summary"]["positions"] == 1
    assert isinstance(brief.ledger["drift"], list)
    assert {"symbol", "target", "current", "diff"} <= set(brief.ledger["drift"][0])
    assert "## 模拟盘账本" in render_markdown(brief)


def test_missing_ledger_is_ignored(sample_panel, tmp_path) -> None:
    brief = build_brief(
        sample_panel, "low_vol_core", params=PARAMS, ledger_path=tmp_path / "nope.json"
    )
    assert brief.ledger is None


def test_beta_reports_standard_error_and_window(sample_panel) -> None:
    weights = pd.Series({"sh600000": 0.3, "sh600001": 0.3, "sh600002": 0.3})
    benchmark = sample_panel["sh600001"]["close"]
    risk = risk_snapshot(weights, sample_panel, benchmark)
    assert risk["beta"] is not None
    # A bare point estimate would imply precision the 60-day window lacks.
    assert risk["beta_se"] is not None and risk["beta_se"] > 0
    assert risk["beta_days"] > 20
    markdown = render_markdown(
        build_brief(sample_panel, "low_vol_core", params=PARAMS, benchmark=benchmark)
    )
    assert "±" in markdown


def _panel_as_of(panel) -> pd.Timestamp:
    return max(frame["close"].dropna().index.max() for frame in panel.values())


def test_data_health_reports_calendar_age(sample_panel) -> None:
    as_of = _panel_as_of(sample_panel)
    health = data_health(sample_panel, today=as_of + pd.Timedelta(days=14))
    assert health["as_of_age_days"] == 14
    assert health["as_of_age_bdays"] > 5


def test_data_health_without_today_leaves_age_empty(sample_panel) -> None:
    health = data_health(sample_panel)
    assert health["as_of_age_days"] is None
    assert health["as_of_age_bdays"] is None


def test_brief_warns_when_the_feed_stops_updating(sample_panel) -> None:
    as_of = _panel_as_of(sample_panel)
    brief = build_brief(
        sample_panel,
        "low_vol_core",
        params=PARAMS,
        today=as_of + pd.Timedelta(days=30),
    )
    assert any("数据源可能未更新" in note for note in brief.notes)


def test_brief_stays_quiet_when_the_feed_is_current(sample_panel) -> None:
    as_of = _panel_as_of(sample_panel)
    brief = build_brief(
        sample_panel,
        "low_vol_core",
        params=PARAMS,
        today=as_of + pd.Timedelta(days=1),
    )
    assert not any("数据源可能未更新" in note for note in brief.notes)
