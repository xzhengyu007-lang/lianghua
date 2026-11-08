"""Tests for the published site: purpose-based pages, no CDN, live optional."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


site = _load("build_site", "build_site.py")
analysis = _load("build_analysis", "build_analysis.py")
dashboard = _load("build_dashboard_for_site", "build_dashboard.py")

DOCS = ROOT / "docs"


def _text(name: str) -> str:
    path = DOCS / name
    if not path.exists():
        pytest.skip("run scripts/build_site.py first")
    return path.read_text(encoding="utf-8")


def test_every_page_is_emitted():
    for name, _label in site.PAGES:
        assert (DOCS / name).exists(), name
    assert (DOCS / ".nojekyll").exists()


def test_the_navigation_is_grouped_by_purpose():
    groups = [g for g, _items in site.NAV]
    assert groups == ["应用", "研究", "档案"]
    apply_pages = [h for h, _l in dict(site.NAV)["应用"]]
    assert apply_pages == ["index.html", "daily.html", "market.html"]


def test_pages_link_to_each_other():
    index = _text("index.html")
    for name, _label in site.PAGES:
        assert 'href="%s"' % name in index


def test_no_page_loads_a_remote_asset():
    for name, _label in site.PAGES:
        text = _text(name)
        assets = re.findall(r'(?:src|href)="(https?://[^"]+)"', text)
        remote = [u for u in assets if "github.io" not in u]
        assert remote == [], (name, remote[:3])


def test_daily_page_ships_holdings_checklist_and_failure_branch():
    text = _text("daily.html")
    assert "const HOLDINGS" in text
    assert "qt.gtimg.cn" in text
    assert "取不到实时数据" in text
    assert "执行检查清单" in text


def test_old_live_url_redirects_to_daily():
    assert "url=daily.html" in _text("live.html")


def test_theme_toggle_is_present_on_every_page():
    for name, _label in site.PAGES:
        text = _text(name)
        assert 'id="theme"' in text, name
        assert "aq-theme" in text, name


def test_deployable_numbers_come_from_the_year_table():
    """Regression: reading a report called *blend* showed an arbitrary study."""
    published = site.deployable()
    assert published, "year_table_blend/summary.json missing"
    assert abs(published["return"] - 0.958701932260305) < 1e-9
    text = _text("index.html")
    assert "+95.87%" in text
    assert "-15.70%" in text


def test_equity_curve_is_built_from_daily_returns():
    curve = site.deployable_curve(max_points=50)
    assert len(curve) == 50
    assert curve[-1][1] > 0


def test_market_page_reports_breadth_and_cadence_semantics():
    text = _text("market.html")
    assert "涨 / 跌" in text
    assert "MA60 上方" in text
    snapshot = ROOT / "reports" / "market" / "snapshot.json"
    if snapshot.exists():
        payload = json.loads(snapshot.read_text(encoding="utf-8"))
        assert "cadence" in payload
        # 20 interleaved phases: a single "next rebalance" would be a lie
        assert "next_phase0_rebalance" in payload["cadence"]
        assert "每个交易日都有一个相位在调仓" in payload["cadence"]["note"]


def test_findings_page_has_filters_and_every_row():
    findings = dashboard.parse_findings(ROOT / "FINDINGS.md")
    text = _text("findings.html")
    assert 'data-f="high"' in text and 'data-f="no"' in text
    assert text.count("<tr data-tag=") == len(findings)


def test_log_page_is_collapsible_and_searchable():
    text = _text("log.html")
    assert text.count("<details") > 100
    assert 'id="q"' in text
    assert "全部展开" in text


def test_analysis_page_renders_the_commentary():
    payload = analysis.analyse()
    text = _text("analysis.html")
    for section in payload["sections"]:
        assert section["heading"] in text


def test_each_page_carries_one_active_nav_item():
    for name, _label in site.PAGES:
        text = _text(name)
        assert text.count('class="on"') >= 1, name
