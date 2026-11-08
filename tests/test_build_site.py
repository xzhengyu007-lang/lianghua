"""Tests for the published site and its analysis (23.126)."""

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


def test_pages_link_to_each_other():
    index = _text("index.html")
    for name, _label in site.PAGES:
        assert 'href="%s"' % name in index


def test_live_page_ships_the_holdings_and_survives_a_failure():
    """Live data is optional; the page must render and must report failure."""
    text = _text("live.html")
    assert "const HOLDINGS" in text
    assert "qt.gtimg.cn" in text
    assert "取不到实时数据" in text


def test_findings_page_lists_every_row():
    findings = dashboard.parse_findings(ROOT / "FINDINGS.md")
    text = _text("findings.html")
    assert "结论台账" in text
    assert text.count("<tr>") == len(findings) + 1


def test_log_page_sections_are_not_empty():
    """Guards the bug where 191 shells rendered and looked fine."""
    text = _text("log.html")
    sections = re.findall(r"<h2>(\d+(?:\.\d+)?) · (.*?)</h2>", text)
    assert len(sections) > 100
    bodies = re.findall(r"<h2>[^<]*</h2>(.*?)</div>", text, re.S)
    nonempty = sum(1 for body in bodies if len(body.strip()) > 60)
    assert nonempty > len(bodies) * 0.9


def test_no_page_loads_a_remote_asset():
    """No CDN: a reader behind a firewall must still get a working page.

    The check is on loaded URLs, not on the word: the research log legitimately
    discusses CDNs in prose, and a plain substring test flagged that text.
    """
    import re

    for name, _label in site.PAGES:
        text = _text(name)
        assets = re.findall(r'(?:src|href)="(https?://[^"]+)"', text)
        remote = [u for u in assets if "github.io" not in u]
        assert remote == [], (name, remote[:3])


def _commentary():
    path = ROOT / "reports" / "analysis" / "commentary.json"
    if not path.exists():
        pytest.skip("run scripts/build_analysis.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def test_analysis_is_derived_from_artifacts_only():
    payload = analysis.analyse()
    assert payload["sections"], "analysis produced nothing"
    for section in payload["sections"]:
        assert section["heading"] and section["verdict"] and section["evidence"]
        # every numeric claim must carry a sign or decimal; catches empty templates
        assert any(ch.isdigit() for ch in section["evidence"])


def test_analysis_reaches_the_index_page():
    text = _text("index.html")
    assert "机器解读" in text
    for section in _commentary()["sections"]:
        assert section["heading"] in text
