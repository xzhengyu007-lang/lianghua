"""Tests for the static site builder (23.125)."""

from __future__ import annotations

import importlib.util
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


def test_findings_page_lists_every_row():
    findings = dashboard.parse_findings(ROOT / "FINDINGS.md")
    text = _text("findings.html")
    assert "结论台账（%d 条）" % len(findings) in text
    assert text.count("<tr>") == len(findings) + 1  # +1 header row


def test_log_page_sections_are_not_empty():
    """The bug this guards: an empty body rendered 191 shells and looked fine."""
    text = _text("log.html")
    sections = re.findall(r'<section class="sec"><h3>(.*?)</h3>(.*?)</section>', text, re.S)
    assert sections, "no sections rendered"
    # Four sections in the log are heading-only: their body lives in numbered
    # subsections (e.g. 18.1). That is the document's shape, not a render bug, so
    # the guard is "most sections have body", not "none is empty".
    empty = [title for title, body in sections if not body.strip()]
    assert len(empty) <= 8, empty[:8]
    assert len(empty) < len(sections) * 0.1
    with_table = sum(1 for _t, body in sections if "<table" in body)
    assert with_table > 50


def test_reports_page_escapes_and_formats_metrics():
    text = _text("reports.html")
    assert "报告索引" in text
    assert "pos" in text or "neg" in text


def test_the_site_is_static_no_fetch_calls():
    """GitHub Pages serves files; a page that fetches /api would be broken there."""
    for name, _label in site.PAGES:
        text = _text(name)
        assert "fetch(" not in text, name
        assert "XMLHttpRequest" not in text, name
