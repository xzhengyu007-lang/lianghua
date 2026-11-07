"""Tests for the decision-page claim audit (23.106)."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "audit_decision_claims", ROOT / "scripts" / "audit_decision_claims.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


claims = _load_module()

SCANNER = re.compile(r"[+-]\d+\.\d{2}%|\b\d\.\d{3}\b")


def test_a_missing_section_is_flagged(tmp_path):
    result = claims.audit("见 §23.9", "### 23.1 x", "", tmp_path)
    assert result["missing_sections"] == ["23.9"]


def test_a_present_section_is_accepted(tmp_path):
    result = claims.audit("见 §23.9", "### 23.9 存在", "", tmp_path)
    assert result["missing_sections"] == []


def test_a_missing_report_is_flagged(tmp_path):
    result = claims.audit("见 reports/nope", "", "", tmp_path)
    assert result["missing_reports"] == ["nope"]


def test_an_orphan_figure_is_flagged(tmp_path):
    result = claims.audit("收益 +12.34%", "", "无关语料", tmp_path)
    assert result["orphan_figures"] == ["+12.34%"]


def test_a_figure_found_in_the_corpus_is_accepted(tmp_path):
    result = claims.audit("收益 +12.34%", "", "某文件里写着 +12.34%", tmp_path)
    assert result["orphan_figures"] == []


def test_every_whitelist_entry_is_reachable_by_the_scanner():
    """A whitelist entry the scanner never produces is silently useless."""
    unreachable = [key for key in claims.DERIVED if not SCANNER.search(key)]
    assert unreachable == [], unreachable


def test_the_live_decision_page_passes():
    memo = ROOT / "DECISION.md"
    if not memo.exists():
        pytest.skip("DECISION.md is missing")
    result = claims.audit(
        memo.read_text(encoding="utf-8"),
        (ROOT / "RESEARCH_LOG.md").read_text(encoding="utf-8"),
        claims._corpus(),
        ROOT / "reports",
    )
    assert result["missing_sections"] == []
    assert result["orphan_figures"] == []
