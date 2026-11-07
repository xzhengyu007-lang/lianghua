"""Tests for the dashboard builder (scripts/build_dashboard.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_dashboard", ROOT / "scripts" / "build_dashboard.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dashboard = _load_module()


def test_downsample_keeps_endpoints_and_respects_the_limit():
    values = list(range(1000))
    kept = dashboard.downsample(values, 100)
    assert len(kept) == 100
    assert kept[0] == 0
    assert kept[-1] == 999
    assert kept == sorted(kept)


def test_downsample_is_a_noop_below_the_limit():
    assert dashboard.downsample([1, 2, 3], 10) == [1, 2, 3]


def test_findings_parse_into_numbered_rows():
    rows = dashboard.parse_findings(ROOT / "FINDINGS.md")
    assert len(rows) >= 40
    assert rows[0]["id"] == 1
    assert all(isinstance(r["id"], int) for r in rows)
    assert all(r["claim"] and r["confidence"] for r in rows)


def test_log_sections_have_unique_ids_and_rendered_body():
    sections = dashboard.parse_log(ROOT / "RESEARCH_LOG.md")
    assert len(sections) > 50
    ids = [s["id"] for s in sections]
    assert len(ids) == len(set(ids))
    latest = next(s for s in sections if s["id"] == "23.76")
    assert "<table>" in latest["html"]
    assert "<h4>" in latest["html"]


def test_markdown_renders_tables_lists_and_bold():
    html = dashboard.render_markdown(
        [
            "### 标题",
            "",
            "| a | b |",
            "|---|---:|",
            "| 1 | 2 |",
            "",
            "- 第一项",
            "- 第二项",
            "",
            "正文 **加粗** 与 TICKcodeTICK。".replace("TICK", chr(96)),
        ]
    )
    assert "<h3>标题</h3>" in html
    assert "<table>" in html
    assert "<td>1</td>" in html
    assert html.count("<li>") == 2
    assert "<strong>加粗</strong>" in html
    assert "<code>code</code>" in html


def test_table_separator_row_is_not_rendered_as_data():
    html = dashboard.render_table(["| a | b |", "|---|---|", "| 1 | 2 |"])
    assert html.count("<tr>") == 1 + 1  # header row + one data row
    assert "---" not in html
