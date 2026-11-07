from __future__ import annotations

import json

from scripts.build_report_index import build


def _report(root, name, **metrics):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "metrics.json").write_text(
        json.dumps({"metrics": metrics}, ensure_ascii=False), encoding="utf-8"
    )


def test_index_reads_metrics_and_skips_the_rest(tmp_path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    _report(reports, "run_a", total_return=0.5, sharpe=1.2, max_drawdown=-0.1, calmar=5.0)
    _report(reports, "run_b", total_return=-0.2, sharpe=-0.4, max_drawdown=-0.3, calmar=-0.7)
    (reports / "empty_dir").mkdir()
    (reports / "runs").mkdir()

    output, count = build(reports)
    text = output.read_text(encoding="utf-8")
    assert count == 2
    assert "`run_a`" in text and "`run_b`" in text
    assert "+50.00%" in text and "-20.00%" in text
    assert "empty_dir" not in text
    assert "`runs`" not in text


def test_index_lists_multi_window_reports_separately(tmp_path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    _report(reports, "run_a", total_return=0.5, sharpe=1.2, max_drawdown=-0.1, calmar=5.0)
    panel = reports / "param_panel"
    panel.mkdir()
    (panel / "report.md").write_text("# 三窗口对照\n", encoding="utf-8")

    output, count = build(reports)
    text = output.read_text(encoding="utf-8")
    assert count == 1
    assert "多窗口研究目录" in text
    assert "`param_panel`" in text
    # Metrics table and analysis section must not be confused with each other.
    metrics_table, analysis_table = text.split("## 多窗口研究目录")
    assert "`param_panel`" not in metrics_table
    assert "`run_a`" not in analysis_table
