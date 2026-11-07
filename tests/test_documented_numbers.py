"""Guard: the numbers published in README must match the reports on disk.

README and FINDINGS quote specific metrics ("+43.84%", "1.048", "-16.19%"). They
were transcribed by hand, and one of them silently drifted once already (the
2019-2021 window was quoted with a Sharpe computed from a return series missing
its first day). This test re-reads the report metrics and fails when the
documented value no longer matches.

The reports directory is gitignored, so the test skips when the artifacts are
absent instead of failing on a clean checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

WINDOWS = [
    ("2019-07 .. 2021-12", "reports/wf_pit267_2019_2021"),
    ("2024-01 .. 2026-11", "reports/wf_pit_full267_year"),
    ("2022-01 .. 2023-12", "reports/wf_pit267_2022_trend120"),
]


def _metrics(relative: str) -> dict:
    path = ROOT / relative / "metrics.json"
    if not path.exists():
        pytest.skip(f"{path} not present; run the walk-forward first")
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def _readme_rows() -> dict[str, str]:
    text = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    rows = {}
    for line in text:
        for label, _report in WINDOWS:
            if line.startswith(f"| {label} |"):
                rows[label] = line
    return rows


def test_readme_out_of_sample_table_matches_reports() -> None:
    rows = _readme_rows()
    checked = 0
    for label, report in WINDOWS:
        metrics = _metrics(report)
        row = rows.get(label)
        if row is None:
            pytest.skip(f"README has no row for {label}")
        expected = [
            f"{metrics['total_return'] * 100:+.2f}%",
            f"{metrics['sharpe']:.3f}",
            f"{metrics['max_drawdown'] * 100:.2f}%",
        ]
        for value in expected:
            assert value in row, f"README row for {label} is stale: {value!r} not in {row!r}"
        checked += 1
    assert checked == len(WINDOWS)


def test_findings_section_quotes_the_same_numbers() -> None:
    findings = (ROOT / "FINDINGS.md").read_text(encoding="utf-8")
    for _label, report in WINDOWS:
        metrics = _metrics(report)
        token = f"{metrics['sharpe']:.2f}"
        assert token in findings, f"FINDINGS is missing Sharpe {token} from {report}"


VOL12_WINDOWS = [
    ("2019–2021", "reports/wf_vol12_2019"),
    ("2022–2023", "reports/wf_vol12_2022"),
    ("2024–2026", "reports/wf_vol12_2024"),
]


def test_readme_twelve_percent_rows_match_reports() -> None:
    """The sizing table quotes walk-forward numbers; they drifted once before."""
    text = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    checked = 0
    for label, report in VOL12_WINDOWS:
        metrics = _metrics(report)
        row = next((line for line in text if line.startswith(f"| {label} | 12% |")), None)
        if row is None:
            pytest.skip(f"README has no 12% row for {label}")
        expected = [
            f"{metrics['total_return'] * 100:+.2f}%",
            f"{metrics['annual_volatility'] * 100:.2f}%",
            f"{metrics['sharpe']:.3f}",
            f"{metrics['max_drawdown'] * 100:.2f}%",
        ]
        for value in expected:
            assert value in row, f"README 12% row for {label} is stale: {value!r} not in {row!r}"
        checked += 1
    assert checked == len(VOL12_WINDOWS)


def test_vol_calibration_panel_backs_the_readme_ranges() -> None:
    path = ROOT / "reports" / "vol_calibration" / "panel.csv"
    if not path.exists():
        pytest.skip("run scripts/run_vol_calibration.py first")
    import pandas as pd

    panel = pd.read_csv(path)
    low = panel[panel["target"] == 0.12]["vol_ratio_invested"]
    high = panel[panel["target"] == 0.18]["vol_ratio_invested"]
    assert len(low) == len(high) == 12
    assert low.min() >= 0.95 and low.max() <= 1.14, (low.min(), low.max())
    assert high.min() >= 0.73 and high.max() <= 0.92, (high.min(), high.max())
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "0.95\u20131.14" in readme
    assert "0.74\u20130.91" in readme


def test_cost_interaction_intervals_all_include_zero() -> None:
    path = ROOT / "reports" / "cost_target_interaction" / "bootstrap.csv"
    if not path.exists():
        pytest.skip("run scripts/run_cost_target_interaction.py first")
    import pandas as pd

    boot = pd.read_csv(path)
    assert len(boot) == 12
    assert (boot["low"] < 0).all() and (boot["high"] > 0).all()
    panel = pd.read_csv(ROOT / "reports" / "cost_target_interaction" / "panel.csv")
    wide = panel.pivot_table(index=["scenario", "window"], columns="target", values="max_drawdown")
    assert (wide[0.12] > wide[0.18]).all(), "12% must draw down less in every cell"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "-16.3% vs -22.2%" in readme


def test_findings_covers_the_audit_sections() -> None:
    findings = (ROOT / "FINDINGS.md").read_text(encoding="utf-8")
    for section in ("§23.60", "§23.61", "§23.62", "§23.63", "§23.64", "§23.65"):
        assert section in findings, f"FINDINGS has no row pointing at {section}"

def test_tradability_numbers_back_the_readme() -> None:
    path = ROOT / "reports" / "tradability" / "summary.json"
    if not path.exists():
        pytest.skip("run scripts/run_tradability.py first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    window = next(row for row in payload["frozen"] if row["label"].startswith("2019"))
    assert f"{window['share_days_frozen']:.1%}" in readme, "README misses the frozen-day share"
    assert f"{window['mean_frozen_weight']:.2%}" in readme, "README misses the mean frozen weight"
    assert f"{window['max_frozen_weight']:.1%}" in readme, "README misses the peak frozen weight"


def test_breadth_ic_numbers_back_the_readme() -> None:
    path = ROOT / "reports" / "breadth_ic" / "summary.json"
    if not path.exists():
        pytest.skip("run scripts/run_breadth_ic.py first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for key in ("full", "listed_only", "delisted_only"):
        value = payload[key]["mean_ic"]
        assert f"{value:+.4f}" in readme, f"README misses {key} mean IC {value:+.4f}"
