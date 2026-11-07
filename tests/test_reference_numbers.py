"""Guard: the README's authoritative reference table must match the reports.

README now states which numbers may be quoted and which are historical. Those
numbers drifted repeatedly during one session (23.82 lowered the baseline, 23.85
changed the target, 23.86 changed the full-period figure), and every drift was
found by hand. This test re-derives each cell from the artifacts so a stale cell
fails instead of shipping.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def _overall() -> dict:
    path = ROOT / "reports" / "year_table_blend" / "summary.json"
    if not path.exists():
        pytest.skip("run scripts/run_year_table_blend.py first")
    return json.loads(path.read_text(encoding="utf-8"))["overall"]


def test_reference_table_exists():
    readme = _readme()
    assert "## 引用口径（唯一权威，有测试守着）" in readme
    for header in ("部署口径：混合 + 12% 目标波动", "混合 + 18%", "单相位 @18%（旧默认，运气相位）"):
        assert header in readme


def test_full_period_cells_match_the_year_table():
    readme = _readme()
    overall = _overall()
    for label in ("blend@12%", "blend@18%", "phase0@18%"):
        stats = overall[label]
        assert ("%+.2f%%" % (stats["return"] * 100)) in readme, label
        assert ("%.3f" % stats["sharpe"]) in readme, label
        assert ("%.2f%%" % (stats["max_drawdown"] * 100)) in readme, label


def test_window_cells_match_the_vol_target_panel():
    path = ROOT / "reports" / "voltarget_blend" / "panel.csv"
    if not path.exists():
        pytest.skip("run scripts/run_voltarget_blend.py first")
    panel = pd.read_csv(path)
    readme = _readme()
    for target in (0.12, 0.18):
        row = panel[(panel["window"] == "2024-2026") & (panel["target"] == target)].iloc[0]
        assert ("%+.2f%%" % (row["total_return"] * 100)) in readme, target


def test_phase_median_and_worst_match_the_phase_recheck():
    path = ROOT / "reports" / "phase_recheck" / "phases.csv"
    if not path.exists():
        pytest.skip("run scripts/run_phase_recheck.py first")
    phases = pd.read_csv(path)
    window = phases[(phases["window"] == "2024-2026") & (phases["mode"] == "year")]
    readme = _readme()
    assert ("%+.2f%%" % (window["total_return"].median() * 100)) in readme
    assert ("%+.2f%%" % (window["total_return"].min() * 100)) in readme
    assert ("%.3f" % window["sharpe"].median()) in readme


def test_the_lucky_phase_caveat_is_still_stated():
    readme = _readme()
    assert "相位中位" in readme
    assert "最差相位" in readme
