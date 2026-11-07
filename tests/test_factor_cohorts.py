"""Tests for the factor cohort decomposition (23.77)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_factor_cohorts", ROOT / "scripts" / "run_factor_cohorts.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cohorts = _load_module()


def test_split_columns_partitions_by_the_code_suffix():
    columns = ["sh600519", "sz000858", "sh600145", "sz000001"]
    listed, gone = cohorts.split_columns(columns)
    assert "sh600519" in listed and "sz000858" in listed
    assert "sh600145" in gone
    assert set(listed) | set(gone) == set(columns)
    assert not (set(listed) & set(gone))


def test_every_column_lands_in_exactly_one_cohort():
    columns = ["sh600000", "sz002415", "sh688981", "sz300750"]
    listed, gone = cohorts.split_columns(columns)
    assert len(listed) + len(gone) == len(columns)


def test_horizons_cover_short_and_long():
    assert 5 in cohorts.HORIZONS and 20 in cohorts.HORIZONS and 60 in cohorts.HORIZONS


def test_report_matches_the_summary_when_present():
    summary = ROOT / "reports" / "factor_cohorts" / "summary.json"
    if not summary.exists():
        import pytest

        pytest.skip("run scripts/run_factor_cohorts.py first")
    import json

    payload = json.loads(summary.read_text(encoding="utf-8"))
    low_vol = [r for r in payload["rows"] if r["factor"] == "low_vol" and r["horizon"] == 20]
    by_cohort = {r["cohort"]: r["mean_ic"] for r in low_vol}
    # The listed cohort must carry less signal than the delisted one (23.76/23.77).
    assert by_cohort["listed"] < by_cohort["delisted"]
    assert by_cohort["listed"] > 0
