"""Tests for the 2x2 calendar-year decomposition (23.86)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_year_table_blend", ROOT / "scripts" / "run_year_table_blend.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


yeartable = _load_module()


def test_decomposition_covers_both_axes():
    assert set(yeartable.KINDS) == {"phase0", "blend"}
    assert set(yeartable.TARGETS) == {0.12, 0.18}


def test_phases_cover_one_full_cycle():
    assert yeartable.PHASES == tuple(range(20))


def _stats():
    path = ROOT / "reports" / "year_table_blend" / "year_stats.csv"
    if not path.exists():
        pytest.skip("run scripts/run_year_table_blend.py first")
    return pd.read_csv(path)


def test_the_losing_years_come_from_the_blend_not_the_target():
    """The whole point of the 2x2: attribute the sign flip to one change."""
    data = _stats()
    for target in ("12%", "18%"):
        blend = data[data["config"] == "blend@" + target]
        single = data[data["config"] == "phase0@" + target]
        assert (blend["return"] < 0).sum() >= 1, target
        assert (single["return"] < 0).sum() == 0, target


def test_every_config_covers_the_same_calendar_years():
    data = _stats()
    years = {config: sorted(block["year"].tolist()) for config, block in data.groupby("config")}
    reference = years["phase0@18%"]
    for config, span in years.items():
        assert span == reference, config
