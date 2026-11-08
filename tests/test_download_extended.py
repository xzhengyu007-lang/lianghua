"""Tests for the resumable extended downloader (23.124)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "download_extended", ROOT / "scripts" / "download_extended.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


extended = _load_module()


def _write(cache: Path, symbol: str, suffix: str, rows: int) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    index = pd.bdate_range("2010-01-04", periods=rows)
    pd.DataFrame({"close": 1.0}, index=index).to_parquet(cache / (symbol + suffix))


def test_depth_reads_whichever_suffix_is_present(tmp_path):
    _write(tmp_path, "sh600000", "_raw.parquet", 4000)
    assert extended.depth(tmp_path, "sh600000") == 4000


def test_depth_is_zero_for_an_absent_symbol(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    assert extended.depth(tmp_path, "sh600000") == 0


def test_depth_takes_the_deepest_of_several_suffixes(tmp_path):
    _write(tmp_path, "sh600000", "_qfq.parquet", 1800)
    _write(tmp_path, "sh600000", "_raw.parquet", 4000)
    assert extended.depth(tmp_path, "sh600000") == 4000


def test_plan_splits_on_the_target_depth(tmp_path):
    """The whole point: names already deep enough must not be re-requested."""
    _write(tmp_path, "sh600036", "_raw.parquet", 4000)
    have, need = extended.plan(tmp_path, 4000)
    assert "sh600036" in have
    assert "sh600036" not in need
    assert len(have) + len(need) > 0


def test_dry_run_does_not_touch_the_network(tmp_path, capsys):
    cache = tmp_path / "cache"
    cache.mkdir()
    code = extended.main(["--cache-dir", str(cache), "--dry-run",
                          "--output", str(tmp_path / "out")])
    assert code == 0
    assert "dry run" in capsys.readouterr().out
