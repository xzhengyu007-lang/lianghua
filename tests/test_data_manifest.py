"""Tests for the data cache manifest (23.114)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_data_manifest", ROOT / "scripts" / "build_data_manifest.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manifest = _load_module()


def _entry(name, digest="abcdef0123456789", rows=10):
    return {"file": name, "sha256_16": digest, "rows": rows, "start": "2024-01-02", "end": "2024-01-05"}


def test_identical_caches_compare_clean():
    payload = {"files": [_entry("a.parquet")]}
    assert manifest.compare(payload, payload) == {"missing": [], "extra": [], "changed": []}


def test_a_changed_hash_is_reported_with_its_shape():
    current = {"files": [_entry("a.parquet", digest="1111111111111111", rows=11)]}
    stored = {"files": [_entry("a.parquet", digest="2222222222222222", rows=10)]}
    result = manifest.compare(current, stored)
    assert len(result["changed"]) == 1
    assert result["changed"][0]["manifest_rows"] == 10
    assert result["changed"][0]["current_rows"] == 11


def test_missing_and_extra_files_are_separate_from_changes():
    current = {"files": [_entry("new.parquet")]}
    stored = {"files": [_entry("old.parquet")]}
    result = manifest.compare(current, stored)
    assert result["missing"] == ["old.parquet"]
    assert result["extra"] == ["new.parquet"]
    assert result["changed"] == []


def test_the_live_manifest_matches_the_cache():
    path = ROOT / "data" / "cache_manifest.json"
    if not path.exists():
        pytest.skip("run scripts/build_data_manifest.py first")
    if not (ROOT / "data" / "cache").exists():
        pytest.skip("no cache in this checkout")
    stored = json.loads(path.read_text(encoding="utf-8"))
    current = manifest.build(read_dates=False)
    result = manifest.compare(current, stored)
    assert len(result["changed"]) == 0, result["changed"][:3]
    assert len(result["extra"]) == 0, result["extra"][:3]
