"""Tests for the artifact fingerprint check (23.99)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "check_artifact_fingerprints", ROOT / "scripts" / "check_artifact_fingerprints.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fingerprints = _load_module()


def test_targets_cover_the_five_upstream_studies():
    assert set(fingerprints.TARGETS) == {
        "factor_cohorts",
        "breadth_ic",
        "pnl_cohorts",
        "selection_slack",
        "phase_recheck",
    }


def test_compare_flags_a_changed_digest():
    current = {"r": {"a.json": "newdigest"}}
    baseline = {"r": {"a.json": "olddigest"}}
    problems = fingerprints.compare(current, baseline)
    assert len(problems) == 1
    assert problems[0]["kind"] == "changed"
    assert problems[0]["report"] == "r"


def test_compare_flags_a_missing_artifact():
    problems = fingerprints.compare({"r": {"a.json": None}}, {"r": {"a.json": "x"}})
    assert problems and problems[0]["kind"] == "missing_artifact"


def test_compare_flags_a_file_absent_from_the_baseline():
    problems = fingerprints.compare({"r": {"a.json": "x"}}, {})
    assert problems and problems[0]["kind"] == "not_in_baseline"


def test_compare_is_silent_when_everything_matches():
    same = {"r": {"a.json": "x", "b.csv": "y"}}
    assert fingerprints.compare(same, same) == []


def test_checked_in_artifacts_match_the_baseline():
    """The live invariant: today's artifacts are the ones the baseline recorded."""
    path = ROOT / "data" / "artifact_fingerprints.json"
    if not path.exists():
        pytest.skip("run scripts/check_artifact_fingerprints.py --update first")
    stored = json.loads(path.read_text(encoding="utf-8"))["hashes"]
    current = fingerprints.scan()
    if any(value is None for files in current.values() for value in files.values()):
        pytest.skip("some artifacts are absent; run the studies first")
    assert fingerprints.compare(current, stored) == []
