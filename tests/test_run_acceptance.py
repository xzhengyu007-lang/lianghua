"""Tests for the acceptance runner.

The runner shells out to pytest and to the publish check, so the tests here cover
the parts that decide the verdict -- the desk smoke check, the anchor comparison,
the check list itself -- and never launch the slow checks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_acceptance  # noqa: E402


def test_the_check_list_covers_every_promise() -> None:
    names = [name for name, _fn in run_acceptance.CHECKS]
    assert names == ["pytest", "publish_check", "artifact_fingerprints", "staleness", "desk_smoke"]


def test_the_desk_smoke_check_starts_and_stops_by_itself() -> None:
    ok, detail = run_acceptance.check_desk()
    assert ok is True, detail
    assert "gate=" in detail
    assert "NaN" not in detail


def test_the_anchor_check_records_the_first_run_and_then_compares() -> None:
    anchor = run_acceptance.current_anchor()
    if anchor is None:
        pytest.skip("run scripts/run_publish_check.py first")

    ok, detail = run_acceptance.check_anchor(None)
    assert ok is True and "首次验收" in detail

    same = {"total_return": anchor["total_return"], "sharpe": anchor["sharpe"]}
    ok, detail = run_acceptance.check_anchor(same)
    assert ok is True and "未变" in detail

    moved = {"total_return": anchor["total_return"] + 1e-6, "sharpe": anchor["sharpe"]}
    ok, detail = run_acceptance.check_anchor(moved)
    assert ok is False and "已变" in detail


def test_a_missing_summary_is_a_failed_anchor_check(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run_acceptance, "ROOT", tmp_path)
    ok, detail = run_acceptance.check_anchor(None)
    assert ok is False and "指纹" in detail


def test_the_record_is_only_written_when_everything_passes(tmp_path, monkeypatch) -> None:
    record = tmp_path / "last.json"
    monkeypatch.setattr(run_acceptance, "RECORD", record)
    monkeypatch.setattr(run_acceptance, "CHECKS", [("always_ok", lambda: (True, "fine"))])
    monkeypatch.setattr(run_acceptance, "current_anchor", lambda: None)

    payload = run_acceptance.run(record=True)
    assert payload["ok"] is True
    assert record.exists()
    assert json.loads(record.read_text(encoding="utf-8"))["ok"] is True

    record.unlink()
    monkeypatch.setattr(run_acceptance, "CHECKS", [("always_bad", lambda: (False, "broke"))])
    payload = run_acceptance.run(record=True)
    assert payload["ok"] is False
    assert not record.exists(), "失败的一次不许覆盖上次通过的记录"


def test_a_check_that_raises_counts_as_failed(tmp_path, monkeypatch) -> None:
    def boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(run_acceptance, "RECORD", tmp_path / "last.json")
    monkeypatch.setattr(run_acceptance, "CHECKS", [("boom", boom)])
    monkeypatch.setattr(run_acceptance, "current_anchor", lambda: None)
    payload = run_acceptance.run(record=False)
    assert payload["ok"] is False
    assert "RuntimeError" in payload["checks"][0]["detail"]
