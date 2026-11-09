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


def test_the_anchor_override_needs_a_reason_and_a_green_read_of_that_reason(tmp_path, monkeypatch) -> None:
    """Accepting a moved anchor is a decision, not a convenience.

    Three rules, all pinned here: an empty reason changes nothing; a failing check
    elsewhere is not excused by the override; a real acceptance leaves one line in
    the history *and* updates the record.
    """
    record = tmp_path / "last.json"
    history = tmp_path / "anchor_history.jsonl"
    monkeypatch.setattr(run_acceptance, "RECORD", record)
    monkeypatch.setattr(run_acceptance, "ANCHOR_HISTORY", history)
    monkeypatch.setattr(run_acceptance, "CHECKS", [("publish_check", lambda: (True, "all_ok=True"))])
    monkeypatch.setattr(run_acceptance, "_git_state", lambda: {"head": "abc1234", "dirty": False})
    monkeypatch.setattr(run_acceptance, "current_anchor",
                        lambda: {"total_return": 0.9700000000000001, "sharpe": 0.9100000000000001})
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"anchor": {"total_return": 0.9587019322603088,
                                             "sharpe": 0.901821607911435}}), encoding="utf-8")

    # 1. no reason -> the moved anchor still fails, nothing recorded
    payload = run_acceptance.run(record=True)
    assert payload["ok"] is False
    assert "已变" in payload["checks"][-1]["detail"]
    assert not history.exists()

    # 2. a reason, but some other check is red -> still refused
    monkeypatch.setattr(run_acceptance, "CHECKS", [("publish_check", lambda: (False, "mismatch"))])
    payload = run_acceptance.run(record=True, accept_anchor="补了 11-09 的 bar")
    assert payload["ok"] is False
    assert "未确认锚点" in payload["checks"][-1]["detail"]
    assert not history.exists()

    # 3. a reason and everything else green -> accepted, with one audit line
    monkeypatch.setattr(run_acceptance, "CHECKS", [("publish_check", lambda: (True, "all_ok=True"))])
    payload = run_acceptance.run(record=True, accept_anchor="补了 11-09 的 bar")
    assert payload["ok"] is True, payload
    assert "人工确认" in payload["checks"][-1]["detail"]
    entries = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 1
    assert entries[0]["reason"] == "补了 11-09 的 bar"
    assert entries[0]["head"] == "abc1234" and entries[0]["dirty"] is False
    assert entries[0]["previous"]["total_return"] == 0.9587019322603088
    assert json.loads(record.read_text(encoding="utf-8"))["anchor"]["total_return"] == 0.9700000000000001


def test_an_override_without_a_moved_anchor_is_a_no_op(tmp_path, monkeypatch) -> None:
    record = tmp_path / "last.json"
    history = tmp_path / "anchor_history.jsonl"
    monkeypatch.setattr(run_acceptance, "RECORD", record)
    monkeypatch.setattr(run_acceptance, "ANCHOR_HISTORY", history)
    monkeypatch.setattr(run_acceptance, "CHECKS", [("publish_check", lambda: (True, "all_ok=True"))])
    anchor = {"total_return": 0.9587019322603088, "sharpe": 0.901821607911435}
    monkeypatch.setattr(run_acceptance, "current_anchor", lambda: dict(anchor))
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"anchor": anchor}), encoding="utf-8")

    payload = run_acceptance.run(record=True, accept_anchor="其实没变")
    assert payload["ok"] is True
    assert "未变" in payload["checks"][-1]["detail"]
    assert not history.exists(), "锚点没动就不该写历史"
