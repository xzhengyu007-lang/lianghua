"""Tests for the critical-report refresh runner (23.96)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


refresh = _load("refresh_critical", "refresh_critical.py")
audit = _load("audit_staleness", "audit_staleness.py")


def test_registry_covers_exactly_the_audit_critical_set():
    """The runner and the audit must agree, or one of them silently drifts."""
    planned = {name for name, _script, _args in refresh.PLAN}
    assert planned == set(audit.CRITICAL)


def test_no_duplicate_targets():
    names = [name for name, _script, _args in refresh.plan_entries()]
    assert len(names) == len(set(names))


def test_every_registered_script_exists():
    for name, script, _args in refresh.plan_entries():
        assert (ROOT / "scripts" / script).exists(), name


def test_post_steps_include_the_two_self_checks():
    post = {name for name, _script, _args in refresh.POST}
    assert post == {"publish_check", "verification"}


def test_dry_run_exits_cleanly():
    assert refresh.main(["--dry-run"]) == 0


def test_unknown_target_is_rejected():
    assert refresh.main(["--only", "does_not_exist"]) == 2
