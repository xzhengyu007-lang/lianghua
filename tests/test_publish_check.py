"""Tests for the deployable-configuration self-check (23.87)."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_publish_check", ROOT / "scripts" / "run_publish_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


publish = _load_module()


def test_publish_config_is_the_blend_at_twelve_percent():
    assert publish.PHASES == tuple(range(20))
    assert publish.TARGET_VOL == 0.12


def test_fingerprint_accepts_the_recorded_values():
    metrics = SimpleNamespace(**publish.CANONICAL)
    check = publish.check_fingerprint(metrics)
    assert check["ok"] is True
    assert all(abs(v) < publish.TOLERANCE for v in check["drift"].values())


def test_fingerprint_rejects_a_drifted_return():
    drifted = dict(publish.CANONICAL)
    drifted["total_return"] += 1e-6
    check = publish.check_fingerprint(SimpleNamespace(**drifted))
    assert check["ok"] is False
    assert abs(check["drift"]["total_return"] - 1e-6) < 1e-12


def test_fingerprint_covers_every_published_metric():
    assert set(publish.CANONICAL) == {
        "total_return",
        "sharpe",
        "max_drawdown",
        "annual_volatility",
    }
