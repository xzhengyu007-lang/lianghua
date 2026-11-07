"""Tests for the 52-week-high overlay (23.78)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_52w_overlay", ROOT / "scripts" / "run_52w_overlay.py"
    )
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules[cls.__module__], so the
    # module has to be registered before it is executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


overlay = _load_module()


def test_base_pins_every_other_overlay_to_zero():
    """If BASE ever turns on illiquidity or momentum, the control breaks."""
    assert overlay.BASE["illiquidity_weight"] == 0.0
    assert overlay.BASE["momentum_weight"] == 0.0
    assert overlay.BASE["vol_weight"] == 0.8
    assert overlay.BASE["trend_weight"] == 0.2


def test_overlay_defaults_to_no_extra_signal():
    strategy = overlay.High52wOverlay(**overlay.BASE)
    assert strategy.high_52w_weight == 0.0
    assert strategy.high_52w_window == 252


def test_windows_are_ordered_and_disjoint():
    spans = [overlay.WINDOWS[key] for key in ("2019-2021", "2022-2023", "2024-2026")]
    assert len(spans) == 3
    for start, end in spans:
        assert start < end
    assert spans[0][1] < spans[1][0]
    assert spans[1][1] < spans[2][0]


def test_overlay_is_a_low_vol_plus_variant():
    from quant_system.strategies import LowVolPlusStrategy

    assert issubclass(overlay.High52wOverlay, LowVolPlusStrategy)
