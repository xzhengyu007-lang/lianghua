"""Tests for the sample-extent probe (23.121).

The probe itself needs the network and did not work through the direct _request
path (see 23.121), so this only guards the parts that must stay stable: the
candidate dates are ordered and reach back further than the current window.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_sample_extent", ROOT / "scripts" / "run_sample_extent.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_module()


def test_candidate_dates_are_ordered_and_reach_before_the_current_window():
    dates = list(probe.CANDIDATES)
    assert dates == sorted(dates)
    assert dates[0] < "2019-06-06"


def test_the_probe_is_documented_as_not_yet_working():
    """23.121 records a 24/24 failure; the docstring must not claim success."""
    source = (ROOT / "scripts" / "run_sample_extent.py").read_text(encoding="utf-8")
    assert "count=1" in source
    assert "95%" in source
