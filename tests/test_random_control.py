"""Tests for the random-selection control (23.102)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_random_control", ROOT / "scripts" / "run_random_control.py"
    )
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations via sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


control = _load_module()


def _panel(symbols=("sh600000", "sh600001", "sh600002"), days=200):
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2024-01-02", periods=days)
    panel = {}
    for symbol in symbols:
        close = pd.Series(10.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, days))), index=index)
        volume = pd.Series(rng.integers(1000, 5000, days).astype(float), index=index)
        panel[symbol] = pd.DataFrame(
            {
                "open": close.shift(1).fillna(close),
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": volume,
            }
        )
    return panel


def test_the_control_keeps_eligibility_identical_to_the_real_strategy():
    """Only the ranking may differ; the eligible set must be byte-identical."""
    panel = _panel()
    real = control.LowVolCoreStrategy(**control.BASE)
    rand = control.RandomSelection(seed=1, **control.BASE)
    _, eligible_real = real.score(panel)
    _, eligible_rand = rand.score(panel)
    assert eligible_real.equals(eligible_rand)


def test_the_control_replaces_the_ranking():
    panel = _panel()
    real = control.LowVolCoreStrategy(**control.BASE)
    rand = control.RandomSelection(seed=1, **control.BASE)
    scores_real, _ = real.score(panel)
    scores_rand, _ = rand.score(panel)
    assert not scores_real.equals(scores_rand)


def test_the_control_is_seed_reproducible():
    panel = _panel()
    first, _ = control.RandomSelection(seed=42, **control.BASE).score(panel)
    second, _ = control.RandomSelection(seed=42, **control.BASE).score(panel)
    assert first.equals(second)


def _summary():
    path = ROOT / "reports" / "random_control" / "summary.json"
    if not path.exists():
        pytest.skip("run scripts/run_random_control.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def test_no_random_draw_reaches_the_real_strategy():
    payload = _summary()
    assert payload["random"]["sharpe_max"] < payload["real"]["sharpe"]
    assert payload["real_percentile_among_random"] == 1.0


def test_the_random_book_is_worse_on_every_axis():
    payload = _summary()
    random_stats = payload["random"]
    assert random_stats["cagr_median"] < payload["real"]["cagr"]
    assert random_stats["max_drawdown_median"] < payload["real"]["max_drawdown"]
    assert random_stats["turnover_median"] > payload["real"]["turnover"]
