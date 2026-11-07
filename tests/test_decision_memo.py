"""Guard: the decision memo must match a fresh render of the artifacts.

The memo is the one page someone reads before deciding to fund this. Prose drifts;
a generated file that is compared against its own generator cannot. Every number in
it is re-derived from reports/ by build_decision_memo.collect().
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MEMO = ROOT / "DECISION.md"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_decision_memo", ROOT / "scripts" / "build_decision_memo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


memo = _load_module()


def _payload():
    if not MEMO.exists():
        pytest.skip("run scripts/build_decision_memo.py first")
    return memo.collect()


def test_checked_in_memo_matches_a_fresh_render():
    payload = _payload()
    assert memo.render(payload) == MEMO.read_text(encoding="utf-8")


def test_memo_states_the_deployable_configuration():
    text = MEMO.read_text(encoding="utf-8")
    for token in ("混合 20 相位", "top_n=8", "目标年化波动 12%", "每 20 个交易日"):
        assert token in text, token


def test_memo_quotes_the_interval_not_only_the_point():
    """A point estimate alone is exactly the failure mode 23.82 was about."""
    text = MEMO.read_text(encoding="utf-8")
    assert "自助法 95%" in text
    assert "检出力" in text
    assert "最差相位" in text


def test_memo_lists_explicit_falsification_conditions():
    text = MEMO.read_text(encoding="utf-8")
    assert "什么会推翻" in text
    assert text.count("\n- ") >= 4


def test_deployable_sharpe_is_below_the_lucky_phase():
    payload = _payload()
    full = payload["full_period"]
    assert full["deployable"]["sharpe"] < full["phase0_18"]["sharpe"]


def test_phase_median_sits_below_the_reported_window():
    payload = _payload()
    assert payload["phase_spread"]["median_return"] < payload["window_2024_2026"]["total_return"]
