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


def test_memo_opens_with_the_three_things_that_matter():
    """If a reader stops after three lines, they must still be correctly informed."""
    text = MEMO.read_text(encoding="utf-8")
    assert "如果你只记得三件事" in text
    assert "相位中位 Sharpe" in text
    assert "「有效」不等于「已被证明」" in text


def test_memo_carries_the_parts_ledger():
    """The description must match where the contribution actually comes from."""
    text = MEMO.read_text(encoding="utf-8")
    assert "零件账" in text
    assert "趋势过滤" in text
    assert "几乎是装饰" in text
    assert "§23.110" in text and "§23.111" in text


def test_memo_does_not_overclaim_against_the_obvious_alternative():
    """23.102's random-control win must sit next to the naive-benchmark tie."""
    text = MEMO.read_text(encoding="utf-8")
    assert "最自然的替代品" in text
    assert "6 格中 0 格显著" in text


def test_memo_tabulates_every_route_to_more_power():
    """The three falsified routes must stay visible next to the two live ones."""
    text = MEMO.read_text(encoding="utf-8")
    assert "只有下面五条路" in text
    for section in ("§23.89", "§23.93", "§23.91", "§23.101", "§23.104"):
        assert section in text, section
    assert text.count("不是杠杆") >= 3


def test_memo_says_where_the_value_comes_from():
    """23.102's positive result must sit next to 23.40/23.77's caveat."""
    text = MEMO.read_text(encoding="utf-8")
    assert "这台机器值在哪" in text
    assert "§23.102" in text
    assert "§23.40" in text


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
