"""Contract tests for the advice card (stage 2).

The card is the one artifact a human acts on, so the tests are written as the six
mandatory elements plus the hard gate -- one test per promise, and the gate tested
in the direction that matters: it must block a stale day even when every other
input is perfect.

No network, no PIT cache: a synthetic panel plus a temporary ``reports/`` tree
carrying the uncertainty artifacts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_system.advice import (
    DISCLAIMER,
    AdviceCard,
    UncertaintyBlock,
    build_card,
    load_uncertainty,
    write_card,
)
from quant_system.signals import latest_signal

ROOT = Path(__file__).resolve().parent.parent
STALE_AS_OF = "2026-11-05"
LATEST_AS_OF = "2026-11-06"


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _panel(n: int = 300, names: int = 12, seed: int = 3) -> dict[str, pd.DataFrame]:
    """A calm cross-section: low-vol names first, all above their trend line."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2025-01-02", periods=n)
    panel = {}
    for i in range(names):
        sigma = 0.006 + i * 0.0012
        returns = rng.normal(0.0009 - i * 0.00002, sigma, n)
        close = pd.Series(12.0 * np.exp(np.cumsum(returns)), index=index)
        panel["sh6000%02d" % i] = pd.DataFrame(
            {
                "open": close * 1.0005,
                "high": close * 1.004,
                "low": close * 0.996,
                "close": close,
                "volume": 800_000.0 + i * 1000.0,
            },
            index=index,
        )
    return panel


def _artifacts(tmp: Path, *, drop: str | None = None) -> Path:
    """Write the four uncertainty artifacts; ``drop`` removes one to test the gate."""
    reports = tmp / "reports"
    (reports / "year_table_blend").mkdir(parents=True)
    (reports / "phase_recheck").mkdir(parents=True)
    (reports / "power_recheck").mkdir(parents=True)

    if drop != "deployment":
        (reports / "year_table_blend" / "summary.json").write_text(
            json.dumps(
                {
                    "overall": {
                        "blend@12%": {
                            "return": 0.9587019322603088,
                            "sharpe": 0.9020722183394804,
                            "max_drawdown": -0.15702821433502623,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
    if drop != "phases":
        rows = []
        for offset in range(20):
            rows.append(
                {
                    "mode": "year",
                    "offset": offset,
                    "window": "2024-2026",
                    "total_return": 0.0913 if offset == 3 else 0.25,
                    "sharpe": 0.52 if offset == 3 else 0.668,
                    "max_drawdown": -0.12,
                }
            )
        pd.DataFrame(rows).to_csv(reports / "phase_recheck" / "phases.csv", index=False)
    if drop != "power":
        (reports / "power_recheck" / "summary.json").write_text(
            json.dumps(
                {
                    "recorded": {
                        "sharpe_point": 0.901821607911435,
                        "sharpe_low": 0.09763249157530336,
                        "sharpe_high": 1.6772761175556623,
                        "years": 7.142857142857143,
                    },
                    "power": [
                        {"sharpe": 0.9, "years": 6.0, "power": 0.429},
                        {"sharpe": 0.9, "years": 7.3, "power": 0.564},
                        {"sharpe": 0.9, "years": 8.0, "power": 0.615},
                    ],
                }
            ),
            encoding="utf-8",
        )
    return reports


# --------------------------------------------------------------------------- #
# element 3: the numbers, with their uncertainty attached
# --------------------------------------------------------------------------- #
def test_uncertainty_is_read_from_artifacts_not_typed(tmp_path) -> None:
    block = load_uncertainty(_artifacts(tmp_path))
    assert block.complete
    assert block.total_return == pytest.approx(0.9587019322603088)
    assert block.sharpe == pytest.approx(0.9020722183394804)
    assert block.phase_median_sharpe == pytest.approx(0.668)
    assert block.phase_worst_return == pytest.approx(0.0913)
    assert block.ci_low == pytest.approx(0.09763249157530336)
    assert block.ci_high == pytest.approx(1.6772761175556623)
    # nearest computed grid cell, not interpolated
    assert block.power == pytest.approx(0.5642, abs=5e-4)  # 56% after rounding


def test_uncertainty_reports_what_is_missing(tmp_path) -> None:
    block = load_uncertainty(_artifacts(tmp_path, drop="phases"))
    assert block.complete is False
    assert any("phase_recheck" in item for item in block.missing)


def test_headline_never_renders_without_the_caveats() -> None:
    block = UncertaintyBlock(
        total_return=0.9587, sharpe=0.902, max_drawdown=-0.157,
        phase_median_sharpe=0.668, phase_worst_return=0.0913,
        power=0.564, power_years=7.142857, ci_low=0.0976, ci_high=1.677,
    )
    assert "+95.87%" in block.headline()
    caveats = " ".join(block.caveats())
    assert "0.668" in caveats           # phase-median Sharpe
    assert "+9.13%" in caveats          # worst phase
    assert "56%" in caveats             # detection power
    assert "[0.098, 1.677]" in caveats  # bootstrap interval


# --------------------------------------------------------------------------- #
# elements 1, 2, 4, 5 and the disclaimer: the rendered card
# --------------------------------------------------------------------------- #
@pytest.fixture
def card(tmp_path) -> AdviceCard:
    panel = _panel()
    reports = _artifacts(tmp_path)
    return build_card(
        panel,
        strategy="low_vol_core",
        params={"top_n": 4, "rebalance_days": 20},
        equity=200_000.0,
        today=pd.Timestamp("2026-02-20"),
        reports_dir=reports,
        last_trading_day=LATEST_AS_OF if "2026-11-06" in str(panel["sh600000"].index[-1].date()) else None,
    )


def _built_card(tmp_path, *, last_trading_day=None, equity=200_000.0, params=None, quotes=None, min_coverage=0.95):
    panel = _panel()
    return build_card(
        panel,
        strategy="low_vol_core",
        params=params or {"top_n": 4},
        equity=equity,
        today=pd.Timestamp("2026-02-20"),
        reports_dir=_artifacts(tmp_path),
        last_trading_day=last_trading_day,
        quotes=quotes,
        min_coverage=min_coverage,
    )


def test_card_is_not_blocked_on_a_fresh_day(tmp_path) -> None:
    card = _built_card(tmp_path)
    assert card.blocked is False, card.block_reasons
    assert card.items, "a fresh day must produce a basket"


def test_element_1_every_row_has_price_timestamp_shares_and_limit_band(tmp_path) -> None:
    card = _built_card(tmp_path)
    for item in card.items:
        assert item.price > 0
        assert item.price_time, item.symbol
        assert item.shares % item.lot_size == 0
        assert item.shares > 0
        assert item.limit_low < item.price < item.limit_high
        assert item.weight > 0


def test_element_2_each_name_explains_why_it_is_in_the_basket(tmp_path) -> None:
    card = _built_card(tmp_path)
    for item in card.items:
        joined = " ".join(item.reasons)
        assert "低波分位" in joined
        assert "日均线" in joined
        assert "行业内第" in joined
        assert "距上次调仓" in joined
        assert item.score_pct is not None and 0.0 <= item.score_pct <= 1.0
        assert item.trend_gap is not None and item.trend_gap > 0  # these names are above trend
        assert item.sector_rank


def test_element_2_states_the_real_pool_size(tmp_path) -> None:
    """The percentile denominator must come from the signal, not from a template."""
    card = _built_card(tmp_path)
    signal = latest_signal(_panel(), "low_vol_core", params={"top_n": 4}, equity=200_000.0)
    assert signal.pool_size and signal.pool_size > 0
    assert ("%d 只合格横截面" % signal.pool_size) in " ".join(card.items[0].reasons)


def test_element_3_uncertainty_sits_on_the_same_screen_as_the_return(tmp_path) -> None:
    card = _built_card(tmp_path)
    body = card.markdown()
    section_one = body.split("### 2.")[0]
    assert "+95.87%" in section_one
    for marker in ("相位中位 Sharpe 只有 0.668", "最差相位 +9.13%", "56%", "[0.098, 1.677]"):
        assert marker in section_one, marker


def test_element_4_risk_block_has_sizing_caps_and_t1(tmp_path) -> None:
    card = _built_card(tmp_path)
    body = card.markdown()
    assert "目标波动 12%" in body
    assert "单票上限 25%" in body
    assert "行业上限 2 只" in body
    assert any("T+1" in risk for item in card.items for risk in item.risks)


def test_element_5_data_health_block_reports_the_cached_session(tmp_path) -> None:
    card = _built_card(tmp_path)
    body = card.markdown()
    assert "### 2. 数据健康" in body
    assert card.data["as_of"] in body
    assert "覆盖率" in body


def test_element_6_disclaimer_is_present_and_never_collapsed(tmp_path) -> None:
    card = _built_card(tmp_path)
    md = card.markdown()
    html = card.html()
    assert DISCLAIMER in md
    assert DISCLAIMER in html
    assert "不构成投资建议" in html
    # The disclaimer is the last thing on the page: after every foldable block,
    # and nothing in the stylesheet can hide it.
    assert html.find('<section class="disclaimer"') > html.rfind("</details>")
    assert "display:none" not in html
    assert "visibility:hidden" not in html


def test_card_forbids_a_bare_return_number(tmp_path) -> None:
    """+95.87% may not appear anywhere without the uncertainty block beside it."""
    card = _built_card(tmp_path)
    body = card.markdown()
    for occurrence in re.finditer(r"\+?95\.87%", body):
        window = body[max(0, occurrence.start() - 200): occurrence.end() + 600]
        assert "0.668" in window and "56%" in window, window


# --------------------------------------------------------------------------- #
# the hard gate
# --------------------------------------------------------------------------- #
def test_gate_blocks_a_stale_session_even_when_everything_else_is_perfect(tmp_path) -> None:
    card = _built_card(tmp_path, last_trading_day="2026-02-23")  # panel ends 2026-02-20
    assert card.blocked is True
    assert any("上一交易日" in reason for reason in card.block_reasons)
    assert card.items == [], "a blocked card must not carry orders"


def test_gate_blocks_on_low_coverage(tmp_path) -> None:
    card = _built_card(tmp_path, min_coverage=1.01)
    assert card.blocked is True
    assert any("覆盖率" in reason for reason in card.block_reasons)
    assert card.items == []


def test_gate_blocks_when_the_uncertainty_artifacts_are_missing(tmp_path) -> None:
    panel = _panel()
    reports = _artifacts(tmp_path, drop="power")
    card = build_card(
        panel, strategy="low_vol_core", params={"top_n": 4}, equity=200_000.0,
        today=pd.Timestamp("2026-02-20"), reports_dir=reports,
    )
    assert card.blocked is True
    assert any("不确定性" in reason for reason in card.block_reasons)


def test_a_blocked_card_renders_alerts_only(tmp_path) -> None:
    card = AdviceCard(
        as_of="2026-02-20", generated_at="2026-02-20 10:00:00", strategy="low_vol_core",
        equity=200_000.0, data={"as_of": "2026-02-19", "coverage": 0.9},
        uncertainty=UncertaintyBlock(), risk={},
    )
    card.blocked = True
    card.block_reasons = ["缓存最新交易日 2026-02-19 ≠ 上一交易日 2026-02-20"]
    body = card.markdown()
    assert "⛔ 本日不出建议" in body
    assert "2026-02-19" in body
    assert DISCLAIMER in body
    assert "| 标的 |" not in body  # no order table at all


# --------------------------------------------------------------------------- #
# quotes, JSON, and the artifacts on disk
# --------------------------------------------------------------------------- #
class _FakeBoard:
    def __init__(self, quotes, degraded=False, errors=None):
        self.quotes = quotes
        self.degraded = degraded
        self.errors = errors or []


class _FakeQuote:
    def __init__(self, symbol, price, *, stale=False, suspect=False, name="样本", status="normal"):
        from quant_system.realtime import PriceLimit

        self.symbol = symbol
        self.name = name
        self.price = price
        self.quote_time = pd.Timestamp("2026-02-20 14:30:00", tz="Asia/Shanghai")
        self.stale = stale
        self.suspect = suspect
        self.status = status
        self.limit = PriceLimit(0.10, round(price * 1.1, 2), round(price * 0.9, 2), "provider")
        self.message = "现价 14:30:00"


def test_live_quotes_replace_the_close_and_are_stamped(tmp_path) -> None:
    panel = _panel()
    signal = latest_signal(panel, "low_vol_core", params={"top_n": 4}, equity=200_000.0)
    first = signal.rows.iloc[0]
    quote = _FakeQuote(first["symbol"], price=round(float(first["price"]) * 1.01, 2))
    card = _built_card(tmp_path, quotes=_FakeBoard({quote.symbol: quote}))
    item = next(i for i in card.items if i.symbol == quote.symbol)
    assert item.price_kind == "live"
    assert item.price == pytest.approx(quote.price)
    assert "14:30:00" in item.price_time
    assert item.limit_basis == "交易所公布的涨跌停价"


def test_a_stale_quote_is_flagged_and_blocks_buying_advice_on_that_name(tmp_path) -> None:
    panel = _panel()
    signal = latest_signal(panel, "low_vol_core", params={"top_n": 4}, equity=200_000.0)
    first = signal.rows.iloc[0]
    quote = _FakeQuote(first["symbol"], price=float(first["price"]), stale=True)
    card = _built_card(tmp_path, quotes=_FakeBoard({quote.symbol: quote}))
    item = next(i for i in card.items if i.symbol == quote.symbol)
    assert "行情滞后" in item.flags
    assert any("滞后" in risk for risk in item.risks)


def test_a_suspect_quote_tells_the_user_not_to_use_that_price(tmp_path) -> None:
    panel = _panel()
    signal = latest_signal(panel, "low_vol_core", params={"top_n": 4}, equity=200_000.0)
    first = signal.rows.iloc[0]
    quote = _FakeQuote(first["symbol"], price=float(first["price"]), suspect=True)
    card = _built_card(tmp_path, quotes=_FakeBoard({quote.symbol: quote}))
    item = next(i for i in card.items if i.symbol == quote.symbol)
    assert "双源不一致" in item.flags
    assert any("价差" in risk for risk in item.risks)


def test_names_come_from_the_offline_universe(tmp_path) -> None:
    """Synthetic symbols are real codes, so the manifest resolves their names.

    The manifest keys are *bare* codes ("600000"), so a lookup with "sh600000"
    against it silently yields an empty column -- which is how the broker file
    shipped with a blank 证券名称 once.  Normalising the keys is the fix.
    """
    from quant_system.data import normalize_symbol, pit_universe

    manifest = {normalize_symbol(key): value for key, value in pit_universe().items()}
    card = _built_card(tmp_path)
    assert any(item.name for item in card.items), "at least one synthetic code is in the manifest"
    for item in card.items:
        assert item.name == manifest.get(item.symbol, ""), item.symbol


def test_card_json_round_trips_the_six_elements(tmp_path) -> None:
    card = _built_card(tmp_path)
    payload = card.to_dict()
    assert payload["blocked"] is False
    assert payload["uncertainty"]["complete"] is True
    assert payload["disclaimer"] == DISCLAIMER
    assert payload["items"] and all(
        {"price", "price_time", "shares", "limit_low", "limit_high", "reasons", "risks"} <= set(item)
        for item in payload["items"]
    )
    json.dumps(payload, ensure_ascii=False)  # must be serialisable as-is


def test_write_card_emits_three_files(tmp_path) -> None:
    card = _built_card(tmp_path)
    out = write_card(card, tmp_path / "advice")
    assert (out / "card.md").exists()
    assert (out / "card.json").exists()
    assert (out / "card.html").exists()
    html = (out / "card.html").read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "不构成投资建议" in html


def test_the_real_reports_directory_yields_the_documented_numbers() -> None:
    """Ties the card to the repository's own artifacts, not to the fixture."""
    block = load_uncertainty(ROOT / "reports")
    if not block.complete:
        pytest.skip("reports/ artifacts not generated in this checkout")
    assert "%+.2f%%" % (block.total_return * 100) == "+95.87%"
    assert round(block.sharpe, 3) == 0.902
    assert round(block.phase_median_sharpe, 3) == 0.668
    assert round(block.phase_worst_return * 100, 2) == 9.13
    assert round(block.ci_low, 3) == 0.098 and round(block.ci_high, 3) == 1.677
    assert block.power == pytest.approx(0.5642, abs=5e-4)  # 56% after rounding
