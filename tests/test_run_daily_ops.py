"""Tests for the close-of-day pipeline (`scripts/run_daily_ops.py`).

Red line #9 says the statement is generated automatically after the close.  These
tests pin the things that make "automatically" believable:

* one command produces a report with all five sections;
* a blocked card or a halted gate is *visible* in the report and in the exit code;
* a reconciliation difference is not swallowed;
* re-running the same day does not double-book anything;
* "the rule working" (T+1 on shares bought today) does not look like a problem,
  while a real refusal (limits, missing cash) does.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from quant_system.paper import Ledger  # noqa: E402
from quant_system.trading import RiskLimits, TradingGate, TradingLevel  # noqa: E402
from run_daily_ops import run_daily_ops, split_rejections, _exit_code, Step  # noqa: E402

TRADE_DATE = "2026-11-06"  # the session the synthetic cache actually covers


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _panel(n: int = 300, names: int = 10, *, last: str = "2026-11-06", seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end=pd.Timestamp(last), periods=n)
    panel = {}
    for i in range(names):
        returns = rng.normal(0.0008 - i * 0.00002, 0.006 + i * 0.0012, n)
        close = pd.Series(12.0 * np.exp(np.cumsum(returns)), index=index)
        panel["sh6000%02d" % i] = pd.DataFrame(
            {"open": close * 1.0005, "high": close * 1.004, "low": close * 0.996,
             "close": close, "volume": 800_000.0},
            index=index,
        )
    return panel


def _artifacts(reports: Path) -> Path:
    """The three uncertainty artifacts the advice card refuses to run without."""
    (reports / "year_table_blend").mkdir(parents=True, exist_ok=True)
    (reports / "phase_recheck").mkdir(parents=True, exist_ok=True)
    (reports / "power_recheck").mkdir(parents=True, exist_ok=True)
    (reports / "year_table_blend" / "summary.json").write_text(
        json.dumps({"overall": {"blend@12%": {"return": 0.9587019322603088,
                                              "sharpe": 0.9020722183394804,
                                              "max_drawdown": -0.15702821433502623}}}),
        encoding="utf-8",
    )
    rows = [
        {"mode": "year", "offset": i, "window": "2024-2026",
         "total_return": 0.0913 if i == 3 else 0.25,
         "sharpe": 0.52 if i == 3 else 0.668, "max_drawdown": -0.12}
        for i in range(20)
    ]
    pd.DataFrame(rows).to_csv(reports / "phase_recheck" / "phases.csv", index=False)
    (reports / "power_recheck" / "summary.json").write_text(
        json.dumps({
            "recorded": {"sharpe_point": 0.901821607911435,
                         "sharpe_low": 0.09763249157530336,
                         "sharpe_high": 1.6772761175556623,
                         "years": 7.142857142857143},
            "power": [{"sharpe": 0.9, "years": 7.3, "power": 0.564}],
        }),
        encoding="utf-8",
    )
    return reports


@pytest.fixture()
def env(tmp_path) -> dict:
    reports = _artifacts(tmp_path / "reports")
    return {
        "reports": reports,
        "trading": reports / "trading",
        "ledger": reports / "paper" / "ledger.json",
        "panel": _panel(),
    }


def _seed_ledger(env, *, symbol: str = "sh600000", shares: int = 1000, cost: float = 10.0,
                 cash: float = 100_000.0) -> Ledger:
    """A ledger that already holds something: planning alone never creates positions."""
    from quant_system.paper import Position

    ledger = Ledger(cash=cash, positions={symbol: Position(shares=shares, avg_cost=cost)},
                    as_of="2026-11-06")
    env["ledger"].parent.mkdir(parents=True, exist_ok=True)
    ledger.save(env["ledger"])
    return ledger


def _run(env, **kwargs) -> dict:
    defaults = dict(
        trade_date=TRADE_DATE,
        equity=200_000.0,
        reports_dir=env["reports"],
        trading_dir=env["trading"],
        ledger_path=env["ledger"],
        init_cash=200_000.0,
        use_quotes=False,
        panel=env["panel"],
        params={"top_n": 4},
    )
    defaults.update(kwargs)
    return run_daily_ops(**defaults)


# --------------------------------------------------------------------------- #
# the happy path
# --------------------------------------------------------------------------- #
def test_the_report_is_filed_under_the_session_that_closed(env) -> None:
    """A run after midnight still belongs to the session that just ended.

    ``trade_date`` stays the decision day (it is what the orders execute in and
    what the idempotency keys carry), but the report is a statement about the
    session that closed -- filing Friday's close under "Saturday" would make two
    different runs of the same session land in two different folders.
    """
    summary = _run(env, trade_date="2026-11-08")  # a Saturday
    assert summary["session_date"] == "2026-11-06", summary
    assert summary["trade_date"] == "2026-11-08"
    assert Path(summary["report"]).parent.name == "2026-11-06"
    report = Path(summary["report"]).read_text(encoding="utf-8")
    assert "# 收盘作业 2026-11-06" in report
    assert "决策日 2026-11-08" in report


def test_a_session_whose_bar_is_missing_blocks_the_whole_run(env) -> None:
    """Monday's close with Friday's cache: no card, no orders, exit code 1."""
    summary = _run(env, trade_date="2026-11-09")
    assert summary["exit_code"] == 1, summary
    signal = next(step for step in summary["steps"] if step["name"] == "signal")
    assert signal["status"] in {"blocked", "warn"}, signal
    assert not (env["trading"] / "orders.jsonl").exists()
    assert summary["session_date"] == "2026-11-09"


def test_one_command_produces_a_report_with_every_section(env) -> None:
    summary = _run(env)
    assert summary["exit_code"] == 0, summary
    report = Path(summary["report"]).read_text(encoding="utf-8")
    for marker in ("# 收盘作业", "## 1. 信号", "## 2. 建议卡", "## 3. 下单计划",
                   "## 4. 对账单", "## 5. 券商对账", "## 需要人做的事"):
        assert marker in report, marker
    assert "不构成投资建议" in report
    assert (env["reports"] / "daily_ops" / TRADE_DATE / "summary.json").exists()


def test_the_signal_and_card_are_written_where_the_desk_reads_them(env) -> None:
    _run(env)
    assert (env["reports"] / "signal_today" / "signal.json").exists()
    assert (env["reports"] / "advice_today" / "card.json").exists()
    card = json.loads((env["reports"] / "advice_today" / "card.json").read_text(encoding="utf-8"))
    assert card["blocked"] is False and card["items"]
    assert card["uncertainty"]["complete"] is True


def test_the_plan_is_recorded_even_when_there_is_nothing_to_do(env) -> None:
    summary = _run(env)
    plan = next(step for step in summary["steps"] if step["name"] == "plan")
    assert plan["status"] in {"ok", "skipped"}
    assert "accepted" in plan["detail"]


def test_the_equity_curve_gets_exactly_one_point_per_day(env) -> None:
    _seed_ledger(env)
    _run(env)
    _run(env)  # same day again must upsert, not append
    curve = (env["trading"] / "equity.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(curve) == 1
    statement = json.loads((env["trading"] / "statement.json").read_text(encoding="utf-8"))
    assert statement["positions"] == 1
    assert statement["equity"] > 0


# --------------------------------------------------------------------------- #
# the paths that must not look fine
# --------------------------------------------------------------------------- #
def test_a_stale_panel_blocks_the_day_and_says_so(env) -> None:
    stale = _panel(last="2026-11-04")  # two sessions behind
    summary = _run(env, panel=stale)
    assert summary["exit_code"] == 1
    signal = summary["steps"][0]
    assert signal["status"] == "blocked"
    assert any("最近已收盘交易日" in note for note in signal["notes"])
    report = Path(summary["report"]).read_text(encoding="utf-8")
    assert "⛔" in report or "被闸门拦住" in report


def test_the_kill_switch_halts_the_plan_but_still_writes_a_report(env) -> None:
    TradingGate(env["trading"] / "gate.json").kill("测试停机")
    summary = _run(env)
    assert summary["exit_code"] == 1
    plan = next(step for step in summary["steps"] if step["name"] == "plan")
    assert plan["status"] == "halted"
    assert "kill switch" in plan["notes"][0]
    assert Path(summary["report"]).exists()


def test_a_reconciliation_difference_is_never_swallowed(env) -> None:
    _seed_ledger(env)
    _run(env)
    ledger = Ledger.load(env["ledger"])
    positions = {symbol: pos.shares for symbol, pos in ledger.positions.items()}
    first = sorted(positions)[0]
    positions[first] += 100                      # broker says we hold more
    broker = env["reports"] / "broker.json"
    broker.write_text(json.dumps({"cash": ledger.cash, "positions": positions}), encoding="utf-8")

    summary = _run(env, broker_path=broker)
    assert summary["exit_code"] == 1
    recon = next(step for step in summary["steps"] if step["name"] == "reconcile")
    assert recon["status"] == "mismatch"
    assert recon["detail"]["alerts"]
    assert (env["trading"] / "recon" / "reconciliation.md").exists()
    # the ledger itself must be untouched
    assert Ledger.load(env["ledger"]).positions[first].shares == positions[first] - 100


def test_a_real_refusal_needs_attention(env) -> None:
    """A tiny per-ticket limit refuses every order -- that is a human's problem."""
    summary = _run(env, limits=RiskLimits(max_order_notional=1_000.0))
    plan = next(step for step in summary["steps"] if step["name"] == "plan")
    assert plan["status"] == "warn"
    assert plan["detail"]["rejected_needs_attention"] > 0
    assert summary["exit_code"] == 1


# --------------------------------------------------------------------------- #
# classification of rejections
# --------------------------------------------------------------------------- #
def test_t_plus_one_and_duplicates_are_expected_not_alarming() -> None:
    expected, attention = split_rejections([
        {"symbol": "a", "reasons": ["T+1：持有 100 股，今日买入 100 股，可卖只有 0 股"]},
        {"symbol": "b", "reasons": ["2026-11-09 b buy 已有一张有效订单（x，状态 nottraded）"]},
        {"symbol": "c", "reasons": ["现金不足：需要 1,000 元（含费用），账本只有 10 元"]},
        {"symbol": "d", "reasons": ["sh600001 停牌，无法成交"]},
    ])
    assert [row["symbol"] for row in expected] == ["a", "b"]
    assert [row["symbol"] for row in attention] == ["c", "d"]


def test_exit_codes_track_the_worst_step() -> None:
    assert _exit_code([Step("a", "ok"), Step("b", "skipped")]) == 0
    assert _exit_code([Step("a", "ok"), Step("b", "warn")]) == 1
    assert _exit_code([Step("a", "blocked")]) == 1
    assert _exit_code([Step("a", "failed")]) == 2


# --------------------------------------------------------------------------- #
# the plan actually trades when the ledger is empty
# --------------------------------------------------------------------------- #
def test_an_empty_ledger_gets_a_full_first_day_of_planned_orders(env) -> None:
    """Planning is not execution: the ledger is untouched until fills are booked.

    This is the property that makes the pipeline safe to re-run -- `plan` writes
    orders and audit, `fills` moves money, and the two are separate commands.
    """
    summary = _run(env, init_cash=200_000.0)
    plan = next(step for step in summary["steps"] if step["name"] == "plan")
    assert plan["status"] == "ok", plan
    assert plan["detail"]["accepted"] > 0
    ledger = Ledger.load(env["ledger"])
    assert ledger.positions == {}, "plan 不落成交，账本必须保持原样"
    assert ledger.cash == pytest.approx(200_000.0)


def test_fills_booked_for_the_day_move_the_ledger_and_the_fee_line(env) -> None:
    """`--fills` is the step that actually trades, and it shows up in the statement."""
    summary = _run(env, init_cash=200_000.0)
    plan = json.loads((env["reports"] / "trading" / "plan.json").read_text(encoding="utf-8"))
    orders = plan["orders"]
    assert orders, "first day should plan something"
    fills = env["reports"] / "fills.csv"
    pd.DataFrame([
        {"symbol": row["symbol"], "side": row["side"], "shares": row["shares"],
         "price": row["limit_price"]}
        for row in orders
    ]).to_csv(fills, index=False, encoding="utf-8-sig")

    summary = _run(env, fills_path=fills)
    statement = next(step for step in summary["steps"] if step["name"] == "statement")
    assert statement["detail"]["positions"] > 0
    assert statement["detail"]["fees_today_total"] > 0
    assert Ledger.load(env["ledger"]).positions
