"""One command for the close-of-day loop: signal → advice → plan → statement → recon.

Red line #9 asks for an automatic end-of-day statement.  The pieces already existed
(``daily_statement``, ``append_equity_point``, ``reconcile``) but nobody had joined
them, so "收盘后会自动生成对账单" was only true if a human remembered five commands
in the right order.  This script is that order, in one place, with a report:

    1. signal      target book from the last close (research path, unchanged)
    2. advice      the six-element card, gated on data freshness
    3. plan        validate → limits → idempotency → gate → audit (pipeline)
    4. statement   today's fills, fees and equity; appends the equity curve point
    5. reconcile   only if a broker statement is supplied -- differences alert
    6. report      reports/daily_ops/<date>/report.md + summary.json

Exit codes matter for a scheduler: ``0`` everything fine, ``1`` something needs a
human (card blocked / plan halted / reconciliation mismatch), ``2`` the run itself
failed.  It never exits 0 while hiding a problem.

Idempotency: re-running the same day is safe.  The plan dedups through
``orders.jsonl``, the equity point is upserted per ``trade_date``, and the report is
overwritten.  Running it twice is how you *check* it, not how you break it.

Usage::

    python scripts/run_daily_ops.py                       # today, L1 paper
    python scripts/run_daily_ops.py --date 2026-11-09 --broker reports/paper/broker.json
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.advice import build_card, write_card  # noqa: E402
from quant_system.brief import data_health  # noqa: E402
from quant_system.config import CACHE_DIR, CostConfig  # noqa: E402
from quant_system.data import load_symbol, normalize_symbol, pit_universe  # noqa: E402
from quant_system.paper import Ledger, apply_fills, plan_orders  # noqa: E402
from quant_system.realtime import (  # noqa: E402
    DEFAULT_STORE,
    SnapshotStore,
    default_source,
    last_completed_session,
    load_board,
    now_cn,
    refresh,
    session_gap,
)
from quant_system.signals import latest_signal  # noqa: E402
from quant_system.trading import (  # noqa: E402
    AuditLog,
    IdempotencyBook,
    RiskLimits,
    TradingGate,
    reconcile,
    signal_fingerprint,
    write_alerts,
)
from quant_system.trading.pipeline import default_paths, plan_session  # noqa: E402
from quant_system.trading.recon import append_equity_point, daily_statement, equity_curve  # noqa: E402


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #
@dataclass
class Step:
    name: str
    status: str = "ok"          # ok | skipped | blocked | halted | mismatch | failed
    detail: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "detail": self.detail, "notes": self.notes}


def load_rebuilt_panel(symbols, cache_dir: Path, adjustment: str = "rebuilt") -> dict[str, pd.DataFrame]:
    """Panel under one explicit price convention (normalized keys!)."""
    panel: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        normalized = normalize_symbol(str(symbol))
        try:
            if adjustment == "cached":
                frame = load_symbol(normalized, cache_dir, adjust="qfq")
            else:
                frame = load_symbol(normalized, cache_dir, adjust="tr")
        except FileNotFoundError:
            continue
        if len(frame) > 200:
            panel[normalized] = frame
    return panel


def run_daily_ops(
    *,
    trade_date: str | None = None,
    equity: float = 1_000_000.0,
    strategy: str = "low_vol_core",
    params: dict | None = None,
    cache_dir: Path = CACHE_DIR,
    universe: str = "pit",
    adjustment: str = "rebuilt",
    reports_dir: Path = ROOT / "reports",
    trading_dir: Path = ROOT / "reports" / "trading",
    ledger_path: Path | None = None,
    broker_path: Path | None = None,
    fills_path: Path | None = None,
    store_path: Path = DEFAULT_STORE,
    use_quotes: bool = True,
    band_bps: float = 50.0,
    init_cash: float | None = None,
    limits: RiskLimits | None = None,
    panel: dict[str, pd.DataFrame] | None = None,
) -> dict:
    """Run the five steps for one session and write the report.  Never raises for a
    business-level problem -- that is what the status fields are for."""
    # An explicit --date means "run this session's close", so it is evaluated at
    # the end of that day; without one the clock decides.  Either way the gate asks
    # which session's bar must already exist -- never "the day before today".
    moment = (
        pd.Timestamp(trade_date) + pd.Timedelta(hours=23, minutes=59)
        if trade_date
        else now_cn()
    )
    # The gate compares against exchange prints, so it wants the aware moment;
    # everything downstream (signals, cash, data health) is naive wall-clock.
    day = pd.Timestamp(moment)
    day = day.tz_localize(None) if day.tzinfo is not None else day
    day = day.normalize()
    day_str = str(day.date())
    last_trading = str(last_completed_session(moment))
    # The report is a statement about the session that just closed, so it is filed
    # under that session even when the run happens after midnight (trade_date stays
    # the decision day: it is what the orders are executed in, and what the
    # idempotency keys and the audit log carry).
    report_date = last_trading
    paths = default_paths(trading_dir)
    gate = TradingGate(paths["gate"])
    audit = AuditLog(paths["audit"])
    book = IdempotencyBook(paths["idempotency"])
    steps: list[Step] = []
    summary: dict = {
        "trade_date": day_str,
        "session_date": report_date,
        "last_trading_day": last_trading,
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": strategy,
        "equity": float(equity),
        "gate": gate.describe(day.date()),
        "steps": steps,
    }

    # ---- 1. signal -------------------------------------------------------
    signal_step = Step("signal")
    if panel is None:                      # test seam: callers may inject a panel
        if universe == "pit":
            symbols = list(pit_universe())
        else:
            from quant_system.config import BROAD_UNIVERSE, DEFAULT_UNIVERSE

            symbols = list(BROAD_UNIVERSE if universe == "broad" else DEFAULT_UNIVERSE)
        panel = load_rebuilt_panel(symbols, Path(cache_dir), adjustment)
    health = data_health(panel, today=day)
    signal_step.detail = {
        "symbols": len(panel),
        "as_of": health["as_of"],
        "coverage": health["coverage"],
        "on_latest_bar": health["on_latest_bar"],
    }
    if health["as_of"] != last_trading:
        signal_step.status = "blocked"
        signal_step.notes.append(
            "缓存最新交易日 %s ≠ 最近已收盘交易日 %s：今天不出建议（先 download；差 %d 个交易日）"
            % (health["as_of"], last_trading, session_gap(health["as_of"], last_trading))
        )
        steps.append(signal_step)
        return _finish(summary, steps, reports_dir, report_date)

    signal = latest_signal(
        panel, strategy, params=params, equity=float(equity), today=day
    )
    (Path(reports_dir) / "signal_today").mkdir(parents=True, exist_ok=True)
    (Path(reports_dir) / "signal_today" / "orders.csv").write_text(
        signal.rows.to_csv(index=False), encoding="utf-8-sig"
    )
    (reports_dir / "signal_today" / "signal.json").write_text(
        json.dumps(signal.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    fingerprint = signal_fingerprint(str(signal.as_of.date()), signal.target_weights.to_dict(), strategy)
    signal_step.detail.update(
        {"names": int(len(signal.target_weights)), "gross": round(float(signal.gross_exposure), 4),
         "fingerprint": fingerprint, "last_rebalance": signal.last_rebalance and str(signal.last_rebalance.date()),
         "bars_since_rebalance": signal.bars_since_rebalance}
    )
    if signal.rows.empty:
        signal_step.notes.append("目标为空（100%% 现金）：%s" % "；".join(signal.warnings or ["无合格标的"]))
    steps.append(signal_step)

    # ---- quotes (optional, for the card) --------------------------------
    board = None
    if use_quotes:
        wanted = list(signal.target_weights.index) or symbols[:8]
        try:
            board = refresh(
                wanted,
                source=default_source(cross_check=True, attempts=1),
                store=SnapshotStore(Path(store_path)),
            )
        except Exception as exc:  # keep going: the card can run on closes
            board = None
            steps[-1].notes.append("行情不可用，卡片改用收盘价：%s" % exc)

    # ---- 2. advice -------------------------------------------------------
    advice_step = Step("advice")
    # The exchange's newest print is evidence the calendar lacks; when it disagrees
    # with the calendar guess (a holiday beyond coverage), the print wins and the
    # disagreement is written into the report instead of silently resolved.
    if board is not None and board.print_date is not None:
        witnessed = str(last_completed_session(moment, witness=board.print_date))
        if witnessed != last_trading:
            signal_step.notes.append(
                "交易所最新打印日 %s 与日历推断的已收盘交易日 %s 不一致，按打印日校验"
                % (board.print_date, last_trading)
            )
            last_trading = witnessed
            summary["last_trading_day"] = last_trading
    card = build_card(
        panel,
        strategy=strategy,
        params=params,
        equity=float(equity),
        quotes=board,
        ledger=_load_ledger(ledger_path, init_cash),
        today=day,
        band_bps=band_bps,
        reports_dir=Path(reports_dir),
        last_trading_day=last_trading,
    )
    write_card(card, Path(reports_dir) / "advice_today")
    advice_step.detail = {
        "blocked": bool(card.blocked),
        "block_reasons": list(card.block_reasons),
        "items": len(card.items),
        "disclaimer": card.disclaimer,
    }
    if card.blocked:
        advice_step.status = "blocked"
    steps.append(advice_step)

    # ---- 3. plan ---------------------------------------------------------
    plan_step = Step("plan")
    ledger = _load_ledger(ledger_path, init_cash)
    if ledger is None:
        plan_step.status = "skipped"
        plan_step.notes.append("没有账本（--ledger 或 --init-cash）：跳过下单")
    elif card.blocked:
        plan_step.status = "skipped"
        plan_step.notes.append("建议卡被闸门拦住：按设计不排单")
    else:
        prices = pd.Series({item.symbol: float(item.price) for item in card.items}, dtype=float)
        for symbol in ledger.positions:                      # held names need a mark too
            if symbol in prices.index and prices[symbol] > 0:
                continue
            mark = _mark_price(symbol, board, panel, Path(cache_dir))
            if mark is not None:
                prices[symbol] = mark
        outcome = plan_session(
            ledger=ledger,
            target_weights=signal.target_weights,
            prices=prices,
            gate=gate,
            trade_date=day_str,
            equity=float(equity),
            book=book,
            audit=audit,
            quotes=board,
            limits=limits or RiskLimits(),
            strategy=strategy,
            signal_as_of=str(signal.as_of.date()),
            snapshot_time=(board.fetched_at.isoformat() if board is not None and board.fetched_at is not None else ""),
            names=_symbol_names(),
            output_dir=paths["orders"],
        )
        expected, attention = split_rejections(outcome.rejected)
        plan_step.detail = {
            "accepted": len(outcome.orders),
            "rejected_expected": len(expected),
            "rejected_needs_attention": len(attention),
            "halted": bool(outcome.halted),
            "halt_reason": outcome.halt_reason,
            "broker_file": bool(outcome.written),
            "attention_detail": attention[:5],
            "expected_detail": expected[:5],
        }
        if outcome.halted:
            plan_step.status = "halted"
            plan_step.notes.append(outcome.halt_reason)
        elif attention:
            plan_step.status = "warn"
            plan_step.notes.append(
                "%d 张被拒且需要人看：%s"
                % (len(attention), "；".join(str(r) for r in (attention[0].get("reasons") or [])))
            )
        elif outcome.orders:
            plan_step.status = "ok"
        else:
            plan_step.status = "skipped"
            if expected:
                plan_step.notes.append(
                    "%d 张被预期内拒绝（T+1 / 幂等 / 低于最小额），今天不需要动作" % len(expected)
                )
        (Path(reports_dir) / "trading").mkdir(parents=True, exist_ok=True)
        (Path(reports_dir) / "trading" / "plan.json").write_text(
            json.dumps(outcome.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        (Path(reports_dir) / "trading" / "plan.md").write_text(outcome.markdown(), encoding="utf-8")
    steps.append(plan_step)

    # ---- 4. statement ----------------------------------------------------
    statement_step = Step("statement")
    ledger = _load_ledger(ledger_path, init_cash)
    if ledger is None:
        statement_step.status = "skipped"
        statement_step.notes.append("没有账本：跳过对账单")
    else:
        if fills_path and Path(fills_path).exists():
            fills = pd.read_csv(fills_path, encoding="utf-8-sig")
            ledger = apply_fills(ledger, fills, date=day_str, costs=CostConfig())
            ledger.save(Path(ledger_path))
            statement_step.notes.append("登记了 %d 笔成交（来自 %s）" % (len(fills), fills_path))
        marks = _marks(ledger, board, Path(cache_dir), panel)
        statement = daily_statement(ledger, marks, trade_date=day_str)
        append_equity_point(paths["equity"], statement)
        curve = equity_curve(paths["equity"])
        statement_step.detail = {
            "equity": statement["equity"],
            "cash": statement["cash"],
            "positions": statement["positions"],
            "fills": statement["fills"],
            "fees_today_total": statement["fees_today_total"],
            "equity_points": int(len(curve)),
        }
        (Path(reports_dir) / "trading").mkdir(parents=True, exist_ok=True)
        (Path(reports_dir) / "trading" / "statement.json").write_text(
            json.dumps({k: v for k, v in statement.items() if k != "trades"},
                       ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    steps.append(statement_step)

    # ---- 5. reconcile ----------------------------------------------------
    recon_step = Step("reconcile")
    ledger = _load_ledger(ledger_path, init_cash)
    if ledger is None or not broker_path or not Path(broker_path).exists():
        recon_step.status = "skipped"
        recon_step.notes.append("没有券商流水（--broker）：本次不对账")
    else:
        broker = json.loads(Path(broker_path).read_text(encoding="utf-8"))
        report = reconcile(
            ledger,
            {str(k): int(v) for k, v in (broker.get("positions") or {}).items()},
            broker_cash=broker.get("cash"),
            tolerance=float(broker.get("tolerance", 0.0)),
            trade_date=day_str,
        )
        write_alerts(report, paths["recon"])
        recon_step.detail = {"ok": report.ok, "differences": report.differences, "alerts": report.alerts}
        recon_step.status = "ok" if report.ok else "mismatch"
    steps.append(recon_step)

    return _finish(summary, steps, reports_dir, report_date)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
# Reasons a rejection is *the rule working*, not a problem to fix: shares bought
# today cannot be sold today, a re-run of the same plan is deduped, and a ticket
# below the broker minimum is refused on purpose.  Anything else ("现金不足",
# "停牌", "限价高于涨停价") is a real signal that a human must look.
EXPECTED_REJECTIONS = ("T+1", "已有一张有效订单", "低于最小下单额")


def split_rejections(rejected: list[dict]) -> tuple[list[dict], list[dict]]:
    expected, attention = [], []
    for row in rejected:
        text = " ".join(str(reason) for reason in (row.get("reasons") or []))
        (expected if any(token in text for token in EXPECTED_REJECTIONS) else attention).append(row)
    return expected, attention
def _symbol_names() -> dict[str, str]:
    """Readable names keyed by normalized symbol -- for any universe.

    ``pit_universe()`` carries all 268 names (current + delisted), so it also covers
    the broad/default pools.  Keying it by normalised symbol matters: the raw keys
    are bare codes, and a "sh600519" lookup against them silently yields blanks.
    """
    try:
        return {normalize_symbol(str(key)): str(value) for key, value in pit_universe().items()}
    except Exception:  # pragma: no cover - a checkout without the manifest
        return {}


def _load_ledger(path: Path | None, init_cash: float | None) -> Ledger | None:
    if path is None:
        return None
    target = Path(path)
    if target.exists():
        return Ledger.load(target)
    if init_cash:
        ledger = Ledger(cash=float(init_cash), positions={}, as_of=None)
        ledger.save(target)
        return ledger
    return None


def _mark_price(symbol: str, board, panel, cache_dir: Path) -> float | None:
    """One price for one held name: live quote, then the panel, then the cache."""
    quote = board.quotes.get(symbol) if board is not None else None
    if quote is not None and quote.price > 0:
        return float(quote.price)
    if panel is not None and symbol in panel:
        try:
            return float(panel[symbol]["close"].dropna().iloc[-1])
        except (IndexError, KeyError):
            pass
    try:
        return float(load_symbol(symbol, cache_dir, adjust="tr")["close"].iloc[-1])
    except (FileNotFoundError, IndexError, KeyError):
        return None


def _marks(ledger: Ledger, board, cache_dir: Path, panel=None) -> pd.Series:
    """Close marks for every held name; a name we cannot price is left out so the
    statement fails loudly (`paper.market_value` raises) instead of inventing one."""
    prices: dict[str, float] = {}
    for symbol in ledger.positions:
        price = _mark_price(symbol, board, panel, cache_dir)
        if price is not None:
            prices[symbol] = price
    return pd.Series(prices, dtype=float)


def _exit_code(steps: list[Step]) -> int:
    statuses = {step.status for step in steps}
    if "failed" in statuses:
        return 2
    if statuses & {"blocked", "halted", "mismatch", "warn"}:
        return 1
    return 0


def _report_markdown(summary: dict) -> str:
    session = summary.get("session_date") or summary["trade_date"]
    title = "# 收盘作业 %s" % session
    if session != summary["trade_date"]:
        title += "（决策日 %s，下一交易日开盘执行）" % summary["trade_date"]
    lines = [title, ""]
    gate = summary["gate"]
    lines.append(
        "- 级别 **%s** ｜ kill switch **%s** ｜ 最近已收盘交易日 %s"
        % (gate["level_label"], "开" if gate["kill_switch"] else "关", summary["last_trading_day"])
    )
    lines.append("")
    labels = {
        "signal": "1. 信号", "advice": "2. 建议卡", "plan": "3. 下单计划",
        "statement": "4. 对账单", "reconcile": "5. 券商对账",
    }
    for step in summary["steps"]:
        lines.append("## %s — %s" % (labels.get(step["name"], step["name"]), _status_zh(step["status"])))
        for key, value in step["detail"].items():
            lines.append("- %s：%s" % (key, value))
        for note in step["notes"]:
            lines.append("- ⚠ %s" % note)
        lines.append("")
    todo = [
        step["name"] for step in summary["steps"]
        if step["status"] in {"blocked", "halted", "mismatch", "failed", "warn"}
    ]
    lines.append("## 需要人做的事")
    lines.append("")
    if todo:
        for name in todo:
            lines.append("- %s：见上" % labels.get(name, name))
    else:
        lines.append("- 无。明天同一时间再跑一次即可。")
    lines.append("")
    lines.append("> 历史回测结果，不构成投资建议。所有数字来自本仓产物，可由 `scripts/run_daily_ops.py` 复跑。")
    return "\n".join(lines) + "\n"


def _status_zh(status: str) -> str:
    return {
        "ok": "✅ 正常", "skipped": "⏭ 跳过", "warn": "⚠ 需人看",
        "blocked": "⛔ 被闸门拦住", "halted": "🛑 停机",
        "mismatch": "❌ 有差异", "failed": "💥 失败",
    }.get(status, status)


def _finish(summary: dict, steps: list[Step], reports_dir: Path, day: str) -> dict:
    summary["steps"] = [step.to_dict() for step in steps]
    summary["exit_code"] = _exit_code(steps)
    target = Path(reports_dir) / "daily_ops" / day
    target.mkdir(parents=True, exist_ok=True)
    (target / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (target / "report.md").write_text(_report_markdown(summary), encoding="utf-8")
    summary["report"] = str(target / "report.md")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="收盘作业：信号→建议→计划→对账单→对账，一条命令")
    parser.add_argument("--date", default=None, help="交易日（YYYY-MM-DD），默认今天")
    parser.add_argument("--equity", type=float, default=1_000_000.0)
    parser.add_argument("--strategy", default="low_vol_core")
    parser.add_argument("--params", default=None, help="策略参数 JSON")
    parser.add_argument("--universe", choices=["pit", "broad", "default"], default="pit")
    parser.add_argument("--adjustment", choices=["rebuilt", "cached"], default="rebuilt")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--reports-dir", default=str(ROOT / "reports"))
    parser.add_argument("--trading-dir", default=str(ROOT / "reports" / "trading"))
    parser.add_argument("--ledger", default=str(ROOT / "reports" / "paper" / "ledger.json"))
    parser.add_argument("--init-cash", type=float, default=None, help="账本不存在时用这个现金初始化")
    parser.add_argument("--fills", default=None, help="当日成交 CSV（登记后出对账单）")
    parser.add_argument("--broker", default=None, help="券商流水 JSON（{cash, positions}）")
    parser.add_argument("--no-quotes", action="store_true", help="不取实时行情（卡片与估值改用收盘价）")
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    args = parser.parse_args(argv)

    try:
        summary = run_daily_ops(
            trade_date=args.date,
            equity=args.equity,
            strategy=args.strategy,
            params=json.loads(args.params) if args.params else None,
            cache_dir=Path(args.cache_dir),
            universe=args.universe,
            adjustment=args.adjustment,
            reports_dir=Path(args.reports_dir),
            trading_dir=Path(args.trading_dir),
            ledger_path=Path(args.ledger) if args.ledger else None,
            broker_path=Path(args.broker) if args.broker else None,
            fills_path=Path(args.fills) if args.fills else None,
            store_path=Path(args.store),
            use_quotes=not args.no_quotes,
            init_cash=args.init_cash,
        )
    except Exception:
        traceback.print_exc()
        return 2

    for step in summary["steps"]:
        print("%-10s %s" % (step["name"], _status_zh(step["status"])), flush=True)
    print("\n报告：%s" % summary["report"], flush=True)
    return int(summary["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
