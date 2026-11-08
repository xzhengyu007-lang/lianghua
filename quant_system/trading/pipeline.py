"""One session of the trading loop, in the order the red lines require.

The order of operations is the safety property, so it is spelled out here and
tested one step at a time:

   1. **kill switch / daily-loss halt**  -- nothing below runs if the system is down
   2. **plan** (``paper.plan_orders``)   -- sells before buys, cash-constrained
   3. **validate** (per order)           -- nine human reasons, all reported at once
   4. **portfolio limits** (batch)       -- single ticket, position, daily exposure, count
   5. **idempotency** (per order)        -- one live order per (day, symbol, side), on disk
   6. **gate**                           -- L1 writes nothing, L2 writes broker files,
                                            L3 additionally records a submit intent
   7. **audit**                          -- every accept / reject / halt with the signal
                                            fingerprint and the price snapshot time

Two decisions worth stating, because the alternative is the common bug:

* Validation failure **rejects that order only**.  One suspended name must not
  block the other seven, and a batch that dies on the first bad row teaches the
  operator to stop reading the errors.
* A *portfolio-level* violation (daily exposure, active-order count) rejects the
  **whole batch**.  Those limits protect the account, not the ticket; trimming the
  batch to fit would silently change the strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from ..config import BacktestConfig, CostConfig
from ..paper import Ledger, apply_fills, plan_orders
from .audit import AuditLog, signal_fingerprint
from .gates import TradingGate
from .idempotency import DuplicateOrderError, IdempotencyBook
from .order_state import Order, OrderState
from .orders_file import BrokerFileResult, write_broker_files
from .risk_rules import RiskLimits, check_limits, daily_loss_halt
from .validate import validate_order


@dataclass
class PlanOutcome:
    trade_date: str
    orders: list[Order] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    risk: dict = field(default_factory=dict)
    gate: dict = field(default_factory=dict)
    written: BrokerFileResult | None = None
    halted: bool = False
    halt_reason: str = ""
    signal_fingerprint: str = ""

    @property
    def ok(self) -> bool:
        return not self.halted and not self.rejected

    def to_dict(self) -> dict:
        return {
            "trade_date": self.trade_date,
            "ok": bool(self.ok),
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "signal_fingerprint": self.signal_fingerprint,
            "gate": self.gate,
            "risk": self.risk,
            "orders": [order.to_dict() for order in self.orders],
            "rejected": self.rejected,
            "warnings": self.warnings,
            "written": self.written.to_dict() if self.written else None,
        }

    def markdown(self) -> str:
        lines = ["## 交易计划（%s）" % self.trade_date, ""]
        gate = self.gate or {}
        lines.append(
            "- 级别 **%s** ｜ kill switch **%s** ｜ 可提交 **%s**"
            % (gate.get("level_label", "—"), "开" if gate.get("kill_switch") else "关",
               "是" if gate.get("can_submit") else "否")
        )
        lines.append("- 信号指纹 `%s`" % (self.signal_fingerprint or "—"))
        if self.halted:
            lines += ["", "### ⛔ 已停机：%s" % self.halt_reason, ""]
            lines.append("本日不生成任何订单。停机后必须人工 `trade --action reset-kill` 才会恢复。")
            return "\n".join(lines) + "\n"
        lines += ["", "### 通过校验的订单", ""]
        if self.orders:
            lines.append("| 标的 | 方向 | 股数 | 限价 | 金额 | 状态 |")
            lines.append("|---|---|---:|---:|---:|---|")
            for order in self.orders:
                lines.append(
                    "| %s | %s | %d | %.2f | %.0f | %s |"
                    % (order.symbol, "买入" if order.side == "buy" else "卖出", order.shares,
                       order.limit_price, order.notional, order.state.value)
                )
        else:
            lines.append("（无）")
        if self.rejected:
            lines += ["", "### 被拒的订单（原因逐条列出，不是静默丢弃）", ""]
            for row in self.rejected:
                lines.append("- **%s %s**：%s" % (row.get("symbol"), row.get("side"), "；".join(row.get("reasons", []))))
        if self.warnings:
            lines += ["", "### 提示", ""]
            for warning in self.warnings:
                lines.append("- %s" % warning)
        if self.written is not None:
            lines += [
                "",
                "### 生成的券商导入文件（L2：只生成，不提交）",
                "",
                "- `%s`" % self.written.csv_path,
                "- `%s`" % self.written.checklist_path,
            ]
        lines += ["", "> 免责：历史回测结果，不构成投资建议。", ""]
        return "\n".join(lines)


def _infer_bought_today(ledger: Ledger | None, trade_date: str) -> dict[str, int]:
    bought: dict[str, int] = {}
    if ledger is None:
        return bought
    for trade in ledger.trades:
        if str(trade.get("date")) == str(trade_date) and str(trade.get("side")) == "buy":
            symbol = str(trade.get("symbol"))
            bought[symbol] = bought.get(symbol, 0) + int(trade.get("shares", 0))
    return bought


def plan_session(
    *,
    ledger: Ledger | None,
    target_weights: pd.Series,
    prices: pd.Series,
    gate: TradingGate,
    trade_date: str | date,
    equity: float,
    book: IdempotencyBook,
    audit: AuditLog,
    quotes=None,
    limits: RiskLimits | None = None,
    lot_size: int = 100,
    costs: CostConfig | None = None,
    max_position_weight: float = 0.25,
    strategy: str = "",
    signal_as_of: str = "",
    snapshot_time: str = "",
    names: dict | None = None,
    output_dir: str | Path | None = None,
    equity_prev_close: float | None = None,
    min_order_value: float = 1_000.0,
) -> PlanOutcome:
    """Run steps 1-7 for one session.  Pure with respect to the ledger (no writes)."""
    day = str(trade_date)
    cfg_limits = limits or RiskLimits()
    cfg_limits.validate()
    fingerprint = signal_fingerprint(signal_as_of or day, target_weights.to_dict(), strategy)
    outcome = PlanOutcome(
        trade_date=day,
        signal_fingerprint=fingerprint,
        gate=gate.describe(day),
    )

    # ---- 1. is the system allowed to trade at all? ------------------------
    if gate.state.kill_switch:
        outcome.halted = True
        outcome.halt_reason = "kill switch 处于开启状态：%s" % (gate.state.halt_reason or "人工停机")
        audit.record(
            "halt", trade_date=day, source="kill-switch", signal_fingerprint=fingerprint,
            level=int(gate.level), detail={"reason": outcome.halt_reason},
        )
        return outcome
    if equity_prev_close:
        halt, message = daily_loss_halt(equity, equity_prev_close, limits=cfg_limits)
        outcome.warnings.append(message)
        if halt:
            gate.kill(message)
            outcome.halted = True
            outcome.halt_reason = message
            outcome.gate = gate.describe(day)
            audit.record(
                "halt", trade_date=day, source="daily-loss", signal_fingerprint=fingerprint,
                detail={"reason": message, "equity": equity, "equity_prev_close": equity_prev_close},
            )
            return outcome

    # ---- 2. plan ---------------------------------------------------------
    if ledger is None:
        outcome.halted = True
        outcome.halt_reason = "没有账本：无法确认现金与持仓，拒绝生成订单"
        return outcome
    planned = plan_orders(
        ledger, target_weights, prices, lot_size=lot_size, costs=costs,
        min_order_value=min_order_value, stale_symbols=set(),
    )
    if planned.empty:
        outcome.warnings.append("没有需要下的单：目标与持仓一致，或全部被现金约束吃掉")
        return outcome

    # ---- 3/4. validate + portfolio limits -------------------------------
    bought_today = _infer_bought_today(ledger, day)
    active = len(book.active())
    candidate_orders: list[Order] = []
    for row in planned.itertuples(index=False):
        quote = quotes.quotes.get(row.symbol) if quotes is not None else None
        result = validate_order(
            symbol=row.symbol, side=row.side, shares=int(row.shares), limit_price=float(row.price),
            trade_date=day, ledger=ledger, quote=quote, lot_size=lot_size,
            min_order_value=min_order_value, costs=costs,
            bought_today=bought_today.get(row.symbol, 0), max_position_weight=max_position_weight,
            equity=equity,
        )
        if not result.ok:
            outcome.rejected.append(
                {"symbol": row.symbol, "side": row.side, "shares": int(row.shares),
                 "price": float(row.price), "reasons": result.reasons}
            )
            audit.record(
                "order.rejected", trade_date=day, symbol=row.symbol, side=row.side,
                shares=int(row.shares), price=float(row.price), source="validate",
                signal_fingerprint=fingerprint, snapshot_time=snapshot_time,
                detail={"reasons": result.reasons},
            )
            continue
        outcome.warnings.extend(result.warnings)
        candidate_orders.append(
            Order(
                order_id="%s-%s-%s-%d" % (day.replace("-", ""), row.symbol, row.side, int(row.shares)),
                trade_date=day, symbol=row.symbol, side=row.side, shares=int(row.shares),
                limit_price=round(float(row.price), 2), reason=str(getattr(row, "reason", "")),
                signal_fingerprint=fingerprint, snapshot_time=snapshot_time,
            )
        )

    risk = check_limits(
        candidate_orders, limits=cfg_limits, equity=equity,
        holdings={s: p.shares for s, p in ledger.positions.items()},
        active_orders=active,
        existing_daily_buys=sum(
            float(t.get("notional", 0.0)) for t in ledger.trades
            if str(t.get("date")) == day and str(t.get("side")) == "buy"
        ),
    )
    outcome.risk = risk.to_dict()
    if not risk.ok:
        portfolio_hits = [v for v in risk.violations if v.scope == "portfolio"]
        if portfolio_hits:
            outcome.halted = True
            outcome.halt_reason = "组合限额被触发：" + "；".join(v.message for v in portfolio_hits)
            audit.record(
                "halt", trade_date=day, source="risk-limits", signal_fingerprint=fingerprint,
                detail={"violations": [v.to_dict() for v in risk.violations]},
            )
            return outcome
        # Per-order violations: drop exactly those, keep the rest.
        blocked = {v.symbol for v in risk.violations if v.symbol}
        for order in list(candidate_orders):
            if order.symbol in blocked:
                reasons = [v.message for v in risk.violations if v.symbol == order.symbol]
                outcome.rejected.append(
                    {"symbol": order.symbol, "side": order.side, "shares": order.shares,
                     "price": order.limit_price, "reasons": reasons}
                )
                audit.record(
                    "order.rejected", trade_date=day, symbol=order.symbol, side=order.side,
                    shares=order.shares, price=order.limit_price, source="risk-limits",
                    signal_fingerprint=fingerprint, detail={"reasons": reasons},
                )
                candidate_orders.remove(order)

    # ---- 5. idempotency --------------------------------------------------
    reserved: list[Order] = []
    for order in candidate_orders:
        try:
            book.reserve(trade_date=day, symbol=order.symbol, side=order.side, order_id=order.order_id)
        except DuplicateOrderError as exc:
            outcome.rejected.append(
                {"symbol": order.symbol, "side": order.side, "shares": order.shares,
                 "price": order.limit_price, "reasons": [str(exc)]}
            )
            audit.record(
                "order.duplicate", trade_date=day, symbol=order.symbol, side=order.side,
                shares=order.shares, price=order.limit_price, source="idempotency",
                signal_fingerprint=fingerprint, detail={"error": str(exc)},
            )
            continue
        reserved.append(order)

    # ---- 6. gate + 7. audit ---------------------------------------------
    can_submit = gate.can_submit(day)
    outcome.gate = gate.describe(day)
    for order in reserved:
        book.update(key=order.key, state=OrderState.NOTTRADED, order_id=order.order_id)
        order.state = OrderState.NOTTRADED
        outcome.orders.append(order)
        audit.record(
            "order.accepted", trade_date=day, symbol=order.symbol, side=order.side,
            shares=order.shares, price=order.limit_price, order_id=order.order_id,
            signal_fingerprint=fingerprint, snapshot_time=snapshot_time,
            level=int(gate.level), source="signal",
            detail={"reason": order.reason, "can_submit": can_submit},
        )

    if can_submit and outcome.orders and output_dir is not None:
        outcome.written = write_broker_files(
            outcome.orders, output_dir, trade_date=day, names=names, lot_size=lot_size,
            signal_fingerprint=fingerprint, snapshot_time=snapshot_time,
            data_as_of=signal_as_of, level_label=gate.level.label,
            extra_notes=[
                "本次通过校验 %d 张、被拒 %d 张（原因见计划文件）"
                % (len(outcome.orders), len(outcome.rejected))
            ],
        )
        audit.record(
            "orders_file.written", trade_date=day, order_id=None, source="gate",
            signal_fingerprint=fingerprint, snapshot_time=snapshot_time,
            level=int(gate.level),
            detail={"csv": str(outcome.written.csv_path), "rows": outcome.written.rows},
        )
    elif outcome.orders and not can_submit:
        outcome.warnings.append(
            "级别 %s：只写入审计与计划，不生成券商导入文件（L2 及以上才会生成）"
            % gate.level.label
        )
    return outcome


def apply_session_fills(
    ledger: Ledger,
    fills: pd.DataFrame,
    *,
    trade_date: str,
    costs: CostConfig | None = None,
) -> Ledger:
    """Thin wrapper so the session loop has one obvious place to book fills."""
    return apply_fills(ledger, fills, date=trade_date, costs=costs)


def default_paths(root: str | Path = Path("reports") / "trading") -> dict[str, Path]:
    base = Path(root)
    return {
        "base": base,
        "gate": base / "gate.json",
        "idempotency": base / "orders.jsonl",
        "audit": base / "audit.jsonl",
        "orders": base / "orders_today",
        "equity": base / "equity.jsonl",
        "recon": base / "recon",
    }
