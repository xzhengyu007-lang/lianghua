"""Pre-trade validation: the nine reasons an order is refused, in plain Chinese.

Every check answers a question the operator would otherwise have to remember at
07:25 with the market about to open:

    cash? T+1 sellable? limit-up/limit-down? suspended? whole lots? price fresh?
    order value? sector cap? single-name cap?

The function returns *all* the reasons, not the first one: a ticket with three
problems should not need three attempts to discover them.  Failure is data, not an
exception -- the caller decides whether to skip one name or stop the whole batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ..config import CostConfig
from ..paper import Ledger, Position


@dataclass
class ValidationResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cash_after: float | None = None

    def __bool__(self) -> bool:
        return self.ok


def cash_delta(side: str, shares: int, price: float, costs: CostConfig) -> float:
    """Signed change in cash: negative for a buy (spend), positive for a sell.

    Signing it once, here, is why the "cash is enough" check cannot be written
    backwards in two places -- the bug that made a 1M buy look affordable against
    50k of cash (found by test_buying_beyond_cash_is_refused_with_the_shortfall).
    """
    notional = float(shares) * float(price)
    commission = max(costs.min_commission, notional * costs.commission_rate)
    transfer = notional * costs.transfer_fee_rate
    if side == "buy":
        return -(notional + commission + transfer)
    stamp = notional * costs.stamp_duty_rate
    return notional - commission - transfer - stamp


def validate_order(
    *,
    symbol: str,
    side: str,
    shares: int,
    limit_price: float,
    trade_date: str,
    ledger: Ledger | None = None,
    quote=None,
    lot_size: int = 100,
    min_order_value: float = 1_000.0,
    costs: CostConfig | None = None,
    bought_today: int = 0,
    frozen: int = 0,
    max_position_weight: float = 0.25,
    equity: float | None = None,
) -> ValidationResult:
    """Check one order.  ``quote`` is a :class:`quant_system.realtime.Quote`."""
    cfg = costs or CostConfig()
    reasons: list[str] = []
    warnings: list[str] = []

    if side not in ("buy", "sell"):
        reasons.append("方向只能是 buy / sell，收到 %r" % side)
    if shares <= 0:
        reasons.append("股数必须为正，收到 %d" % shares)
    elif shares % lot_size:
        reasons.append("不是为了 %d 股整手：%d 股（A 股最小交易单位 100 股）" % (lot_size, shares))
    if limit_price <= 0:
        reasons.append("限价必须为正，收到 %.2f" % limit_price)

    notional = float(shares) * float(limit_price)
    if shares > 0 and limit_price > 0 and notional < min_order_value:
        warnings.append(
            "金额 %.0f 元低于最小下单额 %.0f 元，多数券商会拒单或按最小佣金计费"
            % (notional, min_order_value)
        )

    # ---- quote-dependent checks -----------------------------------------
    if quote is not None:
        if getattr(quote, "price", 0.0) <= 0:
            reasons.append("没有有效报价（%s），不能定价下单" % symbol)
        if getattr(quote, "stale", False):
            reasons.append("行情滞后：%s" % (getattr(quote, "message", "") or "报价不是最新"))
        if getattr(quote, "suspect", False):
            reasons.append("双源价差不一致，今日不要按这个价挂单：%s" % symbol)
        status = getattr(quote, "status", "normal")
        if status == "suspended":
            reasons.append("%s 停牌，无法成交" % symbol)
        elif status == "no_quote":
            reasons.append("%s 无报价（停牌或未上市）" % symbol)
        elif status == "limit_up" and side == "buy":
            reasons.append("%s 涨停：买不到，别把资金挂在封板上" % symbol)
        elif status == "limit_down" and side == "sell":
            reasons.append("%s 跌停：卖不掉，挂单只会排在封单后面" % symbol)
        limit = getattr(quote, "limit", None)
        if limit is not None and limit.up > 0:
            if limit_price > limit.up + 1e-6:
                reasons.append("限价 %.2f 高于涨停价 %.2f" % (limit_price, limit.up))
            if limit_price < limit.down - 1e-6:
                reasons.append("限价 %.2f 低于跌停价 %.2f" % (limit_price, limit.down))

    # ---- position-dependent checks --------------------------------------
    if ledger is not None and side == "sell":
        held = ledger.positions.get(symbol, Position()).shares
        sellable = max(0, held - int(bought_today) - int(frozen))
        if shares > held:
            reasons.append("卖出 %d 股超过持仓 %d 股" % (shares, held))
        elif shares > sellable:
            reasons.append(
                "T+1：持有 %d 股，今日买入 %d 股、冻结 %d 股，可卖只有 %d 股"
                % (held, int(bought_today), int(frozen), sellable)
            )

    cash_after = None
    if ledger is not None:
        delta = cash_delta(side, max(shares, 0), max(limit_price, 0.0), cfg)
        cash_after = float(ledger.cash) + delta
        if side == "buy" and cash_after < -1e-6:
            reasons.append(
                "现金不足：需要 %.0f 元（含费用），账本只有 %.0f 元"
                % (-delta, float(ledger.cash))
            )

    if ledger is not None and side == "buy" and equity:
        held = ledger.positions.get(symbol, Position())
        after = (held.shares * float(limit_price) + notional) / float(equity)
        if after > max_position_weight + 1e-9:
            reasons.append(
                "买入后单票权重 %.1f%% 超过上限 %.0f%%"
                % (after * 100, max_position_weight * 100)
            )

    if trade_date and ledger is not None and ledger.as_of and str(ledger.as_of) > str(trade_date):
        warnings.append(
            "账本日期 %s 晚于本次交易日 %s：请确认不是在重放旧账本" % (ledger.as_of, trade_date)
        )

    return ValidationResult(ok=not reasons, reasons=reasons, warnings=warnings, cash_after=cash_after)
