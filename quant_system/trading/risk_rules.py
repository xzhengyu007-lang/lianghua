"""Portfolio-level limits and the daily loss circuit breaker.

Five rules, taken from `vnpy_riskmanager`'s list (MIT) and re-implemented against
this repo's own ledger, because its rules live in a GUI main engine and count from
an in-memory ``defaultdict`` that forgets everything on restart:

| rule | what it protects | failure mode it prevents |
|---|---|---|
| ``max_order_notional`` | single ticket size | a fat finger or a unit bug |
| ``max_position_weight`` | single-name concentration | the "8 names" book quietly becoming 2 |
| ``max_daily_new_exposure`` | how much is added today | re-running the plan three times |
| ``max_active_orders`` | order count per session | a retry storm against a flaky broker |
| ``daily_loss_halt`` | the account | holding a bad day to "let it come back" |

The daily-loss rule is a **state transition**, not an opinion: once the day's loss
exceeds ``daily_loss_halt`` the caller must stop generating orders and set the kill
switch.  Thresholds live in one frozen dataclass so a limit cannot be edited by
accident somewhere downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RiskLimits:
    max_order_notional: float = 200_000.0
    max_position_weight: float = 0.25
    # ``None`` means "0.95 x equity" -- the gross-exposure ceiling.  A hard-coded
    # 500k blocked the very first full deployment of a 1M account (934k of buys),
    # which taught the operator to raise the limit rather than to read it.  An
    # absolute cap still wins when the operator sets one.
    max_daily_new_exposure: float | None = None
    max_active_orders: int = 12
    max_single_name_shares: int = 100_000
    daily_loss_halt: float = 0.03      # 单日亏损 3% 停机
    min_order_notional: float = 1_000.0

    def validate(self) -> None:
        if self.max_order_notional <= 0:
            raise ValueError("单笔限额必须为正")
        if self.max_daily_new_exposure is not None and self.max_daily_new_exposure <= 0:
            raise ValueError("单日新增仓位上限必须为正（或留空表示 0.95 x 权益）")
        if not 0 < self.max_position_weight <= 1:
            raise ValueError("单票上限必须在 (0, 1] 内")
        if self.max_active_orders <= 0:
            raise ValueError("活动订单上限必须为正")
        if not 0 < self.daily_loss_halt < 1:
            raise ValueError("日亏阈值必须在 (0, 1) 内")


@dataclass
class RiskViolation:
    code: str
    message: str
    scope: str = "order"          # "order" | "portfolio"
    symbol: str | None = None

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "scope": self.scope, "symbol": self.symbol}


@dataclass
class RiskReport:
    ok: bool = True
    violations: list[RiskViolation] = field(default_factory=list)
    totals: dict = field(default_factory=dict)

    def add(self, code: str, message: str, *, scope: str = "order", symbol: str | None = None) -> None:
        self.violations.append(RiskViolation(code=code, message=message, scope=scope, symbol=symbol))
        self.ok = False

    def to_dict(self) -> dict:
        return {
            "ok": bool(self.ok),
            "totals": dict(self.totals),
            "violations": [violation.to_dict() for violation in self.violations],
        }


def check_limits(
    orders,
    *,
    limits: RiskLimits | None = None,
    equity: float | None = None,
    holdings: dict | None = None,
    prices: dict | None = None,
    active_orders: int = 0,
    existing_daily_buys: float = 0.0,
) -> RiskReport:
    """Check a batch of planned orders against the limits, all violations at once.

    ``existing_daily_buys`` is the notional already bought today (from the ledger),
    so re-running a plan cannot spend the daily budget twice.
    """
    cfg = limits or RiskLimits()
    cfg.validate()
    report = RiskReport()
    holdings = holdings or {}
    prices = prices or {}

    gross_buy = 0.0
    for order in orders:
        side = getattr(order, "side", "buy")
        symbol = getattr(order, "symbol", "?")
        shares = int(getattr(order, "shares", 0))
        price = float(getattr(order, "limit_price", getattr(order, "price", 0.0)) or 0.0)
        notional = shares * price

        if notional > cfg.max_order_notional:
            report.add(
                "max_order_notional",
                "%s 单笔 %.0f 元超过上限 %.0f 元" % (symbol, notional, cfg.max_order_notional),
                symbol=symbol,
            )
        if shares > cfg.max_single_name_shares:
            report.add(
                "max_single_name_shares",
                "%s 单笔 %d 股超过上限 %d 股" % (symbol, shares, cfg.max_single_name_shares),
                symbol=symbol,
            )
        if side == "buy":
            gross_buy += notional
            if equity:
                held = float(holdings.get(symbol, 0)) * price
                after = (held + notional) / float(equity)
                if after > cfg.max_position_weight + 1e-9:
                    report.add(
                        "max_position_weight",
                        "%s 买入后权重 %.1f%% 超过 %.0f%%"
                        % (symbol, after * 100, cfg.max_position_weight * 100),
                        symbol=symbol,
                    )

    total_new = gross_buy + float(existing_daily_buys)
    cap = cfg.max_daily_new_exposure
    if cap is None:
        cap = 0.95 * float(equity) if equity else None
    report.totals = {
        "planned_buy_notional": round(gross_buy, 2),
        "existing_daily_buys": round(float(existing_daily_buys), 2),
        "total_new_exposure": round(total_new, 2),
        "daily_new_exposure_cap": None if cap is None else round(float(cap), 2),
        "order_count": len(list(orders)),
        "active_orders": int(active_orders),
    }
    if cap is not None and total_new > cap + 1e-6:
        report.add(
            "max_daily_new_exposure",
            "今日新增仓位 %.0f 元（含已成交 %.0f）超过上限 %.0f 元"
            % (total_new, float(existing_daily_buys), float(cap)),
            scope="portfolio",
        )
    if active_orders >= cfg.max_active_orders:
        report.add(
            "max_active_orders",
            "活动订单 %d 张已达上限 %d 张" % (active_orders, cfg.max_active_orders),
            scope="portfolio",
        )
    return report


def daily_loss_halt(
    equity_now: float,
    equity_prev_close: float,
    *,
    limits: RiskLimits | None = None,
) -> tuple[bool, str]:
    """Has today's loss crossed the circuit breaker?

    Returns ``(halt, message)``.  A non-positive previous equity is treated as
    "cannot judge" and does **not** halt: refusing to trade because an input is
    missing would be a different bug wearing the same clothes.
    """
    cfg = limits or RiskLimits()
    if equity_prev_close is None or equity_prev_close <= 0:
        return False, "无法判断（缺少昨收权益）"
    change = float(equity_now) / float(equity_prev_close) - 1.0
    if change <= -cfg.daily_loss_halt:
        return True, "单日亏损 %.2f%% 触发熔断（阈值 %.0f%%）：停机并停止生成新单" % (
            change * 100,
            cfg.daily_loss_halt * 100,
        )
    return False, "单日变动 %+.2f%%（阈值 -%.0f%%）" % (change * 100, cfg.daily_loss_halt * 100)
