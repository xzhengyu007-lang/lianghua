"""Order state machine (six states, borrowed from vnpy's `Status`, MIT).

`paper.py` only knows fills: it goes from "planned" to "applied".  A real order
has a life in between -- submitted, partially filled, cancelled -- and the desk
must be able to say which of those it is in, because "unknown" is what makes an
operator place the same order twice.

The transition table is the whole point: anything not listed raises.  A state
machine that accepts any transition is just a string field.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum


class OrderState(str, Enum):
    """The six states an A-share order can be in."""

    SUBMITTING = "submitting"     # handed to the broker/file, no ack yet
    NOTTRADED = "nottraded"       # accepted, zero filled
    PARTTRADED = "parttraded"     # accepted, partially filled
    ALLTRADED = "alltraded"       # fully filled
    CANCELLED = "cancelled"
    REJECTED = "rejected"

    @property
    def terminal(self) -> bool:
        return self in (OrderState.ALLTRADED, OrderState.CANCELLED, OrderState.REJECTED)

    @property
    def active(self) -> bool:
        """Counts against the "one live order per (date, symbol, side)" rule."""
        return not self.terminal


ALLOWED: dict[OrderState, frozenset[OrderState]] = {
    OrderState.SUBMITTING: frozenset(
        {OrderState.NOTTRADED, OrderState.PARTTRADED, OrderState.ALLTRADED,
         OrderState.REJECTED, OrderState.CANCELLED}
    ),
    OrderState.NOTTRADED: frozenset(
        {OrderState.PARTTRADED, OrderState.ALLTRADED, OrderState.CANCELLED, OrderState.REJECTED}
    ),
    OrderState.PARTTRADED: frozenset(
        {OrderState.PARTTRADED, OrderState.ALLTRADED, OrderState.CANCELLED, OrderState.REJECTED}
    ),
    OrderState.ALLTRADED: frozenset(),
    OrderState.CANCELLED: frozenset(),
    OrderState.REJECTED: frozenset(),
}


def can_transition(source: OrderState, target: OrderState) -> bool:
    return target in ALLOWED.get(source, frozenset())


@dataclass
class Order:
    """One intended trade, with the evidence needed to audit it later.

    ``signal_fingerprint`` and ``snapshot_time`` are not decoration: they answer
    "which signal produced this order, and how old was the price it used", which
    is the first question asked after any bad fill.
    """

    order_id: str
    trade_date: str
    symbol: str
    side: str                      # "buy" | "sell"
    shares: int
    limit_price: float
    state: OrderState = OrderState.SUBMITTING
    filled_shares: int = 0
    filled_price: float = 0.0
    reason: str = ""
    signal_fingerprint: str = ""
    snapshot_time: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    note: str = ""

    def __post_init__(self) -> None:
        if self.side not in ("buy", "sell"):
            raise ValueError(f"{self.symbol}: side must be buy or sell, got {self.side!r}")
        if self.shares <= 0:
            raise ValueError(f"{self.symbol}: shares must be positive")
        if self.limit_price <= 0:
            raise ValueError(f"{self.symbol}: limit price must be positive")

    @property
    def key(self) -> tuple[str, str, str]:
        """The idempotency key: one live order per (day, symbol, side)."""
        return (self.trade_date, self.symbol, self.side)

    @property
    def notional(self) -> float:
        return float(self.shares) * float(self.limit_price)

    def transition(self, target: OrderState, *, filled_shares: int | None = None,
                   filled_price: float | None = None, note: str = "") -> "Order":
        if not can_transition(self.state, target):
            raise ValueError(
                f"{self.order_id}: illegal transition {self.state.value} -> {target.value}"
            )
        filled = self.filled_shares if filled_shares is None else int(filled_shares)
        if filled > self.shares:
            raise ValueError(f"{self.order_id}: filled {filled} > ordered {self.shares}")
        if target is OrderState.ALLTRADED and filled != self.shares:
            raise ValueError(f"{self.order_id}: ALLTRADED needs filled == shares")
        if target in (OrderState.NOTTRADED, OrderState.CANCELLED) and filled not in (0, self.shares):
            raise ValueError(f"{self.order_id}: {target.value} with a partial fill is PARTTRADED")
        return replace(
            self,
            state=target,
            filled_shares=filled,
            filled_price=float(filled_price if filled_price is not None else self.filled_price),
            note=(note or self.note),
        )

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id, "trade_date": self.trade_date, "symbol": self.symbol,
            "side": self.side, "shares": self.shares, "limit_price": self.limit_price,
            "state": self.state.value, "filled_shares": self.filled_shares,
            "filled_price": self.filled_price, "reason": self.reason,
            "signal_fingerprint": self.signal_fingerprint, "snapshot_time": self.snapshot_time,
            "created_at": self.created_at, "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Order":
        data = dict(payload)
        data["state"] = OrderState(data.get("state", OrderState.SUBMITTING.value))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
