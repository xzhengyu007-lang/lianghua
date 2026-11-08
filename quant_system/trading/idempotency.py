"""Idempotency: one live order per (trade_date, symbol, side).  On disk.

This is the single most dangerous place in the whole system.  A crash between
"orders written" and "orders acknowledged", a retried CLI call, two terminal tabs
-- each one re-plans the same intent and, without a ledger of *intentions*, sends
the same buy twice.  The paper ledger cannot catch it: two legitimate-looking buys
of the same name are indistinguishable from a doubled order once they land.

So the key is persisted before anything is submitted, and released only when the
order reaches a terminal state.  The file is JSONL -- append-only, human-readable,
greppable after an incident -- and every write is followed by ``fsync`` because an
order book that silently loses its last line is worse than one that refuses.

Design notes:

* Key is ``(trade_date, symbol, side)``.  Two buys of the same name on the same day
  are forbidden *by design*: the strategy rebalances once a day, so a second buy
  means a bug, not a strategy.
* ``release`` is explicit; nothing expires on a timer.  A stale key is visible in
  ``list_active`` and the operator resolves it, instead of the machine guessing.
* Re-planning after a terminal state is allowed -- that is how a rejected order is
  retried deliberately.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .order_state import OrderState


@dataclass
class Reservation:
    key: tuple[str, str, str]
    order_id: str
    state: OrderState
    created_at: str

    def to_dict(self) -> dict:
        return {
            "trade_date": self.key[0],
            "symbol": self.key[1],
            "side": self.key[2],
            "order_id": self.order_id,
            "state": self.state.value,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Reservation":
        return cls(
            key=(payload["trade_date"], payload["symbol"], payload["side"]),
            order_id=payload["order_id"],
            state=OrderState(payload["state"]),
            created_at=payload.get("created_at", ""),
        )


class DuplicateOrderError(RuntimeError):
    """Raised when a live order already exists for the same key."""

    def __init__(self, key: tuple[str, str, str], existing: Reservation) -> None:
        super().__init__(
            "%s %s %s 已有一张有效订单（%s，状态 %s）；"
            "重复生成已被拒绝——同一交易日同一标的同一方向只允许一张"
            % (key[0], key[1], key[2], existing.order_id, existing.state.value)
        )
        self.key = key
        self.existing = existing


class IdempotencyBook:
    """Append-only intent ledger with an in-memory index of the live keys."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._active: dict[tuple[str, str, str], Reservation] = {}
        self._load()

    # ---------- persistence ------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except ValueError:  # pragma: no cover - a truncated last line
                continue
            reservation = Reservation.from_dict(payload)
            if reservation.state.active:
                self._active[reservation.key] = reservation
            else:
                self._active.pop(reservation.key, None)

    def _append(self, payload: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    # ---------- api --------------------------------------------------------
    def reserve(self, *, trade_date: str, symbol: str, side: str, order_id: str) -> Reservation:
        """Claim the key, or refuse with the live order that already holds it."""
        key = (trade_date, symbol, side)
        existing = self._active.get(key)
        if existing is not None:
            raise DuplicateOrderError(key, existing)
        reservation = Reservation(
            key=key, order_id=order_id, state=OrderState.SUBMITTING,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        self._append(reservation.to_dict())
        self._active[key] = reservation
        return reservation

    def update(self, *, key: tuple[str, str, str], state: OrderState, order_id: str | None = None) -> None:
        reservation = self._active.get(key)
        order_id = order_id or (reservation.order_id if reservation else "")
        updated = Reservation(
            key=key,
            order_id=order_id,
            state=state,
            created_at=reservation.created_at if reservation else "",
        )
        self._append(updated.to_dict())
        if state.active:
            self._active[key] = updated
        else:
            self._active.pop(key, None)

    def release(self, key: tuple[str, str, str], *, state: OrderState = OrderState.CANCELLED) -> None:
        self.update(key=key, state=state)

    def active(self) -> dict[tuple[str, str, str], Reservation]:
        return dict(self._active)

    def is_reserved(self, *, trade_date: str, symbol: str, side: str) -> bool:
        return (trade_date, symbol, side) in self._active

    def __len__(self) -> int:
        return len(self._active)
