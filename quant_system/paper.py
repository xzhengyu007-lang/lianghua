"""Paper-trading ledger: signal -> orders -> fills -> reconciliation.

The backtester is a research tool; this module is the operating loop that sits
between the daily signal and a broker account. It deliberately holds no broker
credentials and makes no network calls: it reads and writes a versioned JSON
ledger, so the same code works for a manual account, a simulated account, or a
replay of the backtester's own trades.

Design rules:

* Money and shares are updated only by explicit fills. Planning orders never
  mutates the ledger.
* Sells are planned before buys and fund them, and buys cannot exceed the cash
  raised plus the existing balance.
* Every fill is checked against the held position: you cannot sell more than
  you own and you cannot buy with money you do not have.
* Each fill records commission, stamp duty, transfer fee and realized PnL, so
  the ledger reconciles against a broker statement line by line.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date as date_cls
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CostConfig

LEDGER_VERSION = 1


@dataclass
class Position:
    shares: int = 0
    avg_cost: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.shares > 0


@dataclass
class Ledger:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    as_of: str | None = None
    last_signal_date: str | None = None
    trades: list[dict] = field(default_factory=list)
    version: int = LEDGER_VERSION

    # ---------- persistence -------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "as_of": self.as_of,
            "last_signal_date": self.last_signal_date,
            "cash": round(float(self.cash), 2),
            "positions": {
                symbol: {"shares": int(pos.shares), "avg_cost": round(float(pos.avg_cost), 6)}
                for symbol, pos in sorted(self.positions.items())
                if pos.shares
            },
            "trades": self.trades,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Ledger":
        if int(payload.get("version", 0)) != LEDGER_VERSION:
            raise ValueError(
                f"unsupported ledger version {payload.get('version')!r}; expected {LEDGER_VERSION}"
            )
        positions = {
            symbol: Position(shares=int(row["shares"]), avg_cost=float(row["avg_cost"]))
            for symbol, row in (payload.get("positions") or {}).items()
        }
        return cls(
            cash=float(payload["cash"]),
            positions=positions,
            as_of=payload.get("as_of"),
            last_signal_date=payload.get("last_signal_date"),
            trades=list(payload.get("trades") or []),
            version=LEDGER_VERSION,
        )

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temp.replace(target)
        return target

    @classmethod
    def load(cls, path: Path | str) -> "Ledger":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ---------- accounting --------------------------------------------------
    def market_value(self, prices: pd.Series) -> float:
        total = 0.0
        for symbol, pos in self.positions.items():
            if pos.shares <= 0:
                continue
            price = prices.get(symbol, np.nan)
            if pd.isna(price):
                raise ValueError(f"missing price for held symbol {symbol}")
            total += pos.shares * float(price)
        return total

    def equity(self, prices: pd.Series) -> float:
        return float(self.cash + self.market_value(prices))


def _as_date(value: str | date_cls | pd.Timestamp) -> str:
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if isinstance(value, date_cls):
        return str(value)
    return str(pd.Timestamp(value).date())


def plan_orders(
    ledger: Ledger,
    target_weights: pd.Series,
    prices: pd.Series,
    *,
    lot_size: int = 100,
    min_order_value: float = 1_000.0,
    cash_buffer: float = 0.0,
    costs: CostConfig | None = None,
    stale_symbols: set[str] | None = None,
) -> pd.DataFrame:
    """Turn target weights into share orders against the current ledger.

    Target shares are sized on *current equity*, not on the cash balance, which
    is what a real account needs. Sells are netted first so their proceeds can
    fund the buys. Buys are then capped by cash actually available after fees,
    so the last order can be a few lots smaller than the signal's estimate; the
    signal is a sizing guide, the ledger is the executable book.

    ``stale_symbols`` names positions whose latest bar predates the signal date
    (suspension or delisting). They are neither valued nor traded: using a stale
    close would size orders against a price that cannot be executed today.
    """
    cfg = costs or CostConfig()
    stale = set(stale_symbols or ())
    weights = target_weights.astype(float).fillna(0.0)
    # A held name that is absent from the target must be liquidated, otherwise
    # a dropped position would stay in the book forever.
    missing = [s for s in ledger.positions if s not in weights.index]
    if missing:
        weights = pd.concat([weights, pd.Series(0.0, index=missing)])
    weights = weights[weights.index.notna()]
    equity = ledger.equity(prices)
    available = ledger.cash * (1.0 - cash_buffer)

    rows: list[dict] = []
    for symbol, weight in weights.sort_values(ascending=False).items():
        if symbol in stale:
            continue
        price = prices.get(symbol, np.nan)
        if pd.isna(price) or float(price) <= 0:
            raise ValueError(f"missing or non-positive price for {symbol}")
        target_shares = int(np.floor(max(weight, 0.0) * equity / float(price) / lot_size) * lot_size)
        held = ledger.positions.get(symbol, Position()).shares
        delta = target_shares - held
        if delta == 0:
            continue
        if delta < 0:
            shares = int(min(-delta, held))
            if shares <= 0:
                continue
            notional = shares * float(price)
            rows.append(
                {
                    "symbol": symbol,
                    "side": "sell",
                    "shares": shares,
                    "price": round(float(price), 4),
                    "notional": round(notional, 2),
                    "reason": "target_rebalance" if target_shares > 0 else "exit",
                }
            )
            available += notional * (1 - cfg.commission_rate - cfg.stamp_duty_rate - cfg.transfer_fee_rate)

    buys: list[dict] = []
    for symbol, weight in weights[weights > 1e-9].sort_values(ascending=False).items():
        if symbol in stale:
            continue
        price = prices.get(symbol, np.nan)
        if pd.isna(price) or float(price) <= 0:
            continue
        target_shares = int(np.floor(weight * equity / float(price) / lot_size) * lot_size)
        held = ledger.positions.get(symbol, Position()).shares
        delta = target_shares - held
        if delta <= 0:
            continue
        unit_cost = float(price) * (1 + cfg.commission_rate + cfg.transfer_fee_rate)
        affordable = int(np.floor(available / unit_cost / lot_size) * lot_size)
        shares = int(min(delta, affordable))
        if shares <= 0 or shares * float(price) < min_order_value:
            continue
        available -= shares * unit_cost
        buys.append(
            {
                "symbol": symbol,
                "side": "buy",
                "shares": shares,
                "price": round(float(price), 4),
                "notional": round(shares * float(price), 2),
                "reason": "target_entry",
            }
        )

    orders = pd.DataFrame(rows + buys)
    if orders.empty:
        return orders
    return orders.sort_values(["side", "symbol"], ascending=[False, True]).reset_index(drop=True)


def apply_fills(
    ledger: Ledger,
    fills: pd.DataFrame,
    *,
    date: str | pd.Timestamp,
    costs: CostConfig | None = None,
) -> Ledger:
    """Apply executed fills and return an updated ledger (input is not mutated)."""
    cfg = costs or CostConfig()
    trade_date = _as_date(date)
    cash = float(ledger.cash)
    positions = {
        symbol: Position(pos.shares, pos.avg_cost) for symbol, pos in ledger.positions.items()
    }
    trades = [dict(row) for row in ledger.trades]

    missing = [c for c in ("symbol", "price", "shares") if c not in fills.columns]
    if missing:
        raise ValueError(
            "fills file needs columns symbol, price, shares (optional: side, note); "
            f"missing {missing}"
        )

    for _, fill in fills.iterrows():
        symbol = str(fill["symbol"])
        price = float(fill["price"])
        shares = int(fill["shares"])
        # Brokers usually export only symbol/price/shares. When `side` is absent,
        # infer it: a symbol not currently held can only be a buy.
        raw_side = fill.get("side", "")
        if raw_side is None or (isinstance(raw_side, float) and np.isnan(raw_side)) or str(raw_side).strip() == "":
            side = "sell" if positions.get(symbol, Position()).shares > 0 else "buy"
        else:
            side = str(raw_side).strip().lower()
        if shares <= 0:
            raise ValueError(f"{symbol}: fill shares must be positive")
        if price <= 0:
            raise ValueError(f"{symbol}: fill price must be positive")
        pos = positions.setdefault(symbol, Position())
        notional = shares * price
        if side == "buy":
            commission = max(cfg.min_commission, notional * cfg.commission_rate)
            transfer = notional * cfg.transfer_fee_rate
            stamp = 0.0
            total = notional + commission + transfer
            if total > cash + 1e-6:
                raise ValueError(
                    f"{symbol}: buy needs {total:,.2f} but only {cash:,.2f} is available"
                )
            previous = pos.avg_cost * pos.shares
            pos.shares += shares
            pos.avg_cost = (previous + total) / pos.shares
            cash -= total
            realized = 0.0
        elif side == "sell":
            if shares > pos.shares:
                raise ValueError(
                    f"{symbol}: cannot sell {shares} shares, only {pos.shares} held"
                )
            commission = max(cfg.min_commission, notional * cfg.commission_rate)
            transfer = notional * cfg.transfer_fee_rate
            stamp = notional * cfg.stamp_duty_rate
            total = notional - commission - transfer - stamp
            realized = shares * (price - pos.avg_cost) - commission - transfer - stamp
            cash += total
            pos.shares -= shares
            if pos.shares == 0:
                pos.avg_cost = 0.0
        else:
            raise ValueError(f"{symbol}: unknown side {side!r}")

        trades.append(
            {
                "date": trade_date,
                "symbol": symbol,
                "side": side,
                "shares": shares,
                "price": round(price, 4),
                "notional": round(notional, 2),
                "commission": round(commission, 2),
                "stamp_duty": round(stamp, 2),
                "transfer_fee": round(transfer, 2),
                "cash_flow": round(total if side == "buy" else -total, 2),
                "realized_pnl": round(realized, 2),
                "note": str(fill.get("note", "") or ""),
            }
        )

    for symbol in [s for s, p in positions.items() if p.shares == 0]:
        del positions[symbol]

    return Ledger(
        cash=cash,
        positions=positions,
        as_of=trade_date,
        last_signal_date=ledger.last_signal_date,
        trades=trades,
        version=LEDGER_VERSION,
    )


def reconcile(
    ledger: Ledger,
    broker_positions: dict[str, int],
    *,
    broker_cash: float | None = None,
    tolerance: float = 0.0,
) -> pd.DataFrame:
    """Compare the ledger with a broker statement and return the differences."""
    symbols = sorted(set(ledger.positions) | set(broker_positions))
    rows = []
    for symbol in symbols:
        ours = ledger.positions.get(symbol, Position()).shares
        theirs = int(broker_positions.get(symbol, 0))
        if ours == theirs:
            continue
        rows.append(
            {
                "item": symbol,
                "ledger": ours,
                "broker": theirs,
                "difference": ours - theirs,
            }
        )
    if broker_cash is not None and abs(float(broker_cash) - float(ledger.cash)) > tolerance:
        rows.append(
            {
                "item": "cash",
                "ledger": round(float(ledger.cash), 2),
                "broker": round(float(broker_cash), 2),
                "difference": round(float(ledger.cash) - float(broker_cash), 2),
            }
        )
    return pd.DataFrame(rows, columns=["item", "ledger", "broker", "difference"])


def positions_frame(ledger: Ledger, prices: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    equity = ledger.equity(prices) if prices is not None else float("nan")
    for symbol, pos in sorted(ledger.positions.items()):
        price = prices.get(symbol, np.nan) if prices is not None else np.nan
        value = pos.shares * float(price) if pd.notna(price) else np.nan
        rows.append(
            {
                "symbol": symbol,
                "shares": pos.shares,
                "avg_cost": round(pos.avg_cost, 4),
                "price": round(float(price), 4) if pd.notna(price) else np.nan,
                "market_value": round(value, 2) if pd.notna(value) else np.nan,
                "weight": round(value / equity, 6) if pd.notna(value) and equity else np.nan,
            }
        )
    return pd.DataFrame(
        rows, columns=["symbol", "shares", "avg_cost", "price", "market_value", "weight"]
    )


def summarize(ledger: Ledger, prices: pd.Series | None = None) -> dict:
    payload = {
        "as_of": ledger.as_of,
        "cash": round(float(ledger.cash), 2),
        "positions": len(ledger.positions),
        "trades": len(ledger.trades),
        "realized_pnl": round(
            float(sum(float(t.get("realized_pnl", 0.0)) for t in ledger.trades)), 2
        ),
    }
    if prices is not None:
        payload["market_value"] = round(ledger.market_value(prices), 2)
        payload["equity"] = round(ledger.equity(prices), 2)
    return payload
