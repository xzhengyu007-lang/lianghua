"""Transaction cost model for mainland A-share cash equities."""

from __future__ import annotations

from dataclasses import dataclass

from .config import CostConfig


@dataclass(frozen=True)
class CostBreakdown:
    commission: float
    stamp_duty: float
    transfer_fee: float
    slippage: float

    @property
    def fees(self) -> float:
        """Explicit cash fees. Slippage is already inside the executed price."""
        return self.commission + self.stamp_duty + self.transfer_fee

    @property
    def total(self) -> float:
        """All-in cost including slippage; use only for reporting, never for cash."""
        return self.fees + self.slippage


def impact_bps(shares: int, adv_shares: float, config: CostConfig) -> float:
    """Square-root impact in basis points for one order.

    ``adv_shares`` is trailing average daily volume already expressed in
    shares. The model is intentionally one-parameter and monotone: doubling the
    order raises impact by sqrt(2), which is the shape every published
    meta-analysis finds for equity markets.
    """
    if config.impact_coefficient_bps <= 0 or shares <= 0 or adv_shares <= 0:
        return 0.0
    participation = min(float(shares) / float(adv_shares), 1.0)
    return float(config.impact_coefficient_bps * participation ** 0.5)


def execution_price(
    side: str,
    open_price: float,
    config: CostConfig,
    *,
    shares: int | None = None,
    adv_shares: float | None = None,
) -> float:
    """Apply adverse slippage and optional market impact to the next-day open."""
    if side not in {"buy", "sell"}:
        raise ValueError("side must be buy or sell")
    slip = config.slippage_bps / 10_000
    if shares is not None and adv_shares is not None:
        slip += impact_bps(shares, adv_shares, config) / 10_000
    return float(open_price * (1 + slip if side == "buy" else 1 - slip))


def transaction_costs(
    side: str,
    shares: int,
    price: float,
    *,
    open_price: float,
    config: CostConfig,
) -> CostBreakdown:
    if shares < 0:
        raise ValueError("shares cannot be negative")
    notional = shares * price
    commission = max(config.min_commission, notional * config.commission_rate) if shares else 0.0
    stamp = notional * config.stamp_duty_rate if side == "sell" else 0.0
    transfer = notional * config.transfer_fee_rate
    slip = shares * abs(price - open_price)
    return CostBreakdown(commission, stamp, transfer, slip)
