"""Trading layer: tiered automation around the daily signal.

The pieces, in the order an order travels through them:

    signal ──► plan (paper.PlanOrders) ──► validate ──► limits ──► idempotency
                                                                    │
                                              gate (L0..L3) ◄───────┘
                                                    │
                                    orders file (L2) / nothing (L1)
                                                    │
                                              fills ──► ledger ──► recon

Why a package and not one file: each of those steps has exactly one job and one
failure mode, and the red lines in ``docs/实时行情与交易架构.md`` §11 map to them
one-to-one.  A single 800-line module would make "the idempotency check ran before
the limit check" impossible to test.

Nothing in here talks to a broker.  There is no network code in this package at
all -- the strongest possible form of "L3 默认关闭": the capability does not exist
yet, and ``gates.py`` is the only thing that could ever authorise adding it.
"""

from .audit import AuditLog, AuditEvent, signal_fingerprint
from .gates import GateState, TradingGate, TradingLevel
from .idempotency import IdempotencyBook
from .order_state import Order, OrderState, can_transition
from .orders_file import write_broker_files
from .recon import (
    ReconciliationReport,
    append_equity_point,
    daily_statement,
    equity_curve,
    reconcile,
    write_alerts,
)
from .risk_rules import RiskLimits, RiskViolation, check_limits, daily_loss_halt
from .pipeline import default_paths, plan_session
from .validate import ValidationResult, validate_order

__all__ = [
    "AuditEvent",
    "AuditLog",
    "GateState",
    "IdempotencyBook",
    "Order",
    "OrderState",
    "ReconciliationReport",
    "RiskLimits",
    "RiskViolation",
    "TradingGate",
    "TradingLevel",
    "ValidationResult",
    "can_transition",
    "check_limits",
    "daily_loss_halt",
    "daily_statement",
    "reconcile",
    "signal_fingerprint",
    "validate_order",
    "write_broker_files",
    "write_alerts",
    "append_equity_point",
    "equity_curve",
]
