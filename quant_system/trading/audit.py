"""Audit log: immutable, append-only, and answerable after the fact.

Red line #6 asks for six facts on every order: time, symbol, shares, price, the
**signal fingerprint**, the **data snapshot time**, and what triggered it.  The
first four are easy and nearly useless on their own -- the question asked after a
bad fill is "which signal, based on which snapshot, decided this?", and that is
exactly what a fingerprint and a snapshot timestamp answer.

The log is JSONL: append-only, one event per line, ``fsync`` after each write (an
audit entry that exists only in a page cache is not an audit entry).  It contains
no secrets: broker credentials never enter this module, and red line #7 is tested
by scanning the file for anything that looks like a key.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


def signal_fingerprint(as_of: str, weights: dict | list, strategy: str) -> str:
    """Stable hash of *what the signal said*, independent of formatting.

    ``weights`` accepts the ``{symbol: weight}`` mapping or the order rows; both are
    sorted before hashing, so re-serialising the same book cannot change the
    fingerprint and mask a real change.
    """
    if isinstance(weights, dict):
        normalised = sorted((str(k), round(float(v), 8)) for k, v in weights.items())
    else:
        normalised = sorted(
            (str(row.get("symbol")), round(float(row.get("weight", 0.0)), 8)) for row in weights
        )
    payload = json.dumps(
        {"as_of": str(as_of), "strategy": str(strategy), "weights": normalised},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class AuditEvent:
    event: str
    scope: str = "order"
    at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    trade_date: str | None = None
    symbol: str | None = None
    side: str | None = None
    shares: int | None = None
    price: float | None = None
    order_id: str | None = None
    signal_fingerprint: str | None = None
    snapshot_time: str | None = None
    level: int | None = None
    source: str | None = None       # "signal" | "operator" | "kill-switch" | "recon"
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "at": self.at, "event": self.event, "scope": self.scope,
            "trade_date": self.trade_date, "symbol": self.symbol, "side": self.side,
            "shares": self.shares, "price": self.price, "order_id": self.order_id,
            "signal_fingerprint": self.signal_fingerprint, "snapshot_time": self.snapshot_time,
            "level": self.level, "source": self.source, "detail": self.detail,
        }


class AuditLog:
    """Append-only JSONL writer/reader."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: AuditEvent) -> AuditEvent:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return event

    def record(self, event: str, **kwargs) -> AuditEvent:
        return self.append(AuditEvent(event=event, **kwargs))

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        events: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except ValueError:  # pragma: no cover - truncated tail after a crash
                continue
        return events

    def count(self, event: str | None = None) -> int:
        rows = self.events()
        return len(rows) if event is None else sum(1 for row in rows if row.get("event") == event)

    def fingerprints(self) -> set[str]:
        return {row["signal_fingerprint"] for row in self.events() if row.get("signal_fingerprint")}

    def scan_for_secrets(self, patterns: tuple[str, ...] = (
        "password", "passwd", "token", "secret", "api_key", "apikey", "private_key",
    )) -> list[str]:
        """Red line #7: what must never reach the log.  Returns offending lines."""
        hits: list[str] = []
        for line in self.path.read_text(encoding="utf-8").splitlines() if self.path.exists() else []:
            lowered = line.lower()
            if any(pattern in lowered for pattern in patterns):
                hits.append(line)
        return hits
