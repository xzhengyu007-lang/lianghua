"""Stable fingerprints for backtest artifacts.

A published number is only useful if a later run can prove it is the same run.
``metrics.json`` alone is too coarse: two different paths can share a total
return, and a reordering of trades or a shifted equity date would not show up.

The fingerprint hashes canonical, order-independent content:

* the equity curve (dates + values, rounded to a fixed precision),
* the trade log (date, symbol, side, shares, price, reason) sorted,
* the result metadata (costs, window, symbol count) minus volatile keys.

Rounding is deliberate: float noise from BLAS or pandas versions should not
invalidate a genuinely identical run, but a real change in a single trade must.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

# Keys that legitimately differ between two identical runs (timestamps, paths).
VOLATILE_KEYS = {"generated_at", "output", "output_dir", "run_id", "elapsed_seconds"}
PRECISION = 8


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, PRECISION)
    if isinstance(value, dict):
        return {str(k): _round(v) for k, v in sorted(value.items()) if k not in VOLATILE_KEYS}
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    return value


def _digest(payload: Any) -> str:
    text = json.dumps(_round(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def equity_digest(equity: pd.Series) -> str:
    series = equity.dropna().sort_index()
    rows = [[str(pd.Timestamp(index).date()), float(value)] for index, value in series.items()]
    return _digest(rows)


def trades_digest(trades: pd.DataFrame) -> str:
    if trades is None or len(trades) == 0:
        return _digest([])
    columns = [c for c in ("date", "symbol", "side", "shares", "price", "reason") if c in trades.columns]
    frame = trades.loc[:, columns].copy()
    frame["date"] = frame["date"].astype(str)
    for column in ("symbol", "side", "reason"):
        if column in frame.columns:
            frame[column] = frame[column].astype(str)
    for column in ("shares", "price"):
        if column in frame.columns:
            frame[column] = frame[column].astype(float).round(PRECISION)
    frame = frame.sort_values(columns, kind="mergesort").reset_index(drop=True)
    return _digest(frame.to_dict(orient="records"))


def metadata_digest(metadata: dict) -> str:
    return _digest(metadata or {})


def fingerprint(equity: pd.Series, trades: pd.DataFrame, metadata: dict) -> dict:
    parts = {
        "equity": equity_digest(equity),
        "trades": trades_digest(trades),
        "metadata": metadata_digest(metadata),
    }
    combined = _digest(parts)
    return {"parts": parts, "combined": combined, "bars": int(len(equity.dropna()))}


def compare(first: dict, second: dict) -> dict:
    """Field-by-field comparison, never collapsing a mismatch into one boolean."""
    a, b = first.get("parts", {}), second.get("parts", {})
    keys = sorted(set(a) | set(b))
    return {
        "equal": first.get("combined") == second.get("combined"),
        "differences": [key for key in keys if a.get(key) != b.get(key)],
        "first": first.get("combined"),
        "second": second.get("combined"),
    }
