"""Tests for the SLA measurement script.

The script itself talks to the live endpoints; these tests cover the parts that
decide *what the published table says*, because a table is only worth quoting if the
code that builds it is pinned:

* ``timed_round`` must report what the desk would have shown;
* ``report_markdown`` must emit one row per documented metric, with the target;
* the JSON payload must be strict-serialisable (it is cited by `docs/`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from quant_system.realtime import parse_payload  # noqa: E402
from measure_realtime_sla import HOLDINGS, TARGETS, report_markdown, timed_round  # noqa: E402


def _payload(symbol: str = "sh600519", when: str = "20261106135930") -> str:
    values = [""] * 88
    values[0], values[1], values[2] = "1", "样本", symbol[2:]
    values[3], values[4], values[5], values[6] = "100.0", "99.0", "99.5", "1000"
    values[9], values[19] = "99.99", "100.01"
    values[30], values[33], values[34], values[37] = when, "101.0", "98.0", "100.0"
    values[47], values[48] = "108.9", "89.1"
    return 'v_%s="%s";' % (symbol, "~".join(values))


class FakeSource:
    name = "fake"

    def __init__(self, symbols):
        self._quotes = parse_payload("".join(_payload(s) for s in symbols), fetched_at=pd.Timestamp("2026-11-06 14:00:00", tz="Asia/Shanghai"))
        self.warnings: list[str] = []
        self.retry_log: list[dict] = []

    def fetch(self, symbols):
        return list(self._quotes), [], []


def test_timed_round_reports_what_the_desk_would_show() -> None:
    symbols = ["sh600519", "sz300750"]
    result = timed_round(FakeSource(symbols), symbols)
    assert result["elapsed"] >= 0.0
    assert result["quotes"] == 2
    assert result["degraded"] is False
    assert result["errors"] == []


def test_report_markdown_covers_every_documented_metric() -> None:
    payload = {
        "quote_4_single": {"elapsed": 0.31},
        "quote_4_cross": {"elapsed": 0.33},
        "quote_50_cross": {"elapsed": 0.34},
        "calendar": {"cold": 0.84, "disk": 0.04},
        "degraded_single": {"elapsed": 4.51},
        "degraded_cross": {"elapsed": 4.50},
        "reconnect": {"elapsed": 0.33},
        "throttle": {"request_gap": 3.0},
    }
    table = report_markdown(payload)
    for marker in ("行情刷新（4 只，单源）", "双源交叉", "50 只", "交易日历",
                   "降级可见（单源）", "降级可见（双源都挂）", "重连", "请求间隔"):
        assert marker in table, marker
    assert table.count("\n|") == 9  # header + separator + 8 metrics
    assert "0.31" in table and "4.51" in table


def test_the_targets_are_the_ones_the_docs_promise() -> None:
    assert TARGETS["quote_4_single"] == 3.0
    assert TARGETS["degraded_single"] == 5.0
    assert TARGETS["reconnect"] == 5.0


def test_the_holdings_sample_covers_the_desk_cases() -> None:
    """A normal name, a suspended name and a 科创板 name: three different statuses."""
    assert "sz002129" in HOLDINGS   # 停牌样本（volume=0/今开 0）
    assert "sh688981" in HOLDINGS   # 科创板 ±20%
    assert "sh600519" in HOLDINGS


def test_the_payload_is_strict_json() -> None:
    payload = {"throttle": {"request_gap": 3.0}, "degraded_single": {"elapsed": None}}
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
