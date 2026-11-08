"""Measure the realtime/advice SLAs the docs quote, so they cannot drift silently.

Every number in `docs/实时行情与交易架构.md` §9 comes from this script.  Until now
it was a throwaway file under `.tmp_*`, which meant the published table could only
be reproduced by whoever still had that file -- exactly the "hand-copied number"
this repo forbids (RUNBOOK §7 rule 3).

What it measures, all against the live free endpoints:

    quote round (4 names, one source)      -- the desk's steady-state cost
    quote round (4 names, cross-check)     -- two feeds in parallel
    quote round (50 names, cross-check)    -- one batch at the per-request ceiling
    calendar build (cold / from disk)      -- the advice command's fixed overhead
    degraded latency (single / both dead)  -- proxy pointed at a closed port
    reconnect latency                      -- first healthy round afterwards
    throttle gap                           -- two rounds on one source (>= 3 s)

Usage::

    python scripts/measure_realtime_sla.py              # table to stdout
    python scripts/measure_realtime_sla.py --json       # + reports/realtime/sla.json

Takes ~20 s.  It deliberately blocks the network for two measurements by setting
HTTPS_PROXY to a closed port; if this machine reaches the internet through a proxy,
pass `--skip-block` so the run does not fight the local proxy configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.realtime import (  # noqa: E402
    CONNECT_TIMEOUT,
    DEFAULT_ATTEMPTS,
    REFRESH_MIN_INTERVAL,
    RETRY_BASE_WAIT,
    SnapshotStore,
    TencentQuoteSource,
    default_source,
    load_calendar,
    refresh,
)

HOLDINGS = ["sh600519", "sz300750", "sz002129", "sh688981"]
TARGETS = {
    "quote_4_single": 3.0,
    "quote_4_cross": 3.0,
    "quote_50_cross": 3.0,
    "degraded_single": 5.0,
    "degraded_cross": 5.0,
    "reconnect": 5.0,
}


def timed_round(source, symbols, *, store=None) -> dict:
    """One refresh round: wall time plus the summary the desk would show."""
    start = time.perf_counter()
    board = refresh(list(symbols), source=source, store=store)
    elapsed = time.perf_counter() - start
    return {
        "elapsed": round(elapsed, 3),
        "quotes": len(board.quotes),
        "tradable": board.tradable,
        "stale": len(board.stale_symbols),
        "degraded": bool(board.degraded),
        "errors": board.errors[:1],
        "warnings": board.warnings[:1],
    }


def measure_degraded(symbols, *, cross_check: bool) -> dict:
    """Block the network for real (proxy -> closed port), then measure the round."""
    saved = {key: os.environ.get(key) for key in ("HTTP_PROXY", "HTTPS_PROXY")}
    os.environ["HTTP_PROXY"] = os.environ["HTTPS_PROXY"] = "http://127.0.0.1:9"
    try:
        return timed_round(default_source(cross_check=cross_check, attempts=DEFAULT_ATTEMPTS), symbols)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def measure_throttle(symbols, *, store=None) -> dict:
    """Two consecutive rounds on one source: measure the *request* spacing itself.

    Timing the rounds understates the gap (the second round starts before the limiter
    sleeps, so its elapsed time hides the first round's tail).  Stamping the transport
    calls measures what the endpoint actually sees.
    """
    from quant_system.realtime import default_transport

    stamps: list[float] = []

    def transport(url: str, **kwargs):
        stamps.append(time.monotonic())
        return default_transport(url, **kwargs)

    source = TencentQuoteSource(transport=transport, attempts=1)
    first = timed_round(source, symbols, store=store)
    second = timed_round(source, symbols, store=store)
    gap = (stamps[1] - stamps[0]) if len(stamps) >= 2 else None
    return {
        "first": first["elapsed"],
        "second": second["elapsed"],
        "request_gap": None if gap is None else round(gap, 3),
        "request_gap_ok": gap is not None and gap >= REFRESH_MIN_INTERVAL - 0.05,
    }


def measure_calendar(cache_dir: Path | None = None) -> dict:
    """Cold build vs the on-disk cache a fresh process gets for free."""
    _ = load_calendar(cache_dir, force=True)
    start = time.perf_counter()
    calendar = load_calendar(cache_dir, force=True)
    cold = time.perf_counter() - start
    from quant_system import realtime as rt

    rt._CALENDAR_CACHE.clear()
    start = time.perf_counter()
    cached = load_calendar(cache_dir)
    warm = time.perf_counter() - start
    return {
        "cold": round(cold, 3),
        "disk": round(warm, 3),
        "days": len(calendar),
        "same_days": len(cached) == len(calendar),
        "first": str(calendar.first),
        "last": str(calendar.last),
    }


def report_markdown(payload: dict) -> str:
    """The exact table shape the architecture doc quotes."""
    rows = [
        ("行情刷新（4 只，单源）", "≤3s", payload["quote_4_single"]["elapsed"], "一次请求"),
        ("行情刷新（4 只，双源交叉）", "≤3s", payload["quote_4_cross"]["elapsed"], "两源并行"),
        ("行情刷新（50 只，单请求上限，双源）", "≤3s", payload["quote_50_cross"]["elapsed"], "一次请求"),
        ("首次调用额外开销（建交易日历）", "—", payload["calendar"]["cold"], "落盘后 %ss" % payload["calendar"]["disk"]),
        ("断线 → 降级可见（单源）", "≤5s", payload["degraded_single"]["elapsed"], "代理指向关闭端口"),
        ("断线 → 降级可见（双源都挂）", "≤5s", payload["degraded_cross"]["elapsed"], "并行取价，不叠加预算"),
        ("断线重连（恢复后第一轮）", "≤5s", payload["reconnect"]["elapsed"], "—"),
        ("两轮请求间隔（同一 source）", "≥3s", payload["throttle"]["request_gap"], "由限频器保证"),
    ]
    lines = ["| 指标 | 目标 | 实测 | 条件 |", "|---|---:|---:|---|"]
    for label, target, value, note in rows:
        # ``--skip-block`` leaves the two network-blocked measurements as None: print
        # "跳过" rather than crashing on ``%.2f`` (it did, silently, from the acceptance
        # runner -- which is how the runner's empty detail line was noticed).
        shown = "跳过" if value is None else "**%.2fs**" % float(value)
        lines.append("| %s | %s | %s | %s |" % (label, target, shown, note))
    return "\n".join(lines)


def run(*, symbols=HOLDINGS, wide=None, store=None, skip_block: bool = False,
        cache_dir: Path | None = None) -> dict:
    if wide is None:
        import glob

        names = {os.path.basename(p).split("_")[0] for p in glob.glob(str(ROOT / "data" / "cache" / "*_tr.parquet"))}
        wide = sorted(names)[:50]
    payload: dict = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "symbols": list(symbols),
        "attempts": DEFAULT_ATTEMPTS,
        "connect_timeout": CONNECT_TIMEOUT,
        "retry_base_wait": RETRY_BASE_WAIT,
        "throttle_seconds": REFRESH_MIN_INTERVAL,
    }
    payload["calendar"] = measure_calendar(cache_dir)
    payload["quote_4_single"] = timed_round(default_source(cross_check=False), symbols, store=store)
    payload["quote_4_cross"] = timed_round(default_source(cross_check=True), symbols, store=store)
    payload["quote_50_cross"] = timed_round(default_source(cross_check=True), wide, store=store)
    if skip_block:
        payload["degraded_single"] = {"elapsed": None, "skipped": "—skip-block"}
        payload["degraded_cross"] = {"elapsed": None, "skipped": "—skip-block"}
    else:
        payload["degraded_single"] = measure_degraded(symbols, cross_check=False)
        payload["degraded_cross"] = measure_degraded(symbols, cross_check=True)
    payload["reconnect"] = timed_round(default_source(cross_check=True), symbols, store=store)
    payload["throttle"] = measure_throttle(symbols, store=store)
    payload["targets"] = dict(TARGETS)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="实测行情/建议链路的 SLA（会真的访问免费接口）")
    parser.add_argument("--json", action="store_true", help="同时写 reports/realtime/sla.json")
    parser.add_argument("--output", default=str(ROOT / "reports" / "realtime" / "sla.json"))
    parser.add_argument("--skip-block", action="store_true", help="跳过断网两项（本机有代理时用）")
    parser.add_argument("--store", default=str(ROOT / "reports" / "realtime" / "sla.sqlite"))
    args = parser.parse_args(argv)

    store_path = Path(args.store)
    store_path.unlink(missing_ok=True)
    store = SnapshotStore(store_path)
    payload = run(store=store, skip_block=args.skip_block)
    print(report_markdown(payload))
    print()
    print("口径：attempts=%d，connect 超时 %ss，重试基础等待 %ss，限频 %ss"
          % (DEFAULT_ATTEMPTS, CONNECT_TIMEOUT, RETRY_BASE_WAIT, REFRESH_MIN_INTERVAL))
    if args.json:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("-> %s" % output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
